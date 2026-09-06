"""player_style.py —— per-player 操作画像 + 战术画像（route B Phase 2 / 2.5）。

读取 data/player_behavior.parquet（per-round-per-player 行为）+ data/player_util.parquet
（per-throw 道具）+ data/duel_dataset.parquet（per-kill 战术），对每个选手按情景
（side × round_class × score_bucket × buy_tier）统计两层签名：

【操作层】（行为）
  - 侵略性：first_contact 率 / 首杀率 / 首死率（分边、分比分、分阶段、分装备）
  - 武器：主武器偏好（步枪/狙/冲锋）+ awp 击杀占比
  - 买枪优先级：eco/force 局的 equip_offset（相对队伍均值，正 = 被塞枪）
  - 道具：分类型每回合投掷率 + 时序（早/中/晚）

【战术层】（duel_dataset 的 per-kill 维度，语义见 engagements.py）
  - 交火距离分布：近(<350)/中/远(>1000) —— brawler vs 架点手
  - 人数差局面：man_down/even/man_up 占比 + 残局(man_down)胜率
  - 抱团程度：subject_nearby = 身边队友数 —— 团队 vs 独狼
  - 先手/被先手：subject_facing & ~opponent_facing —— 谁先架到谁
  - 盲打率：subject_flashed —— 被闪时还在打
  - 对手位置：opponent_place —— 他杀的人在哪（进点 vs 架点）

产出 reports/player_style_memory.json，并打印 donk 画像 + 与 sh1ro/magixx/zont1x 对照。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config

# 比分处境归并
BEHIND = ["behind1-2", "behind3+"]
AHEAD = ["ahead1-2", "ahead3+"]

# eco/force 档（非全起）——买枪优先级看这里
ECON_TIERS = {"eco0", "eco_armor", "force_low", "force_high"}

# trade 窗口（同 player_behavior）：死亡后 5s 内队友补枪算 trade
TRADE_TICKS = 5 * 64


def _rate(vals: pd.Series) -> float:
    v = vals.dropna()
    return float(v.mean()) if len(v) >= 5 else float("nan")


def load_behavior(path=None) -> pd.DataFrame:
    return pd.read_parquet(path or (config.DATA_DIR / "player_behavior.parquet"))


def load_util(path=None) -> pd.DataFrame:
    return pd.read_parquet(path or (config.DATA_DIR / "player_util.parquet"))


def load_duels(steamids: set[str]) -> pd.DataFrame:
    """duel_dataset 里这些选手的 per-kill 战术行（全字段）。"""
    dd = pd.read_parquet(config.DATA_DIR / "duel_dataset.parquet")
    return dd[dd["player_id"].astype(str).isin(steamids)]


def _weapon_family(w: str) -> str:
    w = str(w).lower()
    if "awp" in w or "scout" in w:
        return "awp"
    if any(x in w for x in ("ak47", "m4", "galil", "famas", "sg", "aug")):
        return "rifle"
    if any(x in w for x in ("mp", "p90", "bizon", "ump", "mac", "nova", "xm", "mag", "sawed")):
        return "smg_shotgun"
    if any(x in w for x in ("deagle", "glock", "usp", "p2000", "cz", "tec9", "fiveseven", "p250")):
        return "pistol"
    if "knife" in w:
        return "knife"
    return "other"


def _dist_bucket(d: float) -> str:
    if d < 350:
        return "close"
    if d < 1000:
        return "mid"
    return "long"


def _cond(vals: pd.Series, mask: pd.Series) -> float:
    return _rate(vals[mask])


def duel_tactics(duels: pd.DataFrame) -> dict:
    """一个选手的战术签名（per-kill 维度）。"""
    out: dict = {}
    if len(duels) < 5:
        return out

    # 交火距离：近身 brawler vs 远距架点
    dist = duels["distance"].map(_dist_bucket)
    out["duel_distance"] = {k: int(v) for k, v in dist.value_counts().items()}

    # 人数差局面 + 残局胜率
    md = duels["man_diff"]
    out["duels_man_down"] = round(float((md < 0).mean()), 3)
    out["duels_man_even"] = round(float((md == 0).mean()), 3)
    out["duels_man_up"] = round(float((md > 0).mean()), 3)
    down = duels[md < 0]
    if len(down) >= 5:
        out["clutch_winrate"] = round(float(down["label"].mean()), 3)
        out["n_clutch"] = int(len(down))

    # 抱团：身边队友数（可 trade 支持），高=团队打，低=独狼
    out["mean_teammates_nearby"] = round(float(duels["subject_nearby"].mean()), 2)
    out["lone_duel_rate"] = round(float((duels["subject_nearby"] == 0).mean()), 3)

    # 先手 vs 被先手 vs 正面
    sf = duels["subject_facing"].astype(bool)
    of = duels["opponent_facing"].astype(bool)
    out["got_drop"] = round(float((sf & ~of).mean()), 3)   # 我先架到对方
    out["caught"] = round(float((of & ~sf).mean()), 3)     # 被对方先架到
    out["head_on"] = round(float((sf & of).mean()), 3)     # 正面

    # 盲打：被闪时还在打
    out["fights_flashed"] = round(float(duels["subject_flashed"].mean()), 3)
    out["flashes_opponent"] = round(float(duels["opponent_flashed"].mean()), 3)

    # 对手位置：他杀的人在哪（进点 vs 架点）
    op = duels["opponent_place"].astype(str).value_counts()
    out["opponent_place"] = {k: int(v) for k, v in op.head(5).items()}
    return out


def load_kills(path=None) -> pd.DataFrame:
    return pd.read_parquet(path or (config.DATA_DIR / "player_kills.parquet"))


def entry_profile(kills: pd.DataFrame, sids: list[str], n_rounds: int) -> dict:
    """per-kill 维度：HLTV 的 Opening/Entrying/Trading + 首杀时机（opener vs entry）。

    Opening   —— 首杀率/首死率/首杀成功率（opening duel success）
    Entrying  —— 死亡被队友 trade 的占比（traded death，entry 的"可被补"）
    Trading   —— 自己击杀里补队友的占比（trade kill）
    首杀时机  —— 他卷入 round 首杀时，发生在回合的早/中/晚（早=opener 默认阶段，
                 中/晚=entry 执行阶段）

    sids 是同一选手的全部 steamid（按 name 合并，处理跨 demo 多 steamid 分裂）。
    """
    sids = {str(s) for s in sids}
    out: dict = {}
    my_kills = kills[kills["attacker"].isin(sids)]
    my_deaths = kills[kills["victim"].isin(sids)]
    if not len(my_kills) and not len(my_deaths):
        return out

    ok = int((my_kills["is_first_kill"] == True).sum())
    od = int((my_deaths["is_first_kill"] == True).sum())
    out["opening_kills"] = ok
    out["opening_deaths"] = od
    out["opening_duel_success"] = round(ok / (ok + od), 3) if (ok + od) else None
    out["opening_participation"] = round((ok + od) / n_rounds, 3) if n_rounds else None
    out["trade_kill_rate"] = round(float(my_kills["trade"].mean()), 3) if len(my_kills) else None

    # Entrying：死亡被队友补枪的占比（5s 内同边队友杀掉杀我的那个人）。
    # 必须限定同一 demo：不同 demo 的 tick 各自从零起，跨 demo 比对会假阳性。
    # 按 tick 排序后用 searchsorted 圈出 5s 窗口，避免 O(deaths*kills) 广播。
    ks = kills.sort_values("tick")
    kt_s = ks["tick"].to_numpy()
    kv_s = ks["victim"].to_numpy()
    kas_s = ks["attacker_side"].to_numpy()
    kdp_s = ks["demo_path"].to_numpy()
    traded = 0
    for T, E, side, demo in my_deaths[["tick", "attacker", "victim_side", "demo_path"]].itertuples(index=False):
        lo = int(np.searchsorted(kt_s, T, side="right"))
        hi = int(np.searchsorted(kt_s, T + TRADE_TICKS, side="right"))
        if lo >= hi:
            continue
        seg = slice(lo, hi)
        if ((kdp_s[seg] == demo) & (kas_s[seg] == side) & (kv_s[seg] == E)).any():
            traded += 1
    out["traded_death_rate"] = round(traded / len(my_deaths), 3) if len(my_deaths) else None
    out["n_deaths"] = int(len(my_deaths))

    # 首杀时机：我卷入 first kill 时，rel_tick / round_len_ticks
    fk = kills[(kills["is_first_kill"] == True)
               & (kills["attacker"].isin(sids) | kills["victim"].isin(sids))]
    if len(fk) and "round_len_ticks" in fk.columns and (fk["round_len_ticks"] > 0).any():
        frac = fk["rel_tick"] / fk["round_len_ticks"].replace(0, np.nan)
        frac = frac.dropna()
        if len(frac) >= 5:
            out["first_kill_early"] = round(float((frac < 0.3).mean()), 3)
            out["first_kill_mid"] = round(float(((frac >= 0.3) & (frac <= 0.6)).mean()), 3)
            out["first_kill_late"] = round(float((frac > 0.6).mean()), 3)
            out["first_kill_median_frac"] = round(float(frac.median()), 3)
    return out


def print_entry(ep: dict) -> None:
    print(f"  HLTV 属性:")
    print(f"    Opening: 首杀 {ep.get('opening_kills')} / 首死 {ep.get('opening_deaths')}"
          f" | 首杀成功率 {_fmt(ep.get('opening_duel_success'))}"
          f" | 卷入率 {_fmt(ep.get('opening_participation'))}")
    print(f"    Entrying(trade死亡): {_fmt(ep.get('traded_death_rate'))}  (n死={ep.get('n_deaths')})")
    print(f"    Trading(trade击杀): {_fmt(ep.get('trade_kill_rate'))}")
    if "first_kill_median_frac" in ep:
        print(f"    首杀时机: 早 {_fmt(ep.get('first_kill_early'))}"
              f" / 中 {_fmt(ep.get('first_kill_mid'))} / 晚 {_fmt(ep.get('first_kill_late'))}"
              f" | 中位 {_fmt(ep.get('first_kill_median_frac'))} 回合进度")


def player_portrait(beh: pd.DataFrame, util: pd.DataFrame,
                    duels: pd.DataFrame, steamid: str) -> dict:
    """一个选手的操作+战术签名。beh/util/duels 是该选手已切好的子集。"""
    n = len(beh)
    out: dict = {"steamid": steamid, "n_rounds": int(n)}
    if n == 0:
        return out

    out["name"] = str(beh["name"].mode().iloc[0])
    out["roster_key"] = str(beh["roster_key"].mode().iloc[0])

    # --- 侵略性 ---
    out["first_contact_rate"] = round(_rate(beh["first_contact"]), 3)
    out["opening_kill_rate"] = round(_rate(beh["opening_kill"]), 3)
    out["opening_death_rate"] = round(_rate(beh["opening_death"]), 3)
    ct = beh["side"] == "CT"
    out["fc_ct"] = round(_cond(beh["first_contact"], ct), 3)
    out["fc_t"] = round(_cond(beh["first_contact"], ~ct), 3)
    behind = beh["score_bucket"].isin(BEHIND)
    ahead = beh["score_bucket"].isin(AHEAD)
    even = beh["score_bucket"] == "even"
    out["fc_behind"] = round(_cond(beh["first_contact"], behind), 3)
    out["fc_even"] = round(_cond(beh["first_contact"], even), 3)
    out["fc_ahead"] = round(_cond(beh["first_contact"], ahead), 3)

    # 分回合阶段：手枪局 / 下半场手枪局 / 常规局
    rc = beh["round_class"]
    out["fc_pistol"] = round(_cond(beh["first_contact"], rc == "pistol"), 3)
    out["fc_post_pistol"] = round(_cond(beh["first_contact"], rc == "post_pistol"), 3)
    out["fc_regular"] = round(_cond(beh["first_contact"], rc == "regular"), 3)

    # 分装备档：eco/force 局 vs 全起局，他还抢不抢第一交火（穷也冲 = 真突破手）
    econ_mask = beh["buy_tier"].isin(ECON_TIERS)
    full_mask = beh["buy_tier"] == "full"
    out["fc_on_eco"] = round(_cond(beh["first_contact"], econ_mask), 3)
    out["fc_on_full"] = round(_cond(beh["first_contact"], full_mask), 3)

    # --- 武器 ---
    w = beh["main_weapon"].dropna().astype(str).map(_weapon_family)
    out["weapon_dist"] = {k: int(v) for k, v in w.value_counts().items()}
    out["awp_kills_per_round"] = round(float(beh["awp_kills"].mean()), 3)

    # --- 买枪优先级（eco/force 局相对队伍均值）---
    econ = beh["buy_tier"].isin(ECON_TIERS)
    off = beh.loc[econ, "equip_offset"] if econ.any() else pd.Series(dtype=float)
    out["buy_priority_offset"] = round(_rate(off), 1)  # 正 = 被塞枪
    out["n_econ_rounds"] = int(econ.sum())

    # --- 道具 ---
    if len(util):
        throws_per_round = len(util) / max(n, 1)
        out["throws_per_round"] = round(throws_per_round, 2)
        fam = util["family"].value_counts()
        out["util_family"] = {k: int(v) for k, v in fam.items()}
        if "round_len_ticks" in util.columns and (util["round_len_ticks"] > 0).any():
            frac = util["rel_tick"] / util["round_len_ticks"].replace(0, np.nan)
            frac = frac.dropna()
            if len(frac) >= 5:
                out["util_early"] = round(float((frac < 0.33).mean()), 2)
                out["util_mid"] = round(float(((frac >= 0.33) & (frac <= 0.66)).mean()), 2)
                out["util_late"] = round(float((frac > 0.66).mean()), 2)
    else:
        out["throws_per_round"] = 0.0

    # --- 战术（duel 维度）---
    out["tactics"] = duel_tactics(duels)

    # --- 站位（duel 里的交火位置）---
    if len(duels):
        pc = duels["subject_place"].astype(str).value_counts()
        out["top_places"] = {k: int(v) for k, v in pc.head(8).items()}
    return out


def _fmt(v, nd=3):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}"


def print_portrait(p: dict) -> None:
    print(f"\n===== {p.get('name')} 操作画像（{p.get('n_rounds')} 回合）=====")
    print(f"  侵略性: first_contact {_fmt(p.get('first_contact_rate'))} | "
          f"首杀 {_fmt(p.get('opening_kill_rate'))} | 首死 {_fmt(p.get('opening_death_rate'))}")
    print(f"          分边   CT {_fmt(p.get('fc_ct'))} / T {_fmt(p.get('fc_t'))}")
    print(f"          分比分 落后 {_fmt(p.get('fc_behind'))} / 持平 {_fmt(p.get('fc_even'))}"
          f" / 领先 {_fmt(p.get('fc_ahead'))}")
    print(f"          分阶段 手枪 {_fmt(p.get('fc_pistol'))} / 手枪后 {_fmt(p.get('fc_post_pistol'))}"
          f" / 常规 {_fmt(p.get('fc_regular'))}")
    print(f"          分装备 eco {_fmt(p.get('fc_on_eco'))} / 全起 {_fmt(p.get('fc_on_full'))}"
          f"   (穷也冲=真突破手)")
    print(f"  武器: {p.get('weapon_dist')} | awp击杀/回合 {_fmt(p.get('awp_kills_per_round'))}")
    print(f"  买枪优先级: eco/force局 equip_offset={_fmt(p.get('buy_priority_offset'), 1)}"
          f" (n={p.get('n_econ_rounds')})  正=被塞枪")
    print(f"  道具: {_fmt(p.get('throws_per_round'), 2)} 次/回合 {p.get('util_family')}"
          f"  时序 早{_fmt(p.get('util_early'), 2)}/中{_fmt(p.get('util_mid'), 2)}/晚{_fmt(p.get('util_late'), 2)}")
    t = p.get("tactics") or {}
    if t:
        print(f"  战术: 距离 {t.get('duel_distance')}")
        print(f"        人数差 少{_fmt(t.get('duels_man_down'))} 平{_fmt(t.get('duels_man_even'))}"
              f" 多{_fmt(t.get('duels_man_up'))} | 残局胜率 {_fmt(t.get('clutch_winrate'))}"
              f" (n={t.get('n_clutch')})")
        print(f"        抱团 身边队友均值 {_fmt(t.get('mean_teammates_nearby'), 2)}"
              f" / 独狼率 {_fmt(t.get('lone_duel_rate'))}")
        print(f"        先手 {_fmt(t.get('got_drop'))} / 被先手 {_fmt(t.get('caught'))}"
              f" / 正面 {_fmt(t.get('head_on'))}")
        print(f"        盲打 {_fmt(t.get('fights_flashed'))} / 闪到对方 {_fmt(t.get('flashes_opponent'))}")
        print(f"        对手在哪 {t.get('opponent_place')}")
    if p.get("top_places"):
        tp = p["top_places"]
        print(f"  站位(交火最多): "
              + ", ".join(f"{k}({v})" for k, v in list(tp.items())[:5]))


def main(behavior_path=None, util_path=None, kills_path=None) -> None:
    beh = load_behavior(behavior_path)
    util = load_util(util_path)
    print(f"player_behavior: {len(beh)} 行 / {beh['steamid'].nunique()} 选手 / "
          f"{beh['demo_path'].nunique()} 图")
    print(f"player_util: {len(util)} 行道具")
    kills = load_kills(kills_path)
    print(f"player_kills: {len(kills)} 行击杀")

    sid2name = beh.groupby("steamid")["name"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    steamids = set(beh["steamid"].astype(str))
    duels = load_duels(steamids)

    # donk + 队友对照。按 name 合并：同一选手可能跨 demo 有多个 steamid
    #（如 donk 在 93 图里出现过两个 steamid），只按名字聚合才能拼全。
    target_names = ["donk", "sh1ro", "magixx", "zont1x"]
    sids = {n: [s for s, nm in sid2name.items() if str(nm).lower().strip() == n]
            for n in target_names}
    portraits = {}
    for n in target_names:
        if not sids[n]:
            continue
        b = beh[beh["steamid"].isin(sids[n])]
        u = util[util["steamid"].isin(sids[n])] if len(util) else pd.DataFrame()
        dl = duels[duels["player_id"].astype(str).isin(sids[n])]
        p = player_portrait(b, u, dl, sids[n][0])
        portraits[n] = p
        print_portrait(p)
        if len(kills):
            ep = entry_profile(kills, sids[n], len(b))
            portraits[n]["hlvt"] = ep
            print_entry(ep)

    # 落盘学习记忆
    mem = {"version": 2, "built_at": datetime.now(timezone.utc).isoformat(),
           "players": portraits}
    out = config.DATA_DIR / "player_style_memory.json"
    out.write_text(json.dumps(mem, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n操作画像已写入 {out}（{len(portraits)} 名选手）")


if __name__ == "__main__":
    import sys

    bp = sys.argv[1] if len(sys.argv) > 1 else None
    up = sys.argv[2] if len(sys.argv) > 2 else None
    kp = sys.argv[3] if len(sys.argv) > 3 else None
    main(behavior_path=bp, util_path=up, kills_path=kp)
