"""round_events.py —— 回合级事件富特征（分析师 round-level 信号落地）。

在 round_dataset（已缓存 equip/roster/winner）之上，补分析师共识的回合级结构信号：
  1. 装备 6 档分层（current_equip_value 重分类，替换 3 档 eco/force/full）
  2. 团队存活人数 + 存活总 HP（player_hurt 累伤反推，击杀方判死）
  3. 双方击杀 + 首杀边 + RWAFF（首杀方是否赢下本回合）
  4. trade 击杀（5s 复仇窗口）
  5. utility damage（HE/火 伤害，player_hurt.weapon）
  6. eco 击杀折扣（杀 eco 对手的击杀数，价值减半）

关键：只用快速 parse_event（player_death / player_hurt / round_freeze_end / player_info），
装备值、roster、胜方从 round_dataset 按 (demo_path, round_num) merge，**不重复跑慢的
parse_ticks**（装备值已缓存）。

产物：data/round_events.parquet（每回合一行 = round_dataset 增列）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

from . import config
from .demo_features import _group_roster, _group_side_first_half, _steamid_to_group

FIRST_HALF_ROUNDS = 12
TRADE_TICKS = 5 * 64  # 5s @ 64 tick/s


def _classify6(v: float) -> str:
    """装备价值 → 6 档（粗略 CS2 购买档位，值 = 队伍人均 current_equip_value）。"""
    if v < 700:
        return "eco0"       # 裸奔/手枪无甲
    if v < 1400:
        return "eco_armor"  # 手枪+甲
    if v < 2800:
        return "force_low"  # SMG/沙鹰+甲
    if v < 4100:
        return "force_high"  # 咖喱/法玛斯+甲+少量道具
    if v < 4700:
        return "semi"       # 长枪+甲 但道具不全
    return "full"           # 全起


_ECO_TIERS = {"eco0", "eco_armor"}


def _is_utility(weapon: str) -> bool:
    w = str(weapon).lower()
    return ("hegrenade" in w or "molotov" in w or "incendiary" in w or "inferno" in w)


def _side_of(g: int, r: int, first_half_side: dict[int, str]) -> str | None:
    s = first_half_side.get(g)
    if s is None:
        return None
    s = str(s).upper()
    return s if r <= FIRST_HALF_ROUNDS else ("T" if s == "CT" else "CT")


def compute_round_events(dem_path: str, sub: pd.DataFrame) -> pd.DataFrame:
    """解析一张 demo，返回其每回合的富特征（增列，key=round_num）。"""
    parser = DemoParser(dem_path)
    player_info = parser.parse_player_info()
    if player_info is None or not len(player_info):
        return pd.DataFrame()
    sid2group = _steamid_to_group(player_info)
    group_roster = _group_roster(player_info)
    groups = sorted(set(sid2group.values()))
    if len(groups) != 2:
        return pd.DataFrame()

    freeze = parser.parse_event("round_freeze_end")
    if freeze is None or not len(freeze):
        return pd.DataFrame()
    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())

    # 第 1 回合的边（用于把 group 映射回 CT/T，跨半场翻转）
    ticks_start = parser.parse_ticks(["team_name"], ticks=[freeze_ticks[0]])
    first_half_side = _group_side_first_half(ticks_start, sid2group)
    if len(first_half_side) != 2:
        return pd.DataFrame()

    kills = parser.parse_event("player_death")
    hurts = parser.parse_event("player_hurt")

    def round_of(tick: int) -> int:
        r = 1
        for i, t in enumerate(freeze_ticks):
            if tick >= t:
                r = i + 1
            else:
                break
        return r

    # ---- 击杀：group 归属 + 首杀 + trade ----
    k = kills.copy()
    k["round"] = k["tick"].astype(int).map(round_of)
    k["att_g"] = k["attacker_steamid"].astype(str).map(sid2group)
    k["vic_g"] = k["user_steamid"].astype(str).map(sid2group)
    k = k.dropna(subset=["att_g", "vic_g"])
    k["att_g"] = k["att_g"].astype(int)
    k["vic_g"] = k["vic_g"].astype(int)

    # trade 击杀（队友 5s 内阵亡后的复仇）——沿用 demo_features 语义
    k_sorted = k.sort_values("tick").reset_index(drop=True)
    ticks_arr = k_sorted["tick"].to_numpy(dtype=float)
    att_sid = k_sorted["attacker_steamid"].astype(str).to_numpy()
    vic_sid = k_sorted["user_steamid"].astype(str).to_numpy()
    vg = k_sorted["vic_g"].to_numpy()
    ag = k_sorted["att_g"].to_numpy()
    trade_flag = np.zeros(len(k_sorted), dtype=bool)
    for i in range(len(k_sorted)):
        trade_flag[i] = ((att_sid == vic_sid[i]) & (vg == ag[i])
                         & (ticks_arr >= ticks_arr[i] - TRADE_TICKS)
                         & (ticks_arr < ticks_arr[i])).any()
    k_sorted["trade"] = trade_flag

    # 首杀：每回合 tick 最早的那一杀
    first = k_sorted.sort_values("tick").groupby("round").head(1)

    # ---- 伤害：group 归属 + 累计 dmg_health（HP 反推）+ utility damage ----
    h = hurts.copy()
    h["round"] = h["tick"].astype(int).map(round_of)
    h["att_g"] = h["attacker_steamid"].astype(str).map(sid2group)
    h["vic_g"] = h["user_steamid"].astype(str).map(sid2group)
    h = h.dropna(subset=["vic_g"])          # HP 只关心受害者，attacker 可为 0/world
    h["vic_g"] = h["vic_g"].astype(int)
    h["att_g"] = h["att_g"].fillna(-1).astype(int)
    h["dmg"] = h["dmg_health"].astype(float)

    # 每回合每 group 的存活血量：100 - 该回合受伤害；被击杀者记 0
    # (CS2 每回合开始血量重置为 100)

    # 击杀者（受害者）集合，按回合
    dead = k_sorted[["round", "vic_g", "user_steamid"]].copy()
    dead["sid"] = dead["user_steamid"].astype(str)

    # 每回合每名选手的累计受伤害（HP 反推用，逐选手）
    dmg_by_sid = h.groupby(["round", "user_steamid"])["dmg"].sum().reset_index()
    dmg_map = {(int(r), str(s)): float(d) for r, s, d in dmg_by_sid.itertuples(index=False)}

    # 组装每回合
    eq_map = {int(r.round_num): (r.ct_equip, r.t_equip) for r in sub.itertuples(index=False)}
    out_rows = []
    for _, rr in sub.iterrows():
        rnd = int(rr.round_num)
        ct_g = next((g for g in groups if _side_of(g, rnd, first_half_side) == "CT"), None)
        t_g = next((g for g in groups if _side_of(g, rnd, first_half_side) == "T"), None)
        if ct_g is None or t_g is None:
            continue
        ct_equip, t_equip = eq_map.get(rnd, (np.nan, np.nan))
        ct_tier6 = _classify6(ct_equip) if pd.notna(ct_equip) else None
        t_tier6 = _classify6(t_equip) if pd.notna(t_equip) else None

        kr = k_sorted[k_sorted["round"] == rnd]
        fr = first[first["round"] == rnd]

        ct_kills = int((kr["att_g"] == ct_g).sum())
        t_kills = int((kr["att_g"] == t_g).sum())
        ct_trade = int((kr["trade"] & (kr["att_g"] == ct_g)).sum())
        t_trade = int((kr["trade"] & (kr["att_g"] == t_g)).sum())
        # eco 击杀折扣：杀装备在 eco 档的对手，价值减半
        t_is_eco = t_tier6 in _ECO_TIERS
        ct_is_eco = ct_tier6 in _ECO_TIERS
        ct_eco_kills = int(((kr["att_g"] == ct_g) & (kr["vic_g"] == t_g)).sum() * (1 if t_is_eco else 0))
        t_eco_kills = int(((kr["att_g"] == t_g) & (kr["vic_g"] == ct_g)).sum() * (1 if ct_is_eco else 0))

        # 首杀
        first_kill_side = None
        if len(fr):
            fg = int(fr.iloc[0]["att_g"])
            first_kill_side = "CT" if fg == ct_g else "T"
        ct_first = 1 if first_kill_side == "CT" else 0
        t_first = 1 if first_kill_side == "T" else 0

        # utility damage（att_g 归属，weapon 为 HE/火）
        hr = h[h["round"] == rnd]
        ct_util = 0.0
        t_util = 0.0
        if len(hr):
            u = hr[hr["weapon"].astype(str).map(_is_utility)]
            ct_util = float(u[u["att_g"] == ct_g]["dmg"].sum())
            t_util = float(u[u["att_g"] == t_g]["dmg"].sum())

        # 存活人数 + 存活总 HP
        ct_sids = [s for s, g in sid2group.items() if g == ct_g]
        t_sids = [s for s, g in sid2group.items() if g == t_g]
        ct_dead = set(dead[(dead["round"] == rnd) & (dead["vic_g"] == ct_g)]["sid"])
        t_dead = set(dead[(dead["round"] == rnd) & (dead["vic_g"] == t_g)]["sid"])
        ct_alive = sum(1 for s in ct_sids if s not in ct_dead)
        t_alive = sum(1 for s in t_sids if s not in t_dead)

        # HP：存活者 100 - 各自受伤害；死亡者贡献 0
        def _team_hp(sids, dead_set):
            tot = 0.0
            for s in sids:
                if s in dead_set:
                    continue  # 死亡者贡献 0
                d = dmg_map.get((rnd, s), 0.0)
                tot += max(0.0, 100.0 - d)
            return round(tot, 1)
        ct_hp = _team_hp(ct_sids, ct_dead)
        t_hp = _team_hp(t_sids, t_dead)

        out_rows.append({
            "demo_path": dem_path,
            "round_num": rnd,
            "ct_tier6": ct_tier6, "t_tier6": t_tier6,
            "ct_alive": ct_alive, "t_alive": t_alive,
            "ct_hp": ct_hp, "t_hp": t_hp,
            "ct_kills": ct_kills, "t_kills": t_kills,
            "ct_first": ct_first, "t_first": t_first,
            "first_kill_side": first_kill_side,
            "ct_trade": ct_trade, "t_trade": t_trade,
            "ct_util": ct_util, "t_util": t_util,
            "ct_eco_kills": ct_eco_kills, "t_eco_kills": t_eco_kills,
        })
    return pd.DataFrame(out_rows)


def build() -> pd.DataFrame:
    """全量：round_dataset 每张 demo 补富特征，写 data/round_events.parquet。"""
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    frames = []
    n_fail = 0
    demos = rd["demo_path"].unique()
    print(f"round_dataset: {len(rd)} 回合 / {len(demos)} 图")
    for i, dp in enumerate(demos):
        sub = rd[rd["demo_path"] == dp]
        try:
            fr = compute_round_events(dp, sub)
            if len(fr):
                frames.append(fr)
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  跳过 {dp.split(chr(92))[-1]}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(demos)}")
    events = pd.concat(frames, ignore_index=True)
    # merge 回 round_dataset（装备值/roster/胜方/round_class/map 等）
    rd["round_num"] = rd["round_num"].astype(int)
    events["round_num"] = events["round_num"].astype(int)
    out = rd.merge(events, on=["demo_path", "round_num"], how="inner")
    # RWAFF：首杀方 == 胜方
    out["rwaff"] = (out["first_kill_side"] == out["winner_side"]).astype(int)
    out["rwaff"] = out["rwaff"].where(out["first_kill_side"].notna(), np.nan)
    out["equip_gap"] = out["ct_equip"] - out["t_equip"]
    out["hp_gap"] = out["ct_hp"] - out["t_hp"]
    out["alive_gap"] = out["ct_alive"] - out["t_alive"]

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.DATA_DIR / "round_events.parquet", index=False)
    print(f"完成：{len(out)} 回合富事件 / {out['demo_path'].nunique()} 图（跳过 {n_fail}）")
    return out


if __name__ == "__main__":
    build()
