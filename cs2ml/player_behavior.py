"""player_behavior.py —— per-player 行为提取（route B 第一步，pilot 先行）。

从游戏本身提取「每个选手在每回合做了什么操作」——不是技术指标，是打法层：

  1. 回合开局(freeze_end tick) parse_ticks：per-player 装备值 current_equip_value
     → 谁买枪、买多少（买枪优先级：eco/force 局谁被塞枪）
  2. 击杀(player_death)：attacker/victim steamid、weapon、headshot、thrusmoke、
     assistedflash、tick → 谁杀谁、什么武器、首杀/首死/first_contact（侵略性）、
     awp/trade 击杀
  3. 道具(parse_grenades 的 Projectile 态)：steamid、family(flash/smoke/molly/he)、
     tick、x/y/z → 谁在什么时刻丢什么道具（道具时序）
  4. 位置：不重新解析，直接复用 data/duel_dataset.parquet 的 subject_place（363 图
     已解析的 per-kill 位置），在 player_style.py 里 join。

join 回合上下文(round_dataset + round_states)：side、round_class、score_bucket、
winner、team equip。产出：
  data/player_behavior.parquet —— per-player per-round 行为聚合行
  data/player_util.parquet     —— per-throw 原始行（道具时序用）

注意：parse_ticks 在 freeze 时刻 last_place_name=spawn / active_weapon_name=knife
无意义（buy 阶段未出枪），故站位/武器不从这里取，只取装备值 + side。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

from . import config
from .demo_features import _grenade_family, _group_roster, _steamid_to_group
from .round_transition import _classify6, _round_class
from .team_strategy import _score_bucket

TRADE_TICKS = 5 * 64  # 5s @ 64 tick/s

# freeze 时刻需要的 tick 字段：装备值 + side（位置/武器在这里无意义，不取）
_FREEZE_FIELDS = ["steamid", "current_equip_value", "team_name"]


def _weapon_family(w: str) -> str:
    w = str(w).lower()
    if "awp" in w:
        return "awp"
    if "ak47" in w or "m4" in w or "galil" in w or "famas" in w or "sg" in w or "aug" in w:
        return "rifle"
    if "mp" in w or "p90" in w or "bizon" in w or "ump" in w or "mac10" in w or "nova" in w or "xm" in w or "mag" in w or "sawed" in w:
        return "smg_shotgun"
    if "deagle" in w or "glock" in w or "usp" in w or "p2000" in w or "cz" in w or "tec9" in w or "fiveseven" in w or "p250" in w:
        return "pistol"
    if "knife" in w or "knife" in w:
        return "knife"
    return "other"


def pilot_rosters(n: int = 10) -> list[str]:
    """按图数取 top N roster（含 Spirit=donk 所在队），作为 pilot 选片。"""
    pf = pd.read_parquet(config.DATA_DIR / "player_features.parquet")
    top = (pf.groupby("roster_key")["demo_path"].nunique()
             .sort_values(ascending=False).head(n))
    return list(top.index)


def pilot_demos(rosters: list[str]) -> list[str]:
    """这些 roster 出现过的全部 demo_path（去重排序）。"""
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    rs = set(rosters)
    hit = rd[(rd["ct_roster"].isin(rs)) | (rd["t_roster"].isin(rs))]
    return sorted(hit["demo_path"].unique())


def build_round_context(demo_paths: list[str]) -> pd.DataFrame:
    """(demo_path, round_num) → 回合上下文：map/round_class/winner/equip/比分。

    比分 = 进入该回合前双方累计分（CT 视角 gap = ct_score - t_score）。
    """
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    rs = pd.read_parquet(config.DATA_DIR / "round_states.parquet")
    rd = rd[["demo_path", "match_id", "map_name", "round_num", "winner_side"]].copy()
    rs = rs[["demo_path", "round_num", "ct_equip0", "t_equip0"]].copy()
    m = rd.merge(rs, on=["demo_path", "round_num"], how="inner")
    m = m[m["demo_path"].isin(set(demo_paths))]
    m["round_num"] = m["round_num"].astype(int)
    m = m.sort_values(["demo_path", "round_num"]).reset_index(drop=True)

    rows = []
    for dp, sub in m.groupby("demo_path"):
        sub = sub.sort_values("round_num").reset_index(drop=True)
        winners = sub["winner_side"].tolist()
        ct = t = 0
        for i in range(len(sub)):
            rows.append({
                "demo_path": dp,
                "round_num": int(sub["round_num"].iloc[i]),
                "ct_score": ct, "t_score": t,
            })
            if winners[i] == "CT":
                ct += 1
            else:
                t += 1
    sc = pd.DataFrame(rows)
    m = m.merge(sc, on=["demo_path", "round_num"], how="left")
    m["round_class"] = m["round_num"].map(_round_class)
    return m


def compute_player_behavior(dem_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """解析一张 demo，返回 (behavior 行 per-round-per-player, util 行 per-throw)。"""
    dem_path = str(dem_path)
    match_id = Path(dem_path).parent.name
    parser = DemoParser(dem_path)

    player_info = parser.parse_player_info()
    if player_info is None or not len(player_info):
        return pd.DataFrame(), pd.DataFrame()
    sid2name = {str(r.steamid): str(r.name or "") for r in player_info.itertuples(index=False)}
    sid2group = _steamid_to_group(player_info)
    group_roster = _group_roster(player_info)
    groups = sorted(set(sid2group.values()))
    if len(groups) != 2:
        return pd.DataFrame(), pd.DataFrame()
    roster_of = {g: ",".join(group_roster[g]) for g in groups}
    sid2roster = {sid: roster_of[g] for sid, g in sid2group.items()}

    header = parser.parse_header() or {}
    map_name = header.get("map_name", Path(dem_path).stem)

    freeze = parser.parse_event("round_freeze_end")
    if freeze is None or not len(freeze):
        return pd.DataFrame(), pd.DataFrame()
    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())

    def round_of(tick: int) -> int:
        r = 1
        for i, t in enumerate(freeze_ticks):
            if tick >= t:
                r = i + 1
            else:
                break
        return r

    # ---- 1. freeze 装备 + side（per-player per-round）----
    start_ticks = [t + 1 for t in freeze_ticks]
    st = parser.parse_ticks(_FREEZE_FIELDS, ticks=start_ticks)
    st["steamid"] = st["steamid"].astype(str)
    st["side"] = st["team_name"].map(
        lambda s: "CT" if str(s).upper().startswith("CT") else "T")
    st["round_num"] = st["tick"].astype(int).map(round_of)
    st["equip0"] = st["current_equip_value"].astype(float)
    start = st[["steamid", "round_num", "side", "equip0"]].copy()
    # 队伍均值 equip（每 side 每 round）
    team_eq = start.groupby(["round_num", "side"])["equip0"].mean().rename("team_equip0")

    # ---- 2. 击杀（per-player per-round 聚合）----
    kills = parser.parse_event("player_death")
    if kills is None or not len(kills):
        kills = pd.DataFrame(columns=["attacker_steamid", "user_steamid", "weapon",
                                      "headshot", "thrusmoke", "assistedflash", "tick"])
    k = kills.copy()
    k["round"] = k["tick"].astype(int).map(round_of)
    k["att"] = k["attacker_steamid"].astype(str)
    k["vic"] = k["user_steamid"].astype(str)
    k = k.dropna(subset=["att", "vic"])
    k["att_g"] = k["att"].map(sid2group)
    k["vic_g"] = k["vic"].map(sid2group)
    k = k.dropna(subset=["att_g", "vic_g"]).copy()
    k["att_g"] = k["att_g"].astype(int)
    k["vic_g"] = k["vic_g"].astype(int)

    # trade 击杀（复仇）：attacker A 杀 V，而 V 在 5s 内杀过 A 的队友
    k_sorted = k.sort_values("tick").reset_index(drop=True)
    ticks_arr = k_sorted["tick"].to_numpy(dtype=float)
    att_sid = k_sorted["att"].to_numpy()
    vic_sid = k_sorted["vic"].to_numpy()
    vg = k_sorted["vic_g"].to_numpy()
    ag = k_sorted["att_g"].to_numpy()
    trade = np.zeros(len(k_sorted), dtype=bool)
    for i in range(len(k_sorted)):
        trade[i] = ((att_sid == vic_sid[i]) & (vg == ag[i])
                    & (ticks_arr >= ticks_arr[i] - TRADE_TICKS)
                    & (ticks_arr < ticks_arr[i])).any()
    k_sorted["trade"] = trade
    k = k_sorted

    # 首杀（每回合最早一杀）
    first = k.sort_values("tick").groupby("round").head(1)

    # ---- 2b. per-kill 落库（战术/时序分析：首杀时机、trade 死亡率、opening duel）----
    first_ticks = first.set_index("round")["tick"].to_dict()
    side_lookup = {(str(s), int(r)): side for s, r, side
                   in start[["steamid", "round_num", "side"]].itertuples(index=False)}
    kills_df = k[["round", "att", "vic", "weapon", "headshot", "thrusmoke",
                  "assistedflash", "trade", "tick"]].copy()
    kills_df = kills_df.rename(columns={"att": "attacker", "vic": "victim"})
    kills_df["freeze_tick"] = kills_df["round"].map(lambda r: freeze_ticks[int(r) - 1])
    kills_df["rel_tick"] = kills_df["tick"].astype(int) - kills_df["freeze_tick"].astype(int)
    kills_df["first_tick"] = kills_df["round"].map(first_ticks)
    kills_df["is_first_kill"] = kills_df["tick"].astype(int) == kills_df["first_tick"].astype(int)
    kills_df["attacker_side"] = [side_lookup.get((a, int(r))) for a, r in
                                 zip(kills_df["attacker"], kills_df["round"])]
    kills_df["victim_side"] = [side_lookup.get((v, int(r))) for v, r in
                               zip(kills_df["victim"], kills_df["round"])]
    kills_df["demo_path"] = dem_path
    kills_df["match_id"] = match_id
    kills_df["map_name"] = map_name
    kills_df = kills_df[["demo_path", "match_id", "map_name", "round", "tick", "rel_tick",
                         "is_first_kill", "attacker", "victim", "attacker_side",
                         "victim_side", "weapon", "headshot", "thrusmoke",
                         "assistedflash", "trade"]]

    att_kills = k.groupby(["round", "att"]).agg(
        n_kills=("weapon", "size"),
        n_headshot=("headshot", lambda s: (s == True).sum()),
        n_thrusmoke=("thrusmoke", lambda s: (s == True).sum()),
        n_flash_assist=("assistedflash", lambda s: (s == True).sum()),
        awp_kills=("weapon", lambda s: (s.astype(str).str.lower() == "awp").sum()),
        trade_kills=("trade", "sum"),
    ).reset_index().rename(columns={"att": "steamid"})
    mode = (k.groupby(["round", "att"])["weapon"]
              .agg(lambda s: s.astype(str).value_counts().idxmax())
              .rename("main_weapon").reset_index().rename(columns={"att": "steamid"}))
    att_kills = att_kills.merge(mode, on=["round", "steamid"], how="left")
    vic_deaths = k.groupby(["round", "vic"]).size().rename("n_deaths").reset_index()
    vic_deaths = vic_deaths.rename(columns={"vic": "steamid"})

    # 首杀 / 首死 / first_contact
    opening = []
    for _, fr in first.iterrows():
        opening.append({"round": int(fr["round"]), "ok_steamid": fr["att"], "od_steamid": fr["vic"]})
    op = pd.DataFrame(opening)
    if len(op):
        op = op.groupby("round").agg(
            ok_steamid=("ok_steamid", "first"), od_steamid=("od_steamid", "first")).reset_index()

    # ---- 3. 道具（per-throw，Projectile 态 = 真正丢出去）----
    util_rows = []
    g = parser.parse_grenades()
    if g is not None and len(g):
        proj = g[g["grenade_type"].astype(str).str.contains("Projectile", case=False)]
        if len(proj):
            th = (proj.groupby("grenade_entity_id")
                     .agg(tick=("tick", "min"), steamid=("steamid", "first"),
                          gtype=("grenade_type", "first"),
                          x=("x", "first"), y=("y", "first"), z=("z", "first"))
                     .reset_index())
            th["family"] = th["gtype"].astype(str).map(_grenade_family)
            th = th.dropna(subset=["family"])
            th["round"] = th["tick"].astype(int).map(round_of)
            for r in th.itertuples(index=False):
                rnd = int(r.round)
                ft = freeze_ticks[rnd - 1]
                util_rows.append({
                    "demo_path": dem_path, "match_id": match_id, "map_name": map_name,
                    "round_num": rnd, "steamid": str(r.steamid),
                    "family": r.family, "tick": int(r.tick),
                    "freeze_tick": ft, "rel_tick": int(r.tick) - ft,
                    "x": r.x, "y": r.y, "z": r.z,
                })

    # ---- 组装 per-round-per-player ----
    players = sorted(sid2group.keys())
    side_of = start.set_index(["steamid", "round_num"])["side"]
    equip_of = start.set_index(["steamid", "round_num"])["equip0"]

    rows = []
    for rnd in range(1, len(freeze_ticks) + 1):
        for sid in players:
            side = None
            try:
                side = side_of.loc[(sid, rnd)]
            except KeyError:
                continue
            equip0 = float(equip_of.loc[(sid, rnd)])
            ak = att_kills[(att_kills["round"] == rnd) & (att_kills["steamid"] == sid)]
            dk = vic_deaths[(vic_deaths["round"] == rnd) & (vic_deaths["steamid"] == sid)]
            a = ak.iloc[0].to_dict() if len(ak) else {}
            n_deaths = int(dk["n_deaths"].iloc[0]) if len(dk) else 0
            ok = od = fc = 0
            if len(op):
                o = op[op["round"] == rnd]
                if len(o):
                    ok = int(o["ok_steamid"].iloc[0] == sid)
                    od = int(o["od_steamid"].iloc[0] == sid)
                    fc = int(ok or od)
            rows.append({
                "demo_path": dem_path, "match_id": match_id, "map_name": map_name,
                "round_num": rnd, "steamid": sid, "name": sid2name.get(sid, sid[:8]),
                "roster_key": sid2roster.get(sid, ""), "side": side,
                "equip0": equip0,
                "n_kills": a.get("n_kills", 0),
                "n_deaths": n_deaths,
                "opening_kill": ok, "opening_death": od, "first_contact": fc,
                "n_headshot": a.get("n_headshot", 0),
                "n_thrusmoke": a.get("n_thrusmoke", 0),
                "n_flash_assist": a.get("n_flash_assist", 0),
                "awp_kills": a.get("awp_kills", 0),
                "trade_kills": a.get("trade_kills", 0),
                "main_weapon": a.get("main_weapon"),
            })
    beh = pd.DataFrame(rows)
    util = pd.DataFrame(util_rows)
    return beh, util, kills_df


def _build(demo_paths: list[str], limit: int | None = None,
           out_beh: str = "player_behavior.parquet",
           out_util: str = "player_util.parquet",
           out_kills: str = "player_kills.parquet") -> None:
    ctx = build_round_context(demo_paths)
    beh_frames, util_frames, kill_frames = [], [], []
    n_fail = 0
    demos = demo_paths[:limit] if limit else demo_paths
    for i, dp in enumerate(demos):
        try:
            b, u, kd = compute_player_behavior(dp)
            if len(b):
                beh_frames.append(b)
            if len(u):
                util_frames.append(u)
            if len(kd):
                kill_frames.append(kd)
        except Exception as e:
            n_fail += 1
            if n_fail <= 8:
                print(f"  跳过 {Path(dp).name}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(demos)}")
    beh = pd.concat(beh_frames, ignore_index=True) if beh_frames else pd.DataFrame()
    util = pd.concat(util_frames, ignore_index=True) if util_frames else pd.DataFrame()
    kills = pd.concat(kill_frames, ignore_index=True) if kill_frames else pd.DataFrame()

    # join 回合上下文：round_class / score_bucket / winner / team equip
    if len(beh):
        ctx["round_num"] = ctx["round_num"].astype(int)
        beh = beh.merge(ctx, on=["demo_path", "round_num"], how="left")
        beh["score_gap"] = np.where(beh["side"] == "CT",
                                    beh["ct_score"] - beh["t_score"],
                                    beh["t_score"] - beh["ct_score"])
        beh["score_bucket"] = beh["score_gap"].map(_score_bucket)
        beh["team_equip0"] = np.where(beh["side"] == "CT", beh["ct_equip0"], beh["t_equip0"])
        beh["equip_offset"] = beh["equip0"] - beh["team_equip0"]
        beh["buy_tier"] = beh["equip0"].map(_classify6)
        beh["won_round"] = (beh["winner_side"] == beh["side"]).astype(int)

    rs = pd.read_parquet(config.DATA_DIR / "round_states.parquet")[
        ["demo_path", "round_num", "round_len_ticks"]]
    rs["round_num"] = rs["round_num"].astype(int)

    if len(util):
        ctx["round_num"] = ctx["round_num"].astype(int)
        util = util.merge(ctx, on=["demo_path", "round_num"], how="left")
        # 回合时长：Phase 2 用 rel_tick / round_len_ticks 算道具时序（早/中/晚）
        util = util.merge(rs, on=["demo_path", "round_num"], how="left")

    if len(kills):
        kills = kills.rename(columns={"round": "round_num"})
        kills["round_num"] = kills["round_num"].astype(int)
        kills = kills.merge(rs, on=["demo_path", "round_num"], how="left")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    beh.to_parquet(config.DATA_DIR / out_beh, index=False)
    util.to_parquet(config.DATA_DIR / out_util, index=False)
    kills.to_parquet(config.DATA_DIR / out_kills, index=False)
    print(f"\n完成：{len(beh)} 行为行 / {beh['demo_path'].nunique()} 图  |  "
          f"{len(util)} 道具行  |  {len(kills)} 击杀行  |  跳过 {n_fail}")


def main(limit: int | None = None, n_rosters: int = 10) -> None:
    rosters = pilot_rosters(n_rosters)
    print(f"pilot roster：{len(rosters)} 支（top {n_rosters} by map count）")
    demos = pilot_demos(rosters)
    print(f"pilot demo：{len(demos)} 张图")
    _build(demos, limit=limit)


if __name__ == "__main__":
    import sys

    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
