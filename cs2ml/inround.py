"""inround.py —— 回合内 per-event 状态轨迹（预测→检验→校正闭环的数据底座）。

对接用户的新策略（不是预测最终 map winner，而是逐回合、逐事件滚动的
「下一回合 / 接下来谁更接近赢」概率流，用来低买高卖）：

  宏观层：第 k 回合的结果(类型) → 经济决策 → 第 k+1 回合结果
  微观层：回合内部，每击杀/下包/拆包一个事件，实时更新「本回合谁更接近赢」

这里只产出**数据底座**（两张表），模型在后续脚本：
  1. inround_events —— 回合内每个事件的「事后状态」轨迹（事件级）
  2. round_states  —— 回合级（结果分类 + 经济决策 + 炸弹 + 时长 + 回合开始态）

技术要点：
  - 全程只用 SIDE 空间（team_name = "CT"/"TERRORIST"），无需 steamid→group 映射；
    队伍身份(roster)由下游 merge round_dataset 获得。
  - 关键优化：只对「事件 tick」（freeze + 击杀 + 炸弹事件）parse_ticks，而非 full tick，
    就能拿到每个事件时刻的 位置/HP/存活/经济/朝向/致盲（BLAST 的 per-event 粒度）。
  - 回合边界用 round_freeze_end；半场换边由 round_end.winner("CT"/"T") 直接给 side。
  - 位置/视野聚合成紧凑特征（接触距离/队形松散度/瞄准数），丢弃原始坐标控体积。

产物：data/inround_events.parquet、data/round_states.parquet。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from demoparser2 import DemoParser

from . import config
from .duel import discover_demos

FIRST_HALF_ROUNDS = 12
TICKS_PER_SECOND = 64

# 位置/视野聚合阈值
CONTACT_RANGE = 600.0            # 交战距离（CS2 距离单位）
_FACING_COS = math.cos(math.radians(45.0))   # 45° 锥内算「正看着对方」

# 每张 demo 只 parse_ticks 这些字段（事件 tick 上）
_TICK_FIELDS = [
    "X", "Y", "Z", "health", "armor_value", "current_equip_value",
    "is_alive", "team_name", "steamid", "active_weapon_name", "yaw", "flash_duration",
]


def _round_of(tick: int, freeze_ticks: list[int]) -> int:
    """tick → 回合号（freeze_end 即该回合开始）。"""
    r = 1
    for i, t in enumerate(freeze_ticks):
        if tick >= t:
            r = i + 1
        else:
            break
    return r


def _dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))


def _facing(x1: float, y1: float, yaw1: float, x2: float, y2: float) -> bool:
    """p1 是否朝 p2 看（水平面，45° 锥内）。yaw 单位度，forward=(cos,sin)。"""
    dx, dy = x2 - x1, y2 - y1
    ln = math.hypot(dx, dy)
    if ln < 1e-6:
        return False
    r = math.radians(yaw1)
    return (math.cos(r) * dx + math.sin(r) * dy) / ln > _FACING_COS


def _position_aggs(rows: list[dict]) -> dict:
    """给定某 tick 某边存活玩家的 (x,y,yaw)，算位置/视野聚合特征。

    rows: [{x, y, yaw}...]（已过滤 is_alive 且坐标有效）。空列表 → 全 NaN。
    """
    n = len(rows)
    if n == 0:
        return {"contact": np.nan, "spread": np.nan, "aiming": np.nan}
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    # 队形松散度：存活者两两距离均值
    if n >= 2:
        d = []
        for i in range(n):
            for j in range(i + 1, n):
                d.append(_dist2d(xs[i], ys[i], xs[j], ys[j]))
        spread = float(np.mean(d))
    else:
        spread = 0.0
    return {"contact": np.nan, "spread": spread, "aiming": np.nan}


def _cross_team(side_rows: dict[str, list[dict]], side: str, other: str) -> dict:
    """side 队 vs other 队：接触距离 + 瞄准数。"""
    mine = side_rows.get(side, [])
    theirs = side_rows.get(other, [])
    if not mine or not theirs:
        return {"contact": np.nan, "aiming": np.nan}
    dists = []
    aiming = 0
    for m in mine:
        best = None
        for t in theirs:
            d = _dist2d(m["x"], m["y"], t["x"], t["y"])
            best = d if best is None or d < best else best
            if d <= CONTACT_RANGE and _facing(m["x"], m["y"], m["yaw"], t["x"], t["y"]):
                aiming += 1
                break
        dists.append(best)
    return {"contact": float(np.min(dists)), "aiming": float(aiming)}


def _alive_rows(snap_t: pd.DataFrame, side: str) -> list[dict]:
    sub = snap_t[(snap_t["team_name"] == side) & (snap_t["is_alive"].to_numpy(dtype=bool))]
    out = []
    for r in sub.itertuples(index=False):
        if pd.isna(r.X) or pd.isna(r.Y):
            continue
        out.append({"x": float(r.X), "y": float(r.Y), "yaw": float(r.yaw)})
    return out


def compute_inround(dem_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """解析一张 demo，返回 (事件轨迹 events, 回合态 rounds)，失败返回 None。"""
    dem_path = str(dem_path)
    parser = DemoParser(dem_path)
    header = parser.parse_header()
    map_name = header.get("map_name", Path(dem_path).stem)

    freeze = parser.parse_event("round_freeze_end")
    re_ = parser.parse_event("round_end")
    if freeze is None or not len(freeze) or re_ is None or not len(re_):
        return None
    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())
    if not freeze_ticks:
        return None

    kills = parser.parse_event("player_death")
    planted = parser.parse_event("bomb_planted")
    defused = parser.parse_event("bomb_defused")
    exploded = parser.parse_event("bomb_exploded")

    # ---- 事件 tick 集合：freeze + 击杀 + 炸弹事件 ----
    bomb_events: dict[str, dict[int, str]] = {}  # tick -> event_type（planted/defused/exploded）
    for ev, name in ((planted, "bomb_planted"), (defused, "bomb_defused"),
                     (exploded, "bomb_exploded")):
        if ev is not None and len(ev):
            for t in ev["tick"].tolist():
                bomb_events[int(t)] = name

    kill_ticks: list[int] = []
    if kills is not None and len(kills):
        kill_ticks = sorted(int(t) for t in kills["tick"].tolist())

    event_ticks = sorted(set(freeze_ticks + kill_ticks + list(bomb_events.keys())))
    tick2round = {t: _round_of(t, freeze_ticks) for t in event_ticks}

    # ---- 一次批量 parse_ticks（事件 tick 上）----
    snap = parser.parse_ticks(_TICK_FIELDS, ticks=event_ticks)
    if snap is None or not len(snap):
        return None
    snap["team_name"] = snap["team_name"].astype(str).replace("TERRORIST", "T")
    snap["steamid"] = snap["steamid"].astype(str)
    snap["is_alive"] = snap["is_alive"].astype(bool)
    snap["round"] = snap["tick"].map(tick2round)

    teams = {t for t in snap["team_name"].unique() if t in ("CT", "T")}
    if len(teams) != 2:
        return None

    # ---- 每回合赢家 + 炸弹 + 时长 ----
    re2 = re_[re_["winner"].notna()].copy()
    re2["round"] = re2["round"].astype(int)
    re2 = re2[re2["round"] >= 1]
    winner_of = {int(r.round): str(r.winner).upper() for r in re2.itertuples(index=False)}
    reason_of = {int(r.round): str(r.reason) for r in re2.itertuples(index=False)}
    freeze_start = {i + 1: t for i, t in enumerate(freeze_ticks)}

    bomb_ticks_by_round: dict[int, dict[str, int]] = {}
    for t, ev in bomb_events.items():
        r = tick2round[t]
        bomb_ticks_by_round.setdefault(r, {})[ev] = t

    # ---- 击杀事件明细 ----
    kill_rows: dict[int, list[dict]] = {}
    if kills is not None and len(kills):
        k = kills.copy()
        k["round"] = k["tick"].astype(int).map(lambda t: tick2round.get(t))
        k = k.dropna(subset=["round"])
        k["round"] = k["round"].astype(int)
        # 击杀事件的 side：attacker/victim 的 team_name 从 snap 反查（该 tick 双方都在表中）
        sid2team = {}
        for t, sub in snap.groupby("tick"):
            sid2team[int(t)] = dict(zip(sub["steamid"], sub["team_name"]))
        for r in k.itertuples(index=False):
            att_team = sid2team.get(int(r.tick), {}).get(str(r.attacker_steamid), "")
            vic_team = sid2team.get(int(r.tick), {}).get(str(r.user_steamid), "")
            kill_rows.setdefault(int(r.round), []).append({
                "tick": int(r.tick), "event_type": "kill",
                "att_side": att_team, "vic_side": vic_team,
                "weapon": str(r.weapon) if r.weapon is not None else None,
                "headshot": bool(r.headshot) if r.headshot is not None else None,
                "thrusmoke": bool(r.thrusmoke) if r.thrusmoke is not None else None,
                "assistedflash": bool(r.assistedflash) if r.assistedflash is not None else None,
            })

    # ---- 组装：每个事件一行 ----
    event_rows: list[dict] = []
    round_state: dict[int, dict] = {}

    for t in event_ticks:
        r = tick2round[t]
        snap_t = snap[snap["tick"] == t]
        ct = snap_t[snap_t["team_name"] == "CT"]
        tr = snap_t[snap_t["team_name"] == "T"]
        ct_alive = int(ct["is_alive"].sum())
        t_alive = int(tr["is_alive"].sum())
        ct_hp = float(ct.loc[ct["is_alive"], "health"].sum())
        t_hp = float(tr.loc[tr["is_alive"], "health"].sum())

        # 炸弹状态：0 无 / 1 已下包 / 2 已拆 / 3 已爆
        bomb_state = 0
        bts = bomb_ticks_by_round.get(r, {})
        if bts.get("bomb_exploded") is not None and t >= bts["bomb_exploded"]:
            bomb_state = 3
        elif bts.get("bomb_defused") is not None and t >= bts["bomb_defused"]:
            bomb_state = 2
        elif bts.get("bomb_planted") is not None and t >= bts["bomb_planted"]:
            bomb_state = 1

        # 位置/视野聚合
        side_rows = {"CT": _alive_rows(snap_t, "CT"), "T": _alive_rows(snap_t, "T")}
        ct_pos = _cross_team(side_rows, "CT", "T")
        t_pos = _cross_team(side_rows, "T", "CT")
        ct_spread = _position_aggs(side_rows["CT"])["spread"]
        t_spread = _position_aggs(side_rows["T"])["spread"]

        # 该 tick 若是 freeze → round_start 基线事件
        is_freeze = t in freeze_ticks

        if is_freeze:
            # 回合开始态：经济决策 = freeze 时刻已购买的 equip 均值
            ct_equip = float(ct["current_equip_value"].mean()) if len(ct) else np.nan
            t_equip = float(tr["current_equip_value"].mean()) if len(tr) else np.nan
            round_state[r] = {
                "demo_path": dem_path, "map_name": map_name, "round_num": r,
                "ct_equip0": ct_equip, "t_equip0": t_equip,
                "ct_hp0": ct_hp, "t_hp0": t_hp,
            }

        if is_freeze or t in bomb_events or t in kill_ticks:
            ev_type = "round_start" if is_freeze else (bomb_events.get(t) or "kill")
            # 击杀明细
            att_side = vic_side = None
            weapon = headshot = thrusmoke = assistedflash = None
            if ev_type == "kill":
                for kr in kill_rows.get(r, []):
                    if kr["tick"] == t:
                        att_side, vic_side = kr["att_side"], kr["vic_side"]
                        weapon, headshot, thrusmoke, assistedflash = (
                            kr["weapon"], kr["headshot"], kr["thrusmoke"], kr["assistedflash"])
                        break
            event_rows.append({
                "demo_path": dem_path, "map_name": map_name, "round_num": r,
                "tick": t, "event_type": ev_type,
                "att_side": att_side, "vic_side": vic_side,
                "weapon": weapon, "headshot": headshot,
                "thrusmoke": thrusmoke, "assistedflash": assistedflash,
                "ct_alive": ct_alive, "t_alive": t_alive,
                "alive_gap": ct_alive - t_alive,
                "ct_hp": ct_hp, "t_hp": t_hp, "hp_gap": ct_hp - t_hp,
                "bomb_state": bomb_state,
                "ct_contact": ct_pos["contact"], "t_contact": t_pos["contact"],
                "ct_spread": ct_spread, "t_spread": t_spread,
                "ct_aiming": ct_pos["aiming"], "t_aiming": t_pos["aiming"],
            })

    events = pd.DataFrame(event_rows)
    if events.empty:
        return None

    # ---- 回合级：首杀、结果分类、炸弹、时长 ----
    # 首杀：每回合第一笔 kill 事件
    first_kill = {}
    for r, krs in kill_rows.items():
        krs_sorted = sorted(krs, key=lambda x: x["tick"])
        first_kill[r] = krs_sorted[0]

    rows2: list[dict] = []
    for r, rs in sorted(round_state.items()):
        w = winner_of.get(r)
        if w is None:
            continue
        fk = first_kill.get(r)
        fk_side = None
        if fk is not None and fk["att_side"] in ("CT", "T"):
            fk_side = fk["att_side"]

        # 本回合轨迹：winner 视角的 alive_gap 序列
        ev_r = events[events["round_num"] == r].sort_values("tick")
        gaps = []
        if len(ev_r):
            for er in ev_r.itertuples(index=False):
                g = er.alive_gap if w == "CT" else -er.alive_gap
                if pd.notna(g):
                    gaps.append(g)
        max_gap_winner = max(gaps) if gaps else 0
        min_gap_winner = min(gaps) if gaps else 0
        # 终局余量（winner alive - loser alive），注意爆炸/拆弹赢可能 0 存活
        final_margin = 0
        if len(ev_r):
            last = ev_r.iloc[-1]
            if w == "CT":
                final_margin = int(last.ct_alive - last.t_alive)
            else:
                final_margin = int(last.t_alive - last.ct_alive)

        bts = bomb_ticks_by_round.get(r, {})
        planted_b = "bomb_planted" in bts
        defused_b = "bomb_defused" in bts
        exploded_b = "bomb_exploded" in bts

        # 时长
        fstart = freeze_start.get(r)
        rlen = 0
        rlen_sec = 0.0
        if fstart is not None:
            rend = None
            for rr in re2.itertuples(index=False):
                if int(rr.round) == r:
                    rend = int(rr.tick)
                    break
            if rend is not None and rend >= fstart:
                rlen = rend - fstart
                rlen_sec = round(rlen / TICKS_PER_SECOND, 2)

        # 结果分类（winner 视角）
        outcome = "even"
        if fk_side is not None and fk_side != w:
            outcome = "comeback"                      # 丢首杀仍赢
        elif min_gap_winner <= -2:
            outcome = "comeback"                      # 曾少打 2+ 人仍赢
        elif (fk_side == w) and (max_gap_winner >= 2) and (final_margin >= 2):
            outcome = "stomp"                         # 首杀+全程优势+大余量
        # else: even

        rows2.append({
            "demo_path": dem_path, "map_name": map_name, "round_num": r,
            "winner_side": w, "first_kill_side": fk_side,
            "outcome_type": outcome,
            "max_gap_winner": max_gap_winner, "min_gap_winner": min_gap_winner,
            "final_margin": final_margin,
            "bomb_planted": planted_b, "bomb_defused": defused_b, "bomb_exploded": exploded_b,
            "round_len_ticks": rlen, "round_len_sec": rlen_sec,
            **{k2: v for k2, v in rs.items() if k2 not in ("demo_path", "map_name", "round_num")},
        })

    rounds = pd.DataFrame(rows2)
    if rounds.empty:
        return None
    return events, rounds


def build(max_demos: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全量解析，写两张 parquet。max_demos 用于冒烟测试。"""
    demos = discover_demos()
    if max_demos is not None:
        demos = demos[:max_demos]
    ev_frames, rd_frames = [], []
    n_fail = 0
    print(f"解析 {len(demos)} 张 demo ...")
    for i, dp in enumerate(demos):
        try:
            res = compute_inround(dp)
            if res is not None:
                ev_frames.append(res[0])
                rd_frames.append(res[1])
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  跳过 {Path(dp).name}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(demos)}")
    events = pd.concat(ev_frames, ignore_index=True)
    rounds = pd.concat(rd_frames, ignore_index=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(config.DATA_DIR / "inround_events.parquet", index=False)
    rounds.to_parquet(config.DATA_DIR / "round_states.parquet", index=False)
    print(f"完成：{len(events)} 事件 / {len(rounds)} 回合 / "
          f"{events['demo_path'].nunique()} 图（跳过 {n_fail}）")
    return events, rounds


if __name__ == "__main__":
    build()
