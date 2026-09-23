"""inround.py —— 回合内 per-event 状态轨迹（预测→检验→校正闭环的数据底座）。

对接用户的新策略（不是预测最终 map winner，而是逐回合、逐事件滚动的
「下一回合 / 接下来谁更接近赢」概率流，用来低买高卖）：

  宏观层：第 k 回合的结果(类型) → 经济决策 → 第 k+1 回合结果
  微观层：回合内部，每击杀/下包/拆包一个事件，实时更新「本回合谁更接近赢」

这里只产出**数据底座**（两张表），模型在后续脚本：
  1. inround_events —— 回合内每个事件的「事后状态」轨迹（事件级）
  2. round_states  —— 回合级（结果分类 + 经济决策 + 炸弹 + 时长 + 回合开始态）

技术要点：
  - SIDE 空间（team_name = "CT"/"TERRORIST"）直接绑定当前快照的五人 SteamID；
    不从 legacy round_dataset 推断队伍身份、换边或最终地图赢家。
  - 关键优化：只对「事件 tick」（freeze + 击杀 + 炸弹事件）parse_ticks，而非 full tick，
    就能拿到每个事件时刻的 位置/HP/存活/经济/朝向/致盲（BLAST 的 per-event 粒度）。
  - 回合边界用 round_freeze_end；半场换边由 round_end.winner("CT"/"T") 直接给 side。
  - 位置/视野聚合成紧凑特征（接触距离/队形松散度/瞄准数），丢弃原始坐标控体积。

v2 产物：data/inround_v2/inround_events.parquet、round_states.parquet。
状态采样在事件后一 tick；明确保留事件 tick、状态 tick 和回合起止 tick。
同一 tick 的多个事件是一个观察批次，不伪造批次内部的先后状态。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from demoparser2 import DemoParser

from . import config
from .duel import discover_demos
from .map1_data import first_ct_is_ct, roster_key, terminal_score

FIRST_HALF_ROUNDS = 12
TICKS_PER_SECOND = 64
SCHEMA_VERSION = 2
EXTRACTOR_VERSION = "inround_v2_identity_audit_v1"
AUDIT_EXAMPLE_LIMIT = 20

# 位置/视野聚合阈值
CONTACT_RANGE = 600.0            # 交战距离（CS2 距离单位）
_FACING_COS = math.cos(math.radians(45.0))   # 45° 锥内算「正看着对方」

# 每张 demo 只 parse_ticks 这些字段（事件 tick 上）
_TICK_FIELDS = [
    "X", "Y", "Z", "health", "armor_value", "current_equip_value",
    "is_alive", "team_name", "steamid", "active_weapon_name", "yaw", "flash_duration",
]
_EVENT_COLUMNS = ["demo_path", "map_name", "round_num", "tick", "event_type", "event_tick", "state_tick",
                  "round_start_tick", "round_end_tick", "schema_version", "extractor_version",
                  "ct_roster", "t_roster", "roster_a", "roster_b", "ct_is_a", "observation_basis",
                  "kills_at_event_tick", "att_side", "vic_side", "weapon", "headshot", "thrusmoke",
                  "assistedflash", "ct_alive", "t_alive", "alive_gap", "ct_hp", "t_hp", "hp_gap",
                  "bomb_state", "ct_contact", "t_contact", "ct_spread", "t_spread", "ct_aiming", "t_aiming"]


def _round_of(tick: int, freeze_ticks: list[int]) -> int:
    """tick → 回合号（freeze_end 即该回合开始）。"""
    r = 0  # warmup/pre-freeze events are not round 1
    for i, t in enumerate(freeze_ticks):
        if tick >= t:
            r = i + 1
        else:
            break
    return r


def observation_tick(event_tick: int) -> int:
    """Request a full-tick snapshot strictly after the event batch.

    Demo ticks are not feed receipt timestamps. Sub-tick order and a live data
    provider's latency cannot be recovered from this offline convention.
    """
    return int(event_tick) + 1


def identity_problem(snapshot: pd.DataFrame) -> str | None:
    """Ledger identity does not depend on HP/equipment feature availability."""
    required = {"steamid", "team_name"}
    if required - set(snapshot):
        return "snapshot_missing_columns"
    if len(snapshot) != 10:
        return "snapshot_not_ten_players"
    if snapshot.steamid.astype(str).nunique() != 10:
        return "snapshot_duplicate_steamids"
    if not snapshot.steamid.astype(str).str.fullmatch(r"[0-9]+").all():
        return "snapshot_invalid_steamids"
    if snapshot.team_name.value_counts().to_dict() != {"CT": 5, "T": 5}:
        return "snapshot_not_five_per_side"
    return None


def snapshot_problem(snapshot: pd.DataFrame) -> str | None:
    """Stable feature diagnostics; missing players are never interpreted as dead."""
    problem = identity_problem(snapshot)
    if problem is not None:
        return problem
    if {"is_alive", "health", "current_equip_value"} - set(snapshot):
        return "snapshot_missing_feature_columns"
    if not snapshot.is_alive.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        return "snapshot_invalid_alive_flags"
    hp = pd.to_numeric(snapshot.health, errors="coerce").to_numpy(dtype=float)
    equipment = pd.to_numeric(snapshot.current_equip_value, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(hp).all() or not ((hp >= 0) & (hp <= 100)).all():
        return "snapshot_invalid_hp"
    if not np.isfinite(equipment).all() or (equipment < 0).any():
        return "snapshot_invalid_equipment"
    alive = snapshot.is_alive.to_numpy(dtype=bool)
    return None if (hp[alive] > 0).all() else "snapshot_alive_without_hp"


def valid_snapshot(snapshot: pd.DataFrame) -> bool:
    """Compatibility predicate; ``snapshot_problem`` supplies an optional reason."""
    return snapshot_problem(snapshot) is None


def _snapshot_rosters(snapshot: pd.DataFrame) -> tuple[str, str]:
    return tuple(roster_key(snapshot.loc[snapshot.team_name.eq(side), "steamid"].tolist())
                 for side in ("CT", "T"))


def _excluded(audit: dict, reason: str, tick: int, round_num: int) -> None:
    counts = audit["excluded_states_by_reason"]
    counts[reason] = counts.get(reason, 0) + 1
    if len(audit["excluded_state_examples"]) < AUDIT_EXAMPLE_LIMIT:
        audit["excluded_state_examples"].append({"reason_code": reason, "event_tick": int(tick),
                                                  "round_num": int(round_num)})


def _reject(audit: dict, reason: str, message: str):
    audit.update(status="rejected", reason_code=reason)
    raise ValueError(message)


def _empty(audit: dict, reason: str):
    audit.update(status="rejected", reason_code=reason)
    return None


def add_round_identity_scores(rounds: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical identities and scores using only preceding round labels.

    Scores become unavailable after a gap; later observed results do not fill a
    missing earlier round. Final labels are only produced by completion_audit.
    """
    result = rounds.sort_values("round_num").copy()
    if result.empty:
        return result
    a, b = sorted((roster_key(result.iloc[0].ct_roster), roster_key(result.iloc[0].t_roster)))
    score_a = score_b = 0
    expected = 1
    known = True
    rows = []
    for row in result.to_dict("records"):
        ct, tt = roster_key(row["ct_roster"]), roster_key(row["t_roster"])
        if {ct, tt} != {a, b}:
            raise ValueError("Roster composition changed within map")
        side = str(row["winner_side"]).upper().replace("TERRORIST", "T")
        winner = ct if side == "CT" else tt if side == "T" else None
        known = known and int(row["round_num"]) == expected
        rows.append({**row, "roster_a": a, "roster_b": b, "ct_is_a": ct == a,
                     "winner_roster": winner,
                     "score_a_before": score_a if known else np.nan,
                     "score_b_before": score_b if known else np.nan})
        if known:
            if winner is None:
                known = False
            else:
                score_a += int(winner == a)
                score_b += int(winner == b)
        expected = int(row["round_num"]) + 1
    return pd.DataFrame(rows)


def completion_audit(rounds: pd.DataFrame, expected_rounds: int) -> dict:
    """Map-label gate from direct roster/round winners, not legacy round caches."""
    result = {"complete": False, "reason_code": None, "roster_a": None, "roster_b": None,
              "score_a": None, "score_b": None, "winner_roster": None,
              "n_rounds": len(rounds), "overtime": len(rounds) > 24,
              "n_overtime_rounds": max(0, len(rounds)-24)}
    if rounds.empty:
        result["reason_code"] = "no_rounds"
        return result
    ordered = rounds.sort_values("round_num")
    if ordered.round_num.tolist() != list(range(1, expected_rounds+1)):
        result["reason_code"] = "incomplete_round_sequence"
        return result
    a, b = sorted((roster_key(ordered.iloc[0].ct_roster), roster_key(ordered.iloc[0].t_roster)))
    result.update(roster_a=a, roster_b=b)
    first_ct = roster_key(ordered.iloc[0].ct_roster)
    other = b if first_ct == a else a
    scores = {a: 0, b: 0}
    side_valid = True
    early_terminal = False
    for number, row in enumerate(ordered.itertuples(index=False), 1):
        ct, tt = roster_key(row.ct_roster), roster_key(row.t_roster)
        if {ct, tt} != {a, b}:
            result["reason_code"] = "roster_changed_within_map"
            return result
        side_valid = side_valid and ct == (first_ct if first_ct_is_ct(number) else other)
        side = str(row.winner_side).upper().replace("TERRORIST", "T")
        if side not in {"CT", "T"}:
            result["reason_code"] = "unknown_round_winner"
            return result
        scores[ct if side == "CT" else tt] += 1
        if terminal_score(scores[a], scores[b]) and number < len(ordered):
            early_terminal = True
    result.update(score_a=scores[a], score_b=scores[b])
    if early_terminal:
        result["reason_code"] = "rounds_after_terminal_score"
    elif not side_valid:
        result["reason_code"] = "nonstandard_side_schedule"
    elif not terminal_score(scores[a], scores[b]):
        result["reason_code"] = "incomplete_or_nonstandard_final_score"
    else:
        result.update(complete=True, winner_roster=a if scores[a] > scores[b] else b)
    return result


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


def _compute_inround(dem_path: str | Path, audit: dict) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    dem_path = str(dem_path)
    parser = DemoParser(dem_path)
    audit["stage"] = "header"
    header = parser.parse_header()
    map_name = header.get("map_name", Path(dem_path).stem)

    audit["stage"] = "round_events"
    freeze = parser.parse_event("round_freeze_end")
    re_ = parser.parse_event("round_end")
    audit["counts"].update(freeze_events=0 if freeze is None else len(freeze),
                           round_end_events=0 if re_ is None else len(re_))
    if freeze is None or not len(freeze):
        return _empty(audit, "missing_freeze_events")
    if re_ is None or not len(re_):
        return _empty(audit, "missing_round_end_events")
    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())
    if len(set(freeze_ticks)) != len(freeze_ticks):
        _reject(audit, "duplicate_freeze_ticks", "Duplicate freeze ticks; restart or ambiguous round numbering")
    if not freeze_ticks:
        return _empty(audit, "missing_freeze_events")

    audit["stage"] = "gameplay_events"
    kills = parser.parse_event("player_death")
    planted = parser.parse_event("bomb_planted")
    defused = parser.parse_event("bomb_defused")
    exploded = parser.parse_event("bomb_exploded")

    # ---- 事件 tick 集合：freeze + 击杀 + 炸弹事件 ----
    bomb_events: dict[int, str] = {}  # one tick batch; terminal bomb event wins ties
    for ev, name in ((planted, "bomb_planted"), (defused, "bomb_defused"),
                     (exploded, "bomb_exploded")):
        if ev is not None and len(ev):
            for t in ev["tick"].tolist():
                bomb_events[int(t)] = name

    kill_ticks: list[int] = []
    if kills is not None and len(kills):
        kill_ticks = sorted(int(t) for t in kills["tick"].tolist())

    # Never treat inter-round cleanup as predictive state of the preceding round.
    audit["stage"] = "round_boundaries"
    raw_round_numbers = pd.to_numeric(re_["round"], errors="coerce")
    invalid_numbers = raw_round_numbers.isna() | ~np.isfinite(raw_round_numbers) | raw_round_numbers.mod(1).ne(0)
    if invalid_numbers.any():
        _reject(audit, "invalid_round_end_numbers", "Round end numbers must be finite integers")
    raw_confirmed = re_[raw_round_numbers.ge(1)]
    audit["raw_round_end_summary"] = {
        "non_play_round_rows": int(raw_round_numbers.lt(1).sum()),
        "in_play_round_rows": len(raw_confirmed),
        "missing_winner_rows": int(raw_confirmed.winner.isna().sum()),
        "round_numbers": [int(n) for n in raw_round_numbers[raw_round_numbers.ge(1)].tolist()],
    }
    re2 = re_[re_["winner"].notna()].copy()
    re2["round"] = re2["round"].astype(int)
    re2 = re2[re2["round"] >= 1]
    re2["winner"] = re2.winner.astype(str).str.upper().replace({"TERRORIST": "T"})
    if not re2.winner.isin(["CT", "T"]).all():
        _reject(audit, "unknown_round_winner", "Round winner is not CT or T")
    audit["counts"]["valid_round_end_events"] = len(re2)
    if re2["round"].duplicated().any():
        _reject(audit, "duplicate_round_end_numbers", "Duplicate round_end numbers; cannot identify round boundaries safely")
    end_of = {int(row.round): int(row.tick) for row in re2.itertuples(index=False)}
    for number, end in end_of.items():
        if number > len(freeze_ticks) or end <= freeze_ticks[number-1]:
            audit["invalid_boundary"] = {"round_num": number, "round_end_tick": end,
                                          "freeze_count": len(freeze_ticks)}
            _reject(audit, "round_end_unmatched_freeze", "Round end cannot be matched to its freeze boundary")
        if number < len(freeze_ticks) and end >= freeze_ticks[number]:
            audit["invalid_boundary"] = {"round_num": number, "round_end_tick": end,
                                          "next_freeze_tick": freeze_ticks[number]}
            _reject(audit, "round_end_overlaps_next_freeze", "Round end overlaps the next round freeze")
    event_ticks = sorted(set(freeze_ticks + kill_ticks + list(bomb_events.keys())))
    tick2round = {t: _round_of(t, freeze_ticks) for t in event_ticks}
    audit["counts"]["candidate_event_ticks"] = len(event_ticks)
    eligible_ticks = []
    for tick in event_ticks:
        number = tick2round[tick]
        if number < 1:
            _excluded(audit, "before_first_freeze", tick, number)
        elif number not in end_of:
            _excluded(audit, "round_without_valid_end", tick, number)
        elif observation_tick(tick) >= end_of[number]:
            _excluded(audit, "at_or_after_round_end", tick, number)
        else:
            eligible_ticks.append(tick)
    event_ticks = eligible_ticks
    audit["counts"]["eligible_event_ticks"] = len(event_ticks)
    if not event_ticks:
        return _empty(audit, "no_active_round_event_ticks")

    # ---- 一次批量 parse_ticks（事件 tick 上）----
    snapshot_ticks = sorted(set([observation_tick(t) for t in event_ticks] +
                                [observation_tick(t) for t in end_of.values()]))
    audit["stage"] = "snapshots"
    audit["counts"]["requested_snapshot_ticks"] = len(snapshot_ticks)
    snap = parser.parse_ticks(_TICK_FIELDS, ticks=snapshot_ticks)
    if snap is None or not len(snap):
        return _empty(audit, "missing_tick_snapshots")
    audit["counts"]["returned_snapshot_rows"] = len(snap)
    snap["team_name"] = snap["team_name"].astype(str).replace("TERRORIST", "T")
    snap["steamid"] = snap["steamid"].astype(str)
    snap["round"] = snap["tick"].map({observation_tick(t): tick2round[t] for t in event_ticks})

    teams = {t for t in snap["team_name"].unique() if t in ("CT", "T")}
    if len(teams) != 2:
        return _empty(audit, "missing_team_sides")

    # ---- 每回合赢家 + 炸弹 + 时长 ----
    winner_of = {int(r.round): str(r.winner).upper() for r in re2.itertuples(index=False)}
    freeze_start = {i + 1: t for i, t in enumerate(freeze_ticks)}
    # Round identities come only from complete first-post-freeze observations.
    # Every later valid snapshot must retain that round's team composition.
    round_rosters = {}
    canonical_pair = None
    for number, tick in freeze_start.items():
        if number not in end_of or observation_tick(tick) >= end_of[number]:
            continue
        initial = snap[snap.tick.eq(observation_tick(tick))]
        problem = identity_problem(initial)
        if problem is not None:
            audit["invalid_round_starts"][str(number)] = problem
            continue
        current = _snapshot_rosters(initial)
        pair = tuple(sorted(current))
        if canonical_pair is not None and pair != canonical_pair:
            _reject(audit, "roster_changed_within_map", "Roster composition changed within map")
        canonical_pair = pair
        round_rosters[number] = current
        feature_problem = snapshot_problem(initial)
        if feature_problem:
            audit["invalid_initial_feature_snapshots"][str(number)] = feature_problem

    bomb_ticks_by_round: dict[int, dict[str, int]] = {}
    for t, ev in bomb_events.items():
        r = tick2round[t]
        if r >= 1 and t <= end_of.get(r, -1):
            bomb_ticks_by_round.setdefault(r, {})[ev] = t

    # ---- 击杀事件明细 ----
    kill_rows: dict[int, list[dict]] = {}
    if kills is not None and len(kills):
        k = kills.copy()
        k["round"] = k["tick"].astype(int).map(lambda t: tick2round.get(t))
        k = k.dropna(subset=["round"])
        k["round"] = k["round"].astype(int)
        k = k[k.apply(lambda row: row["round"] >= 1 and
                     int(row["tick"]) <= end_of.get(int(row["round"]), -1), axis=1)]
        # 击杀事件的 side：attacker/victim 的 team_name 从 snap 反查（该 tick 双方都在表中）
        sid2team = {}
        for t, sub in snap.groupby("tick"):
            sid2team[int(t)] = dict(zip(sub["steamid"], sub["team_name"]))
        for r in k.itertuples(index=False):
            att_team = sid2team.get(observation_tick(r.tick), {}).get(str(r.attacker_steamid), "")
            vic_team = sid2team.get(observation_tick(r.tick), {}).get(str(r.user_steamid), "")
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
    for number, (ct_roster, t_roster) in round_rosters.items():
        initial = snap[snap.tick.eq(observation_tick(freeze_start[number]))]
        feature_valid = snapshot_problem(initial) is None
        values = {}
        for side, key in (("CT", "ct"), ("T", "t")):
            team = initial[initial.team_name.eq(side)]
            values[f"{key}_equip0"] = float(team.current_equip_value.mean()) if feature_valid else np.nan
            values[f"{key}_hp0"] = float(team.loc[team.is_alive, "health"].sum()) if feature_valid else np.nan
        round_state[number] = {"demo_path": dem_path, "map_name": map_name, "round_num": number,
                               "ct_roster": ct_roster, "t_roster": t_roster,
                               "round_start_tick": freeze_start[number], "round_end_tick": end_of[number],
                               "schema_version": SCHEMA_VERSION, "extractor_version": EXTRACTOR_VERSION, **values}
    rejected_snapshots = 0
    audit["stage"] = "event_states"

    for t in event_ticks:
        r = tick2round[t]
        state_tick = observation_tick(t)
        snap_t = snap[snap["tick"] == state_tick]
        problem = identity_problem(snap_t)
        if problem is not None:
            rejected_snapshots += 1
            _excluded(audit, problem, t, r)
            continue
        if r not in round_rosters:
            _excluded(audit, "round_without_valid_initial_snapshot", t, r)
            continue
        ct_roster, t_roster = _snapshot_rosters(snap_t)
        if (ct_roster, t_roster) != round_rosters[r]:
            _reject(audit, "roster_changed_within_round", "Roster side assignment changed within round")
        problem = snapshot_problem(snap_t)
        if problem:
            rejected_snapshots += 1
            _excluded(audit, problem, t, r)
            continue
        ct = snap_t[snap_t["team_name"] == "CT"]
        tr = snap_t[snap_t["team_name"] == "T"]
        ct_alive = int(ct["is_alive"].sum())
        t_alive = int(tr["is_alive"].sum())
        ct_hp = float(ct.loc[ct["is_alive"], "health"].sum())
        t_hp = float(tr.loc[tr["is_alive"], "health"].sum())

        # 炸弹状态：0 无 / 1 已下包 / 2 已拆 / 3 已爆
        bomb_state = 0
        bts = bomb_ticks_by_round.get(r, {})
        if bts.get("bomb_exploded") is not None and state_tick >= bts["bomb_exploded"]:
            bomb_state = 3
        elif bts.get("bomb_defused") is not None and state_tick >= bts["bomb_defused"]:
            bomb_state = 2
        elif bts.get("bomb_planted") is not None and state_tick >= bts["bomb_planted"]:
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
                "round_start_tick": t, "round_end_tick": end_of[r],
                "schema_version": SCHEMA_VERSION,
                "extractor_version": EXTRACTOR_VERSION,
                "ct_roster": ct_roster, "t_roster": t_roster,
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
                "event_tick": t, "state_tick": state_tick,
                "round_start_tick": freeze_start[r], "round_end_tick": end_of[r],
                "schema_version": SCHEMA_VERSION,
                "extractor_version": EXTRACTOR_VERSION,
                "ct_roster": ct_roster, "t_roster": t_roster,
                "roster_a": canonical_pair[0], "roster_b": canonical_pair[1],
                "ct_is_a": ct_roster == canonical_pair[0],
                "observation_basis": "first_full_tick_after_event_batch",
                "kills_at_event_tick": sum(kr["tick"] == t for kr in kill_rows.get(r, [])),
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

    events = pd.DataFrame(event_rows, columns=_EVENT_COLUMNS)
    audit["counts"]["rejected_snapshots"] = rejected_snapshots
    if events.empty and not round_state:
        return _empty(audit, "no_valid_event_states")

    # ---- 回合级：首杀、结果分类、炸弹、时长 ----
    # 首杀：每回合第一笔 kill 事件
    first_kill = {}
    for r, krs in kill_rows.items():
        krs_sorted = sorted(krs, key=lambda x: x["tick"])
        first_kill[r] = krs_sorted[0]

    rows2: list[dict] = []
    audit["stage"] = "round_summaries"
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
        # 终局余量（winner alive - loser alive），注意爆炸/拆弹赢可能 0 存活
        final_margin = 0
        final_snapshot = snap[snap["tick"] == observation_tick(end_of[r])]
        final_identity_problem = identity_problem(final_snapshot)
        if final_identity_problem is None:
            if _snapshot_rosters(final_snapshot) != round_rosters[r]:
                final_identity_problem = "summary_side_assignment_changed"
        if final_identity_problem:
            audit["invalid_round_end_identities"][str(r)] = final_identity_problem
        if final_identity_problem is None and valid_snapshot(final_snapshot):
            alive_counts = final_snapshot.loc[final_snapshot.is_alive].team_name.value_counts()
            ct_final, t_final = int(alive_counts.get("CT", 0)), int(alive_counts.get("T", 0))
            final_margin = ct_final - t_final if w == "CT" else t_final - ct_final
            gaps.append(final_margin)
        else:
            final_margin = np.nan
            audit["invalid_round_end_snapshots"][str(r)] = final_identity_problem or snapshot_problem(final_snapshot)
        max_gap_winner = max(gaps) if gaps else np.nan
        min_gap_winner = min(gaps) if gaps else np.nan

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
        outcome = "even" if np.isfinite(final_margin) else "unknown"
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
            "summary_state_tick": observation_tick(end_of[r]),
            **{k2: v for k2, v in rs.items() if k2 not in ("demo_path", "map_name", "round_num")},
        })

    rounds = pd.DataFrame(rows2)
    if rounds.empty:
        return _empty(audit, "no_valid_round_summaries")
    rounds = add_round_identity_scores(rounds)
    audit["map_completion"] = completion_audit(rounds, len(freeze_ticks))
    raw_summary = audit["raw_round_end_summary"]
    if raw_summary["missing_winner_rows"]:
        audit["map_completion"].update(complete=False, winner_roster=None,
                                       reason_code="unresolved_raw_round_end_winners")
    elif sorted(raw_summary["round_numbers"]) != list(range(1, len(freeze_ticks)+1)):
        audit["map_completion"].update(complete=False, winner_roster=None,
                                       reason_code="raw_round_end_sequence_mismatch")
    # Emit past-only scores in event rows too; current/final results stay in the
    # round label table and the audit, never become model input features.
    events = events.merge(rounds[["round_num", "score_a_before", "score_b_before"]],
                          on="round_num", how="inner", validate="many_to_one")
    return events, rounds


def compute_inround(dem_path: str | Path, *, audit: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Extract one demo; optionally fill structured diagnostics in ``audit``.

    Existing tuple/None/exception behavior is preserved. A supplied audit dict
    is reset for this call and remains populated on None or parser exceptions.
    Additive v2 identity columns retain the established post-event timing contract.
    """
    if audit is not None and not isinstance(audit, dict):
        raise TypeError("audit must be a dict or None")
    report = {} if audit is None else audit
    report.clear()
    report.update(extractor_version=EXTRACTOR_VERSION, schema_version=SCHEMA_VERSION,
                  demo_path=str(dem_path), status="running", reason_code=None, stage="parser",
                  counts={"returned_event_rows": 0, "returned_round_rows": 0},
                  excluded_states_by_reason={}, excluded_state_examples=[],
                  invalid_round_starts={}, invalid_initial_feature_snapshots={},
                  invalid_round_end_snapshots={}, invalid_round_end_identities={},
                  map_completion=completion_audit(pd.DataFrame(), 0))
    try:
        result = _compute_inround(dem_path, report)
    except Exception as exc:
        report["counts"]["excluded_event_ticks"] = sum(report["excluded_states_by_reason"].values())
        if report["status"] != "rejected":
            report.update(status="error", reason_code="parser_or_extraction_error")
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
        raise
    report["counts"]["excluded_event_ticks"] = sum(report["excluded_states_by_reason"].values())
    if result is not None:
        events, rounds = result
        report.update(status="accepted", stage="complete")
        report["counts"].update(returned_event_rows=len(events), returned_round_rows=len(rounds))
        # Preserve compact compatibility fields while supplying full diagnostics.
        metadata = {"rejected_snapshots": report["counts"].get("rejected_snapshots", 0),
                    "observation_basis": "first_full_tick_after_event_batch", **report}
        events.attrs["extraction_audit"] = metadata
        rounds.attrs["extraction_audit"] = metadata
    return result


def build(max_demos: int | None = None, output_dir: Path | None = None,
          overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write versioned v2 artifacts, never overwrite old research by default."""
    output_dir = Path(output_dir) if output_dir is not None else config.DATA_DIR / "inround_v2"
    paths = [output_dir / "inround_events.parquet", output_dir / "round_states.parquet"]
    if output_dir.resolve() == config.DATA_DIR.resolve():
        raise ValueError("Use a versioned output directory; legacy artifacts are protected")
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("Output exists; use a new versioned directory or explicit overwrite=True")
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
    if not ev_frames or not rd_frames:
        raise ValueError("No valid demos produced v2 in-round data")
    events = pd.concat(ev_frames, ignore_index=True)
    rounds = pd.concat(rd_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(paths[0], index=False)
    rounds.to_parquet(paths[1], index=False)
    print(f"完成：{len(events)} 事件 / {len(rounds)} 回合 / "
          f"{events['demo_path'].nunique()} 图（跳过 {n_fail}）")
    return events, rounds


if __name__ == "__main__":
    build()
