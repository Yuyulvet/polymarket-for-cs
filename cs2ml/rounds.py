"""round 级战术特征：每回合开局 15s 的阵型快照（默认站位：抱团 vs 拉开）。

用户洞察（少打多/多打多）：阵型是动态的，关键在"拉开"还是"抱团"。
在每回合 freeze_end + 15s 取一次快照——此时双方已展开到默认站位、通常尚未交火——
度量每队：
  - spread：队员平均两两距离（拉开程度）
  - max_pair：队员最大两两距离（是否有极端脱节 / 抱团半径）
  - n_places：占据的命名区域数（拉开 = 多区域，抱团 = 少区域）
  - stacked：是否全员抱团在 ~400 单位内（开局 rush / 前压堆叠）
聚合到队级：平均 spread、平均区域数、抱团率，作为 match 级"战术/站位"特征。

队伍身份与 demo_features 一致：用 roster（5 个 steamid 排序元组）而非 team_number，
后者在 BO3 跨图会翻转主/客。
"""
from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from demoparser2 import DemoParser

from . import config
from .duel import discover_demos

SETUP_SECONDS = 15
TICKS_PER_SECOND = 64
STACK_RADIUS = 400.0  # 与 engagements.NEARBY_RADIUS 一致
EXECUTE_RADIUS = 600.0  # contact 时刻抱团半径：报点打点/抱团 rush vs 默认拉开

_TICK_FIELDS = ["X", "Y", "Z", "steamid", "team_name", "last_place_name", "is_alive"]


def _pairwise_spread(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    d = [math.hypot(xs[i] - xs[j], ys[i] - ys[j])
         for i, j in combinations(range(len(xs)), 2)]
    return float(np.mean(d))


def _max_pairwise(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    return float(max(math.hypot(xs[i] - xs[j], ys[i] - ys[j])
                     for i, j in combinations(range(len(xs)), 2)))


def compute_round_rows(dem_path: str | Path,
                       setup_seconds: float = SETUP_SECONDS) -> pd.DataFrame:
    """一张 .dem -> 每回合每队一条阵型快照行。"""
    dem_path = Path(dem_path)
    parser = DemoParser(str(dem_path))

    player_info = parser.parse_player_info()
    sid2group = {str(r.steamid): int(r.team_number) for r in player_info.itertuples(index=False)}
    # 队伍身份：roster（5 个 steamid 排序元组），与 demo_features 对齐
    group_roster: dict[int, str] = {}
    for g, sub in player_info.groupby("team_number"):
        group_roster[int(g)] = ",".join(sorted(str(s) for s in sub["steamid"]))

    freeze = parser.parse_event("round_freeze_end")
    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())

    # 批量取所有回合的开局快照 tick（原每回合一次 parse_ticks，现整图一次）
    snap_ticks = [ft + int(setup_seconds * TICKS_PER_SECOND) for ft in freeze_ticks]
    snap = parser.parse_ticks(_TICK_FIELDS, ticks=snap_ticks)
    if snap is None or not len(snap):
        return pd.DataFrame()
    snap["steamid"] = snap["steamid"].astype(str)
    snap["group"] = snap["steamid"].map(sid2group)
    snap = snap.dropna(subset=["group"])
    snap = snap[snap["is_alive"].astype(bool)]
    by_tick = {int(t): g for t, g in snap.groupby("tick")}

    rows: list[dict] = []
    for i, ft in enumerate(freeze_ticks):
        round_num = i + 1
        snap_t = by_tick.get(int(ft + int(setup_seconds * TICKS_PER_SECOND)))
        if snap_t is None or not len(snap_t):
            continue
        for g, sub in snap_t.groupby("group"):
            xs = sub["X"].to_numpy(dtype=float)
            ys = sub["Y"].to_numpy(dtype=float)
            mode = sub["last_place_name"].mode()
            rows.append({
                "match_id": dem_path.parent.name,
                "demo_path": str(dem_path),
                "roster": group_roster.get(int(g), ""),
                "round": round_num,
                "group": int(g),
                "n_alive": int(len(sub)),
                "spread": round(_pairwise_spread(xs, ys), 1),
                "max_pair": round(_max_pairwise(xs, ys), 1),
                "n_places": int(sub["last_place_name"].nunique()),
                "stacked": _max_pairwise(xs, ys) <= STACK_RADIUS,
                "mode_place": str(mode.iloc[0]) if len(mode) else "",
            })
    return pd.DataFrame(rows)


def _round_of(tick: int, freeze_ticks: list[int]) -> int:
    """把 tick 映射到回合号（freeze_end 之后即下一回合开始）。"""
    r = 1
    for i, t in enumerate(freeze_ticks):
        if tick >= t:
            r = i + 1
        else:
            break
    return r


def compute_execute_rows(dem_path: str | Path) -> pd.DataFrame:
    """每回合 contact 快照（first kill - 1s），识别"报点打点"(execute) vs "默认站位"。

    在每回合首次击杀前 1 秒取快照——若在打点，进攻方已聚到某一点位；默认站位仍分散。
    判定 execute：≥3 名存活队员聚在同一个 bombsite 命名区域（site_max >= 3）。
    """
    dem_path = Path(dem_path)
    parser = DemoParser(str(dem_path))
    player_info = parser.parse_player_info()
    sid2group = {str(r.steamid): int(r.team_number) for r in player_info.itertuples(index=False)}
    group_roster: dict[int, str] = {}
    for g, sub in player_info.groupby("team_number"):
        group_roster[int(g)] = ",".join(sorted(str(s) for s in sub["steamid"]))

    freeze = parser.parse_event("round_freeze_end")
    kills = parser.parse_event("player_death")
    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())

    first_kill: dict[int, int] = {}
    if kills is not None and len(kills):
        for kr in kills.itertuples(index=False):
            r = _round_of(int(kr.tick), freeze_ticks)
            first_kill.setdefault(r, int(kr.tick))

    # 批量取所有回合的 contact 快照 tick（原每回合一次 parse_ticks，现整图一次）
    contact_ticks = []
    for i, ft in enumerate(freeze_ticks):
        fk = first_kill.get(i + 1)
        contact_ticks.append((fk - TICKS_PER_SECOND) if fk else (ft + SETUP_SECONDS * TICKS_PER_SECOND))
    snap = parser.parse_ticks(_TICK_FIELDS, ticks=contact_ticks)
    if snap is None or not len(snap):
        return pd.DataFrame()
    snap["steamid"] = snap["steamid"].astype(str)
    snap["group"] = snap["steamid"].map(sid2group)
    snap = snap.dropna(subset=["group"])
    snap = snap[snap["is_alive"].astype(bool)]
    by_tick = {int(t): g for t, g in snap.groupby("tick")}

    rows: list[dict] = []
    for i, ft in enumerate(freeze_ticks):
        round_num = i + 1
        snap_t = by_tick.get(contact_ticks[i])
        if snap_t is None or not len(snap_t):
            continue
        for g, sub in snap_t.groupby("group"):
            xs = sub["X"].to_numpy(dtype=float)
            ys = sub["Y"].to_numpy(dtype=float)
            mp = _max_pairwise(xs, ys)
            places = sub["last_place_name"].astype(str)
            site_places = places[places.str.lower().str.contains("bombsite", na=False)]
            site_max = int(site_places.value_counts().max()) if len(site_places) else 0
            site_name = str(site_places.value_counts().idxmax()) if len(site_places) else ""
            rows.append({
                "match_id": dem_path.parent.name,
                "demo_path": str(dem_path),
                "roster": group_roster.get(int(g), ""),
                "round": round_num,
                "group": int(g),
                "n_alive": int(len(sub)),
                "contact_spread": round(_pairwise_spread(xs, ys), 1),
                "contact_maxpair": round(mp, 1),
                "site_max": site_max,
                "is_execute": site_max >= 3,
                "is_clustered": mp <= EXECUTE_RADIUS,
                "site_name": site_name,
            })
    return pd.DataFrame(rows)


def run(root: Path | None = None) -> pd.DataFrame:
    """发现所有 .dem -> 逐图取【开局阵型】与【contact 打点】快照 -> 聚合到队级落盘。

    输出：
      reports/round_formation.csv     开局 15s 默认站位（拉开 vs 抱团）
      reports/round_execute.csv       contact 时刻打点识别（execute 率 vs 默认站位）
    """
    demos = discover_demos(root)
    setup_frames: list[pd.DataFrame] = []
    exec_frames: list[pd.DataFrame] = []
    for dp in demos:
        try:
            s = compute_round_rows(dp)
            if len(s):
                setup_frames.append(s)
        except Exception as e:  # 单图失败不阻断整体
            print(f"跳过 {dp.name}（阵型）: {e}")
        try:
            e = compute_execute_rows(dp)
            if len(e):
                exec_frames.append(e)
        except Exception as e:
            print(f"跳过 {dp.name}（打点）: {e}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    agg = pd.DataFrame()
    if setup_frames:
        df = pd.concat(setup_frames, ignore_index=True)
        agg = (df.groupby(["match_id", "demo_path", "roster"])
                 .agg(n_rounds=("round", "size"),
                      avg_spread=("spread", "mean"),
                      avg_n_places=("n_places", "mean"),
                      stack_rate=("stacked", "mean"))
                 .reset_index())
        agg["avg_spread"] = agg["avg_spread"].round(1)
        agg["avg_n_places"] = agg["avg_n_places"].round(2)
        agg["stack_rate"] = agg["stack_rate"].round(3)
        agg.to_csv(config.REPORTS_DIR / "round_formation.csv", index=False)
        df.to_csv(config.REPORTS_DIR / "round_formation_raw.csv", index=False)
        print(f"已写 reports/round_formation.csv（{len(agg)} 队·图）")
        print(agg.to_string(index=False))
    else:
        print("无有效阵型快照")

    if exec_frames:
        edf = pd.concat(exec_frames, ignore_index=True)
        eagg = (edf.groupby(["match_id", "demo_path", "roster"])
                  .agg(n_contact_rounds=("round", "size"),
                       execute_rate=("is_execute", "mean"),
                       cluster_rate=("is_clustered", "mean"),
                       avg_site_max=("site_max", "mean"),
                       avg_contact_spread=("contact_spread", "mean"),
                       avg_contact_maxpair=("contact_maxpair", "mean"),
                       top_site=("site_name", lambda s: s.value_counts().idxmax() if len(s) else ""))
                  .reset_index())
        eagg["execute_rate"] = eagg["execute_rate"].round(3)
        eagg["cluster_rate"] = eagg["cluster_rate"].round(3)
        eagg["avg_site_max"] = eagg["avg_site_max"].round(2)
        eagg["avg_contact_spread"] = eagg["avg_contact_spread"].round(1)
        eagg["avg_contact_maxpair"] = eagg["avg_contact_maxpair"].round(1)
        eagg.to_csv(config.REPORTS_DIR / "round_execute.csv", index=False)
        edf.to_csv(config.REPORTS_DIR / "round_execute_raw.csv", index=False)
        print(f"\n已写 reports/round_execute.csv（{len(eagg)} 队·图）")
        print(eagg.to_string(index=False))
    else:
        print("无有效 contact 打点快照")

    return agg


if __name__ == "__main__":
    run()
