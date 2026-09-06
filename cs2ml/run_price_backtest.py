"""run_price_backtest.py —— 用户策略的「价格翻译」回测（分钟级盘中价，全地图）。

策略（2026-09-05 定稿）：均势买点（盘口≈0.5）读一回合经济 → 预测后续 2-3 回合 run 方向
→ 买低卖高赚差价。

对齐（关键，比 #34 更准）：
  每张图用「价格结算时刻」(token 锁 0/1) 作 map 结束锚，向后还原每个回合的 wall-clock。
  round_len_sec 只含 gameplay（freeze_end→round_end），不含 freeze/买枪时间与半场休息，
  故向后累积时每个回合补 freeze_sec，半场换边补 halftime_sec（这两个参数用 Map1 oracle
  校准——对齐正确时 oracle 应为正）。

输出：oracle(预知 run 方向) / signal(读经济) / random，按 map_num 与盘口窗口分层。
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .inplay_data import RAW_CACHE

RS = config.DATA_DIR / "round_states.parquet"


def _mnum(dp: str) -> int | None:
    m = re.search(r"-m(\d)-", Path(dp).name)
    return int(m.group(1)) if m else None


def build_side_map() -> pd.DataFrame:
    con = sqlite3.connect(config.DB_PATH)
    mr = pd.read_sql_query("SELECT demo_path, roster_key, team_number FROM map_rosters", con)
    con.close()
    mr = mr.dropna(subset=["roster_key"])
    m = mr[["roster_key", "team_number"]].drop_duplicates()
    rk2tn = m.groupby("roster_key")["team_number"].agg(lambda s: s.mode().iloc[0]).to_dict()
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    rd = rd[["demo_path", "round_num", "ct_roster"]].copy()
    rd["ct_is_team1"] = rd["ct_roster"].map(rk2tn).map(
        lambda x: x == 2 if pd.notna(x) else np.nan)
    return rd[["demo_path", "round_num", "ct_is_team1"]]


def _price_at(t_arr: np.ndarray, p_arr: np.ndarray, ts: float) -> float | None:
    idx = np.searchsorted(t_arr, ts, side="right") - 1
    if idx < 0:
        return None
    return float(p_arr[idx])


def detect_resolution(t: np.ndarray, p: np.ndarray, thr: float = 0.02) -> float | None:
    unresolved = (p > thr) & (p < 1 - thr)
    idx = np.where(unresolved)[0]
    if len(idx) == 0:
        return None
    last = idx[-1]
    return t[last + 1] if last + 1 < len(t) else t[last]


def build_rows(freeze_sec: float, halftime_sec: float) -> pd.DataFrame:
    raw = pd.read_parquet(RAW_CACHE)
    rs = pd.read_parquet(RS)
    rs = rs.sort_values(["demo_path", "round_num"]).reset_index(drop=True)
    rs["match_id"] = rs["demo_path"].map(lambda p: Path(p).parent.name)
    rs["map_num"] = rs["demo_path"].map(_mnum)
    side = build_side_map()
    rs = rs.merge(side, on=["demo_path", "round_num"], how="left")

    rows = []
    n_maps_ok = 0
    for mid, msub in raw.groupby("match_id"):
        team1 = msub["team1"].iloc[0]
        for mk, tsub in msub.groupby("map_market"):
            mnum = int(re.search(r"Map (\d)", mk).group(1))
            t1 = tsub[tsub["outcome"] == team1]
            if t1.empty:
                continue
            s1 = t1.sort_values("t").reset_index(drop=True)
            t_arr = s1["t"].astype("int64").to_numpy() // 1000  # ms -> s
            p_arr = s1["p"].to_numpy()
            res = detect_resolution(t_arr, p_arr)
            if res is None:
                continue
            sub = rs[(rs.match_id == mid) & (rs.map_num == mnum)].sort_values("round_num")
            if sub.empty or sub["ct_is_team1"].isna().all():
                continue
            n_maps_ok += 1
            lens = sub["round_len_sec"].to_numpy(dtype=float)
            n = len(lens)
            rnum_sorted = sub["round_num"].to_numpy()
            h2 = np.where(rnum_sorted >= 13)[0]
            h2_first = int(h2[0]) if len(h2) else n  # 第二半场第一回合的 index

            # 向后还原回合结束墙钟（含 freeze + 半场）
            end_ts = np.zeros(n)
            end_ts[n - 1] = res
            for k in range(n - 2, -1, -1):
                gap = lens[k + 1] + freeze_sec
                if k + 1 == h2_first:
                    gap += halftime_sec
                end_ts[k] = end_ts[k + 1] - gap

            ct_t1 = sub["ct_is_team1"].to_numpy(dtype=float)
            ct_eq = sub["ct_equip0"].to_numpy(dtype=float)
            t_eq = sub["t_equip0"].to_numpy(dtype=float)
            win_ct = (sub["winner_side"] == "CT").to_numpy().astype(int)
            rnum = sub["round_num"].to_numpy()

            for k in range(n):
                p_entry = _price_at(t_arr, p_arr, end_ts[k])
                if p_entry is None:
                    continue
                if k + 1 >= n:
                    continue
                ct1 = ct_t1[k + 1]
                if pd.isna(ct1):
                    continue
                gap_ct_t = ct_eq[k + 1] - t_eq[k + 1]
                gap_t1 = gap_ct_t if ct1 == 1 else -gap_ct_t
                for h in (1, 2, 3):
                    if k + h >= n:
                        continue
                    p_exit = _price_at(t_arr, p_arr, end_ts[k + h])
                    if p_exit is None:
                        continue
                    seg = win_ct[k + 1:k + 1 + h]
                    t1_wins = sum(1 for i, w in enumerate(seg)
                                  if (w == 1) == (ct_t1[k + 1 + i] == 1))
                    run_t1 = 1 if t1_wins > h - t1_wins else 0
                    sig_dir = 1 if gap_t1 > 0 else -1
                    oracle_dir = 1 if run_t1 == 1 else -1
                    dp = p_exit - p_entry
                    rows.append({
                        "match_id": mid, "map_num": mnum, "round_num": int(rnum[k]),
                        "horizon": h, "p_entry": p_entry, "p_exit": p_exit, "dp": dp,
                        "gap_t1": gap_t1, "sig_dir": sig_dir, "oracle_dir": oracle_dir,
                        "run_t1": run_t1,
                    })
    return pd.DataFrame(rows)


def _pnl(direction: np.ndarray, dp: np.ndarray) -> np.ndarray:
    return np.where(direction == 1, dp, -dp)


def oracle_metric(bt: pd.DataFrame, map_num: int | None, lo=0.40, hi=0.60) -> float:
    """对齐质量指标 = 均势点 h=1 oracle P&L（对齐正确应为正）。"""
    b = bt if map_num is None else bt[bt.map_num == map_num]
    e = b[(b.horizon == 1) & (b.p_entry >= lo) & (b.p_entry <= hi)]
    if e.empty:
        return float("nan")
    return float(_pnl(e["oracle_dir"].to_numpy(), e["dp"].to_numpy()).mean())


def sweep() -> None:
    """校准 freeze_sec / halftime_sec：使 Map1 均势点 oracle 最大（物理范围 0-30s）。"""
    print("校准 freeze_sec / halftime_sec（Map1 均势点 h=1 oracle，正=对齐对）")
    print(f"{'freeze':>7}{'halftime':>9}   oracle(Map1)")
    best = (None, -np.inf)
    # 先固定 halftime=0 扫 freeze
    for fr in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        bt = build_rows(fr, 0.0)
        o = oracle_metric(bt, 1)
        print(f"{fr:>7.0f}{0:>9.0f}   {o:+.5f}")
        if o > best[1]:
            best = (fr, o)
    print(f"\n最优 freeze={best[0]}s → 固定它扫 halftime：")
    fr_best = best[0]
    for ht in (0.0, 60.0, 120.0, 180.0, 240.0):
        bt = build_rows(fr_best, ht)
        o = oracle_metric(bt, 1)
        print(f"{fr_best:>7.0f}{ht:>9.0f}   {o:+.5f}")


def report(bt: pd.DataFrame) -> None:
    print(f"回测行总数：{len(bt)}")
    for mnum, lab in [(1, "Map1"), (2, "Map2"), (3, "Map3"), (None, "全部")]:
        b = bt if mnum is None else bt[bt.map_num == mnum]
        if b.empty:
            continue
        print("\n" + "=" * 76)
        print(f"{lab}  (n={len(b)}  盘口在[0.4,0.6]的比例 "
              f"{((b.p_entry>=0.4)&(b.p_entry<=0.6)).mean():.1%})")
        print("=" * 76)
        for lo, hi, tag in [(0.45, 0.55, "盘口[0.45,0.55]"), (0.40, 0.60, "盘口[0.40,0.60]")]:
            ev = b[(b.p_entry >= lo) & (b.p_entry <= hi)]
            if ev.empty:
                print(f"  {tag}: 无样本")
                continue
            for h in (1, 2, 3):
                e = ev[ev.horizon == h]
                if e.empty:
                    continue
                dp = e["dp"].to_numpy()
                orc = _pnl(e["oracle_dir"].to_numpy(), dp).mean()
                sig = _pnl(e["sig_dir"].to_numpy(), dp).mean()
                rnd = _pnl(np.random.RandomState(0).choice([1, -1], len(e)), dp).mean()
                hit = (e["sig_dir"] == e["oracle_dir"]).mean()
                print(f"  {tag} h={h}: n={len(e):<5} oracle={orc:+.5f}  "
                      f"signal={sig:+.5f}  random={rnd:+.5f}  "
                      f"signal命中={hit:.3f}  |Δp|={e['dp'].abs().mean():.4f}")


if __name__ == "__main__":
    sweep()
