"""backtest_live.py —— #34 回测：回合级经济信号 → Polymarket Map Winner 盘中价 低买高卖。

用户方法：预测回合级趋势（下一回合谁赢），在 Map Winner 盘中价上买低卖高，
不持有到结算。

数据流：
  1. 经济信号：round_transition 的「比分+经济(6档)」模型（AUC 0.745），
     GroupKFold by match 出 out-of-fold P(CT 赢下一回合)。
  2. 盘中价：polymarket_inplay_raw.parquet（fidelity=1 分钟，两队 token）。
  3. 对齐：每张图用「价格结算时刻」（token 锁到 0/1）作 map 结束锚点，
     map 开始 = 赛程开始（Map1）或上一图结算（MapN），图内回合按序号均匀分布。
  4. 边映射：team_identity.build_team1_roster_map()（用 raw Map1 结算反推 team1 roster）定 CT 是哪队。
  5. 回测：预测下回合赢家 → 买该队 token → 回合结束卖 → P&L。

注意：盘口 fidelity=1 分钟，回合约 2 分钟，每回合只有 1-2 个价格点；
     赛程 vs 实际开始有漂移，均匀分布是近似，误差是白噪声（不产生虚假 edge）。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .round_transition import build_transitions
from .backtest_market import build_demo_maps
from .inplay_data import load_events, match_demo_to_market, RAW_CACHE
from .team_identity import build_team1_roster_map

CAT = ["map_name", "round_class", "ct_tier6", "t_tier6"]
NUM = ["score_gap", "equip_gap"]


# ---------------------------------------------------------------- 边映射
def build_side_map() -> pd.DataFrame:
    """demo_path -> ct_is_team1 (bool)，通过 team1 的 roster_key。

    team1 roster 用 build_team1_roster_map()（raw Map1 结算反推，唯一稳定口径）。
    ct_is_team1 = 该回合 ct_roster == team1 roster。无 raw 结算的 match 被 drop。
    """
    t1 = build_team1_roster_map()

    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    rd = rd[["demo_path", "round_num", "match_id", "ct_roster", "t_roster"]].copy()
    rd["team1_roster"] = rd["match_id"].map(t1)
    rd = rd.dropna(subset=["team1_roster"]).copy()
    rd["ct_is_team1"] = rd["ct_roster"] == rd["team1_roster"]
    rd["t_is_team1"] = rd["t_roster"] == rd["team1_roster"]
    # 只返回映射列，避免与调用方 tr 的 match_id 合并冲突
    return rd[["demo_path", "round_num", "ct_is_team1", "t_is_team1"]]


# ---------------------------------------------------------------- 价格结算时刻
def detect_resolution(series: pd.DataFrame, thr: float = 0.02) -> pd.Timestamp | None:
    """map token 的结算时刻 = 最后一个「未结算」价格点之后的第一个点。

    未结算 = 0.02 < p < 0.98。返回该时刻；序列为空/无未结算段返回 None。
    """
    if series.empty:
        return None
    p = series["p"].to_numpy()
    t = series["t"].to_numpy()
    unresolved = (p > thr) & (p < 1 - thr)
    idx = np.where(unresolved)[0]
    if len(idx) == 0:
        return None
    last_un = idx[-1]
    return t[last_un + 1] if last_un + 1 < len(t) else t[last_un]


def align_map_rounds(n_rounds: int, t0: pd.Timestamp, t1: pd.Timestamp) -> np.ndarray:
    """图内 n 个回合的边界时刻（n+1 个）：均匀分布 t0..t1。"""
    if n_rounds <= 0 or t0 >= t1:
        return np.array([])
    return np.linspace(t0.timestamp(), t1.timestamp(), n_rounds + 1)


def _price_at(series: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """ts 之前（含）最近一个价格快照。"""
    pre = series[series["t"] <= ts]
    if pre.empty:
        return None
    return float(pre.iloc[-1]["p"])


# ---------------------------------------------------------------- 经济信号 OOF
def compute_oof_pct() -> pd.DataFrame:
    """build_transitions + 经济特征 GroupKFold OOF P(CT 赢下一回合)。"""
    t = build_transitions()
    t = t.dropna(subset=CAT + NUM + ["label_ct_win"])
    y = t["label_ct_win"].to_numpy()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])
    clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
    ncv = min(5, t["match_id"].nunique())
    p = cross_val_predict(clf, t[CAT + NUM], y, groups=t["match_id"].to_numpy(),
                          cv=GroupKFold(ncv), method="predict_proba")[:, 1]
    t = t.copy()
    t["p_ct"] = p
    return t


def _map_number(dp: str) -> int | None:
    m = re.search(r"-m(\d+)-", Path(dp).name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- 主回测
def backtest() -> dict:
    raw = pd.read_parquet(RAW_CACHE)
    print(f"原始盘中价：{len(raw)} 点 / {raw['match_id'].nunique()} 场 / "
          f"{raw['map_market'].nunique()} 类 map 市场")

    # 事件匹配（match_id -> team1/team2/start_date）
    dmaps = build_demo_maps()
    matched = match_demo_to_market(dmaps, load_events())

    # 经济信号
    tr = compute_oof_pct()
    tr["map_num"] = tr["demo_path"].map(_map_number)

    # 边映射
    side = build_side_map()
    tr = tr.merge(side, on=["demo_path", "round_num"], how="left")

    # 每个 map 市场：检测结算时刻，对齐回合
    rows = []
    for _, mm in matched.iterrows():
        mid = mm["match_id"]
        start = mm["start_date"]
        for map_key, toks in mm["markets"].items():
            mnum = int(re.search(r"Map (\d)", map_key).group(1))
            token1 = toks.get(mm["mkt_team1"]) or toks.get(mm["team1"])
            if token1 is None:
                continue
            s1 = raw[(raw["match_id"] == mid) & (raw["map_market"] == map_key) &
                     (raw["token"] == token1)]
            if s1.empty:
                continue
            s1 = s1.sort_values("t")
            res = detect_resolution(s1)
            if res is None:
                continue
            # 图开始 = 赛程开始（Map1）；MapN 用上一图的结算（简化：赛程开始 + 累积）
            # 这里统一用赛程开始作 Map1 锚点；MapN 用赛程开始 + 45min*(N-1) 近似。
            t0 = start + pd.Timedelta(minutes=45 * (mnum - 1))
            sub = tr[(tr["match_id"] == mid) & (tr["map_num"] == mnum)].sort_values("round_num")
            if sub.empty or sub["ct_is_team1"].isna().all():
                continue
            n = len(sub)
            bounds = align_map_rounds(n, t0, res)
            if len(bounds) < 2:
                continue
            for i, (_, r) in enumerate(sub.iterrows()):
                r_start = pd.Timestamp(bounds[i], unit="s", tz="UTC")
                r_end = pd.Timestamp(bounds[i + 1], unit="s", tz="UTC")
                p0 = _price_at(s1, r_start)
                p1 = _price_at(s1, r_end)
                if p0 is None or p1 is None:
                    continue
                ct_is_t1 = r["ct_is_team1"]
                p_ct = r["p_ct"]
                if pd.isna(ct_is_t1) or pd.isna(p_ct):
                    continue
                pred_t1 = (p_ct > 0.5) if ct_is_t1 else (p_ct < 0.5)
                direction = 1 if pred_t1 else -1
                label_t1 = r["label_ct_win"] if ct_is_t1 else (1 - r["label_ct_win"])
                rows.append({
                    "match_id": mid, "map_num": mnum, "round_num": r["round_num"],
                    "p_ct": p_ct, "direction": direction,
                    "pred_t1": int(pred_t1), "actual_t1": int(label_t1),
                    "correct": int(pred_t1 == label_t1),
                    "conf": abs(p_ct - 0.5), "p0": p0, "p1": p1,
                    "pnl": direction * (p1 - p0),
                    "ct_is_team1": int(ct_is_t1),
                    "score_gap": r["score_gap"],
                })
    if not rows:
        print("无回测行"); return {}
    bt = pd.DataFrame(rows)
    return _report(bt)


def _report(bt: pd.DataFrame) -> dict:
    n = len(bt)
    acc = bt["correct"].mean()
    pnl = bt["pnl"].sum()
    ev = pnl / n
    print("=" * 72)
    print("#34 回测：经济信号 → Map Winner 盘中价 低买高卖")
    print("=" * 72)
    print(f"  回合交易次数: {n}")
    print(f"  经济信号回合方向命中率: {acc:.3f}")
    print(f"  每注 P&L 和: {pnl:+.3f}   每注 EV: {ev:+.5f}")
    print(f"  |p_end-p_start| 均值: {bt['pnl'].abs().mean():.4f}（回合平均价格移动幅度）")

    # 分层：置信度
    print("\n--- 按置信度分层（|p_ct-0.5|）---")
    for lo, hi, lab in [(0.0, 0.1, "低<0.1"), (0.1, 0.2, "中0.1-0.2"),
                        (0.2, 1.0, "高>0.2")]:
        b = bt[(bt["conf"] >= lo) & (bt["conf"] < hi)]
        if b.empty:
            continue
        print(f"  {lab:<10} n={len(b):<6} 命中率={b['correct'].mean():.3f} "
              f"EV={b['pnl'].mean():+.5f}")

    # 分层：比分差（高杠杆 vs 低杠杆，接近赛点/追分时单回合价动大）
    print("\n--- 按比分差分层（|score_gap|，高杠杆=接近）---")
    for lo, hi, lab in [(0, 2, "|gap|<=1"), (2, 5, "|gap|2-4"), (5, 99, "|gap|>=5")]:
        b = bt[bt["score_gap"].abs().between(lo, hi - 0.001)]
        if b.empty:
            continue
        print(f"  {lab:<12} n={len(b):<6} 命中率={b['correct'].mean():.3f} "
              f"EV={b['pnl'].mean():+.5f}  价动={b['pnl'].abs().mean():.4f}")

    # 基准：随机方向（无经济信号）的 EV 应≈0（对冲信号是否真的在动价）
    rnd = (bt["direction"].replace({1: 1, -1: -1}) * np.random.RandomState(0).choice(
        [1, -1], size=n) / bt["direction"].abs()).fillna(0)
    print(f"\n  随机方向 EV（无信号对照）: {((rnd * (bt['p1'] - bt['p0'])).mean()):+.5f}")

    print(f"\n  结论：{'有 edge' if ev > 0 else '无 edge / 负 edge'}（EV={ev:+.5f}）")
    return {"n": n, "acc": float(acc), "pnl": float(pnl), "ev": float(ev)}


if __name__ == "__main__":
    backtest()
