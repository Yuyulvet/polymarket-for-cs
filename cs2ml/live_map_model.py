"""live_map_model.py —— step 2a：live 回合序列模型，round 结构是否叠加到比分。

核心问题（对接用户「回合场景事件」哲学）：
  赛中第 k 回合时，已知「比分（已胜回合差）+ 已完成回合的结构质量（full/eco 胜率、
  首杀转换 RWAFF、trade、utility damage、存活 HP 效率）」，预测最终 map winner。

比分已知会主导（after3 0.72 → after12 0.90）。这里检验：**round 结构质量能否在比分之外
额外抬升**（尤其早盘 k=6 窗口）——即「刷 eco 是噪音、clean full-buy 才是真强度」的假设。

快照 k ∈ {6, 12, 18}，GroupKFold by match。标签 home_win。结构特征 home-away 差分。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config
from .round_model import build_round_dataset
from .structure_form import load_round_events, team_round_table
from .totalrounds import build_map_table

SNAPSHOTS = (6, 12, 18)


def _cumul(tr_map: pd.DataFrame, roster: str, k: int) -> dict:
    sub = tr_map[(tr_map["roster_key"] == roster) & (tr_map["round_num"] <= k)]
    if not len(sub):
        return None
    score = int(sub["won"].sum())
    full_n = int(sub["is_full"].sum()); full_w = float(sub["won_full"].sum())
    eco_n = int(sub["is_eco"].sum()); eco_w = float(sub["won_eco"].sum())
    first_n = int(sub["first"].sum()); first_w = float(sub["won_first"].sum())
    return {
        "score": score, "n": len(sub),
        "full_win": full_w / full_n if full_n else np.nan,
        "eco_win": eco_w / eco_n if eco_n else np.nan,
        "rwaff": first_w / first_n if first_n else np.nan,
        "trade": float(sub["trade"].mean()),
        "util": float(sub["util"].mean()),
        "hp": float(sub["hp"].mean()),
    }


def build_snapshots() -> pd.DataFrame:
    re = load_round_events()
    tr = team_round_table(re)
    mt = build_map_table(re)
    mt = mt[(mt["total"] >= 13) & (mt["total"] <= 24)].copy()
    home_of = mt.set_index("demo_path")["home"].to_dict()
    away_of = mt.set_index("demo_path")["away"].to_dict()
    win_of = mt.set_index("demo_path")["home_win"].to_dict()
    mid_of = mt.set_index("demo_path")["match_id"].to_dict()

    rows = []
    for dp, tr_map in tr.groupby("demo_path"):
        home, away = home_of.get(dp), away_of.get(dp)
        if not home or not away:
            continue
        for k in SNAPSHOTS:
            h = _cumul(tr_map, home, k)
            a = _cumul(tr_map, away, k)
            if h is None or a is None or h["n"] < k or a["n"] < k:
                continue
            r = {"demo_path": dp, "match_id": mid_of.get(dp), "k": k,
                 "home_win": win_of.get(dp),
                 "score_gap": h["score"] - a["score"]}
            for f in ("full_win", "eco_win", "rwaff", "trade", "util", "hp"):
                r[f"{f}_gap"] = (h[f] - a[f]) if (pd.notna(h[f]) and pd.notna(a[f])) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def _auc(d: pd.DataFrame, feats: list[str]) -> tuple[float, int]:
    d = d.dropna(subset=feats + ["home_win"])
    if len(d) < 40 or d["home_win"].nunique() < 2:
        return float("nan"), len(d)
    X = d[feats].to_numpy(); y = d["home_win"].to_numpy()
    n = min(5, d["match_id"].nunique())
    p = cross_val_predict(LogisticRegression(max_iter=3000), X, y,
                          groups=d["match_id"].to_numpy(), cv=GroupKFold(n),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p), len(d)


STRUCT_GAPS = ["full_win_gap", "eco_win_gap", "rwaff_gap", "trade_gap", "util_gap", "hp_gap"]


def evaluate() -> dict:
    snap = build_snapshots()
    print(f"快照样本：{len(snap)} 行 / {snap['match_id'].nunique()} 场 / "
          f"快照 {SNAPSHOTS}")
    print("\n" + "=" * 74)
    print("live map winner：score-only vs score+结构质量（GroupKFold by match）")
    print("=" * 74)
    print(f"{'k':>3}{'score-only':>12}{'score+full_win':>16}{'score+全结构':>14}{'结构-only':>12}{'n':>6}")
    for k in SNAPSHOTS:
        d = snap[snap["k"] == k]
        a0, n0 = _auc(d, ["score_gap"])
        a1, _ = _auc(d, ["score_gap", "full_win_gap"])
        a2, _ = _auc(d, ["score_gap"] + STRUCT_GAPS)
        a3, _ = _auc(d, STRUCT_GAPS)
        print(f"{k:>3}{a0:>12.4f}{a1:>16.4f}{a2:>14.4f}{a3:>12.4f}{n0:>6}")

    # 早盘窗口：结构能否在 k=6 就提供比分之外的增量（关键 edge 窗口）
    print("\n" + "=" * 74)
    print("早盘 k=6 窗口：逐个结构特征叠加到比分，ΔAUC")
    print("=" * 74)
    d6 = snap[snap["k"] == 6]
    a0, _ = _auc(d6, ["score_gap"])
    print(f"  score_gap 单独 = {a0:.4f}")
    for f in STRUCT_GAPS:
        a, _ = _auc(d6, ["score_gap", f])
        print(f"  + {f:<14} AUC={a:.4f}  (Δ{a - a0:+.4f})")
    return {}


if __name__ == "__main__":
    evaluate()
