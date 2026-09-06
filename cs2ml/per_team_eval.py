"""per-team 特征评估：base vs base+per-team LOO 语境胜率（GroupKFold by match）。

回答用户核心假设：下载更多数据后，per-roster 稀疏是否被救回、per-team 特征是否
开始贡献增量（AUC 抬升）。纪律与 round_model 一致（GroupKFold by match_id）。

关键：在「per-team 特征全部非空」的子集上做 apples-to-apples 比较——
+per-team 模型只能在这子集上训练（稀疏 NaN 行被 dropna），base 也要在同一子集算，
否则样本量不同、AUC 不可比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .per_team import compute_features, rate_columns
from .round_model import build_round_dataset

CAT = ["map_name", "round_class", "ct_tier", "t_tier"]
NUM = ["ct_equip", "t_equip", "round_num"]


def _auc(df: pd.DataFrame, features: list[str]) -> float:
    d = df.dropna(subset=features + ["label_ct_win"]).copy()
    if len(d) < 100 or d["label_ct_win"].nunique() < 2:
        return float("nan")
    cat = [c for c in features if c in CAT]
    num = [c for c in features if c not in CAT]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    y = d["label_ct_win"].to_numpy()
    n_splits = min(5, d["match_id"].nunique())
    proba = cross_val_predict(model, d[features], y, groups=d["match_id"].to_numpy(),
                              cv=GroupKFold(n_splits), method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def evaluate(min_n: int = 10) -> dict:
    df = build_round_dataset()
    print(f"round_dataset: {len(df)} 回合 / {df['match_id'].nunique()} 图")
    df = compute_features(df, min_n=min_n)

    rates = rate_columns()
    full_cov = df[rates].notna().all(axis=1).mean()
    print(f"per-team 特征完整覆盖（{len(rates)} 列全非空）: {full_cov:.3f}")

    base = CAT + NUM
    full = CAT + NUM + rates

    a_full = _auc(df, base)
    print(f"全量 base AUC = {a_full:.4f}  (对齐 round_model)")

    sub = df.dropna(subset=rates).copy()
    print(f"per-team 非空子集 n = {len(sub)}")
    a_base_sub = _auc(sub, base)
    a_pt_sub = _auc(sub, full)
    print(f"\n子集 base AUC      = {a_base_sub:.4f}")
    print(f"子集 +per-team AUC = {a_pt_sub:.4f}  (Δ {a_pt_sub - a_base_sub:+.4f})")

    print("\n分语境覆盖率 / 均值：")
    for c in rates:
        print(f"  {c:<18} 非空 {df[c].notna().mean():.3f}  均值 {df[c].mean():.3f}")
    return {"base_full": a_full, "base_sub": a_base_sub, "per_team_sub": a_pt_sub,
            "coverage": full_cov, "n_sub": len(sub)}


if __name__ == "__main__":
    evaluate()
