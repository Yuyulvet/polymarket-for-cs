"""测 30s 信号对「回合让分」与「总回合」盘口的增量（Map Winner 已证无效）。

Map Winner 盘口对单回合不敏感（一回合/整图≈零头）。但：
  - Rounds Handicap（如 -3.5）：结算在【最终回合差】上，每回合 ±1，直接受影响。
  - Total Rounds O/U（如 21.5）：结算在【总回合数】上。
所以这两个盘口理论上更能吃到 30s 信号。

做法：每个回合 N 的状态（带/不带 30s），预测本图【最终回合差】和【总回合数】，
比较 MAE / AUC（是否 cover -3.5）的改善。GroupKFold by match_id。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .edge import _load

CAT_BASE = ["map_name", "round_class", "ct_tier", "t_tier"]
NUM_BASE = ["score_diff", "round_num"]
NUM_30 = ["ct_alive", "t_alive"]
CAT_30 = ["first_kill_side"]


def _build() -> pd.DataFrame:
    df = _load()
    rows = []
    for mid, sub in df.groupby("match_id"):
        sub = sub.sort_values("round_num")
        home = sub.iloc[0]["ct_roster"]
        away = sub.iloc[0]["t_roster"]
        hr = int((sub["winner_roster"] == home).sum())
        ar = int((sub["winner_roster"] == away).sum())
        final_diff = hr - ar
        total = len(sub)
        for r in sub.itertuples(index=False):
            prior = sub[sub["round_num"] < r.round_num]
            hs = int((prior["winner_roster"] == home).sum())
            as_ = int((prior["winner_roster"] == away).sum())
            rows.append({
                "match_id": mid, "round_num": r.round_num,
                "map_name": r.map_name, "round_class": r.round_class,
                "ct_tier": r.ct_tier, "t_tier": r.t_tier,
                "score_diff": hs - as_, "ct_alive": r.ct_alive,
                "t_alive": r.t_alive, "first_kill_side": r.first_kill_side,
                "final_diff": final_diff, "total": total,
            })
    return pd.DataFrame(rows)


def _pre(features: list[str]) -> ColumnTransformer:
    cat = [c for c in features if c in (CAT_BASE + CAT_30)]
    num = [c for c in features if c not in (CAT_BASE + CAT_30)]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])


def _cv_reg(df: pd.DataFrame, features: list[str], target: str) -> float:
    d = df.dropna(subset=features + [target])
    y = d[target].to_numpy(float)
    model = Pipeline([("pre", _pre(features)), ("reg", Ridge())])
    pred = np.empty(len(d))
    n = min(5, d["match_id"].nunique())
    for tr, te in GroupKFold(n).split(d, groups=d["match_id"].to_numpy()):
        model.fit(d.iloc[tr][features], d.iloc[tr][target])
        pred[te] = model.predict(d.iloc[te][features])
    return mean_absolute_error(y, pred)


def _cv_auc(df: pd.DataFrame, features: list[str], target: str) -> float:
    d = df.dropna(subset=features + [target])
    if d[target].nunique() < 2:
        return float("nan")
    y = d[target].to_numpy()
    model = Pipeline([("pre", _pre(features)), ("clf", LogisticRegression(max_iter=2000))])
    proba = np.empty(len(d))
    n = min(5, d["match_id"].nunique())
    for tr, te in GroupKFold(n).split(d, groups=d["match_id"].to_numpy()):
        model.fit(d.iloc[tr][features], d.iloc[tr][target])
        proba[te] = model.predict_proba(d.iloc[te][features])[:, 1]
    return roc_auc_score(y, proba)


def evaluate() -> dict:
    df = _build()
    df["cover_neg35"] = (df["final_diff"] <= -3).astype(int)  # away 赢 3+（即 home -3.5 失败）
    base = CAT_BASE + NUM_BASE
    full = CAT_BASE + NUM_BASE + NUM_30 + CAT_30
    print(f"回合样本 {len(df)} / series {df['match_id'].nunique()}")

    for target, is_reg in [("final_diff", True), ("total", True), ("cover_neg35", False)]:
        if is_reg:
            b = _cv_reg(df, base, target)
            f = _cv_reg(df, full, target)
            print(f"{target:<14} MAE：市场 {b:.3f} -> +30s {f:.3f}  (Δ {b-f:+.3f}, 负=改善)")
        else:
            b = _cv_auc(df, base, target)
            f = _cv_auc(df, full, target)
            print(f"{target:<14} AUC：市场 {b:.4f} -> +30s {f:.4f}  (Δ {f-b:+.4f})")
    return {}


if __name__ == "__main__":
    evaluate()
