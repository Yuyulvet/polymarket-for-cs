"""round_structure_model.py —— 回合级：6 档装备 + 结构信号 的回合胜方预测。

三个层次（都 GroupKFold by match，标签 label_ct_win）：

A. 回合开始态（诚实赛前）：3 档 vs 6 档装备，能否只用回合开始前可知的
   equip 档位 + map + round_class + round_num 预测本回合胜方。
   —— 这是唯一的「赛前」回合信号，检验装备分层粒度。

B. 首杀（赛中第一个可读信号）：base(6档) + 首杀边 → 回合 AUC 抬升多少。
   —— 对接 step 2 的 live 读局（midround 已证 0.75→0.80，这里用 6 档重验）。

C. 回合末结构（事后诊断，上界）：base + alive/hp/击杀/trade/util 差，
   验证这些「结构信号」与回合结果强相关（= 它们作为 map 级画像特征的依据）。

纪律：与 round_model 一致，OneHot(分类) + StandardScaler(数值) + LogisticRegression。
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

from . import config


def load_round_events() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_DIR / "round_events.parquet")


def _fit_auc(d: pd.DataFrame, cat: list[str], num: list[str]) -> tuple[float, int]:
    d = d.dropna(subset=cat + num + ["label_ct_win"])
    if len(d) < 100 or d["label_ct_win"].nunique() < 2:
        return float("nan"), len(d)
    y = d["label_ct_win"].to_numpy()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
    n = min(5, d["match_id"].nunique())
    p = cross_val_predict(clf, d[cat + num], y, groups=d["match_id"].to_numpy(),
                          cv=GroupKFold(n), method="predict_proba")[:, 1]
    return roc_auc_score(y, p), len(d)


def evaluate() -> dict:
    re = load_round_events()
    re = re.dropna(subset=["ct_tier6", "t_tier6"]).copy()
    re["kill_gap"] = re["ct_kills"] - re["t_kills"]
    re["trade_gap"] = re["ct_trade"] - re["t_trade"]
    re["util_gap"] = re["ct_util"] - re["t_util"]
    re["ct_first"] = re["ct_first"].astype(float)
    print(f"round_events: {len(re)} 回合 / {re['match_id'].nunique()} 图")

    base_cat = ["map_name", "round_class"]
    base_num = ["round_num"]

    print("\n=== A. 回合开始态（诚实赛前）===")
    print(f"{'模型':<40}{'AUC':>8}{'n':>7}")
    variants = [
        ("3档 equip（round_model 基线）", ["ct_tier", "t_tier"], []),
        ("6档 equip", ["ct_tier6", "t_tier6"], []),
        ("6档 + equip_gap(数值)", ["ct_tier6", "t_tier6"], ["equip_gap"]),
    ]
    for name, cat_x, num_x in variants:
        a, n = _fit_auc(re, base_cat + cat_x, base_num + num_x)
        print(f"  {name:<40}{a:>8.4f}{n:>7}")

    print("\n=== B. 首杀（赛中第一个可读信号，live）===")
    a0, n0 = _fit_auc(re, base_cat + ["ct_tier6", "t_tier6"], base_num)
    a1, n1 = _fit_auc(re, base_cat + ["ct_tier6", "t_tier6", "first_kill_side"], base_num)
    print(f"  base(6档)                          AUC={a0:.4f}  n={n0}")
    print(f"  base + first_kill_side             AUC={a1:.4f}  n={n1}  (Δ{a1 - a0:+.4f})")

    print("\n=== C. 回合末结构（事后诊断，上界）===")
    print(f"{'叠加结构信号':<40}{'AUC':>8}{'n':>7}")
    steps = [
        ("+ alive_gap", ["alive_gap"]),
        ("+ hp_gap", ["alive_gap", "hp_gap"]),
        ("+ kill_gap", ["alive_gap", "hp_gap", "kill_gap"]),
        ("+ trade/util gap", ["alive_gap", "hp_gap", "kill_gap", "trade_gap", "util_gap"]),
    ]
    for name, num_x in steps:
        a, n = _fit_auc(re, base_cat + ["ct_tier6", "t_tier6"], base_num + num_x)
        print(f"  base(6档) {name:<28}{a:>8.4f}{n:>7}")

    return {}


if __name__ == "__main__":
    evaluate()
