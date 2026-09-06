"""量化 30s 中局信号在「地图胜者」预测上的增量（即 Map Winner 盘口的 edge）。

问题：市场（盯 HLTV 比分）在回合 N 进行中只能看到【到 N-1 为止的比分】；我们在
回合 N + 30s 时额外知道【本回合谁占优（人数差+首杀）】。多知道这一回合，能不能更准
地预测【地图最终谁赢】？

做法（直接模型，非 MC，干净快速）：
  对每个回合 N 的观测状态，预测「上半场 CT 方最终是否赢这张图」：
    - 市场视角（round-start）：比分差 + 经济档位 + 地图 + 回合号
    - 我们的视角（+30s）     ：再加 存活人数差 + 首杀边
  GroupKFold by match_id，比较 AUC，并按【比分胶着程度】分桶——edge 应集中在胶着回合。

产物：打印「整体 + 分比分差桶」的 AUC 对比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .midround import build_midround
from .round_model import build_round_dataset

CAT_BASE = ["map_name", "round_class", "ct_tier", "t_tier"]
NUM_BASE = ["score_diff", "round_num"]
NUM_30 = ["ct_alive", "t_alive"]
CAT_30 = ["first_kill_side"]


def _load() -> pd.DataFrame:
    rd = build_round_dataset()
    mr = build_midround()
    df = rd.merge(mr, on=["demo_path", "round_num"], how="inner", suffixes=("", "_m"))
    df = df.dropna(subset=["ct_equip", "t_equip"]).copy()
    df["ct_roster"] = df["ct_roster"].astype(str)
    df["t_roster"] = df["t_roster"].astype(str)
    df["winner_roster"] = df["winner_roster"].fillna("").astype(str)
    return df


def _add_mapwin(df: pd.DataFrame) -> pd.DataFrame:
    """每回合补：home(上半场CT) 比分、比分差、以及「home 是否赢图」标签。"""
    rows = []
    for mid, sub in df.groupby("match_id"):
        sub = sub.sort_values("round_num")
        home = sub.iloc[0]["ct_roster"]
        away = sub.iloc[0]["t_roster"]
        # 地图胜者 = 赢回合多的 roster（用 winner_roster 计数）
        vc = sub[sub["winner_roster"] != ""]["winner_roster"].value_counts()
        if not len(vc) or vc.index[0] not in (home, away):
            continue
        mapwin_home = int(vc.index[0] == home)
        for r in sub.itertuples(index=False):
            prior = sub[sub["round_num"] < r.round_num]
            hs = int((prior["winner_roster"] == home).sum())
            as_ = int((prior["winner_roster"] == away).sum())
            rows.append({
                "match_id": mid,
                "round_num": r.round_num,
                "map_name": r.map_name,
                "round_class": r.round_class,
                "ct_tier": r.ct_tier,
                "t_tier": r.t_tier,
                "score_diff": hs - as_,
                "ct_alive": r.ct_alive,
                "t_alive": r.t_alive,
                "first_kill_side": r.first_kill_side,
                "label_mapwin_home": mapwin_home,
            })
    return pd.DataFrame(rows)


def _auc(df: pd.DataFrame, features: list[str]) -> float:
    d = df.dropna(subset=features).copy()
    if len(d) < 50 or d["label_mapwin_home"].nunique() < 2:
        return float("nan")
    y = d["label_mapwin_home"].to_numpy()
    cat = [c for c in features if c in (CAT_BASE + CAT_30)]
    num = [c for c in features if c not in (CAT_BASE + CAT_30)]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    proba = np.empty(len(d))
    n_splits = min(5, d["match_id"].nunique())
    gkf = GroupKFold(n_splits)
    for tr, te in gkf.split(d, groups=d["match_id"].to_numpy()):
        model.fit(d.iloc[tr][features], d.iloc[tr]["label_mapwin_home"])
        proba[te] = model.predict_proba(d.iloc[te][features])[:, 1]
    return roc_auc_score(y, proba)


def evaluate() -> dict:
    df = _add_mapwin(_load())
    print(f"回合样本（地图未结束）: {len(df)} / series {df['match_id'].nunique()}")
    print(f"home 胜图基准率: {round(df['label_mapwin_home'].mean(), 3)}\n")

    base_f = CAT_BASE + NUM_BASE
    full_f = CAT_BASE + NUM_BASE + NUM_30 + CAT_30

    overall_base = _auc(df, base_f)
    overall_full = _auc(df, full_f)
    print(f"整体 mapwin 预测 AUC：市场视角 {overall_base:.4f}  ->  +30s {overall_full:.4f}")

    # 按比分差分桶（edge 应集中在胶着回合）
    buckets = [(0, 0, "平局"), (1, 2, "差1-2"), (3, 4, "差3-4"), (5, 99, "差5+")]
    print(f"\n{'比分差桶':<10}{'n':>7}{'市场AUC':>9}{'+30s AUC':>10}{'Δ':>8}")
    out = {"overall": {"base": overall_base, "full": overall_full}}
    for lo, hi, name in buckets:
        sub = df[(df["score_diff"].abs() >= lo) & (df["score_diff"].abs() <= hi)]
        b = _auc(sub, base_f)
        f = _auc(sub, full_f)
        out[name] = {"n": len(sub), "base": b, "full": f}
        if pd.isna(b) or pd.isna(f):
            print(f"{name:<10}{len(sub):>7}{'NA':>9}{'NA':>10}")
        else:
            print(f"{name:<10}{len(sub):>7}{b:>9.4f}{f:>10.4f}{f-b:>+8.4f}")
    return out


if __name__ == "__main__":
    evaluate()
