"""测「前几个回合的阵型」→ 地图胜者，是否有比分之外的增量（用户假设的最后一个宏观信号）。

问题（用户 reframing）：30s 预测不了整图是当然的；真正该试的是「几个回合的阵型」——
一个「这队今天打得紧不紧」的当图、跨回合信号，比分里没有、且不随单回合稀释。

做法：与 edge.py 同构。
  对每个回合 N，取每队在 rounds 1..N-1 的【累计阵型 running 均值】（spread / stack_rate），
  作为「当前形态」信号，预测【home 是否赢图】。比较：
    - 市场视角：比分差 + 经济档 + 地图 + 回合号
    - + 阵型   ：再加两队累计 spread / stack / n_places（及 spread 差）
  GroupKFold by match_id。

数据：round_dataset + reports/round_formation_raw.csv（回合级阵型快照）。
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

from . import config
from .round_model import build_round_dataset

CAT_BASE = ["map_name", "round_class", "ct_tier", "t_tier"]
NUM_BASE = ["score_diff", "round_num"]
FORM = ["ct_spread", "t_spread", "ct_stack", "t_stack", "ct_places", "t_places"]


def _load() -> pd.DataFrame:
    rd = build_round_dataset()
    form = pd.read_csv(config.REPORTS_DIR / "round_formation_raw.csv")
    form = form[["demo_path", "roster", "round", "spread", "n_places", "stacked"]]
    form["roster"] = form["roster"].astype(str)
    form = form.rename(columns={"round": "round_num"})
    form["round_num"] = form["round_num"].astype(int)

    rd["ct_roster"] = rd["ct_roster"].astype(str)
    rd["t_roster"] = rd["t_roster"].astype(str)
    rd["winner_roster"] = rd["winner_roster"].fillna("").astype(str)

    # 每回合每队一行 -> 分别按 ct/t 边 join 到 rd
    ct = form.rename(columns={"roster": "ct_roster", "spread": "ct_spread",
                              "n_places": "ct_places", "stacked": "ct_stacked"})
    t = form.rename(columns={"roster": "t_roster", "spread": "t_spread",
                             "n_places": "t_places", "stacked": "t_stacked"})
    rd = rd.merge(ct[["demo_path", "round_num", "ct_roster", "ct_spread", "ct_places", "ct_stacked"]],
                  on=["demo_path", "round_num", "ct_roster"], how="inner")
    rd = rd.merge(t[["demo_path", "round_num", "t_roster", "t_spread", "t_places", "t_stacked"]],
                  on=["demo_path", "round_num", "t_roster"], how="inner")
    return rd


def _build(df: pd.DataFrame) -> pd.DataFrame:
    """每回合补 running 阵型均值（rounds 1..N-1）与 mapwin 标签。"""
    rows = []
    for mid, sub in df.groupby("match_id"):
        sub = sub.sort_values("round_num")
        home = sub.iloc[0]["ct_roster"]
        away = sub.iloc[0]["t_roster"]
        vc = sub[sub["winner_roster"] != ""]["winner_roster"].value_counts()
        if not len(vc) or vc.index[0] not in (home, away):
            continue
        mapwin_home = int(vc.index[0] == home)

        cum = {home: {"spread": [], "stack": [], "places": []},
               away: {"spread": [], "stack": [], "places": []}}
        for r in sub.itertuples(index=False):
            hs = int((sub[(sub["round_num"] < r.round_num)]["winner_roster"] == home).sum())
            as_ = int((sub[(sub["round_num"] < r.round_num)]["winner_roster"] == away).sum())
            # 取出本回合两队累计形态（1..N-1 的均值）
            ch = cum[home]; ca = cum[away]
            ct_spread = np.mean(ch["spread"]) if ch["spread"] else np.nan
            t_spread = np.mean(ca["spread"]) if ca["spread"] else np.nan
            ct_stack = np.mean(ch["stack"]) if ch["stack"] else np.nan
            t_stack = np.mean(ca["stack"]) if ca["stack"] else np.nan
            ct_places = np.mean(ch["places"]) if ch["places"] else np.nan
            t_places = np.mean(ca["places"]) if ca["places"] else np.nan
            rows.append({
                "match_id": mid, "round_num": r.round_num,
                "map_name": r.map_name, "round_class": r.round_class,
                "ct_tier": r.ct_tier, "t_tier": r.t_tier,
                "score_diff": hs - as_,
                "ct_spread": ct_spread, "t_spread": t_spread,
                "ct_stack": ct_stack, "t_stack": t_stack,
                "ct_places": ct_places, "t_places": t_places,
                "label_mapwin_home": mapwin_home,
            })
            # 更新累计（加入本回合实测形态）
            ch["spread"].append(r.ct_spread); ch["stack"].append(r.ct_stacked)
            ch["places"].append(r.ct_places)
            ca["spread"].append(r.t_spread); ca["stack"].append(r.t_stacked)
            ca["places"].append(r.t_places)
    return pd.DataFrame(rows)


def _auc(df: pd.DataFrame, features: list[str]) -> float:
    d = df.dropna(subset=features + ["label_mapwin_home"]).copy()
    if len(d) < 50 or d["label_mapwin_home"].nunique() < 2:
        return float("nan")
    y = d["label_mapwin_home"].to_numpy()
    cat = [c for c in features if c in CAT_BASE]
    num = [c for c in features if c not in CAT_BASE]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    proba = np.empty(len(d))
    n_splits = min(5, d["match_id"].nunique())
    for tr, te in GroupKFold(n_splits).split(d, groups=d["match_id"].to_numpy()):
        model.fit(d.iloc[tr][features], d.iloc[tr]["label_mapwin_home"])
        proba[te] = model.predict_proba(d.iloc[te][features])[:, 1]
    return roc_auc_score(y, proba)


def evaluate() -> dict:
    df = _build(_load())
    print(f"回合样本（带阵型）: {len(df)} / series {df['match_id'].nunique()}")
    print(f"home 胜图基准率: {round(df['label_mapwin_home'].mean(), 3)}\n")

    base_f = CAT_BASE + NUM_BASE
    full_f = CAT_BASE + NUM_BASE + FORM
    b = _auc(df, base_f)
    f = _auc(df, full_f)
    print(f"整体 mapwin AUC：市场 {b:.4f} -> +阵型 {f:.4f}  (Δ {f-b:+.4f})")

    # 分比分差桶（若阵型有 edge，应集中在胶着回合）
    buckets = [(0, 0, "平局"), (1, 2, "差1-2"), (3, 4, "差3-4"), (5, 99, "差5+")]
    print(f"\n{'比分差桶':<10}{'n':>7}{'市场AUC':>9}{'+阵型':>10}{'Δ':>8}")
    for lo, hi, name in buckets:
        sub = df[(df["score_diff"].abs() >= lo) & (df["score_diff"].abs() <= hi)]
        bb = _auc(sub, base_f); ff = _auc(sub, full_f)
        if pd.isna(bb) or pd.isna(ff):
            print(f"{name:<10}{len(sub):>7}{'NA':>9}{'NA':>10}")
        else:
            print(f"{name:<10}{len(sub):>7}{bb:>9.4f}{ff:>10.4f}{ff-bb:>+8.4f}")
    return {}


if __name__ == "__main__":
    evaluate()
