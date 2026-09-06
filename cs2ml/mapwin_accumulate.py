"""map winner 累计曲线：预测 AUC 随「已见回合数」如何涨，以及我们的信号加多少增量。

回答用户的核心问题：「通过小回合不断累计，让 map winner 预测尽可能准」能做到什么程度。
对每个回合 N，用【截至 N-1 的比分 + 当前经济 + 地图 + 回合类型】预测 home 是否赢图，
再分别加【30s 战术】与【累计阵型】，按回合数分桶报 AUC。GroupKFold by match_id。
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
from .midround import build_midround
from .round_model import build_round_dataset

CAT_BASE = ["map_name", "round_class", "ct_tier", "t_tier"]
NUM_BASE = ["score_diff", "round_num"]
NUM_30 = ["ct_alive", "t_alive"]
CAT_30 = ["first_kill_side"]
FORM = ["ct_spread", "t_spread", "ct_stack", "t_stack"]


def _load() -> pd.DataFrame:
    rd = build_round_dataset()
    mr = build_midround()
    df = rd.merge(mr, on=["demo_path", "round_num"], how="inner", suffixes=("", "_m"))
    df["ct_roster"] = df["ct_roster"].astype(str)
    df["t_roster"] = df["t_roster"].astype(str)
    df["winner_roster"] = df["winner_roster"].fillna("").astype(str)

    form = pd.read_csv(config.REPORTS_DIR / "round_formation_raw.csv")
    form = form[["demo_path", "roster", "round", "spread", "stacked"]].rename(
        columns={"round": "round_num"})
    form["roster"] = form["roster"].astype(str)
    ct = form.rename(columns={"roster": "ct_roster", "spread": "ct_spread", "stacked": "ct_stacked"})
    t = form.rename(columns={"roster": "t_roster", "spread": "t_spread", "stacked": "t_stacked"})
    df = df.merge(ct[["demo_path", "round_num", "ct_roster", "ct_spread", "ct_stacked"]],
                  on=["demo_path", "round_num", "ct_roster"], how="left")
    df = df.merge(t[["demo_path", "round_num", "t_roster", "t_spread", "t_stacked"]],
                  on=["demo_path", "round_num", "t_roster"], how="left")
    return df


def _build(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, sub in df.groupby("match_id"):
        sub = sub.sort_values("round_num")
        home = sub.iloc[0]["ct_roster"]; away = sub.iloc[0]["t_roster"]
        vc = sub[sub["winner_roster"] != ""]["winner_roster"].value_counts()
        if not len(vc) or vc.index[0] not in (home, away):
            continue
        mapwin_home = int(vc.index[0] == home)
        chs = {"spread": [], "stack": []}; cas = {"spread": [], "stack": []}
        for r in sub.itertuples(index=False):
            prior = sub[sub["round_num"] < r.round_num]
            hs = int((prior["winner_roster"] == home).sum())
            as_ = int((prior["winner_roster"] == away).sum())
            rows.append({
                "match_id": mid, "round_num": r.round_num,
                "map_name": r.map_name, "round_class": r.round_class,
                "ct_tier": r.ct_tier, "t_tier": r.t_tier,
                "score_diff": hs - as_,
                "ct_alive": r.ct_alive, "t_alive": r.t_alive,
                "first_kill_side": r.first_kill_side,
                "ct_spread": np.mean(chs["spread"]) if chs["spread"] else np.nan,
                "t_spread": np.mean(cas["spread"]) if cas["spread"] else np.nan,
                "ct_stack": np.mean(chs["stack"]) if chs["stack"] else np.nan,
                "t_stack": np.mean(cas["stack"]) if cas["stack"] else np.nan,
                "label_mapwin_home": mapwin_home,
            })
            chs["spread"].append(r.ct_spread); chs["stack"].append(r.ct_stacked)
            cas["spread"].append(r.t_spread); cas["stack"].append(r.t_stacked)
    return pd.DataFrame(rows)


def _auc(df: pd.DataFrame, features: list[str]) -> float:
    d = df.dropna(subset=features + ["label_mapwin_home"]).copy()
    if len(d) < 40 or d["label_mapwin_home"].nunique() < 2:
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
    for tr, te in GroupKFold(n_splits).split(d, groups=d["match_id"].to_numpy()):
        model.fit(d.iloc[tr][features], d.iloc[tr]["label_mapwin_home"])
        proba[te] = model.predict_proba(d.iloc[te][features])[:, 1]
    return roc_auc_score(y, proba)


def evaluate() -> dict:
    df = _build(_load())
    print(f"回合样本 {len(df)} / series {df['match_id'].nunique()} / home 基准率 {df['label_mapwin_home'].mean():.3f}\n")

    base = CAT_BASE + NUM_BASE
    p30 = CAT_BASE + NUM_BASE + NUM_30 + CAT_30
    pform = CAT_BASE + NUM_BASE + FORM
    pfull = CAT_BASE + NUM_BASE + NUM_30 + CAT_30 + FORM

    buckets = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18), (19, 21), (22, 99)]
    print(f"{'回合段':<10}{'n':>6}{'比分':>8}{'+30s':>8}{'+阵型':>8}{'+两者':>8}{'Δ(两者-比分)':>13}")
    for lo, hi in buckets:
        sub = df[(df["round_num"] >= lo) & (df["round_num"] <= hi)]
        if len(sub) < 40:
            continue
        b = _auc(sub, base)
        a30 = _auc(sub, p30)
        af = _auc(sub, pform)
        afull = _auc(sub, pfull)
        print(f"{lo}-{hi:<8}{len(sub):>6}{b:>8.3f}{a30:>8.3f}{af:>8.3f}{afull:>8.3f}{afull-b:>+13.3f}")

    b = _auc(df, base); f = _auc(df, pfull)
    print(f"\n整体：比分 {b:.4f} -> 比分+30s+阵型 {f:.4f}  (Δ {f-b:+.4f})")
    return {}


if __name__ == "__main__":
    evaluate()
