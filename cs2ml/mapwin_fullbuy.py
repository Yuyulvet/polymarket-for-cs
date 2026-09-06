"""关键长枪局（full vs full）→ map winner：as-of 走向分析（修正版）。

用户洞察：手枪不够，关键长枪局(full vs full)才是真实实力样本；且不必抢 30s 提前量，
回合结果出来后做「关键回合走向」分析即可预测 map winner。

关键纪律：只用 as-of（累计到回合 N）特征，杜绝 look-ahead。每个 horizon N 对比：
  裸比分 score_diff_N  vs  full-vs-full 走向(ff_home, ff_total)  vs  两者合并。
看 full-buy 走向是否在裸比分之上加增量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from .round_model import build_round_dataset

HORIZONS = [3, 6, 9, 12, 15, 18, 21, 24]


def build_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, sub in df.groupby("match_id"):
        sub = sub.sort_values("round_num")
        home = sub.iloc[0]["ct_roster"]
        away = sub.iloc[0]["t_roster"]
        if not home or not away:
            continue
        won = sub[sub["winner_roster"].astype(str) != ""]
        vc = won["winner_roster"].value_counts()
        if not len(vc) or vc.index[0] not in (home, away):
            continue
        home_win = int(vc.index[0] == home)

        sh = sa = fh = ft = 0
        snap: dict[int, tuple[int, int, int]] = {}
        for r in sub.itertuples(index=False):
            w = r.winner_roster
            if not w:
                continue
            if w == home:
                sh += 1
            elif w == away:
                sa += 1
            if r.ct_tier == "full" and r.t_tier == "full":
                ft += 1
                if w == home:
                    fh += 1
            n = sh + sa
            if n in HORIZONS:
                snap[n] = (sh - sa, fh, ft)
        for h in HORIZONS:
            if h in snap:
                sd, fh_, ft_ = snap[h]
                rows.append({"match_id": mid, "horizon": h, "score_diff": sd,
                             "ff_home": fh_, "ff_total": ft_, "home_win": home_win})
    return pd.DataFrame(rows)


def _auc(d: pd.DataFrame, feats: list[str]) -> float:
    d = d.dropna(subset=feats + ["home_win"]).copy()
    if len(d) < 30 or d["home_win"].nunique() < 2:
        return float("nan")
    X = d[feats].to_numpy()
    y = d["home_win"].to_numpy()
    n = min(5, d["match_id"].nunique())
    proba = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                              groups=d["match_id"].to_numpy(), cv=GroupKFold(n),
                              method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def evaluate() -> dict:
    df = build_round_dataset()
    tr = build_trajectory(df)
    print(f"轨迹快照：{len(tr)} 行 / {tr['match_id'].nunique()} 图\n")

    # 每个 horizon：ff_total 均值（full-buy 回合何时出现）
    print(f"{'horizon':>8}{'ff均值':>8}  {'裸比分AUC':>12}{'fullbuyAUC':>12}{'合并AUC':>12}{'Δ(合-比分)':>12}")
    out = {}
    for h in HORIZONS:
        d = tr[tr["horizon"] == h].copy()
        if len(d) < 30:
            continue
        # full-buy 模型只在有 ff 回合的子集上才有意义
        d_ff = d[d["ff_total"] >= 1]
        a_score = _auc(d, ["score_diff"])
        a_ff = _auc(d_ff, ["ff_home", "ff_total"]) if len(d_ff) >= 30 else float("nan")
        a_both = _auc(d, ["score_diff", "ff_home", "ff_total"])
        delta = a_both - a_score
        ff_mean = d["ff_total"].mean()
        print(f"{h:>8}{ff_mean:>8.1f}  {a_score:>12.4f}{a_ff:>12.4f}{a_both:>12.4f}{delta:>+12.4f}")
        out[h] = {"ff_mean": ff_mean, "score": a_score, "fullbuy": a_ff,
                  "both": a_both, "delta": delta}
    return out


if __name__ == "__main__":
    evaluate()
