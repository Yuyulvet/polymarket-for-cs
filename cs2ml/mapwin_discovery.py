"""map winner 预测 + 特征发现：从手枪局与回合关联入手。

用户问题：map winner 是可实时交易的盘。核心假设是「手枪局胜率 → map winner」，
以及「回合之间有关联、会累计到 map winner」。用现有数据拆 train/test，
做 map winner 预测，看什么因素真正驱动预测正确。

纪律：GroupKFold by match_id（防同场泄漏）。label = home(首回合 CT roster) 是否赢图。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from .round_model import build_round_dataset


def _won(sub_row: pd.DataFrame, roster: str) -> float:
    """sub_row 是单回合子集；返回该回合 home 是否赢（无该回合返回 NaN）。"""
    if not len(sub_row):
        return float("nan")
    w = sub_row.iloc[0]["winner_roster"]
    return float(w == roster) if w else float("nan")


def build_map_table(df: pd.DataFrame) -> pd.DataFrame:
    """每 map 一行：label=home 赢图 + 手枪/比分等按信息先后排列的特征。"""
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
            continue  # 平局/异常跳过
        label = int(vc.index[0] == home)

        def score_after(n: int) -> int:
            prior = sub[sub["round_num"] <= n]
            hs = int((prior["winner_roster"] == home).sum())
            aw = int((prior["winner_roster"] == away).sum())
            return hs - aw

        rows.append({
            "match_id": mid,
            "map_name": sub.iloc[0]["map_name"],
            "label_home_win": label,
            "pistol1_home_win": _won(sub[sub["round_num"] == 1], home),
            "pistol2_home_win": _won(sub[sub["round_num"] == 13], home),
            "score_after_2": score_after(2),
            "score_after_3": score_after(3),
            "score_after_6": score_after(6),
            "score_after_12": score_after(12),
        })
    return pd.DataFrame(rows)


def _auc_map(tab: pd.DataFrame, features: list[str]) -> tuple[float, int]:
    d = tab.dropna(subset=features + ["label_home_win"]).copy()
    if len(d) < 30 or d["label_home_win"].nunique() < 2:
        return float("nan"), len(d)
    X = d[features].to_numpy()
    y = d["label_home_win"].to_numpy()
    n = min(5, d["match_id"].nunique())
    proba = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                              groups=d["match_id"].to_numpy(), cv=GroupKFold(n),
                              method="predict_proba")[:, 1]
    return roc_auc_score(y, proba), len(d)


def cond_rate(tab: pd.DataFrame, mask: pd.Series) -> tuple[float, int]:
    s = tab[mask]
    if not len(s):
        return float("nan"), 0
    return float(s["label_home_win"].mean()), len(s)


def evaluate() -> dict:
    df = build_round_dataset()
    print(f"round_dataset: {len(df)} 回合 / {df['match_id'].nunique()} 图\n")
    tab = build_map_table(df)
    print(f"map 级表：{len(tab)} 图 / home 赢图基准率 = {tab['label_home_win'].mean():.3f}\n")

    # ---------- 1) 手枪局 → map winner（核心假设） ----------
    print("=" * 60)
    print("1) 手枪局 → map winner（home 视角）")
    print("=" * 60)
    for name, mask in [
        ("赢手枪1", tab["pistol1_home_win"] == 1),
        ("输手枪1", tab["pistol1_home_win"] == 0),
        ("赢手枪2(round13)", tab["pistol2_home_win"] == 1),
        ("输手枪2", tab["pistol2_home_win"] == 0),
        ("双赢(1+13)", (tab["pistol1_home_win"] == 1) & (tab["pistol2_home_win"] == 1)),
        ("双输(1+13)", (tab["pistol1_home_win"] == 0) & (tab["pistol2_home_win"] == 0)),
        ("一赢一输", (tab["pistol1_home_win"] != tab["pistol2_home_win"])),
    ]:
        r, n = cond_rate(tab, mask)
        print(f"  {name:<16} home 赢图率 = {r:.3f}  (n={n})")

    # 手枪1 单独作为预测器（AUC = 排名能力）
    a_p1, n1 = _auc_map(tab, ["pistol1_home_win"])
    print(f"\n  单用手枪1 预测 map winner：AUC = {a_p1:.4f}  (n={n1})")

    # ---------- 2) 回合关联（transition） ----------
    print("\n" + "=" * 60)
    print("2) 回合关联：赢一回合 → 下一回合也赢（同队）的概率")
    print("=" * 60)
    long = []
    for mid, sub in df.groupby("match_id"):
        sub = sub.sort_values("round_num")
        home = sub.iloc[0]["ct_roster"]
        w = sub[sub["winner_roster"].astype(str) != ""]
        if not len(w) or home not in (w["winner_roster"].unique()):
            continue
        for i in range(1, len(w)):
            prev = w.iloc[i - 1]["winner_roster"]
            cur = w.iloc[i]["winner_roster"]
            long.append({"same": int(prev == cur), "ctx": w.iloc[i]["round_class"]})
    tr = pd.DataFrame(long)
    print(f"  全回合：P(连续赢) = {tr['same'].mean():.3f}  (n={len(tr)})")
    for ctx, s in tr.groupby("ctx"):
        print(f"    {ctx:<12} P(连续赢) = {s['same'].mean():.3f}  (n={len(s)})")

    # ---------- 3) 渐进式 map winner 预测 AUC ----------
    print("\n" + "=" * 60)
    print("3) map winner 预测 AUC：随已见回合数增长（GroupKFold 5 折）")
    print("=" * 60)
    print(f"{'特征':<36}{'AUC':>8}{'n':>6}{'Δ':>8}")
    prev = 0.500
    print(f"{'（基准，猜多数类）':<36}{0.500:>8.3f}{'':>6}{'':>8}")
    for name, feats in [
        ("手枪1", ["pistol1_home_win"]),
        ("手枪1+手枪2", ["pistol1_home_win", "pistol2_home_win"]),
        ("比分 after 2", ["score_after_2"]),
        ("比分 after 3", ["score_after_3"]),
        ("比分 after 6", ["score_after_6"]),
        ("比分 after 12", ["score_after_12"]),
    ]:
        a, n = _auc_map(tab, feats)
        print(f"{name:<36}{a:>8.4f}{n:>6}{a - prev:>+8.4f}")
        prev = a

    # ---------- 4) 特征重要性 ----------
    print("\n" + "=" * 60)
    print("4) 特征重要性（逻辑回归系数，标准化后可比大小）")
    print("=" * 60)
    feats = ["pistol1_home_win", "pistol2_home_win", "score_after_3", "score_after_6"]
    d = tab.dropna(subset=feats + ["label_home_win"])
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(d[feats].to_numpy())
    y = d["label_home_win"].to_numpy()
    m = LogisticRegression(max_iter=2000).fit(Xs, y)
    for f, c in sorted(zip(feats, m.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {f:<22} coef = {c:+.3f}")

    return {"tab": tab, "n_maps": len(tab)}


if __name__ == "__main__":
    evaluate()
