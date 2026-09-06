"""online_filter.py —— 在线滤波闭环（预测 → 检验 → 校正 → 再预测）。

对接用户「预测、检验、校正、再预测」的闭环，量化为两个可证伪的检验：

  A. 校正→再预测（surprise residual）：上一回合「预测 vs 实际」的残差（surprise），
     能否在回合级特征之外，进一步预测下一回合？→ 测「模型错了之后，下回合该怎么调」。

  B. 检验→校正（early detection）：经济先验（回合开始前 0.745）如果预测错了，
     回合内 per-event 曲线（alive_gap+HP+炸弹+经济，0.836）在第几个击杀就能「锁定」
     真实赢家？→ 测「先市场一步」的提前量（kills_in 越早越值钱）。

所有预测都 GroupKFold by match、严格时间序，surprise 只用上一回合已结算的信息（非泄漏）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from . import config
from .round_transition import build_transitions
from .inround_model import load_events

BASE_CAT = ["map_name", "round_class", "ct_tier6", "t_tier6"]
BASE_NUM = ["score_gap", "equip_gap"]


def _model(cat: list[str], num: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])


def _oof_proba(d: pd.DataFrame, cat: list[str], num: list[str]) -> np.ndarray:
    d = d.dropna(subset=cat + num + ["label_ct_win"])
    y = d["label_ct_win"].to_numpy()
    ncv = min(5, d["match_id"].nunique())
    return cross_val_predict(_model(cat, num), d[cat + num], y,
                             groups=d["match_id"].to_numpy(),
                             cv=GroupKFold(ncv), method="predict_proba")[:, 1]


def part_a_correction() -> dict:
    """校正→再预测：surprise 是否在回合级特征之外有增量。"""
    t = build_transitions()
    t = t.dropna(subset=BASE_CAT + BASE_NUM + ["label_ct_win"])
    # 基础模型 out-of-fold 概率（经济+比分，~0.744）
    p_base = _oof_proba(t, BASE_CAT, BASE_NUM)
    t["p_base"] = p_base
    t["surprise"] = t["label_ct_win"].astype(float) - t["p_base"]  # 正值=低估了CT

    # 地图内滚动 surprise（校正记忆：最近 3 回合模型系统性偏了多少）
    t = t.sort_values(["demo_path", "round_num"]).reset_index(drop=True)
    t["surprise_roll3"] = t.groupby("demo_path")["surprise"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    t["surprise_prev"] = t.groupby("demo_path")["surprise"].shift(1)

    def auc_of(d: pd.DataFrame, cat: list[str], num: list[str]) -> tuple[float, int]:
        d = d.dropna(subset=cat + num + ["label_ct_win"])
        y = d["label_ct_win"].to_numpy()
        ncv = min(5, d["match_id"].nunique())
        p = cross_val_predict(_model(cat, num), d[cat + num], y,
                              groups=d["match_id"].to_numpy(),
                              cv=GroupKFold(ncv), method="predict_proba")[:, 1]
        return roc_auc_score(y, p), len(d)

    print("=" * 74)
    print("A. 校正→再预测：surprise 残差是否有增量（GroupKFold by match）")
    print("=" * 74)
    print(f"  {'特征':<46}{'AUC':>8}{'n':>7}")
    combos = [
        ("基础(经济+比分)", BASE_CAT, BASE_NUM),
        ("+ surprise_prev", BASE_CAT, BASE_NUM + ["surprise_prev"]),
        ("+ surprise_roll3", BASE_CAT, BASE_NUM + ["surprise_prev", "surprise_roll3"]),
    ]
    for name, c, n in combos:
        a, nn = auc_of(t, c, n)
        print(f"  {name:<46}{a:>8.4f}{nn:>7}")
    return {}


def part_b_early_detection() -> dict:
    """检验→校正：经济先验错了，回合内曲线第几个击杀锁定真实赢家。"""
    e = load_events()
    # 回合内事件级 out-of-fold 概率（alive_gap+HP+炸弹+经济）
    cat = BASE_CAT
    num = ["alive_gap", "hp_gap", "bomb_state", "equip_gap"]
    e = e.dropna(subset=cat + num + ["label_ct_win"])
    p_in = _oof_proba(e, cat, num)
    e = e.copy()
    e["p_in"] = p_in

    # 回合级经济先验：每个回合用该回合第一个事件的 p_in 不可（那是回合内）。
    # 改用 round_transition 的回合级 out-of-fold 概率（纯回合开始前信息）。
    t = build_transitions()
    t = t.dropna(subset=BASE_CAT + BASE_NUM + ["label_ct_win"])
    t["p_prior"] = _oof_proba(t, BASE_CAT, BASE_NUM)
    prior = t.set_index(["demo_path", "round_num"])[["p_prior", "label_ct_win"]]

    e = e.merge(prior, on=["demo_path", "round_num"], how="left",
                suffixes=("", "_prior"))

    # 回合内按击杀推进：第几个击杀起，事件级模型锁定真实赢家
    ev = e[e["event_type"] != "round_start"].copy()
    ev["kills_in"] = ev.groupby(["demo_path", "round_num"])["event_type"].transform(
        lambda s: (s == "kill").cumsum())
    ev["pred_win"] = (ev["p_in"] >= 0.5).astype(int)  # 预测 CT 赢
    true = ev["label_ct_win"].astype(int)
    ev["correct"] = (ev["pred_win"] == true).astype(int)

    # 锁定击杀数：从该 kills_in 起之后所有事件都预测正确
    rows = []
    for (dp, r), sub in ev.groupby(["demo_path", "round_num"]):
        sub = sub.sort_values("kills_in")
        if len(sub) < 2:
            continue
        lock = None
        for k in sorted(sub["kills_in"].unique()):
            tail = sub[sub["kills_in"] >= k]
            if tail["correct"].all():
                lock = int(k)
                break
        if lock is not None:
            rows.append({
                "demo_path": dp, "round_num": r,
                "lock_in": lock,
                "prior_correct": int(sub["p_prior"].iloc[0] >= 0.5) == int(sub["label_ct_win"].iloc[0]),
            })
    lock = pd.DataFrame(rows)

    print("\n" + "=" * 74)
    print("B. 检验→校正：回合内曲线「锁定真实赢家」的击杀数")
    print("=" * 74)
    all_lock = lock["lock_in"]
    print(f"  全部回合：锁定中位数 kills_in = {all_lock.median():.1f}，"
          f"kills_in<=2 占比 {(all_lock <= 2).mean()*100:.0f}%，"
          f"<=3 占比 {(all_lock <= 3).mean()*100:.0f}%")

    if lock["prior_correct"].notna().sum() > 0:
        wrong = lock[lock["prior_correct"] == False]
        right = lock[lock["prior_correct"] == True]
        print(f"  经济先验【对】时：锁定中位数 = {right['lock_in'].median():.1f} kills"
              f"  (n={len(right)})")
        print(f"  经济先验【错】时：锁定中位数 = {wrong['lock_in'].median():.1f} kills"
              f"  (n={len(wrong)})  ← 先验错了，曲线第几个击杀纠正过来")
        print(f"    先验错但 kills_in<=3 就锁定真实赢家："
              f"{(wrong['lock_in'] <= 3).mean()*100:.0f}%")
    return {}


if __name__ == "__main__":
    part_a_correction()
    part_b_early_detection()
