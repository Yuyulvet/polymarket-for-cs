"""roster 强度先验实验：把「队伍强度」注入回合模型，看 AUC 能否超过 0.75。

路线 A 在地图级（170 图 / 57 roster）用对称交火标签做 strength ≈0.5，失败。
这里换成回合级（6913 回合 / 330 图 / 169 series），且用**不对称**信号：
  - roster 的 CT 边胜率 / T 边胜率 / 综合胜率
在「谁更强」这个维度上，CT/T 边胜率是不对称的（队伍 A 强于 B => A 的胜率>50%）。

纪律（比路线 A 更严）：
  - GroupKFold by match_id（= 一场 BO3），5 折。
  - strength 只从【训练折】的 match_id 算（test 折的 roster 若训练折没见过 => 用训练折均值填充），
    避免「用其他 test 折的信息给自己造特征」的传导泄漏。

产物：打印 baseline AUC（重算确认 0.75）vs 加 strength 后的 AUC + 覆盖率。
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

CAT = ["map_name", "round_class", "ct_tier", "t_tier"]
NUM = ["ct_equip", "t_equip", "round_num"]
STRENGTH = ["ct_strength", "t_strength", "ct_overall", "t_overall"]


def _roster_strength_lookup(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """从训练折逐 roster 统计 CT边胜率 / T边胜率 / 综合胜率。"""
    ct = train.groupby("ct_roster")["label_ct_win"].mean()
    tt = train.groupby("t_roster")["label_t_win"].mean()
    both = pd.concat([
        train[["ct_roster", "label_ct_win"]].rename(columns={"ct_roster": "roster", "label_ct_win": "win"}),
        train[["t_roster", "label_t_win"]].rename(columns={"t_roster": "roster", "label_t_win": "win"}),
    ])
    ov = both.groupby("roster")["win"].mean()
    out: dict[str, dict[str, float]] = {}
    for r in set(ct.index) | set(tt.index) | set(ov.index):
        out[str(r)] = {
            "ct": float(ct.get(r, np.nan)),
            "t": float(tt.get(r, np.nan)),
            "overall": float(ov.get(r, np.nan)),
        }
    return out


def _apply_strength(sub: pd.DataFrame, lookup: dict, fallback: dict[str, float]) -> pd.DataFrame:
    sub = sub.copy()
    sub["ct_strength"] = [lookup.get(r, {}).get("ct", fallback["ct"]) for r in sub["ct_roster"]]
    sub["t_strength"] = [lookup.get(r, {}).get("t", fallback["t"]) for r in sub["t_roster"]]
    sub["ct_overall"] = [lookup.get(r, {}).get("overall", fallback["overall"]) for r in sub["ct_roster"]]
    sub["t_overall"] = [lookup.get(r, {}).get("overall", fallback["overall"]) for r in sub["t_roster"]]
    return sub


def _make_model(features: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in CAT if c in features]),
        ("num", StandardScaler(), [c for c in features if c not in CAT]),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


def _cv_auc(df: pd.DataFrame, features: list[str], with_strength: bool) -> dict:
    y = df["label_ct_win"].to_numpy()
    n_splits = min(5, df["match_id"].nunique())
    gkf = GroupKFold(n_splits)
    proba = np.empty(len(df))
    seen_coverage: list[float] = []
    for train_idx, test_idx in gkf.split(df, groups=df["match_id"].to_numpy()):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        if with_strength:
            lookup = _roster_strength_lookup(train)
            fallback = {"ct": float(train["label_ct_win"].mean()),
                        "t": float(train["label_t_win"].mean()),
                        "overall": float(train["label_ct_win"].mean())}
            tr = _apply_strength(train, lookup, fallback)
            te = _apply_strength(test, lookup, fallback)
            seen = (te["ct_overall"].notna()).mean()
            seen_coverage.append(seen)
        else:
            tr, te = train, test
        model = _make_model(features)
        model.fit(tr[features], tr["label_ct_win"])
        proba[test_idx] = model.predict_proba(te[features])[:, 1]
    return {"auc": roc_auc_score(y, proba),
            "coverage": float(np.mean(seen_coverage)) if seen_coverage else None}


def evaluate() -> dict:
    df = build_round_dataset()
    df = df.dropna(subset=["ct_equip", "t_equip"]).copy()
    df["label_t_win"] = 1 - df["label_ct_win"]
    df["ct_roster"] = df["ct_roster"].astype(str)
    df["t_roster"] = df["t_roster"].astype(str)
    print(f"回合数 {len(df)} / series {df['match_id'].nunique()} / roster {df['ct_roster'].nunique()}")

    base = _cv_auc(df, CAT + NUM, with_strength=False)
    print(f"\nbaseline（经济+边+地图，无 roster）：AUC = {base['auc']:.4f}")

    full = _cv_auc(df, CAT + NUM + STRENGTH, with_strength=True)
    print(f"+ roster 强度（CT/T/综合，训练折算）：AUC = {full['auc']:.4f}  "
          f"(test 折 roster 训练折见过的覆盖率 {full['coverage']:.3f})")

    # 只看 side 强度（去掉 overall）
    side = _cv_auc(df, CAT + NUM + ["ct_strength", "t_strength"], with_strength=True)
    print(f"+ 仅 CT/T 边胜率：AUC = {side['auc']:.4f}")
    return {"baseline": base, "full": full, "side": side}


if __name__ == "__main__":
    evaluate()
