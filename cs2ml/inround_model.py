"""inround_model.py —— 回合内 per-event 胜率曲线（微观层，BLAST 式）。

对接用户闭环的「检验」环节：在回合进行中，每个事件（击杀/下包/拆包）之后，
用当前状态实时更新「本回合谁更接近赢」。标签 = 本回合最终胜方（CT 记 1）。

这是 BLAST 那条 per-event 曲线的复刻：人数差、血量、炸弹状态、位置/视野，
逐事件刷新胜率。检验各信号（人数差/HP/炸弹/经济/位置）的增量贡献。

数据：inround_events.parquet（事件轨迹）merge round_dataset（label_ct_win/经济/图）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .round_transition import _classify6, _round_class


def load_events() -> pd.DataFrame:
    ev = pd.read_parquet(config.DATA_DIR / "inround_events.parquet")
    rs = pd.read_parquet(config.DATA_DIR / "round_states.parquet")
    rs["round_num"] = rs["round_num"].astype(int)
    ev["round_num"] = ev["round_num"].astype(int)
    keep = ["demo_path", "round_num", "winner_side", "ct_equip0", "t_equip0"]
    m = ev.merge(rs[keep], on=["demo_path", "round_num"], how="inner")
    m["label_ct_win"] = (m["winner_side"] == "CT").astype(int)
    m["match_id"] = m["demo_path"].map(lambda p: Path(p).parent.name)
    m["round_class"] = m["round_num"].map(_round_class)
    m["ct_tier6"] = m["ct_equip0"].map(_classify6)
    m["t_tier6"] = m["t_equip0"].map(_classify6)
    m["equip_gap"] = m["ct_equip0"] - m["t_equip0"]
    m["contact_gap"] = m["t_contact"] - m["ct_contact"]   # CT 越近(小)→ 优势记正
    m["aiming_gap"] = m["ct_aiming"] - m["t_aiming"]
    m["spread_gap"] = m["t_spread"] - m["ct_spread"]       # CT 更抱团(小)→ 优势记正
    return m


def _auc(d: pd.DataFrame, cat: list[str], num: list[str]) -> tuple[float, int]:
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
    e = load_events()
    print(f"事件样本：{len(e)} 行 / {e['match_id'].nunique()} 场 / "
          f"{e['demo_path'].nunique()} 图")
    print(f"event_type：{e['event_type'].value_counts().to_dict()}")

    base_cat = ["map_name", "round_class"]
    print("\n" + "=" * 78)
    print("回合内胜率：信号增量贡献（所有事件池化，GroupKFold by match）")
    print("=" * 78)
    print(f"{'特征':<52}{'AUC':>8}{'n':>7}")
    steps = [
        ("人数差 alive_gap only", base_cat, ["alive_gap"]),
        ("+ HP gap", base_cat, ["alive_gap", "hp_gap"]),
        ("+ 炸弹状态", base_cat, ["alive_gap", "hp_gap", "bomb_state"]),
        ("+ 经济决策(6档)", base_cat + ["ct_tier6", "t_tier6"],
         ["alive_gap", "hp_gap", "bomb_state", "equip_gap"]),
        ("+ 位置/视野(contact/aiming/spread)", base_cat + ["ct_tier6", "t_tier6"],
         ["alive_gap", "hp_gap", "bomb_state", "equip_gap",
          "contact_gap", "aiming_gap", "spread_gap"]),
    ]
    for name, c, n in steps:
        a, nn = _auc(e, c, n)
        print(f"  {name:<52}{a:>8.4f}{nn:>7}")

    # 逐事件推进的胜率曲线（按回合内第几个事件）
    print("\n--- 胜率曲线：按「回合内已发生击杀数」分层 ---")
    ev2 = e[e["event_type"] != "round_start"].copy()
    ev2["kills_in"] = ev2.groupby(["demo_path", "round_num"])["event_type"].transform(
        lambda s: (s == "kill").cumsum())
    print(f"{'kills_in':>9}{'AUC':>9}{'n':>7}")
    for k, sub in ev2.groupby("kills_in"):
        if k > 6:
            break
        a, nn = _auc(sub, base_cat + ["ct_tier6", "t_tier6"],
                     ["alive_gap", "hp_gap", "bomb_state", "equip_gap"])
        print(f"{int(k):>9}{a:>9.4f}{nn:>7}")

    # 按事件类型
    print("\n--- 按事件类型 ---")
    for et, sub in e.groupby("event_type"):
        a, nn = _auc(sub, base_cat + ["ct_tier6", "t_tier6"],
                     ["alive_gap", "hp_gap", "bomb_state", "equip_gap"])
        print(f"  {et:<14} n={nn:<7} AUC={a:.4f}")
    return {}


if __name__ == "__main__":
    evaluate()
