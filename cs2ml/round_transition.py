"""round_transition.py —— 回合转移模型（宏观层：第 k 回合 → 第 k+1 回合）。

对接用户闭环的「预测」环节：给定第 k 回合的**结果 + 性质**（碾压/翻盘/均势），
预测第 k+1 回合会发生什么。核心中间变量 = **输掉那队第 k+1 回合的经济决策**
（eco/半起/强起），因为那决定了下一回合走向。

两个子问题（都 GroupKFold by match，严格时间序，防同场泄漏）：

  A. 经济决策策略（policy）：一队刚打完第 k 回合（赢/输 + 性质），第 k+1 回合
     会买什么档位（6 档装备）。→ 回答用户「输掉的队伍会选什么策略」。

  B. 下一回合胜方（outcome）：第 k+1 回合谁会赢，特征 = 比分 + 经济决策 +
     第 k 回合性质 + 动量。→ 核心检验：**回合性质(碾压/翻盘) 能否在比分之外
     预测下一回合**（之前的粗糙版回合依赖模型失败，这次加了性质维度）。

数据：round_states.parquet（outcome_type/经济决策）merge round_dataset.parquet
（roster/match_id/round_class/比分/equip）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

PISTOL_ROUNDS = {1, 13}
POST_PISTOL_ROUNDS = {2, 3, 14, 15}


def _classify6(v: float) -> str:
    if pd.isna(v):
        return None
    if v < 700:
        return "eco0"
    if v < 1400:
        return "eco_armor"
    if v < 2800:
        return "force_low"
    if v < 4100:
        return "force_high"
    if v < 4700:
        return "semi"
    return "full"


def _round_class(r: int) -> str:
    if r in PISTOL_ROUNDS:
        return "pistol"
    if r in POST_PISTOL_ROUNDS:
        return "post_pistol"
    return "regular"


def load_merged() -> pd.DataFrame:
    """自足版：只用 round_states（覆盖全部 1134 图，不依赖 round_dataset 的 912 图）。

    match_id = demo 父目录名；label_ct_win = winner_side；round_class 由 round_num 推；
    经济决策用 freeze 时刻的 equip0。
    """
    rs = pd.read_parquet(config.DATA_DIR / "round_states.parquet")
    rs["round_num"] = rs["round_num"].astype(int)
    rs["match_id"] = rs["demo_path"].map(lambda p: Path(p).parent.name)
    rs["label_ct_win"] = (rs["winner_side"] == "CT").astype(int)
    rs["round_class"] = rs["round_num"].map(_round_class)
    rs["ct_tier6"] = rs["ct_equip0"].map(_classify6)
    rs["t_tier6"] = rs["t_equip0"].map(_classify6)
    rs["equip_gap"] = rs["ct_equip0"] - rs["t_equip0"]
    return rs


def _streak(winners: list[str], idx: int) -> int:
    """进入 idx 的连胜（CT 记正，T 记负）。"""
    if idx == 0:
        return 0
    w = winners[idx - 1]
    s = 0
    for j in range(idx - 1, -1, -1):
        if winners[j] == w:
            s += 1 if w == "CT" else -1
        else:
            break
    return s


def build_transitions() -> pd.DataFrame:
    """每行 = 一张图内 (第 k 回合 → 第 k+1 回合) 的转移。

    特征分两组：
      prev_*  —— 第 k 回合的结果/性质（预测的依据）
      next_*  —— 第 k+1 回合开始时已知（比分/经济决策/round_class）
    标签 = 第 k+1 回合 label_ct_win。
    """
    m = load_merged().sort_values(["demo_path", "round_num"]).reset_index(drop=True)
    rows = []
    for dp, sub in m.groupby("demo_path"):
        sub = sub.sort_values("round_num").reset_index(drop=True)
        winners = sub["winner_side"].tolist()
        score_ct = 0
        score_t = 0
        for i in range(1, len(sub)):
            prev = sub.iloc[i - 1]
            cur = sub.iloc[i]
            # 累计比分（进入 cur 之前）
            if winners[i - 1] == "CT":
                score_ct += 1
            else:
                score_t += 1
            rows.append({
                "match_id": cur["match_id"], "demo_path": dp,
                "round_num": int(cur["round_num"]),
                "map_name": cur["map_name"],
                "round_class": cur["round_class"],
                # 比分 + 动量（第 k+1 开始前已知）
                "score_gap": score_ct - score_t,
                "streak": _streak(winners, i),
                # 第 k 回合性质（预测依据）
                "prev_winner": prev["winner_side"],
                "prev_outcome": prev["outcome_type"],
                "prev_first_kill": prev["first_kill_side"],
                # 第 k+1 经济决策（第 k+1 开始前已知）
                "ct_tier6": cur["ct_tier6"], "t_tier6": cur["t_tier6"],
                "equip_gap": cur["equip_gap"],
                # 标签
                "label_ct_win": int(cur["label_ct_win"]),
            })
    return pd.DataFrame(rows)


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


BASE_CAT = ["map_name", "round_class"]


def evaluate() -> dict:
    t = build_transitions()
    print(f"转移样本：{len(t)} 行 / {t['match_id'].nunique()} 场 / "
          f"{t['demo_path'].nunique()} 图")
    print(f"outcome_type 分布：{t['prev_outcome'].value_counts().to_dict()}")

    print("\n" + "=" * 76)
    print("B. 下一回合胜方：比分 vs +经济决策 vs +回合性质（GroupKFold by match）")
    print("=" * 76)
    print(f"{'特征':<48}{'AUC':>8}{'n':>7}")
    cat_prev = ["prev_winner", "prev_outcome", "prev_first_kill"]
    cat_econ = ["ct_tier6", "t_tier6"]
    combos = [
        ("比分 only", BASE_CAT, ["score_gap"]),
        ("比分 + streak", BASE_CAT, ["score_gap", "streak"]),
        ("比分 + 经济(6档cat+gap)", BASE_CAT + cat_econ, ["score_gap", "equip_gap"]),
        ("比分 + 第k回合性质", BASE_CAT + cat_prev, ["score_gap"]),
        ("比分 + 性质 + 经济", BASE_CAT + cat_prev + cat_econ, ["score_gap", "equip_gap"]),
        ("性质 only(无比分)", BASE_CAT + cat_prev, []),
        ("经济 only(无比分)", BASE_CAT + cat_econ, ["equip_gap"]),
    ]
    for name, c, n in combos:
        a, nn = _auc(t, c, n)
        print(f"  {name:<46}{a:>8.4f}{nn:>7}")

    # 分性质看：翻盘/碾压/均势之后，下一回合的可预测性
    print("\n--- 按第 k 回合性质分层（比分+经济）---")
    for oc, sub in t.groupby("prev_outcome"):
        a, nn = _auc(sub, BASE_CAT + cat_econ, ["score_gap", "equip_gap"])
        print(f"  {oc:<10} n={nn:<6} AUC={a:.4f}")

    return {}


if __name__ == "__main__":
    evaluate()
