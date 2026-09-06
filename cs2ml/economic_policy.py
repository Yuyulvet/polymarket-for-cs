"""economic_policy.py —— 经济决策策略层（Stage A，用户闭环的「预测对方策略」）。

回答：一支队伍刚打完第 k 回合（赢/输 + 碾压/翻盘/均势 + 比分处境），第 k+1 回合
会买什么档位（6 档装备）。这是「另一支输掉的队伍会选择什么样的策略」的数据化。

关键区分：买枪档位 ≈ 钱的代理。我们想知道的是**在钱之外**的战略内容——
比如钱在「强起区间」时，落后方是选择 force 还是 save（这就是读局 / 半职业的 edge 所在）。

输出：
  1. 可解释的策略表（条件概率）：
     - P(下一档 | 上回合结果)  赢 vs 输
     - P(下一档 | 结果 + 性质)  碾压输 / 翻盘输 / 均势
     - 分回合类型看（手枪局后 2 局是经济转换关键）
  2. 可预测性增量（GroupKFold by match）：钱连续性 → +结果 → +性质 +比分
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .round_transition import _round_class, load_merged

TIER_ORDER = ["eco0", "eco_armor", "force_low", "force_high", "semi", "full"]
_TIER_IDX = {t: i for i, t in enumerate(TIER_ORDER)}


def build_team_transitions() -> pd.DataFrame:
    """每行 = 一队在 (第 k 回合 → 第 k+1 回合) 的买枪决策。

    双方都展开（CT 行 + T 行），特征 = 上回合结果/性质 + 比分处境 + 上回合自己档位，
    标签 = 本回合该队买的档位。
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
            if winners[i - 1] == "CT":
                score_ct += 1
            else:
                score_t += 1
            for side, other in (("CT", "T"), ("T", "CT")):
                my_score = score_ct if side == "CT" else score_t
                op_score = score_t if side == "CT" else score_ct
                sl = side.lower()
                rows.append({
                    "match_id": cur["match_id"],
                    "demo_path": dp,
                    "round_num": int(cur["round_num"]),
                    "map_name": cur["map_name"],
                    "side": side,
                    "round_class": cur["round_class"],
                    "prev_result": "win" if winners[i - 1] == side else "lose",
                    "prev_nature": prev["outcome_type"],
                    "score_gap_me": my_score - op_score,
                    "prev_tier": prev[f"{sl}_tier6"],
                    "label_tier": cur[f"{sl}_tier6"],
                })
    return pd.DataFrame(rows)


def _policy_table(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    g = df.dropna(subset=["label_tier"]).groupby(groupby)["label_tier"].value_counts(
        normalize=True).rename("p").reset_index()
    g["p"] = (g["p"] * 100).round(1)
    return g


def evaluate() -> dict:
    t = build_team_transitions()
    t = t.dropna(subset=["label_tier", "prev_tier"])
    n = len(t)
    n_match = t["match_id"].nunique()
    print(f"团队转移样本：{n} 行 / {n_match} 场")
    print(f"档位分布（本回合买入）：{t['label_tier'].value_counts().to_dict()}")

    print("\n" + "=" * 72)
    print("A1. 输赢之后的买枪策略（P(下一档 | 上回合结果)）")
    print("=" * 72)
    print(_policy_table(t, ["prev_result"]).pivot(
        index="label_tier", columns="prev_result", values="p").fillna(0).round(1)
        .reindex(TIER_ORDER).to_string())

    print("\n" + "=" * 72)
    print("A2. 输家策略 × 上回合性质（碾压输 / 翻盘输 / 均势）")
    print("=" * 72)
    loser = t[t["prev_result"] == "lose"]
    print(_policy_table(loser, ["prev_nature"]).pivot(
        index="label_tier", columns="prev_nature", values="p").fillna(0).round(1)
        .reindex(TIER_ORDER).to_string())

    print("\n" + "=" * 72)
    print("A3. 输家策略 × 回合类型（手枪局后2局是经济转换关键）")
    print("=" * 72)
    print(_policy_table(loser, ["round_class"]).pivot(
        index="label_tier", columns="round_class", values="p").fillna(0).round(1)
        .reindex(TIER_ORDER).to_string())

    print("\n" + "=" * 72)
    print("A4. 可预测性增量（GroupKFold by match，多分类 top1/top2 精度）")
    print("=" * 72)
    y = t["label_tier"].map(_TIER_IDX).to_numpy()
    baseline = Counter(y).most_common(1)[0][1] / len(y)
    base_cat = ["map_name", "round_class"]

    def acc_multi(d: pd.DataFrame, cat: list[str], num: list[str]):
        d = d.dropna(subset=cat + num + ["label_tier"])
        yy = d["label_tier"].map(_TIER_IDX).to_numpy()
        groups = d["match_id"].to_numpy()
        ncv = min(5, d["match_id"].nunique())
        cv = GroupKFold(ncv)
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ])
        clf = Pipeline([("pre", pre),
                        ("clf", LogisticRegression(max_iter=4000))])
        p = cross_val_predict(clf, d[cat + num], yy, groups=groups, cv=cv)
        proba = cross_val_predict(clf, d[cat + num], yy, groups=groups, cv=cv,
                                  method="predict_proba")
        top1 = (p == yy).mean()
        top2 = (np.argsort(-proba, axis=1)[:, :2] == yy[:, None]).any(axis=1).mean()
        return top1, top2, len(d)

    print(f"  {'特征':<42}{'top1':>8}{'top2':>8}{'n':>7}")
    print(f"  {'baseline(众数=full)':<42}{baseline:>8.4f}{'-':>8}{n:>7}")
    combos = [
        ("钱连续性(prev_tier only)", base_cat + ["prev_tier"], []),
        ("+ 上回合结果", base_cat + ["prev_tier", "prev_result"], []),
        ("+ 上回合性质", base_cat + ["prev_tier", "prev_result", "prev_nature"], []),
        ("+ 比分处境", base_cat + ["prev_tier", "prev_result", "prev_nature"],
         ["score_gap_me"]),
    ]
    for name, c, num in combos:
        a, b, nn = acc_multi(t, c, num)
        print(f"  {name:<42}{a:>8.4f}{b:>8.4f}{nn:>7}")

    return {}


if __name__ == "__main__":
    evaluate()
