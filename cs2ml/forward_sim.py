"""前向经济仿真：从回合 N 的观测状态出发，蒙特卡洛滚到图结束，得到 P(地图胜者)。

这是「回合开始时预测大方向」的实现：
  - 回合胜率 P(CT胜 | 地图, 回合类型, ct_tier, t_tier) —— 查表（由 round_model 的逻辑回归产出）
  - 经济档位转移 P(tier_{r+1} | tier_r, 输赢) —— 从 6913 回合数据学
  - MC 仿真：采样胜方 → 更新档位 → 继续；半场(第13回合)换边 + 手枪局重置经济

无泄漏纪律：GroupKFold by match，回合模型 + 转移矩阵只在训练折上学，测试折上仿真。
产物：打印「预测准确率 vs 预测时点（回合 N）」曲线，对比朴素基线（押当前领先方）。
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .round_model import _round_class, build_round_dataset

CAT = ["map_name", "round_class", "ct_tier", "t_tier"]
TIERS = ["eco", "force", "full"]
ROUND_CLASSES = ["pistol", "post_pistol", "regular"]
FIRST_HALF_ROUNDS = 12


def _fit_round_lookup(train: pd.DataFrame) -> dict:
    """在训练折上学 P(CT胜|地图,回合类型,ct_tier,t_tier)，做成查表。"""
    enc = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([("cat", enc, CAT)])
    clf = LogisticRegression(max_iter=2000)
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(train[CAT], train["label_ct_win"])

    maps = sorted(train["map_name"].unique())
    combos = pd.DataFrame([(m, rc, ct, t)
                           for m in maps for rc in ROUND_CLASSES
                           for ct in TIERS for t in TIERS], columns=CAT)
    probs = model.predict_proba(combos)[:, 1]
    return {tuple(r): p for r, p in zip(combos.itertuples(index=False), probs)}


def _fit_transition(train: pd.DataFrame) -> dict:
    """P(tier_{r+1} | tier_r, 输赢)，逐队、只在同一半场内相邻回合统计（跳过第12回合的换边）。"""
    counts: dict[tuple, Counter] = defaultdict(Counter)
    for _, sub in train.groupby("match_id"):
        sub = sub.sort_values("round_num")
        for i in range(len(sub) - 1):
            r = sub.iloc[i]
            rn = sub.iloc[i + 1]
            if int(r.round_num) == FIRST_HALF_ROUNDS:  # 12→13 换边，跳过
                continue
            if int(r.round_num) < FIRST_HALF_ROUNDS and int(rn.round_num) > FIRST_HALF_ROUNDS:
                continue
            counts[(r.ct_tier, 1 if r.winner_side == "CT" else 0)][rn.ct_tier] += 1
            counts[(r.t_tier, 1 if r.winner_side == "T" else 0)][rn.t_tier] += 1
    trans: dict = {}
    for k, c in counts.items():
        total = sum(c.values())
        if total:
            trans[k] = {t: v / total for t, v in c.items()}
    return trans


def _sample_tier(trans: dict, tier: str, won: int, rng: np.random.Generator) -> str:
    dist = trans.get((tier, won))
    if dist is None:
        return tier
    ts = list(dist)
    ps = np.asarray([dist[t] for t in ts], dtype=float)
    ps /= ps.sum()
    return str(rng.choice(ts, p=ps))


def _simulate(lookup: dict, trans: dict, map_name: str, start_round: int,
              s0: int, s1: int, t0: str, t1: str, n_sims: int,
              rng: np.random.Generator) -> float:
    """从 (start_round, s0:s1, t0/t1) 出发，滚到某队 13 分，返回 P(roster0 胜)。

    roster0 = 上半场 CT，roster1 = 上半场 T；下半场(第13回合起)换边。
    """
    wins = 0
    for _ in range(n_sims):
        a, b = s0, s1
        ta, tb = t0, t1
        r = start_round
        while a < 13 and b < 13 and r < 60:
            if r == 13:  # 半场换边 + 手枪局经济重置
                ta = tb = "eco"
            if r <= FIRST_HALF_ROUNDS:
                ct_tier, t_tier = ta, tb
            else:
                ct_tier, t_tier = tb, ta
            p = lookup.get((map_name, _round_class(r), ct_tier, t_tier), 0.5)
            ct_wins = rng.random() < p
            roster0_wins = ct_wins if r <= FIRST_HALF_ROUNDS else (not ct_wins)
            if roster0_wins:
                a += 1
                ta = _sample_tier(trans, ta, 1, rng)
                tb = _sample_tier(trans, tb, 0, rng)
            else:
                b += 1
                ta = _sample_tier(trans, ta, 0, rng)
                tb = _sample_tier(trans, tb, 1, rng)
            r += 1
        if a >= 13:
            wins += 1
    return wins / n_sims


def _actual_winner(sub: pd.DataFrame) -> str | None:
    vc = sub[sub["winner_roster"] != ""]["winner_roster"].value_counts()
    return str(vc.idxmax()) if len(vc) else None


def evaluate_forward(n_sims: int = 500) -> dict:
    df = build_round_dataset()
    df = df.dropna(subset=["ct_tier", "t_tier"]).copy()
    df["winner_roster"] = df["winner_roster"].fillna("").astype(str)

    checkpoints = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    acc: dict[int, list[int]] = {c: [] for c in checkpoints}
    base_acc: dict[int, list[int]] = {c: [] for c in checkpoints}
    n_eval: dict[int, int] = {c: 0 for c in checkpoints}

    gkf = GroupKFold(5)
    for train_idx, test_idx in gkf.split(df, groups=df["match_id"].to_numpy()):
        train = df.iloc[train_idx]
        lookup = _fit_round_lookup(train)
        trans = _fit_transition(train)
        rng = np.random.default_rng(0)
        for mid in df.iloc[test_idx]["match_id"].unique():
            sub = df[df["match_id"] == mid].sort_values("round_num")
            if len(sub) < 3:
                continue
            roster0 = str(sub.iloc[0]["ct_roster"])
            roster1 = str(sub.iloc[0]["t_roster"])
            winner = _actual_winner(sub)
            if winner is None or winner not in (roster0, roster1):
                continue
            map_name = str(sub.iloc[0]["map_name"])
            max_r = int(sub["round_num"].max())
            for c in checkpoints:
                if c > max_r:
                    continue
                if c == 1:
                    s0, s1, t0, t1 = 0, 0, "eco", "eco"
                else:
                    prior = sub[sub["round_num"] < c]
                    s0 = int((prior["winner_roster"] == roster0).sum())
                    s1 = int((prior["winner_roster"] == roster1).sum())
                    row = sub[sub["round_num"] == c]
                    if not len(row):
                        continue
                    rr = row.iloc[0]
                    if rr["ct_roster"] == roster0:
                        t0, t1 = rr["ct_tier"], rr["t_tier"]
                    else:
                        t0, t1 = rr["t_tier"], rr["ct_tier"]
                p = _simulate(lookup, trans, map_name, c, s0, s1, t0, t1, n_sims, rng)
                pred_roster0 = p > 0.5
                truth = winner == roster0
                acc[c].append(int(pred_roster0 == truth))
                # 朴素基线：押当前领先方（平局押 CT 即 roster0）
                leader0 = s0 >= s1
                base_acc[c].append(int(leader0 == truth))
                n_eval[c] += 1

    print(f"{'预测时点(回合N)':<16}{'仿真acc':>10}{'押领先方acc':>12}{'n_maps':>8}")
    for c in checkpoints:
        if n_eval[c]:
            a = np.mean(acc[c])
            b = np.mean(base_acc[c])
            print(f"{c:<18}{a:>9.3f}{b:>11.3f}{n_eval[c]:>8}")
    return {"acc": {c: float(np.mean(acc[c])) for c in checkpoints if n_eval[c]},
            "baseline": {c: float(np.mean(base_acc[c])) for c in checkpoints if n_eval[c]},
            "n": n_eval}


if __name__ == "__main__":
    evaluate_forward()
