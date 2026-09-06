"""run_prediction.py —— 用户策略的回合级检验（不涉及墙钟对齐，纯 demo 回合，可靠）。

用户方法（2026-09-05 定稿，最新）：
  在双方均势的买点（比分持平 → 盘口≈0.5），读一回合（经济买枪 + 现场发生的事），
  用模型学到的规律校验，预测接下来 2-3 回合的方向（吃经济滚雪球那波 run），
  买低 → 涨上去卖高，赚这个差价。

本脚本回答「读一回合经济能否预测后续 run 方向」这个纯回合级问题——
这是策略能否成立的前提，与价格对齐无关，结果可靠。

核心机制（假设）：输掉第 k 回合那队的经济决定 run 是否延续——
  输家破产(eco) → 被迫 eco/半起 → 大概率连输 2-3 回合（run 延续）；
  输家有钱(full) → 下回合能强起 → run 反转。

输出（按入场点切片：比分持平 / 近持平 |gap|<=1 / 全部）：
  1. run 方向标签的基准率（动量基线）
  2. 单回合：equip_gap → 下一回合胜方（已知全局 0.745，这里看均势切片）
  3. run 预测：输家破产信号 + 完整模型，horizon 1/2/3 的 AUC/命中率
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

PISTOL = {1, 13}
POST_PISTOL = {2, 3, 14, 15}


def _round_class(r: int) -> str:
    if r in PISTOL:
        return "pistol"
    if r in POST_PISTOL:
        return "post_pistol"
    return "regular"


def _classify6(v: float) -> str | None:
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


def build_runs() -> pd.DataFrame:
    """每行 = 一张图内「第 k 回合 → 未来 1/2/3 回合」的 run 观测。

    入场点 = 第 k 回合结束后（第 k+1 回合 freeze 前）。信号：
      score_gap  —— 第 k 回合结束后比分差
      winner_k   —— 第 k 回合胜方（CT/T）
      loser_tier6—— 第 k 回合输家在第 k+1 回合的装备档（freeze 时刻，现场可见）
      equip_gap  —— 第 k+1 回合 freeze 时 CT-T 装备差
      streak     —— 进入第 k+1 回合的连胜（带符号）
      outcome_k  —— 第 k 回合性质(stomp/comeback/even)
    标签（horizon 1/2/3）：未来 h 回合里 CT 是否赢多数（run_ct_h）。
    """
    rs = pd.read_parquet(config.DATA_DIR / "round_states.parquet")
    rs = rs.sort_values(["demo_path", "round_num"]).reset_index(drop=True)
    rs["match_id"] = rs["demo_path"].map(lambda p: Path(p).parent.name)

    rows = []
    for dp, sub in rs.groupby("demo_path"):
        sub = sub.sort_values("round_num").reset_index(drop=True)
        n = len(sub)
        winners = sub["winner_side"].tolist()
        ct_eq = sub["ct_equip0"].tolist()
        t_eq = sub["t_equip0"].tolist()
        oc = sub["outcome_type"].tolist()
        fk = sub["first_kill_side"].tolist()
        rnum = sub["round_num"].tolist()
        mname = sub["map_name"].tolist()

        # 进入每回合前的累计比分
        score_ct = score_t = 0
        pre_ct, pre_t = [], []
        for w in winners:
            pre_ct.append(score_ct)
            pre_t.append(score_t)
            if w == "CT":
                score_ct += 1
            else:
                score_t += 1

        for i in range(1, n):  # i = 第 k+1 回合
            if i + 2 >= n:     # 需要 k+1, k+2, k+3 都在图内
                continue
            k = i - 1
            sc, st = pre_ct[i], pre_t[i]
            gap = sc - st
            wk = winners[k]
            loser_side = "T" if wk == "CT" else "CT"
            loser_eq = (t_eq[i] if wk == "CT" else ct_eq[i])
            equip_gap = ct_eq[i] - t_eq[i]

            # streak：进入第 i 回合的连胜（CT 正 / T 负）
            s = 0
            if i > 0:
                w = winners[i - 1]
                for j in range(i - 1, -1, -1):
                    if winners[j] == w:
                        s += 1 if w == "CT" else -1
                    else:
                        break

            # run 标签：未来 h 回合 CT 赢多数
            for h in (1, 2, 3):
                seg = winners[i:i + h]
                ct_w = sum(1 for w in seg if w == "CT")
                rows.append({
                    "match_id": sub["match_id"].iloc[0], "demo_path": dp,
                    "map_name": mname[i], "round_num": int(rnum[i]),
                    "round_class": _round_class(int(rnum[i])),
                    "horizon": h,
                    "score_gap": gap, "winner_k": wk, "loser_side": loser_side,
                    "loser_eq": loser_eq, "loser_tier6": _classify6(loser_eq),
                    "equip_gap": equip_gap, "streak": s,
                    "outcome_k": oc[k], "first_kill_k": fk[k],
                    "run_ct": 1 if ct_w > h - ct_w else 0,  # CT 赢多数
                    "ct_wins": ct_w,
                })
    return pd.DataFrame(rows)


def _auc(d: pd.DataFrame, label: str, cat: list[str], num: list[str]) -> tuple[float, int]:
    d = d.dropna(subset=cat + num + [label]).copy()
    if len(d) < 60 or d[label].nunique() < 2:
        return float("nan"), len(d)
    y = d[label].to_numpy()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
    ncv = min(5, d["match_id"].nunique())
    p = cross_val_predict(clf, d[cat + num], y, groups=d["match_id"].to_numpy(),
                          cv=GroupKFold(ncv), method="predict_proba")[:, 1]
    return roc_auc_score(y, p), len(d)


def _slice_report(d: pd.DataFrame, tag: str) -> None:
    print("\n" + "=" * 78)
    print(f"入场点切片：{tag}   (n 图-回合对={len(d)}, 覆盖 {d['match_id'].nunique()} 场)")
    print("=" * 78)
    if d.empty:
        print("  无样本")
        return

    # 1. 动量基线：winner_k 在未来 h 回合是否延续（赢多数）
    print(f"\n[基准] 动量基线 = 第 k 回合胜方在未来 h 回合仍赢多数的比例：")
    for h in (1, 2, 3):
        dd = d[d.horizon == h]
        if dd.empty:
            continue
        cont = ((dd.run_ct == 1) == (dd.winner_k == "CT")).mean()
        # CT/T 各自延续率（消除 winner_k 的边分布影响）
        print(f"  h={h}: P(延续)={cont:.3f}  "
              f"(CT胜后延续={((dd.winner_k=='CT')&(dd.run_ct==1)).sum()/max((dd.winner_k=='CT').sum(),1):.3f}, "
              f"T胜后延续={((dd.winner_k=='T')&(dd.run_ct==0)).sum()/max((dd.winner_k=='T').sum(),1):.3f})")

    # 2. 单回合经济信号：equip_gap → 下一回合胜方
    d1 = d[d.horizon == 1]
    a1, n1 = _auc(d1, "run_ct", ["round_class"], ["equip_gap"])
    print(f"\n[单回合] equip_gap → 下一回合 CT 是否赢   AUC={a1:.3f}  n={n1}")

    # 3. run 信号（horizon 2/3）：输家破产 → 延续；否则反转
    print(f"\n[run 信号] 输家破产(eco) → 预测 winner_k 延续；输家有钱 → 预测反转：")
    for h in (2, 3):
        dd = d[d.horizon == h]
        if dd.empty:
            continue
        broke = dd["loser_eq"] < 1400
        pred_cont = broke  # 破产→延续
        cont = (dd.run_ct == 1) == (dd.winner_k == "CT")
        acc = (pred_cont == cont).mean()
        base = max(cont.mean(), 1 - cont.mean())
        print(f"  h={h}: 命中率={acc:.3f}  (动量基线={base:.3f})  "
              f"破产时延续率={cont[broke].mean():.3f}  有钱时延续率={cont[~broke].mean():.3f}")

    # 4. 完整模型 → run_ct (horizon 1/2/3)
    print(f"\n[模型] 完整特征(装备档+equip_gap+streak+outcome_k) → run_ct AUC：")
    cat = ["round_class", "loser_tier6", "outcome_k"]
    num = ["equip_gap", "streak"]
    for h in (1, 2, 3):
        dd = d[d.horizon == h]
        a, n = _auc(dd, "run_ct", cat, num)
        print(f"  h={h}: AUC={a:.3f}  n={n}")

    # 5. 对照：比分差本身在「近持平」切片里还有没有信息
    cat2 = ["round_class", "loser_tier6", "outcome_k"]
    num2 = ["equip_gap", "streak", "score_gap"]
    dd = d[d.horizon == 3]
    a, n = _auc(dd, "run_ct", cat2, num2)
    print(f"  对照 h=3 +score_gap: AUC={a:.3f}  n={n}")


def main() -> None:
    runs = build_runs()
    print(f"run 观测总数(含 3 个 horizon)：{len(runs)}  |  图 {runs['demo_path'].nunique()}  |  "
          f"场 {runs['match_id'].nunique()}")
    print(f"outcome_k 分布：{runs[runs.horizon==1]['outcome_k'].value_counts().to_dict()}")

    tied = runs[runs.score_gap == 0]
    near = runs[runs.score_gap.abs() <= 1]
    _slice_report(tied, "比分持平 (score_gap==0)")
    _slice_report(near, "比分近持平 (|gap|<=1)")
    _slice_report(runs, "全部回合")


if __name__ == "__main__":
    main()
