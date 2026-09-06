"""team_strategy.py —— 队伍级细粒度买枪策略学习 + 「学习记忆」持久化。

用户要求：不要宏观看「输家买什么」，而是对 demo 里**现存的每一支队伍**（roster 级），
学习它在**不同回合处境**下的买枪策略，并单独存一份**学习记忆**（可随新 demo 更新）。

策略处境（situation）维度：
  - round_class：pistol / post_pistol / regular（经济重置点）
  - score_bucket：落后3+ / 落后1-2 / 持平 / 领先1-2 / 领先3+（压力 → force vs save）
  - prev_result：上回合赢 / 输
  - prev_nature：碾压 / 翻盘 / 均势
  - side：CT / T

每队产出一个「策略签名」（interpretable 倾向率）：
  - force_when_behind：落后时强起率（激进强起型 vs 纪律存钱型的核心分水岭）
  - eco_when_behind：落后时存钱率
  - full_when_ahead：领先时全起率
  - post_pistol_loss_force：手枪局输了之后强起率（经济转换习惯）
  - full_when_even：持平时全起率

输出：
  1. 队伍间策略差异（签名率的分布 → 有没有 per-team 信号）
  2. 前 N 激进强起队 / 纪律存钱队排名
  3. data/team_strategy_memory.json —— 持久化「学习记忆」（签名 + 条件表）
  4. 检验：加队伍身份（roster one-hot）是否比宏观模型更准预测买枪（GroupKFold by match）
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .round_transition import _classify6, _round_class

TIER_ORDER = ["eco0", "eco_armor", "force_low", "force_high", "semi", "full"]
FORCE = {"force_low", "force_high"}
ECO = {"eco0", "eco_armor"}
_MIN_ROUNDS = 20  # 队伍样本下限


def _score_bucket(gap: int) -> str:
    if gap <= -3:
        return "behind3+"
    if gap <= -1:
        return "behind1-2"
    if gap == 0:
        return "even"
    if gap >= 3:
        return "ahead3+"
    return "ahead1-2"


def build_team_rounds() -> pd.DataFrame:
    """每行 = 一支队伍（roster）在某一回合的买枪决策 + 处境。"""
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    rs = pd.read_parquet(config.DATA_DIR / "round_states.parquet")
    keep = ["demo_path", "round_num", "outcome_type", "first_kill_side",
            "ct_equip0", "t_equip0"]
    m = rd.merge(rs[keep], on=["demo_path", "round_num"], how="inner")
    m["round_num"] = m["round_num"].astype(int)
    m = m.sort_values(["demo_path", "round_num"]).reset_index(drop=True)

    rows = []
    for dp, sub in m.groupby("demo_path"):
        sub = sub.sort_values("round_num").reset_index(drop=True)
        winners = sub["winner_side"].tolist()
        score_ct = 0
        score_t = 0
        for i in range(len(sub)):
            cur = sub.iloc[i]
            if i > 0:
                prev = sub.iloc[i - 1]
                if winners[i - 1] == "CT":
                    score_ct += 1
                else:
                    score_t += 1
            for side in ("CT", "T"):
                sl = side.lower()
                roster = cur[f"{sl}_roster"]
                if roster is None or (isinstance(roster, float) and np.isnan(roster)):
                    continue
                my_score = score_ct if side == "CT" else score_t
                op_score = score_t if side == "CT" else score_ct
                rows.append({
                    "roster": str(roster),
                    "demo_path": dp,
                    "match_id": cur["match_id"],
                    "round_num": int(cur["round_num"]),
                    "side": side,
                    "round_class": _round_class(int(cur["round_num"])),
                    "score_bucket": _score_bucket(my_score - op_score),
                    "prev_result": (None if i == 0 else
                                    ("win" if winners[i - 1] == side else "lose")),
                    "prev_nature": (None if i == 0 else prev["outcome_type"]),
                    "buy_tier": _classify6(cur[f"{sl}_equip0"]),
                })
    return pd.DataFrame(rows)


def _rate(df: pd.DataFrame, tier_set: set[str], cond: pd.Series) -> float:
    """给定条件下，买 tier_set 档位的比例（NaN 若样本<5）。"""
    sub = df[cond]
    if len(sub) < 5:
        return float("nan")
    return float(sub["buy_tier"].isin(tier_set).mean())


def team_signature(df: pd.DataFrame) -> dict:
    """一队 df → 策略签名。"""
    regular = df["round_class"] == "regular"
    behind = df["score_bucket"].isin(["behind1-2", "behind3+"])
    ahead = df["score_bucket"].isin(["ahead1-2", "ahead3+"])
    even = df["score_bucket"] == "even"
    post_pistol_lose = (df["round_class"] == "post_pistol") & (df["prev_result"] == "lose")
    return {
        "force_when_behind": _rate(df, FORCE, regular & behind),
        "eco_when_behind": _rate(df, ECO, regular & behind),
        "full_when_ahead": _rate(df, {"full"}, regular & ahead),
        "post_pistol_loss_force": _rate(df, FORCE, post_pistol_lose),
        "full_when_even": _rate(df, {"full"}, regular & even),
    }


def _steamid_to_name() -> dict[str, str]:
    """steamid → 最常用选手昵称（来自 player_features）。"""
    pf = pd.read_parquet(config.DATA_DIR / "player_features.parquet")
    m = pf.groupby("steamid")["name"].agg(lambda s: s.value_counts().idxmax())
    return m.to_dict()


def build_memory() -> dict:
    """学习记忆：每队签名 + 条件表 + 样本数 + 可读选手名单。"""
    t = build_team_rounds()
    t = t.dropna(subset=["buy_tier"])
    sid2name = _steamid_to_name()
    teams = {}
    for roster, sub in t.groupby("roster"):
        if len(sub) < _MIN_ROUNDS:
            continue
        sig = team_signature(sub)
        players = [sid2name.get(s, s[:8]) for s in roster.split(",")]
        # 条件表：round_class | score_bucket | prev_result → 档位分布
        cond = (sub.dropna(subset=["prev_result"])
                .groupby(["round_class", "score_bucket", "prev_result"])["buy_tier"]
                .value_counts(normalize=True).round(3))
        cond_table: dict[str, dict[str, float]] = {}
        for (rc, sb, pr, tier), p in cond.items():
            cond_table.setdefault(f"{rc}|{sb}|{pr}", {})[tier] = round(float(p), 3)
        teams[roster] = {
            "roster": roster.split(","),
            "players": players,
            "label": ", ".join(players),
            "n_rounds": int(len(sub)),
            "signature": {k: (None if pd.isna(v) else round(float(v), 3))
                          for k, v in sig.items()},
            "conditional": cond_table,
        }
    return {"version": 1, "built_at": datetime.now(timezone.utc).isoformat(),
            "n_teams": len(teams), "teams": teams}


def evaluate() -> dict:
    t = build_team_rounds()
    t = t.dropna(subset=["buy_tier"])
    print(f"队伍-回合样本：{len(t)} 行 / {t['roster'].nunique()} 支 roster")

    # 队伍间签名差异（有没有 per-team 信号）
    sig_rows = []
    for roster, sub in t.groupby("roster"):
        if len(sub) < _MIN_ROUNDS:
            continue
        s = team_signature(sub)
        s["roster"] = roster
        s["n_rounds"] = len(sub)
        sig_rows.append(s)
    sig = pd.DataFrame(sig_rows)
    print(f"样本 ≥{_MIN_ROUNDS} 的队伍：{len(sig)} 支")

    print("\n" + "=" * 74)
    print("队伍间策略签名差异（per-team 信号有多强）")
    print("=" * 74)
    keys = ["force_when_behind", "eco_when_behind", "full_when_ahead",
            "post_pistol_loss_force", "full_when_even"]
    print(f"  {'签名':<24}{'min':>6}{'p25':>6}{'中位':>6}{'p75':>6}{'max':>6}{'std':>6}")
    for k in keys:
        v = sig[k].dropna()
        print(f"  {k:<24}{v.min():>6.3f}{v.quantile(.25):>6.3f}{v.median():>6.3f}"
              f"{v.quantile(.75):>6.3f}{v.max():>6.3f}{v.std():>6.3f}")

    # 排名：激进强起 vs 纪律存钱
    print("\n--- 落后时强起率 最高 8 队（激进）---")
    top = sig.dropna(subset=["force_when_behind"]).nlargest(8, "force_when_behind")
    for _, r in top.iterrows():
        print(f"  {r['roster'][:34]}...  force={r['force_when_behind']:.2f} "
              f"eco={r['eco_when_behind']:.2f} n={int(r['n_rounds'])}")
    print("--- 落后时存钱率 最高 8 队（纪律）---")
    bot = sig.dropna(subset=["eco_when_behind"]).nlargest(8, "eco_when_behind")
    for _, r in bot.iterrows():
        print(f"  {r['roster'][:34]}...  eco={r['eco_when_behind']:.2f} "
              f"force={r['force_when_behind']:.2f} n={int(r['n_rounds'])}")

    # 持久化学习记忆
    mem = build_memory()
    out = config.DATA_DIR / "team_strategy_memory.json"
    out.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n学习记忆已写入 {out}（{mem['n_teams']} 支队伍）")

    # 检验：加队伍身份是否比宏观更准（GroupKFold by match，多分类 top1）
    print("\n" + "=" * 74)
    print("队伍身份 vs 宏观模型 预测买枪（GroupKFold by match）")
    print("=" * 74)
    from collections import Counter
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    y = t["buy_tier"].map({x: i for i, x in enumerate(TIER_ORDER)}).to_numpy()
    baseline = Counter(y).most_common(1)[0][1] / len(y)

    def acc_multi(d, cat, num):
        d = d.dropna(subset=cat + num + ["buy_tier"])
        yy = d["buy_tier"].map({x: i for i, x in enumerate(TIER_ORDER)}).to_numpy()
        ncv = min(5, d["match_id"].nunique())
        cv = GroupKFold(ncv)
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ])
        clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=4000))])
        p = cross_val_predict(clf, d[cat + num], yy, groups=d["match_id"].to_numpy(), cv=cv)
        proba = cross_val_predict(clf, d[cat + num], yy, groups=d["match_id"].to_numpy(),
                                  cv=cv, method="predict_proba")
        top1 = (p == yy).mean()
        top2 = (np.argsort(-proba, axis=1)[:, :2] == yy[:, None]).any(axis=1).mean()
        return top1, top2, len(d)

    print(f"  {'特征':<50}{'top1':>8}{'top2':>8}{'n':>7}")
    print(f"  {'baseline(众数)':<50}{baseline:>8.4f}{'-':>8}{len(t):>7}")
    combos = [
        ("宏观: 处境(回合类+比分+上回合结果)", ["round_class", "score_bucket", "prev_result"],
         []),
        ("+ 上回合性质", ["round_class", "score_bucket", "prev_result", "prev_nature"], []),
        ("+ 队伍身份(roster one-hot)", ["round_class", "score_bucket", "prev_result",
                                        "roster"], []),
    ]
    for name, c, num in combos:
        a, b, nn = acc_multi(t, c, num)
        print(f"  {name:<50}{a:>8.4f}{b:>8.4f}{nn:>7}")
    return {}


if __name__ == "__main__":
    evaluate()
