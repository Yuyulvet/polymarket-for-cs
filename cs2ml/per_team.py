"""Per-team（roster）× round-context 转移率特征（LOO，按 match_id 留一）。

用户的「特征化学习」：不是 roster one-hot（稀疏必负），而是显式学每个战队在不同
回合语境下的胜率/倾向：
  - pistol：手枪局胜率
  - post_pistol：赢/输手枪后第 2-3 回合（强弱起/eco）的胜率
  - eco / force / full：各经济档的胜率
  - 以及 赢手枪→赢第2回合 的转换率 vs 输手枪→赢第2回合（翻盘率）

LOO：用除【当前比赛】外的所有回合算每队历史胜率，杜绝泄漏（roster 身份本身是
赛前已知的，所以这是合法的 as-of 特征）。

min_n：某队在某个语境下的历史回合数 < min_n 时置 NaN（避免高方差噪声）。
"""
from __future__ import annotations

import pandas as pd

CONTEXTS = ["pistol", "post_pistol", "eco", "force", "full"]


def _context(round_class: str, tier: str) -> str:
    if round_class == "pistol":
        return "pistol"
    if round_class == "post_pistol":
        return "post_pistol"
    return tier  # eco / force / full


def compute_features(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """给 round_dataset 每行补 per-team LOO 语境胜率（ct/t 两队的 5 个语境）。

    输入 df 需含：match_id, demo_path, round_num, ct_roster, t_roster, winner_roster,
                  round_class, ct_tier, t_tier
    输出在原 df 上新增列：{side}_{ctx}_rate, {side}_{ctx}_n（side=ct/t, ctx=5语境）。
    """
    df = df.copy()
    df["ct_roster"] = df["ct_roster"].astype(str)
    df["t_roster"] = df["t_roster"].astype(str)
    df["winner_roster"] = df["winner_roster"].fillna("").astype(str)

    # 长表：每回合拆成 ct/t 两行，含 roster + 语境 + 是否赢
    long_rows = []
    for _, r in df.iterrows():
        w = r["winner_roster"]
        long_rows.append({
            "match_id": r["match_id"], "roster": r["ct_roster"],
            "ctx": _context(r["round_class"], r["ct_tier"]),
            "won": int(w == r["ct_roster"]) if w else 0,
        })
        long_rows.append({
            "match_id": r["match_id"], "roster": r["t_roster"],
            "ctx": _context(r["round_class"], r["t_tier"]),
            "won": int(w == r["t_roster"]) if w else 0,
        })
    long = pd.DataFrame(long_rows)

    # 全局聚合 (roster, ctx) -> wins, n
    all_agg = (long.groupby(["roster", "ctx"])["won"]
                  .agg(wins="sum", n="size").reset_index())
    # 每场比赛内聚合 (roster, match_id, ctx) -> wins, n（用于 LOO 减去当前场）
    match_agg = (long.groupby(["roster", "match_id", "ctx"])["won"]
                    .agg(wins="sum", n="size").reset_index())

    # dict 预聚合：LOO 查找 O(1)，避免 191k 次 DataFrame 过滤（数据扩到万级后从分钟退化到秒）
    all_map = {(r.roster, r.ctx): (int(r.wins), int(r.n))
               for r in all_agg.itertuples(index=False)}
    match_map = {(r.roster, r.match_id, r.ctx): (int(r.wins), int(r.n))
                 for r in match_agg.itertuples(index=False)}

    def _loo_rate(roster: str, match_id: str, ctx: str):
        key = (roster, ctx)
        if key not in all_map:
            return float("nan"), 0
        w_all, n_all = all_map[key]
        w_cur, n_cur = match_map.get((roster, match_id, ctx), (0, 0))
        w_loo, n_loo = w_all - w_cur, n_all - n_cur
        if n_loo < min_n:
            return float("nan"), n_loo
        return w_loo / n_loo, n_loo

    for side, rcol in (("ct", "ct_roster"), ("t", "t_roster")):
        for ctx in CONTEXTS:
            rates, ns = [], []
            for roster, mid in zip(df[rcol], df["match_id"]):
                rt, n = _loo_rate(roster, mid, ctx)
                rates.append(rt); ns.append(n)
            df[f"{side}_{ctx}_rate"] = rates
            df[f"{side}_{ctx}_n"] = ns

    # 差值特征（home 视角）：ct 率 - t 率
    for ctx in CONTEXTS:
        df[f"{ctx}_rate_diff"] = df[f"ct_{ctx}_rate"] - df[f"t_{ctx}_rate"]
    return df


def rate_columns() -> list[str]:
    out = []
    for side in ("ct", "t"):
        out += [f"{side}_{ctx}_rate" for ctx in CONTEXTS]
    out += [f"{ctx}_rate_diff" for ctx in CONTEXTS]
    return out


if __name__ == "__main__":
    from .round_model import build_round_dataset
    df = compute_features(build_round_dataset())
    print(f"特征行数 {len(df)} / 非空率:")
    for c in rate_columns():
        print(f"  {c:<18} 非空 {df[c].notna().mean():.3f}  均值 {df[c].mean():.3f}")
