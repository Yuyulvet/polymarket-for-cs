"""lead_test.py —— task #47：领先性测试（读局能否领先分钟级盘中价 + 波段 P&L）。

用户方法（低买高卖）：在别人认为焦灼时，我们已经读出更可能的「后续」，先买入，
后续兑现（回合胜负 → 比分变化 → Map Winner token 移动）时卖出，赚波段差。

这里把 Spirit 的 demo 回合对齐到 Polymarket「Map N Winner」盘中**逐分钟**价，
测四种策略在回合粒度上的 P&L（买 Spirit token，round 末卖）：

  oracle  —— 预知回合胜者（趋势上限：波段到底存不存在）
  leader  —— 押比分领先方（市场的充分统计量基线）
  signal  —— 经济 equip_gap（freeze 可观察）
  exec    —— 执行层假打/真打（mid-round 可观察，本 pilot 新增）

对齐：demos.start_date + round_len_sec 累积（inplay_data.build_round_timeline）。
注意：跨半场/技术暂停会引入分钟级漂移——见结果里的稳健性说明。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .phase4_closed_loop import load_rounds, roster_to_team
from .inplay_data import build_round_timeline

_MAP_NUM = re.compile(r"-m(\d)-")


def _map_number(dp: str) -> int | None:
    m = _MAP_NUM.search(dp)
    return int(m.group(1)) if m else None


def load_spirit_rounds() -> pd.DataFrame:
    """Spirit 出战的每回合：Spirit 中心化的 equip_gap / 胜负 / 比分 / 地图序号。"""
    rounds = load_rounds()
    r2t = roster_to_team(rounds)
    rounds["ct_team"] = rounds["ct_roster"].map(r2t)
    rounds["t_team"] = rounds["t_roster"].map(r2t)

    sp = rounds[(rounds["ct_team"] == "spirit") | (rounds["t_team"] == "spirit")].copy()
    sp["spirit_side"] = np.where(sp["ct_team"] == "spirit", "CT", "T")
    # equip_gap 原义 = ct_equip0 - t_equip0；转成「Spirit 装备 - 对手装备」
    sp["equip_gap_spirit"] = np.where(sp["spirit_side"] == "CT",
                                      sp["equip_gap"], -sp["equip_gap"])
    sp["spirit_win"] = (sp["winner_side"] == sp["spirit_side"]).astype(int)
    sp["map_num"] = sp["demo_path"].map(_map_number)

    # 每图内累计比分（Spirit 领先差）
    sp = sp.sort_values(["demo_path", "round_num"]).reset_index(drop=True)
    sp["spirit_score"] = sp.groupby("demo_path")["spirit_win"].cumsum()
    sp["opp_score"] = sp.groupby("demo_path")["spirit_win"].transform("count") \
        - sp["spirit_score"]  # 不准确：对手赢 = 该图内非本回合累计
    # 更干净：领先差 = 2*累计胜利 - 已打回合数
    sp["score_lead"] = sp["spirit_score"] - (sp.groupby("demo_path").cumcount())
    sp["score_lead"] = sp["score_lead"].astype(int)
    return sp


def add_execution(sp: pd.DataFrame) -> pd.DataFrame:
    """merge 执行层 fake（Spirit T 侧才有）。"""
    ex = pd.read_csv(config.DATA_DIR / "spirit_execution.csv")
    ex = ex[["demo_path", "round_num", "fake"]].copy()
    sp = sp.merge(ex, on=["demo_path", "round_num"], how="left")
    sp["fake"] = sp["fake"].fillna(0).astype(int)
    return sp


def _spirit_series(raw: pd.DataFrame, match_id: str, map_num: int) -> pd.DataFrame:
    """某场某图的 Spirit token 分钟价序列 {t, p}。"""
    s = raw[(raw["match_id"] == match_id)
            & (raw["map_market"] == f"Map {map_num} Winner")
            & (raw["outcome"] == "Spirit")][["t", "p"]]
    return s.sort_values("t").reset_index(drop=True)


def align_pnl(sp: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """每回合对齐 Spirit token 的 price_start/price_end，算 dp = p_end - p_start。"""
    rows = []
    for rw in sp.itertuples(index=False):
        if pd.isna(rw.map_num):
            continue
        s = _spirit_series(raw, rw.match_id, int(rw.map_num))
        if s.empty:
            continue
        t0, t1 = rw.round_start_utc, rw.round_end_utc
        pre = s[s["t"] <= t0]
        if pre.empty:
            continue
        p_start = pre.iloc[-1]["p"]
        post = s[s["t"] >= t1]
        p_end = post.iloc[0]["p"] if not post.empty else s.iloc[-1]["p"]
        rows.append({
            "demo_path": rw.demo_path, "round_num": rw.round_num,
            "spirit_side": rw.spirit_side, "spirit_win": rw.spirit_win,
            "equip_gap_spirit": rw.equip_gap_spirit, "fake": rw.fake,
            "score_lead": rw.score_lead,
            "p_start": p_start, "p_end": p_end, "dp": p_end - p_start,
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    n = len(df)
    if n == 0:
        print(f"\n=== {label} ===  (0 回合，跳过)")
        return
    dpr = df["dp"]

    def pnl(mask) -> float:
        return float(np.where(mask, dpr, -dpr).mean())

    oracle = pnl(df["spirit_win"] == 1)
    leader = pnl(df["score_lead"] > 0)
    signal = pnl(df["equip_gap_spirit"] > 0)
    execf = pnl(df["fake"] == 1)
    exec_c = pnl((df["fake"] == 1) & (df["equip_gap_spirit"] > 0))
    rand = pnl(np.random.RandomState(0).randint(0, 2, n) == 1)
    corr = df["equip_gap_spirit"].corr(dpr)
    corr_fake = df["fake"].corr(dpr)

    print(f"\n=== {label} ===  n_rounds={n}  |平均|Δp|={dpr.abs().mean():.4f}  "
          f"| 非零Δp 回合={int((dpr.abs() > 1e-9).sum())}")
    print(f"  Q1 oracle(预知胜者):      mean P&L = {oracle:+.4f}   (波段存在吗)")
    print(f"     leader(押比分领先):     mean P&L = {leader:+.4f}")
    print(f"  Q2 signal(经济equip):     mean P&L = {signal:+.4f}   corr(equip,Δp)={corr:+.3f}")
    print(f"  Q3 exec(假打):            mean P&L = {execf:+.4f}   corr(fake,Δp)={corr_fake:+.3f}")
    print(f"     exec+经济(假打且优装):  mean P&L = {exec_c:+.4f}")
    print(f"     random(随机方向):       mean P&L = {rand:+.4f}   (bid-ask 摩擦基准)")
    hit = ((df["equip_gap_spirit"] > 0) == (df["spirit_win"] == 1)).mean()
    hit_f = ((df["fake"] == 1) == (df["spirit_win"] == 1)).mean()
    print(f"  方向命中率: 经济={hit:.2f}  假打={hit_f:.2f}")


def main() -> None:
    sp = load_spirit_rounds()
    sp = add_execution(sp)
    print(f"Spirit 回合：{len(sp)} / {sp['demo_path'].nunique()} 图")
    print(f"  其中 T 侧 {int((sp['spirit_side']=='T').sum())} / CT 侧 "
          f"{int((sp['spirit_side']=='CT').sum())}")

    tl = build_round_timeline()
    tl = tl[["demo_path", "round_num", "match_id", "round_start_utc", "round_end_utc"]]
    sp = sp.merge(tl, on=["demo_path", "round_num"], how="inner")
    print(f"对齐到时间线后：{len(sp)} 回合")

    raw = pd.read_parquet(config.DATA_DIR / "polymarket_inplay_raw.parquet")
    raw = raw[raw["outcome"] == "Spirit"].copy()
    df = align_pnl(sp, raw)
    print(f"对齐到分钟价后：{len(df)} 回合 / {df['demo_path'].nunique()} 图")

    report(df, "全部 Spirit 回合")
    # 焦灼切片：比分持平（用户说的「别人认为焦灼」时刻）
    tied = df[df["score_lead"] == 0]
    report(tied, "焦灼切片（比分持平）")
    close = df[df["score_lead"].abs() <= 1]
    report(close, "近焦灼切片（|比分差|<=1）")
    # 仅 T 侧（执行层有意义的地方）
    tside = df[df["spirit_side"] == "T"]
    report(tside, "Spirit T 侧回合")


if __name__ == "__main__":
    main()
