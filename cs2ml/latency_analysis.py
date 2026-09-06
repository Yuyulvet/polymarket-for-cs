"""latency_analysis.py —— 赛后 latency/趋势分析（task #39）。

用今晚 G2 vs Falcons (event 944152)：
  1) 两张 demo（inferno + dust2）→ 回合级 ground truth（胜者/比分/经济/ticks）
  2) 实时录价 jsonl → Map1/2 Winner 秒级 mid 价

核心问题（用户方法 = 低买高卖「趋势」，不预测谁赢）：
  Q1 oracle：完美预知回合胜者，freeze 买胜者、round 末卖，趋势存在吗？
  Q2 signal：经济信号 equip_gap 能否预测短窗口价格方向（读局可货币化吗？）
  Q3 latency：经济可观察时刻 比 价格响应时刻 早多久。

对齐：demo tick → 墙钟，用「map 结算(价格锁 ~1.0)」做锚，tick/64=秒。
      （回合内无暂停，freeze→end 线性映射准确；跨半场暂停会引入漂移，已用
        NaN 回合(13-15)剔除技术暂停，但漂移仍存在——见结果里的稳健性说明。）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

from . import config

DEMO_DIR = config.DATA_DIR / "demos" / "hltv-2396947-110982"
M1 = DEMO_DIR / "falcons-vs-g2-m1-inferno.dem"
M2 = DEMO_DIR / "falcons-vs-g2-m2-dust2.dem"
M0NESY_SID = "76561198074762801"
TICK_RATE = 64.0
FIRST_HALF_ROUNDS = 12


def _latest_price_file() -> Path:
    return sorted(config.DATA_DIR.glob("realtime/event_944152_*.jsonl"))[-1]


# ---------------------------------------------------------------- demo 解析
def parse_demo(dem_path: Path) -> pd.DataFrame:
    p = DemoParser(str(dem_path))
    header = p.parse_header()
    map_name = header.get("map_name", dem_path.stem)
    pi = p.parse_player_info()
    sid2g = {str(r.steamid): int(r.team_number) for r in pi.itertuples(index=False)}
    groups = sorted(set(sid2g.values()))
    falcons_g = next((g for g in groups if M0NESY_SID in
                      {str(s) for s in pi[pi.team_number == g].steamid}), None)
    g2_g = next(g for g in groups if g != falcons_g)

    re_ = p.parse_event("round_end")
    frz = p.parse_event("round_freeze_end")
    re_ = re_[re_["winner"].notna()].copy()
    re_["round"] = re_["round"].astype(int)
    freeze_ticks = sorted(int(t) for t in frz["tick"].tolist())

    t0 = freeze_ticks[0] + 1
    ts0 = p.parse_ticks(["team_name"], ticks=[t0])
    group_side = {}
    for r in ts0.itertuples(index=False):
        g = sid2g.get(str(r.steamid))
        if g is not None:
            group_side[g] = "CT" if str(r.team_name).upper().startswith("CT") else "T"

    def side_of(g, r):
        s = group_side.get(g, "")
        return s if r <= FIRST_HALF_ROUNDS else ("T" if s == "CT" else "CT")

    start_ticks = [t + 1 for t in freeze_ticks]
    eq = p.parse_ticks(["current_equip_value"], ticks=start_ticks)
    eq["group"] = eq["steamid"].astype(str).map(sid2g)
    eq = eq.dropna(subset=["group"])
    eq_by_tick = {}
    for t, sub in eq.groupby("tick"):
        eq_by_tick[int(t)] = {int(g): float(v) for g, v in
                              sub.groupby("group")["current_equip_value"].mean().items()}

    rows = []
    score_f = score_g = 0
    for rr in re_.itertuples(index=False):
        r = int(rr.round)
        end_tick = int(rr.tick)
        wside = str(rr.winner).upper()
        falcons_win = int(side_of(falcons_g, r) == wside)
        score_f += falcons_win
        score_g += 1 - falcons_win
        cand = [t for t in freeze_ticks if t < end_tick]
        frz_tick = cand[-1] if cand else None
        ef = eg = np.nan
        if frz_tick is not None:
            d = eq_by_tick.get(frz_tick + 1, {})
            ef = d.get(falcons_g, np.nan)
            eg = d.get(g2_g, np.nan)
        rows.append({
            "map_name": map_name, "round": r, "frz_tick": frz_tick, "end_tick": end_tick,
            "winner_side": wside, "falcons_win": falcons_win,
            "score_f": score_f, "score_g": score_g, "falcons_side": side_of(falcons_g, r),
            "equip_f": round(ef, 1) if not pd.isna(ef) else np.nan,
            "equip_g": round(eg, 1) if not pd.isna(eg) else np.nan,
        })
    df = pd.DataFrame(rows)
    df["equip_gap_f"] = df["equip_f"] - df["equip_g"]
    df["dur_ticks"] = df["end_tick"] - df["frz_tick"]
    df["dur_s"] = df["dur_ticks"] / TICK_RATE
    return df


# ---------------------------------------------------------------- 价格
def load_price(price_file: Path) -> pd.DataFrame:
    rows = []
    with open(price_file, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("market") in ("Map 1 Winner", "Map 2 Winner") and d.get("mid") is not None:
                rows.append((d["market"], d["outcome"], float(d["mid"]), float(d["local_ts"])))
    return pd.DataFrame(rows, columns=["market", "outcome", "mid", "ts"])


def mid_series(price, market, outcome):
    s = price[(price.market == market) & (price.outcome == outcome)]
    return s.groupby("ts")["mid"].last().sort_index()


def resolution_time(series, thr=0.999):
    """结算锚 = token 首次 mid 锁到 ~1.0（>=0.999）。

    「赛点瞬峰」最多到 ~0.995（mid），真结算才到 0.9995，故 0.999 可区分。
    空 book(0.5) 发生在结算之后，不影响「首次 >=0.999」。"""
    s = series[series.notna()]
    if s.empty:
        return None
    grid = np.arange(s.index.min(), s.index.max() + 1, 1.0)
    v = s.reindex(s.index.union(pd.Index(grid))).sort_index().ffill().reindex(grid).to_numpy()
    hi = np.where(v >= thr)[0]
    if not len(hi):
        return None
    return float(grid[hi[0]])


def _nearest_mid(series, t):
    """mid series -> 时刻 t 的 mid（最近一次 <= t 的值）。"""
    idx = series.index.to_numpy()
    pos = np.searchsorted(idx, t, side="right") - 1
    if pos < 0:
        pos = 0
    return float(series.iloc[pos])


def align_and_test(df, sf, res, label):
    """把 demo 回合对齐到墙钟，测 Q1/Q2/Q3。"""
    last_end = df.end_tick.max()
    rows = []
    for rr in df.itertuples(index=False):
        t_end = res - (last_end - rr.end_tick) / TICK_RATE
        t_frz = t_end - rr.dur_s
        p_frz = _nearest_mid(sf, t_frz)
        p_end = _nearest_mid(sf, t_end)
        dp = p_end - p_frz  # Falcons token 在回合内涨跌
        rows.append({
            "round": rr.round, "falcons_win": rr.falcons_win, "equip_gap_f": rr.equip_gap_f,
            "score_lead_f": rr.score_f - rr.score_g, "t_frz": t_frz, "t_end": t_end,
            "p_frz": p_frz, "p_end": p_end, "dp": dp, "dur_s": rr.dur_s,
        })
    r = pd.DataFrame(rows)

    def pnl(bet_falcons):
        """bet_falcons=1 买 Falcons token，0 买 G2 token。P&L = 方向校正的 Falcons 价差。"""
        return np.where(bet_falcons == 1, r.dp, -r.dp)

    n = len(r)
    oracle = pnl(r.falcons_win).mean()
    signal = pnl((r.equip_gap_f > 0).astype(int)).mean()
    leader = pnl((r.score_lead_f > 0).astype(int)).mean()
    rand = pnl(np.random.RandomState(0).randint(0, 2, n)).mean()
    corr = r.equip_gap_f.corr(r.dp)

    print(f"\n=== {label} ===  n_rounds={n}  resolution={pd.to_datetime(res, unit='s')}")
    print(f"  平均回合时长 {r.dur_s.mean():.0f}s  |  Falcons token 回合内平均|Δp|={r.dp.abs().mean():.4f}")
    print(f"  Q1 oracle (预知胜者):   mean P&L = {oracle:+.4f}   (趋势存在吗)")
    print(f"  Q2 signal(经济equip):   mean P&L = {signal:+.4f}   corr(equip_gap, Δp)={corr:+.3f}")
    print(f"      leader(押比分领先): mean P&L = {leader:+.4f}")
    print(f"      random(随机方向):   mean P&L = {rand:+.4f}   (bid-ask 摩擦成本)")
    # Q3 latency：经济可观察（freeze）比回合结束（价格响应）早多久
    print(f"  Q3 latency: 经济在 freeze 可观察，价格在 round 末响应 → 领先 ~{r.dur_s.mean():.0f}s")
    # 信号命中率
    hit = ((r.equip_gap_f > 0) == (r.falcons_win == 1)).mean()
    print(f"  signal 方向命中率(equip_gap 符号 == 实际胜者): {hit:.2f}")
    return r


def _split_after_timeout(df, thr_s=200.0):
    """按回合间 demo 时间大间隙(技术暂停)切分，返回 (前段, 后段) 的 index 掩码。"""
    gap = df.end_tick.diff()
    big = gap[gap / TICK_RATE > thr_s]
    if not len(big):
        return None
    split_tick = big.index[0]  # 第一个大间隙之后的 round 属于「后段」
    return split_tick


def main():
    price_file = _latest_price_file()
    print(f"price file: {price_file.name}")
    df1 = parse_demo(M1)
    df2 = parse_demo(M2)
    price = load_price(price_file)

    for df, mk, label in [(df1, "Map 1 Winner", "inferno"),
                          (df2, "Map 2 Winner", "dust2")]:
        sf = mid_series(price, mk, "Team Falcons")
        res = resolution_time(sf)
        if res is None:
            print(f"{label}: no resolution anchor")
            continue
        align_and_test(df, sf, res, label)
        split = _split_after_timeout(df)
        if split is not None:
            tail = df[df.index >= split]
            head = df[df.index < split]
            print(f"  [robust] 检测到技术暂停: 前段 {len(head)} 回合 / 后段 {len(tail)} 回合")
            align_and_test(tail, sf, res, f"{label} (post-pause only)")
        # 比分持平时切片（#34 唯一 latency 线索）
        tied = df[df.score_f == df.score_g]
        if len(tied):
            print(f"  [tied] 比分持平回合数 = {len(tied)}")


if __name__ == "__main__":
    main()
