"""Cross-token arb scan — 同市场两 token 报价和偏离 1.0 的频率/深度/持续性.

结构: Map Winner 等二元市场的两个 token 兑付和恒为 $1. ask_A + ask_B < 1
→ 双边买入锁利; bid_A + bid_B > 1 → 双边卖出(铸币)锁利. 不预测回合,
只抓报价错误 —— 与 latency 约束兼容的家族.

数据源: live_sessions 的 raw book (含档位量). 口径:
  - 每次任一 token 的 book 更新, 重算 ask_sum/bid_sum (另一个 token 前向填充);
  - 违规 = sum 偏离 ≥1pt (tick 0.01);
  - 即时利润 = |1 − sum| × min(两侧盘口量);
  - 持续性 = 违规在多少后续 book 更新后仍存活 (人速 ~秒级可执行性的代理).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SESS = ROOT / "data/live_sessions"
OUT = ROOT / "reports/arb_scan_20260923"
EPS = 0.01          # 1pt 以上才算
N_PERSIST = 5       # 报告「存活≥N 次更新」的深度违规数


def books_by_market(sess: Path):
    """{market_id: {asset: DataFrame[t_recv, bid, bid_sz, ask, ask_sz]}}"""
    per_asset: dict[str, list] = {}
    with open(sess / "market/events.jsonl", encoding="utf-8") as h:
        for line in h:
            row = json.loads(line)
            if row.get("type") != "market_raw":
                continue
            recv = pd.Timestamp(row["received_at"])
            try:
                payload = json.loads(row.get("raw") or "null")
            except json.JSONDecodeError:
                continue
            for msg in (payload if isinstance(payload, list) else [payload]):
                if not isinstance(msg, dict) or "bids" not in msg:
                    continue
                bids = msg.get("bids") or []
                asks = msg.get("asks") or []
                bid = max((float(b["price"]), float(b["size"])) for b in bids) \
                    if bids else (np.nan, 0.0)
                ask = min((float(a["price"]), float(a["size"])) for a in asks) \
                    if asks else (np.nan, 0.0)
                per_asset.setdefault(str(msg["asset_id"]), []).append(
                    (recv, str(msg["market"]), bid[0], bid[1], ask[0], ask[1]))
    out = {}
    for asset, rows in per_asset.items():
        df = pd.DataFrame(rows, columns=["t", "market", "bid", "bid_sz",
                                         "ask", "ask_sz"]).drop_duplicates("t")
        out.setdefault(df["market"].iloc[0], {})[asset] = \
            df.drop(columns="market").sort_values("t").reset_index(drop=True)
    return out


def scan_market(a1: pd.DataFrame, a2: pd.DataFrame) -> dict:
    """两 token 的 best-book 合并扫描."""
    cols = ["t", "bid", "bid_sz", "ask", "ask_sz"]
    l = a1[cols].rename(columns={c: f"{c}_x" for c in cols if c != "t"})
    r = a2[cols].rename(columns={c: f"{c}_y" for c in cols if c != "t"})
    m = pd.merge(l, r, on="t", how="outer").sort_values("t")
    for c in cols[1:]:
        m[f"{c}_x"] = m[f"{c}_x"].ffill()
        m[f"{c}_y"] = m[f"{c}_y"].ffill()
    m = m.dropna(subset=["ask_x", "ask_y"])
    m["ask_sum"] = m["ask_x"] + m["ask_y"]
    m["bid_sum"] = m["bid_x"] + m["bid_y"]
    ask_v = m["ask_sum"] <= 1 - EPS
    bid_v = m["bid_sum"] >= 1 + EPS
    m["ask_profit"] = np.where(ask_v,
                               (1 - m["ask_sum"])
                               * np.minimum(m["ask_sz_x"], m["ask_sz_y"]), 0.0)
    m["bid_profit"] = np.where(bid_v,
                               (m["bid_sum"] - 1)
                               * np.minimum(m["bid_sz_x"], m["bid_sz_y"]), 0.0)
    m["viol"] = ask_v | bid_v
    dead = (m["ask_sum"] < 0.10) | (m["ask_sum"] > 1.90)   # 已决/死市场
    m.loc[dead, "viol"] = False

    def persist(mask: pd.Series) -> pd.Series:
        """每次违规后续连续违规的更新数 (含自身)."""
        n = np.zeros(len(mask), dtype=int)
        run = 0
        vals = mask.values
        for i in range(len(vals) - 1, -1, -1):
            run = run + 1 if vals[i] else 0
            n[i] = run
        return pd.Series(n, index=mask.index)

    m["persist"] = persist(m["viol"])
    v = m[m["viol"]]
    return {
        "n_updates": int(len(m)),
        "n_viol": int(len(v)),
        "n_viol_persist5": int((v["persist"] >= N_PERSIST).sum()),
        "profit_sum": float(v["ask_profit"].sum() + v["bid_profit"].sum()),
        "profit_max": float((v["ask_profit"] + v["bid_profit"]).max()
                            if len(v) else 0.0),
        "gap_max": float((v["ask_sum"].rsub(1).clip(lower=None)
                          .abs().max()) if len(v) else 0.0),
    }, m


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, details = [], {}
    for sess in sorted(SESS.iterdir()):
        ev = sess / "market/events.jsonl"
        if not ev.exists():
            continue
        markets = books_by_market(sess)
        for mid, assets in markets.items():
            if len(assets) != 2:          # 只扫二元市场
                continue
            (a1, a2) = assets.values()
            if min(len(a1), len(a2)) < 10:
                continue
            stats, m = scan_market(a1, a2)
            stats.update({"session": sess.name, "market": mid[:10],
                          "minutes": round((m["t"].max() - m["t"].min())
                                           .total_seconds() / 60, 1)})
            all_rows.append(stats)
            details[f"{sess.name}|{mid[:10]}"] = m
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "per_market.csv", index=False)
    if not len(df):
        print("no binary markets scanned")
        return 0
    tot_v = df["n_viol"].sum()
    tot_up = df["n_updates"].sum()
    print(f"markets scanned: {len(df)}  total book updates: {tot_up}")
    print(f"violations (>=1pt deviation): {tot_v} "
          f"(rate {tot_v / max(tot_up, 1):.2%})")
    print(f"persistent (>= {N_PERSIST} updates): {df['n_viol_persist5'].sum()}")
    print(f"sum of instant profits: ${df['profit_sum'].sum():.2f}")
    print(f"largest single gap: {df['gap_max'].max():.3f}")
    print("\nper-market (violations desc):")
    print(df.sort_values("n_viol", ascending=False).head(15).to_string(
        index=False))

    # ---- episode-level actionable analysis (去重 + 第二腿 1-tick 滑点)
    eps = []
    for key, m in details.items():
        v = m[m["viol"]].copy()
        if not len(v):
            continue
        grp = (v.index.to_series().diff() != 1).cumsum()
        for _, e in v.groupby(grp):
            side = "ask" if e["ask_profit"].iloc[0] > 0 else "bid"
            gap = float((1 - e["ask_sum"]).abs().max()) if side == "ask" \
                else float((e["bid_sum"] - 1).abs().max())
            depth = float(np.minimum(
                e["ask_sz_x"] if side == "ask" else e["bid_sz_x"],
                e["ask_sz_y"] if side == "ask" else e["bid_sz_y"]).min())
            slip = gap - 0.01            # 第二腿滑 1 tick 后的剩余
            eps.append({
                "session": key.split("|")[0], "market": key.split("|")[1],
                "side": side, "n_updates": len(e),
                "gap_max": round(gap, 3), "depth_min": depth,
                "gross": round(gap * depth, 2),
                "net_after_slip": round(max(slip, 0) * depth, 2),
                "actionable": len(e) >= N_PERSIST and depth >= 50
                and slip > 0,
            })
    edf = pd.DataFrame(eps)
    edf.to_csv(OUT / "episodes.csv", index=False)
    if len(edf):
        act = edf[edf["actionable"]]
        print(f"\nepisodes (去重后): {len(edf)}, actionable "
              f"(persist>={N_PERSIST} & depth>=50 & 滑点后仍正): {len(act)}")
        print(f"actionable 毛利: ${act['gross'].sum():.2f} "
              f"净利: ${act['net_after_slip'].sum():.2f}")
        print("\nactionable episodes:")
        print(act.sort_values("net_after_slip", ascending=False).head(20)
              .to_string(index=False))
    # 存一个违规明细最重的 market 供人查
    if tot_v:
        worst = df.sort_values("n_viol", ascending=False).index[0]
        key = df.loc[worst]
        m = details[f"{key['session']}|{key['market']}"]
        m[m["viol"]].to_csv(OUT / "worst_market_violations.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
