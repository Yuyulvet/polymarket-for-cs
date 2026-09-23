"""Bracket-strategy backtest on the historical inplay price library.

User's extreme mechanical strategy (2026-09-22):
1. 5-10 min pre-match: pick the stronger team expected to win the pistol -> buy
2. monitor market: if price rockets past +20% take-profit within ~120s -> sell
3. if price falls past -20% stop -> sell
4. TP/SL mechanically fixed at 20% (relative to entry)

Data: data/polymarket_inplay_raw.parquet (minute-level, 278 matches) joined
with the demo pool's Map-1 pistol outcomes (241 overlapping matches).

Selection arms:
  ALL: buy the market favorite (higher entry price) on every map
  ORACLE_PISTOL: buy only when the favorite actually won the pistol ->
      upper bound of what a perfect pistol read could earn
Reality sits between ALL and ORACLE, scaled by the measured 53.4% pistol
hit-rate ceiling (no model/feature beat it; see cs2-csnet-evaluation).

Caveats: minute bars miss intra-minute band touches (slight lag bias); p is
a single price feed (no book -> no spread cost); entry at first available
price within 10 min pre-match (user's 5-10 min window).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PX = ROOT / "data/polymarket_inplay_raw.parquet"
LABELS = ROOT / "data/map1/rebuild/full-20260916/map_labels.parquet"
ROUND_STATES = ROOT / "data/map1/rebuild/full-20260916/round_states.parquet"
OUT = ROOT / "reports/bracket_backtest_20260922"

TP_MULT, SL_MULT = 1.20, 0.80


def norm_key(path: str) -> str:
    tail = re.sub(r"\.(dem|json\.gz|parquet)$", "", str(path).lower())
    i = tail.rfind("demos")
    if i >= 0:
        tail = tail[i + len("demos"):]
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def pistol_labels() -> pd.DataFrame:
    lab = pd.read_parquet(LABELS, columns=["demo_path", "match_id", "complete"])
    lab = lab[lab["complete"]].copy()
    lab["key"] = lab["demo_path"].map(norm_key)
    m1 = lab[lab["demo_path"].str.contains(r"-m1-", regex=True)]
    rs = pd.read_parquet(ROUND_STATES,
                         columns=["demo_path", "round_num", "ct_is_a",
                                  "winner_side"])
    rs = rs[rs["round_num"] == 1].drop_duplicates(subset=["demo_path"])
    rs["a_won_pistol"] = ((rs["winner_side"] == "CT") == rs["ct_is_a"]).astype(int)
    m1 = m1.merge(rs[["demo_path", "a_won_pistol"]], on="demo_path", how="left")
    return m1[["match_id", "a_won_pistol"]].dropna().drop_duplicates("match_id")


def run_bracket(ts: pd.Series, entry: float):
    """First minute close outside the bands; else last close."""
    tp, sl = TP_MULT * entry, SL_MULT * entry
    for t, p in ts.items():
        if p >= tp:
            return p, "TP"
        if p <= sl:
            return p, "SL"
    return float(ts.iloc[-1]), "END"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(PX, columns=["t", "p", "match_id", "map_market",
                                      "outcome"])
    px = px[px["map_market"] == "Map 1 Winner"]
    labels = pistol_labels()
    rows = []
    for mid, g in px.groupby("match_id"):
        if mid not in set(labels.match_id):
            continue
        wide = g.pivot_table(index="t", columns="outcome", values="p",
                             aggfunc="last")
        if wide.shape[1] != 2 or len(wide) < 30:
            continue
        t0 = wide.index.min()
        pre = wide[wide.index <= t0 + pd.Timedelta(minutes=10)]
        if not len(pre):
            continue
        entry_a = float(pre.iloc[:, 0].dropna().iloc[0])
        entry_b = float(pre.iloc[:, 1].dropna().iloc[0])
        team_a, team_b = wide.columns[0], wide.columns[1]
        fav = team_a if entry_a > entry_b else team_b
        entry = max(entry_a, entry_b)
        series = wide[fav].dropna()
        # skip the entry minute itself when scanning for band touches
        series = series[series.index > series.index[0]]
        exit_p, reason = run_bracket(series, entry)
        rows.append({
            "match_id": mid, "favorite": fav, "other": team_b if fav == team_a else team_a,
            "entry": entry, "exit": exit_p, "reason": reason,
            "pnl": exit_p - entry,
            "pistol_won": bool(labels.set_index("match_id").loc[mid,
                                                              "a_won_pistol"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trades.csv", index=False)
    print(f"maps backtested: {len(df)}")

    summary = {}
    for tag, g in [("ALL_favorite", df),
                   ("ORACLE_pistol_only", df[df.pistol_won])]:
        s = g["pnl"]
        summary[tag] = {
            "n": int(len(g)),
            "pistol_hit_rate": round(float(g.pistol_won.mean()), 3),
            "pnl_mean": round(float(s.mean()), 4),
            "pnl_median": round(float(s.median()), 4),
            "pnl_total": round(float(s.sum()), 2),
            "win_rate": round(float((s > 0).mean()), 3),
            "exit_mix": g["reason"].value_counts().to_dict(),
            "TP_pnl_mean": round(float(g[g.reason == "TP"].pnl.mean()), 4)
                           if (g.reason == "TP").any() else None,
            "SL_pnl_mean": round(float(g[g.reason == "SL"].pnl.mean()), 4)
                           if (g.reason == "SL").any() else None,
            "END_pnl_mean": round(float(g[g.reason == "END"].pnl.mean()), 4)
                            if (g.reason == "END").any() else None,
        }
    # realistic: read accuracy 53.4% -> mixture of oracle and anti-oracle legs
    allf, orc = df, df[df.pistol_won]
    anti = df[~df.pistol_won]
    summary["REALISTIC_53.4pct_read"] = {
        "note": "pnl if pistol read hits 53.4% (measured ceiling): "
                "EV = 0.534*ORACLE + 0.466*ANTI",
        "anti_n": int(len(anti)),
        "anti_pnl_mean": round(float(anti.pnl.mean()), 4),
        "ev_mean": round(0.534 * float(orc.pnl.mean())
                         + 0.466 * float(anti.pnl.mean()), 4),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
