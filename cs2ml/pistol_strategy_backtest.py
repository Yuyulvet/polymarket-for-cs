"""Pistol-round swing strategy backtest (user strategy, 2026-09-22).

Strategy under test:
  v1 (positive direction): model predicts the pistol winner with >=55%
      confidence -> buy that team pre-pistol -> sell the moment the pistol
      ends (eat the sentiment pop; always closed before round 3).
  v3 (user's conditional version): buy the stronger team. If it wins the
      pistol -> sell at pistol end (same as v1). If it LOSES the pistol:
      hold to round-3 end waiting for the economic-round rebound when the
      map-level model still says this team wins the map (p_mapwin >= 0.55);
      exit immediately when the map read is bearish (p_mapwin < 0.55).

Predictions come from pistol_model_v3 (v3_full / control_v2best), both
in-sample (optimistic) and expanding-window OOF (honest). The "stronger
team" is a composite of recent big-event results and player form (all_*
gaps from map1_data, 90-day half-life) plus map-specific strength
(map_* gaps), per the user's definition; market-favorite is reported as a
sensitivity reference only.

CLOCK DISCIPLINE: demos.start_date is the SCHEDULED series time; per-match
offsets to the real Map-1 pistol run from -4 to +45 minutes, so the demo
clock cannot locate the pistol. Two market-derived fixes per match:
  1. side mapping: filename order does NOT reliably encode roster_a/b
     (measured: market-final-winner vs label-implied slug agrees only 50%).
     The match's resolved market outcome (last price >= 0.9) is therefore
     used to bind team names to roster sides; filename slugs are fallback
     only. After this fix, round-1 jump direction agrees with labels 78%.
  2. per-match clock delta: constant offset in [-10, +45] min maximizing
     the number of demo round-ends (rounds 1-10) landing within 45s of a
     >=2pt mirror price jump. t_pistol_market = demo R1 end + delta.
P&L must not be interpreted if jump_side_match collapses toward 50%.

PRE-REGISTERED BENCHMARKS (measured earlier, must not be re-litigated here):
  - stronger-team pistol hit-rate ceiling: 53.4% (n=586, time-stable)
  - tape legs: win pop +6.25pt / loss drop -11.2pt -> breakeven hit 64.2%
  - v3 OOF AUC ~0.48; >=60% subset hits 42-43%
A 55% trigger line has no data support; this backtest runs it anyway and
reports against the benchmarks.

Data: minute-level prices (no book -> mid-price fills, no spread), so the
cost03/fee columns are sensitivity only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# pistol_model_v3 uses bare imports (`from pistol_model_v2 import ...`),
# so its directory must be on sys.path when imported as a top-level module.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pistol_model_v3 as pm3  # noqa: E402

from .inplay_data import build_round_timeline  # noqa: E402
from .map1_data import build_features, load_history  # noqa: E402
from .map1_model import walk_forward  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PX = ROOT / "data/polymarket_inplay_raw.parquet"
PM = ROOT / "data/prematch/prices_full_prematch.parquet"
LABELS = ROOT / "data/map1/rebuild/full-20260916/map_labels.parquet"
ROUND_STATES = ROOT / "data/map1/rebuild/full-20260916/round_states.parquet"
OOF_DIR = ROOT / "reports/pistol_model_v3"
OUT = ROOT / "reports/pistol_strategy_backtest_20260922"

BENCH = {"ceiling_hit": 0.534, "breakeven_hit": 0.6418,
         "tape_win_pop_pt": 6.25, "tape_loss_drop_pt": -11.2,
         "replay_ev_at_ceiling_pt": -1.9, "r2_rebound_rate": 0.444}

COMP_WEIGHTS = {"all_win_gap": 0.30, "all_kd_gap": 0.25,
                "all_round_gap": 0.20, "map_win_gap": 0.15,
                "map_pistol_gap": 0.10}

LAG_GRID = (-2, -1, 0, 1, 2)
CONF_GRID = (0.50, 0.55, 0.60)
JUMP_THRESH = 0.03


# ---------------------------------------------------------------- helpers
def norm_key(path: str) -> str:
    tail = re.sub(r"\.(dem|json\.gz|parquet)$", "", str(path).lower())
    i = tail.rfind("demos")
    if i >= 0:
        tail = tail[i + len("demos"):]
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def team_slugs(demo_path: str) -> tuple[str, str] | None:
    base = Path(str(demo_path)).stem.lower()
    m = re.match(r"([a-z0-9.-]+?)-vs-([a-z0-9.-]+?)-m\d", base)
    return (m.group(1), m.group(2)) if m else None


def resolve_outcome(outcome: str, slug_a: str, slug_b: str) -> str | None:
    n = re.sub(r"[^a-z0-9]+", "",
               re.sub(r"\b(gaming|team|esports|clan)\b", " ",
                      str(outcome).lower()))
    hit = None
    for side, slug in (("a", slug_a), ("b", slug_b)):
        s = re.sub(r"[^a-z0-9]+", "", str(slug).lower())
        if n and s and (n == s or n.startswith(s) or s.startswith(n)):
            if hit and hit != side:
                return None
            hit = side
    return hit


def attach_round1(oof: pd.DataFrame) -> pd.DataFrame:
    """Keep only the round-1 (first pistol) row per map, keyed reliably."""
    rs = pd.read_parquet(ROUND_STATES, columns=["demo_path", "round_num", "ct_is_a"])
    rs["key"] = rs["demo_path"].map(norm_key)
    rs = rs[rs["round_num"] == 1].drop_duplicates(subset=["key"])
    d = oof.merge(rs[["key", "ct_is_a"]], on="key", how="left")
    d = d[d["is_ct"] == d["ct_is_a"]].copy()
    return d.dropna(subset=["ct_is_a"])


def load_labels() -> pd.DataFrame:
    lab = pd.read_parquet(LABELS, columns=["demo_path", "match_id", "map_name",
                                           "roster_a", "roster_b", "complete"])
    lab = lab[lab["complete"]].copy()
    lab = lab[lab["demo_path"].str.contains(r"-m1-", regex=True)].copy()
    lab["key"] = lab["demo_path"].map(norm_key)
    return lab.drop_duplicates(subset=["key"])


def load_round_extras() -> pd.DataFrame:
    """Per key: round-1/2 winners as roster-a perspective (for labels/rebound)."""
    rs = pd.read_parquet(ROUND_STATES, columns=["demo_path", "round_num",
                                                "ct_is_a", "winner_side"])
    rs["a_won"] = ((rs["winner_side"] == "CT") == rs["ct_is_a"]).astype(int)
    piv = rs[rs["round_num"].isin([1, 2])].pivot_table(
        index="demo_path", columns="round_num", values="a_won", aggfunc="first")
    out = piv.reset_index().drop(columns=["demo_path"])
    out["key"] = piv.reset_index()["demo_path"].map(norm_key)
    out.columns = [f"a_won_r{int(c)}" if isinstance(c, int) else c
                   for c in out.columns]
    return out


def first_after(series: pd.Series, t: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    s = series[series.index >= t]
    if s.empty:
        return None
    return s.index[0], float(s.iloc[0])


def plateau(series: pd.Series, t: pd.Timestamp,
            window: pd.Timedelta = pd.Timedelta(minutes=3)) -> tuple[pd.Timestamp, float] | None:
    s = series[(series.index >= t) & (series.index <= t + window)]
    if s.empty:
        return first_after(series, t)
    return t, float(s.median())


def fee(p_in: float, p_out: float) -> float:
    return 0.05 * p_in * (1 - p_in) + 0.05 * p_out * (1 - p_out)


def mirror_jumps(wide: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp,
                 thresh: float = 0.02):
    """Minutes where both tokens moved oppositely by >= thresh (resolution)."""
    out = []
    for t in sorted(set(wide.index)):
        if t < lo or t > hi:
            continue
        rises = {}
        for o in wide.columns:
            s = wide[o].dropna()
            pre = s[s.index <= t - pd.Timedelta(seconds=90)]
            post = s[s.index >= t]
            rises[o] = (post.iloc[0] - pre.iloc[-1]) if len(pre) and len(post) else np.nan
        vals = list(rises.values())
        if all(pd.notna(v) for v in vals) and min(abs(v) for v in vals) >= thresh \
                and vals[0] * vals[1] < 0:
            out.append((t, max(rises, key=lambda k: rises[k])))
    return out


def estimate_delta(wide: pd.DataFrame, round_ends: list[pd.Timestamp],
                   t1_end: pd.Timestamp) -> int | None:
    """Per-match constant clock offset (minutes) maximizing round-end/jump overlap."""
    jumps = mirror_jumps(wide, t1_end - pd.Timedelta(minutes=15),
                         t1_end + pd.Timedelta(minutes=100))
    if len(jumps) < 3:
        return None
    jt = np.array([j[0].value for j in jumps])
    e_arr = np.array([t.value for t in round_ends], dtype=np.int64)
    best = None
    for dm in range(-10, 46):
        shifted = e_arr + int(dm * 60 * 1e9)
        score = 0
        for se in shifted:
            if np.any(np.abs(jt - se) <= 45 * 1e9):
                score += 1
        if best is None or score > best[0]:
            best = (score, dm)
    return best[1]


def build_side_mapping(labels: pd.DataFrame, px_wide: dict) -> pd.DataFrame:
    """Bind market outcome names to roster sides via the RESOLVED final price.

    Filename order does not reliably encode roster_a/b (market-final-winner
    vs label-implied slug agrees only 50%); the settled outcome does.
    """
    scores = pd.read_parquet(LABELS, columns=["match_id", "score_a", "score_b"])
    scores = scores.drop_duplicates("match_id").set_index("match_id")
    rows = []
    for r in labels.itertuples(index=False):
        mid = r.match_id
        if mid not in px_wide or mid not in scores.index:
            continue
        wide = px_wide[mid]
        last = {o: wide[o].dropna().iloc[-1] for o in wide.columns}
        w = max(last, key=lambda k: last[k])
        if last[w] < 0.9:
            continue
        l = [o for o in wide.columns if o != w][0]
        y = int(scores.loc[mid, "score_a"] > scores.loc[mid, "score_b"])
        rows.append({"match_id": mid,
                     "outcome_a": w if y == 1 else l,
                     "outcome_b": l if y == 1 else w,
                     "mapping_src": "market_resolution"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- predictions
def predictions_oof(variant: str) -> pd.DataFrame:
    tag = "v3" if variant == "v3_full" else "control"
    f = OOF_DIR / f"oof_{tag}.csv"
    if not f.exists():
        df = pm3.build_rows()
        pm3.run_oof(df, pm3.VARIANTS[variant])
    oof = pd.read_csv(f)
    return attach_round1(oof)


def predictions_insample(variant: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Phase-1 protocol, Platt applied to ALL round-1 rows (optimistic)."""
    import sklearn.linear_model as lm
    from sklearn.preprocessing import StandardScaler

    d = attach_round1(rows)
    cols = pm3.VARIANTS[variant]
    d = d.dropna(subset=cols + ["y"]).sort_values("mtime").reset_index(drop=True)
    cutoff = d.mtime.quantile(0.7)
    train = d[d.mtime <= cutoff]
    cal_line = train.mtime.quantile(0.8)
    fit_part = train[train.mtime <= cal_line]
    cal_part = train[train.mtime > cal_line]
    if len(cal_part) < 100:
        fit_part, cal_part = train, train
    sc = StandardScaler().fit(fit_part[cols])
    clf = lm.LogisticRegression(max_iter=2000).fit(
        sc.transform(fit_part[cols]), fit_part.y)
    pl = pm3.platt_fit(clf, sc, cal_part[cols], cal_part.y)
    d["p_cal"] = pm3.platt_apply(pl, clf, sc, d[cols])
    return d


# ---------------------------------------------------------------- strength / map model
def strength_and_mapwin() -> pd.DataFrame:
    """Composite strong side (user definition) + honest walk-forward p_mapwin."""
    history, _audit = load_history()
    feats = build_features(history)
    z = feats.copy()
    for c in COMP_WEIGHTS:
        sd = z[c].std()
        z[c] = (z[c] - z[c].mean()) / (sd if sd else 1.0)
    feats["composite"] = sum(z[c] * w for c, w in COMP_WEIGHTS.items())
    print(f"map features: {len(feats)} Map-1 rows; fitting walk_forward map model ...")
    wf = walk_forward(feats, min_train=60)
    keep = feats[["match_id", "roster_a", "roster_b", "start_at"]].merge(
        wf[["match_id", "p_model", "n_train", "reason"]], on="match_id", how="left")
    keep = keep.merge(feats[["match_id"] + list(COMP_WEIGHTS) + ["composite"]],
                      on="match_id", how="left")
    return keep


# ---------------------------------------------------------------- backtest
def backtest(pred: pd.DataFrame, labels: pd.DataFrame, extras: pd.DataFrame,
             strength: pd.DataFrame, timeline: pd.DataFrame,
             px_wide: dict, px_tokens: dict, pm_series: dict,
             side_mapping: pd.DataFrame,
             conf: float) -> tuple[pd.DataFrame, dict]:
    lab = labels[["key", "match_id", "map_name", "demo_path"]].merge(
        extras, on="key", how="left").merge(
        strength.drop(columns=["map_name"], errors="ignore"), on="match_id", how="left")
    d = pred.merge(lab, on="key", how="inner")

    tl = timeline.copy()
    tl["key"] = tl["demo_path"].map(norm_key)
    r1 = tl[tl["round_num"] == 1][["key", "round_start_utc", "round_end_utc"]]
    r3 = tl[tl["round_num"] == 3][["key", "round_end_utc"]].rename(
        columns={"round_end_utc": "r3_end_utc"})
    d = d.merge(r1, on="key", how="left").merge(r3, on="key", how="left")

    d["strong_side"] = np.where(d["composite"] > 0, "a", "b")
    d["strong_side"] = d["strong_side"].where(d["composite"].notna())
    d["pred_side"] = np.where(d["p_cal"] >= conf, "a",
                              np.where(d["p_cal"] <= 1 - conf, "b", None))
    d["relaxed"] = d["p_cal"].notna() & (np.maximum(d["p_cal"], 1 - d["p_cal"]) >= conf)
    d["strict"] = d["relaxed"] & (d["pred_side"] == d["strong_side"])

    rows, skips = [], {}
    map_side = side_mapping.set_index("match_id").to_dict("index") if len(side_mapping) else {}
    deltas, diag_side_match, diag_side_n = {}, 0, 0
    for r in d.itertuples(index=False):
        mid = r.match_id
        if mid not in px_wide:
            skips["no_px_prices"] = skips.get("no_px_prices", 0) + 1
            continue
        if pd.isna(r.round_end_utc):
            skips["no_timeline"] = skips.get("no_timeline", 0) + 1
            continue
        wide = px_wide[mid]

        # side -> outcome: market resolution mapping first, slug fallback
        if mid in map_side:
            side_of = {"a": map_side[mid]["outcome_a"],
                       "b": map_side[mid]["outcome_b"]}
        else:
            slugs = team_slugs(r.demo_path)
            side_of = {}
            if slugs:
                for outcome in wide.columns:
                    s = resolve_outcome(outcome, *slugs)
                    if s:
                        side_of[s] = outcome
            if set(side_of) != {"a", "b"}:
                skips["team_unresolved"] = skips.get("team_unresolved", 0) + 1
                continue

        # per-match clock delta from round-end/jump correlation
        if mid not in deltas:
            sub_tl = tl[tl["key"] == r.key].sort_values("round_num")
            ends = sub_tl[sub_tl["round_num"] <= 10]["round_end_utc"].tolist()
            deltas[mid] = estimate_delta(wide, ends, r.round_end_utc) if len(ends) >= 3 else None
        dm = deltas[mid]
        if dm is None:
            skips["no_delta"] = skips.get("no_delta", 0) + 1
            continue
        t_pistol = r.round_end_utc + pd.Timedelta(minutes=dm)

        # diagnostic: nearest mirror jump to aligned pistol end vs label side
        if pd.notna(r.a_won_r1):
            jumps = mirror_jumps(wide, t_pistol - pd.Timedelta(minutes=2),
                                 t_pistol + pd.Timedelta(minutes=2))
            if jumps:
                diag_side_n += 1
                t_j, jump_out = min(jumps, key=lambda x: abs((x[0] - t_pistol).total_seconds()))
                winner_side = "a" if r.a_won_r1 == 1 else "b"
                if side_of.get(winner_side) == jump_out:
                    diag_side_match += 1

        # duration from pistol end to round-3 end on the demo clock
        hold_delta = None
        if pd.notna(r.r3_end_utc):
            hold_delta = r.r3_end_utc - r.round_end_utc

        for arm_trigger in ("strict", "relaxed"):
            if not getattr(r, arm_trigger):
                continue
            bought = r.pred_side if arm_trigger == "relaxed" else r.strong_side
            outcome = side_of[bought]
            token = px_tokens[mid][outcome]
            series = wide[outcome].dropna()

            # entry: last quiet prematch price at least 3 min before the
            # market-time pistol jump; else first inplay price fallback
            entry = None
            cutoff = t_pistol - pd.Timedelta(minutes=3)
            pm_s = pm_series.get(token)
            if pm_s is not None:
                pre = pm_s[pm_s.index <= cutoff]
                if len(pre):
                    entry = (pre.index[-1], float(pre.iloc[-1]), "prematch")
            if entry is None:
                early = series[series.index <= cutoff]
                if len(early):
                    entry = (early.index[0], float(early.iloc[0]), "inplay_early")
            if entry is None:
                skips[f"no_entry_{arm_trigger}"] = skips.get(f"no_entry_{arm_trigger}", 0) + 1
                continue
            if pd.isna(r.a_won_r1):
                skips[f"no_pistol_label_{arm_trigger}"] = skips.get(f"no_pistol_label_{arm_trigger}", 0) + 1
                continue
            entry_t, entry_p, entry_src = entry

            pistol_won = bool(r.a_won_r1) if bought == "a" else not bool(r.a_won_r1)
            p_mapwin_side = (float(r.p_model) if bought == "a"
                             else 1 - float(r.p_model)) if pd.notna(r.p_model) else np.nan

            ex1 = first_after(series, t_pistol)
            pl1 = plateau(series, t_pistol)
            arm_v3, ex3, ex3_mode = None, None, None
            if pistol_won:
                arm_v3, ex3, ex3_mode = "v3_win_pop", ex1, "pistol_end"
            elif pd.notna(p_mapwin_side) and p_mapwin_side >= 0.55:
                if hold_delta is None:
                    skips[f"no_r3_{arm_trigger}"] = skips.get(f"no_r3_{arm_trigger}", 0) + 1
                    continue
                arm_v3 = "v3_hold"
                ex3 = first_after(series, t_pistol + hold_delta)
                ex3_mode = "r3_end"
            else:
                arm_v3, ex3, ex3_mode = "v3_quick", ex1, "pistol_end"
            if ex1 is None or ex3 is None:
                skips[f"no_exit_{arm_trigger}"] = skips.get(f"no_exit_{arm_trigger}", 0) + 1
                continue
            rebound = None
            if arm_v3 == "v3_hold" and pd.notna(r.a_won_r2):
                rebound = bool(r.a_won_r2) if bought == "a" else not bool(r.a_won_r2)

            rec = {
                "match_id": mid, "key": r.key, "map_name": r.map_name,
                "arm_trigger": arm_trigger, "team_bought": outcome,
                "side_bought": bought, "entry_src": entry_src,
                "entry_t": entry_t, "entry_p": entry_p,
                "p_cal": float(r.p_cal), "composite": r.composite,
                "strong_side": r.strong_side, "pred_side": r.pred_side,
                "pistol_won": pistol_won, "p_mapwin_bought": p_mapwin_side,
                "t_pistol_market": t_pistol, "clock_delta_min": dm,
                "exit_t_v1": ex1[0], "exit_p_v1": ex1[1],
                "pnl_v1_raw": ex1[1] - entry_p,
                "exit_p_v1_plateau": pl1[1] if pl1 else np.nan,
                "arm_v3": arm_v3, "exit_mode_v3": ex3_mode,
                "exit_t_v3": ex3[0], "exit_p_v3": ex3[1],
                "pnl_v3_raw": ex3[1] - entry_p,
                "rebound_r2_won": rebound,
                "lag_scan": {},
            }
            for lag in LAG_GRID:
                ex = first_after(series, t_pistol + pd.Timedelta(minutes=lag))
                if ex:
                    rec["lag_scan"][lag] = ex[1] - entry_p
            rec["pnl_v1_cost03"] = rec["pnl_v1_raw"] - 0.03
            rec["pnl_v1_fee"] = rec["pnl_v1_raw"] - fee(entry_p, ex1[1])
            rec["pnl_v3_cost03"] = rec["pnl_v3_raw"] - 0.03
            rec["pnl_v3_fee"] = rec["pnl_v3_raw"] - fee(entry_p, ex3[1])
            rows.append(rec)

    trades = pd.DataFrame(rows)
    delta_s = pd.Series([v for v in deltas.values() if v is not None])
    summary = {"universe": {
        "n_with_preds": int(len(d)),
        "n_px_overlap": int(sum(m in px_wide for m in d.match_id)),
        "n_triggered_strict": int(d["strict"].sum()),
        "n_triggered_relaxed": int(d["relaxed"].sum()),
        "n_side_mapping": int(len(map_side)),
        "skip_reasons": skips},
        "alignment_diag": {
            "clock_delta_min_median": round(float(delta_s.median()), 1) if len(delta_s) else None,
            "clock_delta_min_p10": round(float(delta_s.quantile(0.1)), 1) if len(delta_s) else None,
            "clock_delta_min_p90": round(float(delta_s.quantile(0.9)), 1) if len(delta_s) else None,
            "jump_side_match": f"{diag_side_match}/{diag_side_n}",
            "by_lag": {},
            "tape_reference": {"win_pop": BENCH["tape_win_pop_pt"],
                               "loss_drop": BENCH["tape_loss_drop_pt"]}}}

    if len(trades):
        rel = trades[trades.arm_trigger == "relaxed"]
        for lag in LAG_GRID:
            col = rel["lag_scan"].map(lambda s: s.get(lag))
            won = col[rel.pistol_won].dropna()
            lost = col[~rel.pistol_won].dropna()
            summary["alignment_diag"]["by_lag"][f"lag_{lag:+d}"] = {
                "n": int(col.dropna().shape[0]),
                "won_pop_mean": round(float(won.mean()), 3) if len(won) else None,
                "lost_pop_mean": round(float(lost.mean()), 3) if len(lost) else None,
            }
    summary["benchmarks"] = BENCH
    summary["arms"] = arm_summary(d, trades)
    return trades, summary


def _hit(g: pd.DataFrame) -> dict:
    if not len(g):
        return {"n": 0}
    k, n = int(g.pistol_won.sum()), len(g)
    p, lo, hi = pm3.wilson(k, n)
    return {"n": n, "hit": round(p, 4), "ci": [round(lo, 3), round(hi, 3)]}


def _pnl(g: pd.DataFrame, col: str) -> dict:
    s = g[col].dropna()
    if not len(s):
        return {"n": 0}
    return {"n": int(len(s)), "pnl_mean": round(float(s.mean()), 4),
            "pnl_median": round(float(s.median()), 4),
            "pnl_total": round(float(s.sum()), 3),
            "win_rate": round(float((s > 0).mean()), 3)}


def arm_summary(frame: pd.DataFrame, trades: pd.DataFrame) -> dict:
    out = {}
    if len(trades):
        for trig in ("strict", "relaxed"):
            g = trades[trades.arm_trigger == trig]
            out[f"v1_{trig}"] = {**_hit(g), **_pnl(g, "pnl_v1_raw"),
                                 "pnl_cost03": _pnl(g, "pnl_v1_cost03"),
                                 "pnl_fee": _pnl(g, "pnl_v1_fee")}
            v3 = g[g.arm_v3 != "v3_win_pop"]
            hold = g[g.arm_v3 == "v3_hold"]
            out[f"v3_{trig}"] = {
                "pnl": _pnl(v3, "pnl_v3_raw"),
                "hold_legs": _pnl(hold, "pnl_v3_raw"),
                "hold_n": int(len(hold)),
                "rebound_hit": ({**_hit(hold), "note": "R2 won during hold"}
                                if len(hold) else {"n": 0}),
            }
    sweep = {}
    ok = frame[frame.p_cal.notna()]
    for c in CONF_GRID:
        sub = ok[np.maximum(ok.p_cal, 1 - ok.p_cal) >= c]
        sweep[f"conf_{c:.2f}"] = {"n": int(len(sub)),
                                  "model_hit": round(float(sub.y.mean()), 4) if len(sub) else None}
    out["conf_sweep_model_hit"] = sweep
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["insample", "oof", "both"], default="both")
    ap.add_argument("--variant", choices=["v3_full", "control_v2best", "both"],
                    default="both")
    ap.add_argument("--conf", type=float, default=0.55)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    extras = load_round_extras()
    strength = strength_and_mapwin()
    timeline = build_round_timeline()

    px = pd.read_parquet(PX, columns=["t", "p", "match_id", "map_market",
                                      "outcome", "token"])
    px = px[px["map_market"] == "Map 1 Winner"]
    px_wide, px_tokens = {}, {}
    for mid, g in px.groupby("match_id"):
        wide = g.pivot_table(index="t", columns="outcome", values="p",
                             aggfunc="last")
        if wide.shape[1] == 2 and len(wide) >= 30:
            px_wide[mid] = wide
            px_tokens[mid] = (g.drop_duplicates("outcome")
                              .set_index("outcome")["token"].to_dict())

    pm = pd.read_parquet(PM, columns=["t", "p", "token"])
    pm_series = {tok: g.set_index("t")["p"].sort_index()
                 for tok, g in pm.groupby("token")}

    side_mapping = build_side_mapping(labels, px_wide)
    side_mapping.to_csv(OUT / "side_mapping.csv", index=False)
    print(f"side mapping (market resolution): {len(side_mapping)} matches")

    variants = (["v3_full", "control_v2best"] if args.variant == "both"
                else [args.variant])
    sources = (["insample", "oof"] if args.source == "both" else [args.source])

    insample_rows = None
    merged = {"universe_px_maps": len(px_wide), "conf": args.conf, "sources": {}}
    for variant in variants:
        if "insample" in sources:
            if insample_rows is None:
                print("[insample] building model rows (once) ...")
                insample_rows = pm3.build_rows()
            pred_in = predictions_insample(variant, insample_rows)
            trades, summary = backtest(
                pred_in, labels, extras, strength, timeline,
                px_wide, px_tokens, pm_series, side_mapping, args.conf)
            trades.to_csv(OUT / f"trades_insample_{variant}.csv", index=False)
            merged["sources"].setdefault("insample", {})[variant] = summary
            print(f"[insample/{variant}] {summary['universe']}")
            print(f"  align: {summary['alignment_diag']['jump_side_match']} side-match, "
                  f"delta median {summary['alignment_diag']['clock_delta_min_median']}min")
        if "oof" in sources:
            pred_oof = predictions_oof(variant)
            trades, summary = backtest(
                pred_oof, labels, extras, strength, timeline,
                px_wide, px_tokens, pm_series, side_mapping, args.conf)
            trades.to_csv(OUT / f"trades_oof_{variant}.csv", index=False)
            merged["sources"].setdefault("oof", {})[variant] = summary
            print(f"[oof/{variant}] {summary['universe']}")
            print(f"  align: {summary['alignment_diag']['jump_side_match']} side-match, "
                  f"delta median {summary['alignment_diag']['clock_delta_min_median']}min")

    (OUT / "summary.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (OUT / "README.md").write_text(render_readme(merged), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


def render_readme(s: dict) -> str:
    lines = [
        "# Pistol swing strategy backtest (2026-09-22)",
        "",
        "User strategy: model calls the pistol winner at >=55% confidence;",
        "v1 = sell the instant the pistol ends; v3 = if the strong team loses",
        "the pistol, hold to round-3 end when the map model still favors it",
        "(p_mapwin >= 0.55), else exit immediately.",
        "",
        "## Pre-registered benchmarks (do not re-litigate)",
        f"- stronger-team pistol hit-rate ceiling: {BENCH['ceiling_hit']}",
        f"- tape legs: win {BENCH['tape_win_pop_pt']:+}pt / loss {BENCH['tape_loss_drop_pt']}pt"
        f" -> breakeven hit {BENCH['breakeven_hit']}",
        f"- replay EV at ceiling: {BENCH['replay_ev_at_ceiling_pt']}pt",
        f"- R2 rebound rate after losing pistol: {BENCH['r2_rebound_rate']}",
        "",
        "A 55% trigger line has NO data support (ceiling 53.4%; v3 OOF AUC ~0.48",
        "with >=60% subset hitting 42-43%). This run executes it anyway.",
        "",
        "## Honesty ledger",
        "- prices are minute-level mids: no spread, no book; cost03/fee columns",
        "  are sensitivity only",
        "- 'stronger team' = composite(all_win/kd/round + map_win/pistol gaps),",
        "  90-day half-life history; HLTV crawl out of scope (user named it the",
        "  ideal source)",
        "- in-sample p_cal is optimistic (Platt applied to all rows); OOF is the",
        "  honest arm. Read OOF first.",
        "- pistol timing is anchored to the first >=3pt 1-min price jump near",
        "  the demo-clock pistol (market-time detection); alignment_diag must",
        "  show a high jump-side match rate before any P&L is interpreted",
        "",
        "## Result tables",
        "```json",
        json.dumps(s, ensure_ascii=False, indent=2, default=str)[:12000],
        "```",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
