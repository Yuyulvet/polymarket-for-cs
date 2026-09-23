"""Offline map-winner checkpoints with identical cohorts and past-only fitting.

Completed-round summaries are observed at the NEXT round's freeze end, not
retroactively at the previous round_end. UTC/availability is a conservative
series-level proxy; this module does not simulate live fills.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .map1_data import first_ct_is_ct, path_key, roster_key, summarize_map, utc
from .map1_model import probability_metrics
from .structure_form import team_round_table

SNAPSHOTS = (6, 12, 18, 24, 27, 30)
STRUCT_GAPS = ["full_win_gap", "eco_win_gap", "rwaff_gap", "trade_gap", "util_gap", "hp_gap"]
VARIANTS = {
    "score_only": ["score_gap"],
    "score_full_buy": ["score_gap", "full_win_gap"],
    "score_structure": ["score_gap", *STRUCT_GAPS],
    "structure_only": STRUCT_GAPS,
}


def normalize_round_events(group: pd.DataFrame, meta) -> pd.DataFrame:
    """Repair legacy one-swap team-oriented columns, not raw winner_side.

    Every ct_/t_ pair in this cache belongs to its recorded roster. Thus all
    pairs, including economy and event counts, must move together.
    """
    d = group.sort_values("round_num").copy()
    summary = summarize_map(d)
    for name in ("map_id", "match_id", "map_name", "roster_a", "roster_b", "y", "rounds"):
        if summary[name] != meta[name]:
            raise ValueError(f"history_mismatch_{name}")
    first_ct, first_t = roster_key(d.iloc[0].ct_roster), roster_key(d.iloc[0].t_roster)
    d["ct_roster"] = d.ct_roster.map(roster_key)
    d["t_roster"] = d.t_roster.map(roster_key)
    expected_ct = d.round_num.map(lambda r: first_ct if first_ct_is_ct(int(r)) else first_t)
    swap = d.ct_roster.ne(expected_ct)
    pairs = [(c, "t_" + c[3:]) for c in d if c.startswith("ct_") and "t_" + c[3:] in d]
    for ct, t in pairs:
        old_ct, old_t = d[ct].copy(), d[t].copy()
        d.loc[swap, ct], d.loc[swap, t] = old_t.loc[swap], old_ct.loc[swap]
    d["label_ct_win"] = d.winner_side.str.upper().eq("CT").astype(int)
    d["winner_roster"] = np.where(d.label_ct_win.eq(1), d.ct_roster, d.t_roster)
    return d


def _cumul(tr_map: pd.DataFrame, roster: str, k: int) -> dict:
    sub = tr_map[(tr_map.roster_key == roster) & (tr_map.round_num <= k)]
    out = {"score": int(sub.won.sum()), "n": len(sub)}
    for name, count, won in (("full_win", "is_full", "won_full"),
                             ("eco_win", "is_eco", "won_eco"),
                             ("rwaff", "first", "won_first")):
        n = sub[count].sum()
        out[name] = float(sub[won].sum() / n) if n else np.nan
    for name in ("trade", "util", "hp"):
        out[name] = float(sub[name].mean())
    return out


def build_snapshots(round_events=None, history=None, checkpoints=SNAPSHOTS) -> pd.DataFrame:
    re = (pd.read_parquet(config.DATA_DIR / "round_events.parquet")
          if round_events is None else round_events.copy())
    hist = (pd.read_parquet(config.DATA_DIR / "map1" / "history.parquet")
            if history is None else history.copy())
    hist["map_id"] = hist.map_id.map(path_key)
    if hist.map_id.duplicated().any():
        raise ValueError("duplicate_clean_map")
    for col in ("start_at", "available_at"):
        hist[col] = hist[col].map(utc)
    if (hist.available_at <= hist.start_at).any():
        raise ValueError("invalid_result_availability")
    for _, series in hist.groupby("match_id"):
        if series.start_at.nunique() != 1 or series.available_at.nunique() != 1:
            raise ValueError("inconsistent_series_metadata")
    if not checkpoints or any(isinstance(k, bool) or int(k) != k or k < 1 for k in checkpoints):
        raise ValueError("invalid_checkpoints")
    metas = hist.set_index("map_id").to_dict("index")
    re["map_id"] = re.demo_path.map(path_key)
    rejected = Counter()
    rows = []
    for mid, group in re.groupby("map_id", sort=True):
        meta = metas.get(mid)
        if meta is None:
            rejected["outside_clean_history"] += 1
            continue
        try:
            d = normalize_round_events(group, {**meta, "map_id": mid})
        except (ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
            continue
        tr = team_round_table(d)
        for k in sorted(set(checkpoints)):
            # A terminal map has no next-round checkpoint. Keep overtime maps.
            if k >= meta["rounds"]:
                continue
            a = _cumul(tr, meta["roster_a"], k)
            b = _cumul(tr, meta["roster_b"], k)
            if a["n"] != k or b["n"] != k or a["score"] + b["score"] != k:
                raise ValueError("non_contiguous_checkpoint")
            row = {"map_id": mid, "demo_path": str(group.iloc[0].demo_path),
                   "match_id": meta["match_id"], "k": k, "y": meta["y"],
                   "start_at": meta["start_at"], "available_at": meta["available_at"],
                   "checkpoint": "next_round_freeze_end", "score_gap": a["score"] - b["score"]}
            for name in ("full_win", "eco_win", "rwaff", "trade", "util", "hp"):
                row[f"{name}_gap"] = a[name] - b[name]
            rows.append(row)
    cols = ["map_id", "demo_path", "match_id", "k", "y", "start_at", "available_at",
            "checkpoint", "score_gap", *STRUCT_GAPS]
    result = pd.DataFrame(rows, columns=cols)
    result.attrs["audit"] = {"source_maps": int(re.map_id.nunique()),
                             "rejected_maps": dict(rejected), "snapshot_rows": len(result)}
    return result


def forward_splits(d: pd.DataFrame, n_blocks: int = 5):
    """Equal-time series stay together; train labels available before block start."""
    if n_blocks < 2:
        raise ValueError("at_least_two_blocks_required")
    starts = d.start_at.map(utc)
    available = d.available_at.map(utc)
    if (available <= starts).any():
        raise ValueError("invalid_result_availability")
    for _, group in d.assign(_start=starts, _available=available).groupby("match_id"):
        if group._start.nunique() != 1 or group._available.nunique() != 1:
            raise ValueError("inconsistent_series_metadata")
    times = sorted(starts.unique())
    if len(times) < 2:
        return
    for block in np.array_split(np.asarray(times, dtype=object), min(n_blocks, len(times)))[1:]:
        test = starts.isin(block)
        train = starts.lt(block[0]) & available.lt(block[0]) & ~d.match_id.isin(d.loc[test, "match_id"])
        if train.any() and test.any():
            yield np.flatnonzero(train), np.flatnonzero(test)


def compare_variants(d: pd.DataFrame, min_train: int = 40) -> tuple[dict, pd.DataFrame]:
    required = sorted({f for features in VARIANTS.values() for f in features})
    valid = np.isfinite(d[required].to_numpy(dtype=float)).all(axis=1) & d.y.isin([0, 1])
    common = d.loc[valid].sort_values(["start_at", "match_id", "map_id"]).reset_index(drop=True).copy()
    if common.duplicated("map_id").any():
        raise ValueError("compare_one_checkpoint_per_map")
    for name in VARIANTS:
        common[f"p_{name}"] = np.nan
    common["train_max_available_at"] = ""
    for tr, te in forward_splits(common):
        train, test = common.iloc[tr], common.iloc[te]
        if len(train) < min_train or train.y.nunique() != 2:
            continue
        # Each series has equal aggregate fit weight; mirrored rows stay in train.
        weights = 1 / train.groupby("match_id").match_id.transform("size").to_numpy()
        weights *= len(weights) / weights.sum()
        for name, features in VARIANTS.items():
            x, y = train[features].to_numpy(), train.y.to_numpy()
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.25, fit_intercept=False, max_iter=2000))
            model.fit(np.concatenate([x, -x]), np.concatenate([y, 1-y]),
                      logisticregression__sample_weight=np.tile(weights, 2))
            common.loc[te, f"p_{name}"] = model.predict_proba(test[features].to_numpy())[:, 1]
        common.loc[te, "train_max_available_at"] = train.available_at.max().isoformat()
    scored = common.dropna(subset=[f"p_{name}" for name in VARIANTS])
    metrics = {name: probability_metrics(scored.y, scored[f"p_{name}"]) for name in VARIANTS}
    # Also expose equal-series loss; map counts are not independent trials.
    for name in VARIANTS:
        if len(scored):
            p = np.clip(scored[f"p_{name}"].to_numpy(), 1e-6, 1-1e-6)
            losses = pd.DataFrame({"match_id": scored.match_id, "brier": (p-scored.y)**2,
                                  "log_loss": -scored.y*np.log(p)-(1-scored.y)*np.log(1-p)})
            metrics[name]["equal_series_loss"] = losses.groupby("match_id")[["brier", "log_loss"]].mean().mean().to_dict()
    return {"input_rows": len(d), "common_complete_rows": len(common), "scored_maps": len(scored),
            "scored_series": int(scored.match_id.nunique()), "metrics": metrics}, common


def evaluate() -> dict:
    snap = build_snapshots()
    report = {"mode": "offline_map_checkpoint_diagnostic_v2", "audit": snap.attrs["audit"],
              "validation": "whole_series_chronological_blocks_past_available_labels_only",
              "time_basis": "series_start_and_series_end_plus_lag_proxy_not_live_receipt",
              "checkpoint": "next_round_freeze_end_after_k_completed_rounds",
              "execution_pnl": None, "live_trading_enabled": False, "checkpoints": {},
              "limitations": ["Historical data already inspected; not a fresh acceptance set.",
                              "Legacy completed-round event-count semantics still need raw-demo audit.",
                              "Common complete-case coverage can be selective; report exclusions.",
                              "No synchronized market comparison or statistical significance claim."]}
    for k, group in snap.groupby("k"):
        report["checkpoints"][str(k)] = compare_variants(group)[0]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON path; existing files are never overwritten")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output_exists_refusing_overwrite")
    report = evaluate()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
