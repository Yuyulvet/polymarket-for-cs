"""Offline current-round prediction with terminal exclusion and forward series splits.

V2 observes the first full tick after an event batch. Legacy same-tick caches
require explicit diagnostic opt-in, not a claim of live timing verification.
Event rows are dependent observations, not independent matches or map trades.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .map1_data import path_key, utc
from .round_transition import _classify6, _round_class


VERSION = "inround_nonterminal_forward_v2"
NUMERIC_FEATURES = ["alive_gap", "hp_gap", "bomb_state", "equip_gap", "contact_gap", "aiming_gap", "spread_gap"]
FEATURE_STEPS = [
    ("alive", ["map_name", "round_class"], ["alive_gap"]),
    ("hp", ["map_name", "round_class"], ["alive_gap", "hp_gap"]),
    ("bomb", ["map_name", "round_class"], ["alive_gap", "hp_gap", "bomb_state"]),
    ("economy", ["map_name", "round_class", "ct_tier6", "t_tier6"], NUMERIC_FEATURES[:4]),
    ("position", ["map_name", "round_class", "ct_tier6", "t_tier6"], NUMERIC_FEATURES),
]


def terminal_reason(ct_alive: int, t_alive: int, bomb_state: int, event_type: str = "") -> str | None:
    """T wiped after plant is not terminal: CT still must defuse the live bomb."""
    if event_type in {"bomb_defused", "bomb_exploded", "round_end"} or bomb_state in (2, 3):
        return "terminal_bomb_or_round_event"
    if ct_alive == 0:
        return "ct_eliminated"
    if t_alive == 0 and bomb_state == 0:
        return "t_eliminated_before_plant"
    return None


def prepare_events(events: pd.DataFrame, rounds: pd.DataFrame, *,
                   allow_legacy_timing: bool = False) -> tuple[pd.DataFrame, dict]:
    """Pure, fail-closed label/timing join. End tick is exclusion metadata only.

    Legacy boundaries can be reconstructed from freeze tick + round length, but
    this cannot repair old same-tick snapshot semantics. Future outcome summaries
    from the round table are never exposed as model features.
    """
    required_events = {"demo_path", "round_num", "tick", "event_type", "map_name",
                       "ct_alive", "t_alive", "ct_hp", "t_hp", "bomb_state"}
    required_rounds = {"demo_path", "round_num", "winner_side", "ct_equip0", "t_equip0"}
    if required_events - set(events) or required_rounds - set(rounds):
        raise ValueError("Missing required event or round-state columns")
    e, r = events.copy(), rounds.copy()
    for frame in (e, r):
        frame["demo_path"] = frame.demo_path.map(path_key)
    keys = ["demo_path", "round_num"]
    if r.duplicated(keys).any():
        raise ValueError("Duplicate round labels; cannot perform a many-to-one merge")
    if e.duplicated([*keys, "tick"]).any():
        raise ValueError("Duplicate tick batches; intra-tick observations must be coalesced")
    boundaries = {"round_start_tick", "round_end_tick"}
    if not boundaries.issubset(r):
        if "round_len_ticks" not in r:
            raise ValueError("Round boundaries or legacy round_len_ticks are required")
        starts = e[e.event_type.eq("round_start")][keys + ["tick"]].rename(columns={"tick": "round_start_tick"})
        if starts.duplicated(keys).any():
            raise ValueError("Multiple round starts; cannot infer boundaries")
        r = r.drop(columns=list(boundaries & set(r))).merge(starts, on=keys, how="left", validate="one_to_one")
        r["round_end_tick"] = r.round_start_tick + r.round_len_ticks
    r = r.sort_values(keys).copy()
    next_start = r.groupby("demo_path").round_start_tick.shift(-1)
    r["round_boundary_valid"] = (r.round_end_tick.gt(r.round_start_tick) &
                                  (next_start.isna() | r.round_end_tick.lt(next_start)))
    e = e.drop(columns=list(((required_rounds | boundaries | {"label_ct_win"}) & set(e)) - set(keys)))
    r["winner_side"] = r.winner_side.astype(str).str.upper().replace({"TERRORIST": "T"})
    keep = [*keys, "winner_side", "ct_equip0", "t_equip0", "round_boundary_valid", *sorted(boundaries)]
    m = e.merge(r[keep], on=keys, how="left", validate="many_to_one", indicator=True)
    excluded = Counter()
    accepted = []
    for row in m.to_dict("records"):
        reason = None
        if row["_merge"] != "both" or row["winner_side"] not in {"CT", "T"}:
            reason = "missing_or_invalid_round_label"
        elif not row["round_boundary_valid"]:
            reason = "invalid_round_boundary"
        try:
            ct, tt, bomb = (float(row[k]) for k in ("ct_alive", "t_alive", "bomb_state"))
            event_tick = float(row.get("event_tick", row["tick"]))
            start, end = float(row["round_start_tick"]), float(row["round_end_tick"])
            v2 = (row.get("schema_version") == 2 and
                  row.get("observation_basis") == "first_full_tick_after_event_batch")
            state_tick = float(row.get("state_tick", row["tick"]))
            hp_ct, hp_t, equip_ct, equip_t = (float(row[k]) for k in ("ct_hp", "t_hp", "ct_equip0", "t_equip0"))
            if not all(np.isfinite(v) and v.is_integer() for v in (ct, tt, bomb, event_tick, state_tick, start, end)):
                reason = reason or "invalid_state_or_boundary"
            elif not (0 <= ct <= 5 and 0 <= tt <= 5 and bomb in (0, 1, 2, 3) and end > start):
                reason = reason or "invalid_state_or_boundary"
            elif (not all(np.isfinite(v) for v in (hp_ct, hp_t, equip_ct, equip_t)) or
                  not (0 <= hp_ct <= ct*100 and 0 <= hp_t <= tt*100 and
                       equip_ct >= 0 and equip_t >= 0)):
                reason = reason or "invalid_hp_or_equipment"
            elif (event_tick != float(row["tick"]) or (v2 and state_tick != event_tick + 1) or
                  (not v2 and not allow_legacy_timing)):
                reason = reason or "unverified_observation_timing"
            elif not (start <= event_tick <= state_tick < end):
                reason = reason or "outside_active_round"
            else:
                reason = reason or terminal_reason(int(ct), int(tt), int(bomb), row["event_type"])
        except (TypeError, ValueError):
            reason = reason or "invalid_state_or_boundary"
            v2 = False
        if reason:
            excluded[reason] += 1
            continue
        row["event_tick"], row["state_tick"] = int(event_tick), int(state_tick)
        row["timing_verified"] = bool(v2)
        row["elapsed_seconds"] = (state_tick - start) / 64.
        row["label_ct_win"] = int(row["winner_side"] == "CT")
        accepted.append(row)
    extra = [name for name in ("event_tick", "state_tick", "timing_verified", "elapsed_seconds", "label_ct_win")
             if name not in m.columns]
    result = pd.DataFrame(accepted, columns=[*m.columns, *extra]).drop(columns=["_merge", "winner_side"])
    for name in ("ct_alive", "t_alive", "ct_hp", "t_hp", "bomb_state", "ct_equip0", "t_equip0"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    result["alive_gap"] = result.ct_alive - result.t_alive
    result["hp_gap"] = result.ct_hp - result.t_hp
    result["equip_gap"] = result.ct_equip0 - result.t_equip0
    result["round_class"] = result.round_num.map(_round_class)
    result["ct_tier6"] = result.ct_equip0.map(_classify6)
    result["t_tier6"] = result.t_equip0.map(_classify6)
    for name, left, right in (("contact_gap", "t_contact", "ct_contact"),
                               ("aiming_gap", "ct_aiming", "t_aiming"),
                               ("spread_gap", "t_spread", "ct_spread")):
        result[name] = (pd.to_numeric(result[left], errors="coerce") - pd.to_numeric(result[right], errors="coerce")
                        if left in result and right in result else np.nan)
    result[NUMERIC_FEATURES] = result[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    result = result.sort_values([*keys, "state_tick"]).reset_index(drop=True)
    return result, {"source_event_rows": len(events), "retained_event_rows": len(result),
                    "retained_rounds": int(result[keys].drop_duplicates().shape[0]),
                    "excluded_rows": dict(sorted(excluded.items())),
                    "legacy_timing_opt_in": allow_legacy_timing,
                    "verified_event_rows": int(result.timing_verified.sum()),
                    "timing_basis": "v2_after_tick_or_explicit_unverified_legacy_diagnostic",
                    "terminal_rule": "T_eliminated_after_plant_remains_predictive;_CT_eliminated_or_bomb_finished_does_not"}


def attach_chronology(events: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Require clean series clocks; never substitute filename order for time."""
    columns = ["map_id", "match_id", "start_at", "available_at"]
    if set(columns) - set(metadata):
        raise ValueError("Chronological evaluation requires clean map/series timestamps")
    meta = metadata[columns].copy()
    meta["map_id"] = meta.map_id.map(path_key)
    if meta.map_id.duplicated().any():
        raise ValueError("Duplicate chronology map_id")
    meta["match_id"] = meta.match_id.map(str)
    for name in ("start_at", "available_at"):
        meta[name] = pd.to_datetime(meta[name].map(utc), utc=True)
    meta["start_at"] = meta.groupby("match_id").start_at.transform("min")
    meta["available_at"] = meta.groupby("match_id").available_at.transform("max")
    if (meta.available_at <= meta.start_at).any():
        raise ValueError("Series result availability must follow series start")
    if "match_id" in events:
        expected = events.demo_path.map(path_key).map(meta.set_index("map_id").match_id)
        known = expected.notna()
        if (events.loc[known, "match_id"].astype(str) != expected[known]).any():
            raise ValueError("Event series identity conflicts with clean chronology")
    e = events.drop(columns=[name for name in ("match_id", "start_at", "available_at") if name in events]).copy()
    e["map_id"] = e.demo_path.map(path_key)
    m = e.merge(meta, on="map_id", how="inner", validate="many_to_one")
    return m, {"rows_missing_clean_chronology": len(events) - len(m),
               "retained_series": int(m.match_id.nunique()), "retained_maps": int(m.map_id.nunique()),
               "availability_basis": "clean_history_match_end_plus_lag_proxy_not_live_receipt"}


def forward_splits(events: pd.DataFrame, n_splits: int = 5, min_train_series: int = 20):
    """Yield positional indices: training availability strictly precedes test start."""
    if n_splits < 1 or min_train_series < 1:
        raise ValueError("Positive split and minimum training-series counts required")
    starts = pd.to_datetime(events.start_at.map(utc), utc=True)
    availability = pd.to_datetime(events.available_at.map(utc), utc=True)
    series = events.match_id.astype(str)
    # Whole-series blocks even when callers pass unaggregated metadata.
    starts = starts.groupby(series).transform("min")
    availability = availability.groupby(series).transform("max")
    times = sorted(starts.unique())
    if len(times) < 2:
        return
    chunks = np.array_split(np.arange(len(times)), min(n_splits + 1, len(times)))
    for chunk in chunks[1:]:
        test_times = [times[int(i)] for i in chunk]
        cutoff = min(test_times)
        test_mask = starts.isin(test_times)
        train_mask = availability.lt(cutoff) & ~series.isin(set(series[test_mask]))
        if series[train_mask].nunique() >= min_train_series:
            yield np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(test_mask.to_numpy())


def hierarchical_weights(events: pd.DataFrame) -> np.ndarray:
    """Equal total mass per series, per round within series, then per event."""
    keys = ["match_id", "demo_path", "round_num"]
    per_round = events.groupby(keys, dropna=False).match_id.transform("size").to_numpy(dtype=float)
    counts = events[keys].drop_duplicates().groupby("match_id").size()
    per_series = events.match_id.map(counts).to_numpy(dtype=float)
    return 1. / (per_round * per_series)


def _metrics(events: pd.DataFrame, probabilities: np.ndarray) -> dict:
    y = events.label_ct_win.to_numpy(dtype=int)
    p = np.clip(np.asarray(probabilities), 1e-8, 1 - 1e-8)
    w = hierarchical_weights(events)
    return {"brier": float(np.average((p-y)**2, weights=w)),
            "log_loss": float(np.average(-(y*np.log(p)+(1-y)*np.log(1-p)), weights=w)),
            "auc": float(roc_auc_score(y, p, sample_weight=w)) if len(set(y)) == 2 else None,
            "event_rows": len(events), "rounds": int(events[["demo_path", "round_num"]].drop_duplicates().shape[0]),
            "series": int(events.match_id.nunique()), "weighting": "equal_series_then_round_then_event"}


def evaluate_frame(events: pd.DataFrame, *, n_splits: int = 5, min_train_series: int = 20) -> dict:
    """Fixed ablations on the same forward cohort, no random-fold fallback."""
    data = events.copy().reset_index(drop=True)
    if data.empty:
        return {"status": "insufficient_data", "models": {}, "folds": []}
    splits = list(forward_splits(data, n_splits, min_train_series))
    predictions = {name: np.full(len(data), np.nan) for name, _, _ in FEATURE_STEPS}
    predictions["constant_50"] = np.full(len(data), np.nan)
    folds = []
    for train, test in splits:
        tr, te = data.iloc[train], data.iloc[test]
        if tr.label_ct_win.nunique() < 2:
            continue
        weights = hierarchical_weights(tr)
        weights = weights / weights.mean()
        for name, categorical, numeric in FEATURE_STEPS:
            pre = ColumnTransformer([
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("numeric", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0., add_indicator=True)),
                                      ("scale", StandardScaler())]), numeric),
            ])
            model = Pipeline([("pre", pre), ("clf", LogisticRegression(C=.25, max_iter=3000))])
            model.fit(tr[categorical+numeric], tr.label_ct_win, clf__sample_weight=weights)
            predictions[name][test] = model.predict_proba(te[categorical+numeric])[:, 1]
        predictions["constant_50"][test] = .5
        folds.append({"train_series": int(tr.match_id.nunique()), "test_series": int(te.match_id.nunique()),
                      "train_event_rows": len(tr), "test_event_rows": len(te),
                      "train_max_available_at": tr.available_at.max().isoformat(),
                      "test_min_start_at": te.start_at.min().isoformat(),
                      "test_max_start_at": te.start_at.max().isoformat(),
                      "series_overlap": len(set(tr.match_id) & set(te.match_id))})
    valid = np.isfinite(predictions["constant_50"])
    if not valid.any():
        return {"status": "insufficient_past_series", "models": {}, "folds": folds}
    scored = data.loc[valid]
    models = {name: _metrics(scored, values[valid]) for name, values in predictions.items()}
    strata = {}
    for event_type, subset in scored.groupby("event_type"):
        positions = subset.index.to_numpy()
        strata[str(event_type)] = {name: _metrics(subset, values[positions]) for name, values in predictions.items()}
    return {"status": "exploratory_only", "models": models, "folds": folds, "by_event_type": strata,
            "source_event_rows": len(data), "scored_series": int(scored.match_id.nunique()),
            "scored_rounds": int(scored[["demo_path", "round_num"]].drop_duplicates().shape[0]),
            "scored_event_rows": int(valid.sum()),
            "limitations": ["Round prediction is not map-winner prediction or executable trading profit.",
                            "No event-row IID confidence intervals; evidence is at most the series blocks.",
                            "Existing history is exploratory, not an untouched acceptance holdout.",
                            "Demo ticks do not establish live source receipt time or market synchronization."]}


def load_events(data_dir: Path | None = None, *, allow_legacy_timing: bool = False) -> pd.DataFrame:
    """Quality audit is attached in DataFrame.attrs; v2 is the safe default."""
    folder = Path(data_dir) if data_dir is not None else config.DATA_DIR / "inround_v2"
    events = pd.read_parquet(folder / "inround_events.parquet")
    rounds = pd.read_parquet(folder / "round_states.parquet")
    result, audit = prepare_events(events, rounds, allow_legacy_timing=allow_legacy_timing)
    result.attrs["quality_audit"] = audit
    return result


def evaluate(data_dir: Path | None = None, metadata_path: Path | None = None, *,
             allow_legacy_timing: bool = False, n_splits: int = 5, min_train_series: int = 20) -> dict:
    folder = Path(data_dir) if data_dir is not None else config.DATA_DIR / "inround_v2"
    meta_path = Path(metadata_path) if metadata_path is not None else config.DATA_DIR / "map1" / "history.parquet"
    e = load_events(folder, allow_legacy_timing=allow_legacy_timing)
    quality = e.attrs["quality_audit"]
    e, chronology = attach_chronology(e, pd.read_parquet(meta_path))
    report = {"version": VERSION, "data_dir": str(folder), "quality": quality, "chronology": chronology,
              **evaluate_frame(e, n_splits=n_splits, min_train_series=min_train_series)}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--allow-legacy-timing", action="store_true")
    parser.add_argument("--min-train-series", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.data_dir, args.metadata_path, allow_legacy_timing=args.allow_legacy_timing,
             min_train_series=args.min_train_series)
