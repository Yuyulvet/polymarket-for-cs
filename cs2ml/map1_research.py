"""Offline Map 1 audit and fixed ablations. Never selects a live model.

python -m cs2ml.map1_research
Writes a new timestamped directory; existing research/desk files are not replaced.
The previously inspected tail is diagnostic, NOT a fresh untouched holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .map1 import now
from .map1_data import FEATURES, build_features, roster_key, terminal_score, utc
from .map1_eligibility import MIN_ROSTER_HISTORY, roster_history_eligibility
from .map1_model import probability_metrics, walk_forward
from .map1_player_research import PLAYER_FEATURES, build_player_features


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_inputs(history, features, verify_features=True):
    """Reject inconsistent labels/cache. This is not external result verification."""
    if history.empty or features.empty:
        raise ValueError("empty_research_inputs")
    if history.map_id.duplicated().any() or history.duplicated(["match_id", "map_no"]).any():
        raise ValueError("duplicate_history_map_identity")
    if features.map_id.duplicated().any() or features.match_id.duplicated().any() or not features.map_no.eq(1).all():
        raise ValueError("expected_one_map1_per_series")
    for row in history.to_dict("records"):
        a, b = roster_key(row["roster_a"]), roster_key(row["roster_b"])
        if a != row["roster_a"] or b != row["roster_b"] or set(a.split(",")) & set(b.split(",")):
            raise ValueError("invalid_history_roster_identity")
        scores = [row["round_wins_a"], row["round_wins_b"]]
        if any(not np.isfinite(x) or x < 0 or int(x) != x for x in scores):
            raise ValueError("invalid_map_score")
        if (not terminal_score(*scores) or sum(scores) != row["rounds"]
                or row["y"] not in (0, 1) or row["y"] != int(scores[0] > scores[1])):
            raise ValueError("score_label_or_terminal_state_mismatch")
        if utc(row["available_at"]) <= utc(row["start_at"]):
            raise ValueError("history_result_not_after_start")
    indexed = history.set_index("map_id")
    if set(features.map_id) != set(history.loc[history.map_no.eq(1), "map_id"]):
        raise ValueError("cached_features_missing_or_extra_map1")
    leads = []
    for row in features.to_dict("records"):
        original = indexed.loc[row["map_id"]]
        for key in ("match_id", "map_no", "map_name", "roster_a", "roster_b", "y"):
            if row[key] != original[key]:
                raise ValueError(f"cached_target_identity_mismatch:{key}")
        for key in ("start_at", "available_at"):
            if utc(row[key]) != utc(original[key]):
                raise ValueError(f"cached_target_time_mismatch:{key}")
        decision = utc(row["decision_at"])
        lead = (utc(row["start_at"]) - decision).total_seconds()
        if lead < 0 or decision >= utc(row["available_at"]):
            raise ValueError("invalid_prediction_cutoff")
        leads.append(lead)
        latest = row.get("max_history_available_at")
        if pd.notna(latest) and utc(latest) >= decision:
            raise ValueError("future_feature_history")
    if len(set(leads)) != 1 or not np.isfinite(features[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("invalid_feature_values_or_mixed_lead_times")
    if verify_features:
        regenerated = build_features(history, lead_seconds=leads[0]).set_index("map_id").loc[features.map_id]
        for column in FEATURES + ["history_all_a", "history_all_b", "history_map_a", "history_map_b"]:
            if not np.allclose(regenerated[column].to_numpy(dtype=float), features[column].to_numpy(dtype=float),
                               rtol=1e-10, atol=1e-10, equal_nan=False):
                raise ValueError(f"feature_cache_rebuild_mismatch:{column}")
    return {"clean_maps": len(history), "map1_targets": len(features), "lead_seconds": leads[0],
            "label_audit": "internal_score_terminal_state_and_roster_consistency_only",
            "independent_external_result_audit": "not_performed",
            "cached_features_recomputed": bool(verify_features),
            "time_basis": "series_end_plus_publication_lag_proxy_not_recorded_receipt",
            "feature_whitelist_only": True}


def elo_predictions(history, targets, k=20.0, scale=400.0):
    """Fixed roster Elo from available past maps; same-time results update together.

    No target series is allowed to update its own prior. Ratings can incorporate
    earlier results inside the diagnostic tail, just as the aggregate historical
    features do; K and scale are fixed, never fitted on that tail.
    """
    if not np.isfinite([k, scale]).all() or k < 0 or scale <= 0:
        raise ValueError("invalid_elo_parameters")
    h = history.copy()
    h["available_at"] = h.available_at.map(utc)
    groups = [(at, group.to_dict("records")) for at, group in h.groupby("available_at", sort=True)]
    result = []
    for target in targets.to_dict("records"):
        ratings = defaultdict(lambda: 1500.0)
        cutoff = utc(target["decision_at"])
        for available, group in groups:
            if available >= cutoff:
                break
            changes = defaultdict(float)
            for row in group:
                if row["match_id"] == target["match_id"]:
                    continue
                a, b = row["roster_a"], row["roster_b"]
                p = 1 / (1 + 10 ** np.clip((ratings[b] - ratings[a]) / scale, -12, 12))
                delta = k * (row["y"] - p)
                changes[a] += delta
                changes[b] -= delta
            for roster, delta in changes.items():
                ratings[roster] += delta
        result.append(1 / (1 + 10 ** np.clip((ratings[target["roster_b"]] - ratings[target["roster_a"]]) / scale, -12, 12)))
    return pd.Series(result, index=targets.index, dtype=float, name="p_roster_elo")


def model_definitions():
    clean = [f for f in FEATURES if f not in {"all_adr_gap", "map_adr_gap"}]
    return {
        "current_roster16": list(FEATURES),
        "roster14_no_legacy_damage": clean,
        "player_only": list(PLAYER_FEATURES),
        "roster14_plus_player": clean + list(PLAYER_FEATURES),
    }


def cohort_metrics(frame, columns):
    common = frame.loc[np.isfinite(frame[list(columns.values())].to_numpy(dtype=float)).all(axis=1)]
    metrics = {name: probability_metrics(common.y, common[column]) for name, column in columns.items()}
    paired = {}
    if len(common):
        y = common.y.to_numpy(dtype=float)
        for name, column in columns.items():
            loss = (common[column].to_numpy(dtype=float) - y) ** 2
            paired[name] = {reference: float(np.mean(loss - (common[columns[reference]].to_numpy(dtype=float) - y) ** 2))
                            for reference in ("constant_0_5", "roster_elo")}
    return {"n": len(common), "series_n": int(common.match_id.nunique()),
            "decision_days": int(common.decision_at.map(utc).dt.normalize().nunique()),
            "metrics": metrics, "paired_brier_difference_model_minus_reference": paired,
            "comparison_status": "descriptive_fixed_candidates_not_significance_or_selection_test"}


def compare_models(history, features, player_maps, *, min_train=60,
                   min_history=MIN_ROSTER_HISTORY, holdout_fraction=.2, progress=None):
    emit = progress or (lambda message: None)
    emit("Auditing labels, time boundaries and cached features ...")
    audit = audit_inputs(history, features)
    d = features.sort_values(["decision_at", "match_id"]).reset_index(drop=True).copy()
    eligibility = [roster_history_eligibility(row, min_history) for row in d.to_dict("records")]
    d["history_eligible"] = [item["eligible"] for item in eligibility]
    d["history_eligibility_reason"] = [item["reason"] for item in eligibility]
    emit("Building past-only per-player profiles ...")
    granular, player_audit = build_player_features(history, player_maps, d)
    if not granular.index.equals(d.index) or set(granular.columns) & set(d.columns):
        raise ValueError("player_features_changed_target_alignment")
    d = pd.concat([d, granular], axis=1)
    d["p_constant_0_5"] = .5
    d["p_roster_elo"] = elo_predictions(history, d)
    columns = {"constant_0_5": "p_constant_0_5", "roster_elo": "p_roster_elo"}
    definitions = model_definitions()
    for name, feature_columns in definitions.items():
        emit(f"Time-forward fixed model: {name} ({len(feature_columns)} features) ...")
        predictions = walk_forward(d, min_train, holdout_fraction, feature_columns)
        if not predictions.map_id.equals(d.map_id):
            raise ValueError("prediction_target_alignment_changed")
        column = "p_" + name
        d[column] = predictions.p_model
        columns[name] = column
        if name == "current_roster16":
            d["partition"] = predictions.partition.replace({"holdout": "diagnostic_tail"})
            d["n_train"] = predictions.n_train
            d["train_max_available_at"] = predictions.train_max_available_at
            d["prediction_reason"] = predictions.reason
    partitions = {}
    for partition, group in d.groupby("partition"):
        partitions[partition] = {"target_rows": len(group),
            "history_eligible_rows": int(group.history_eligible.sum()),
            "history_exclusions": dict(Counter(group.history_eligibility_reason)),
            "all_common": cohort_metrics(group, columns),
            "history_eligible_common": cohort_metrics(group[group.history_eligible], columns)}
    report = {
        "schema_version": 1, "mode": "offline_conditional_map_known_model_audit",
        "generated_at": now(), "audit": audit, "player_feature_audit": player_audit,
        "funnel": {"clean_maps": len(history), "map1_targets": len(d),
                   "history_eligible": int(d.history_eligible.sum()),
                   "common_predictions": int(np.isfinite(d[list(columns.values())].to_numpy(dtype=float)).all(axis=1).sum())},
        "parameters": {"min_train": min_train, "min_roster_history_each": min_history,
                       "tail_fraction_by_unique_decision_time": holdout_fraction,
                       "logistic_C": .25, "mirrored_after_time_split": True,
                       "elo_K": 20., "elo_scale": 400.},
        "models": {"constant_0_5": {"kind": "uninformed_probability"},
                   "roster_elo": {"kind": "available_past_maps_opponent_adjusted_roster_rating"},
                   **{name: {"kind": "fixed_regularized_logistic", "feature_columns": cols}
                      for name, cols in definitions.items()}},
        "partitions": partitions,
        "validation_design": {
            "tail_status": "previously_inspected_diagnostic_not_untouched_test",
            "tail_coefficients": "frozen_before_boundary",
            "historical_profiles": "may_update_using_only_results_available_before_each_prediction",
            "eligibility_scope": "same_roster_history_gate_as_capture_not_market_veto_or_execution_eligibility",
            "candidate_selection": "none_no_automatic_promotion_or_tuning",
            "final_acceptance": "requires_new_forward_time_window_after_protocol_freeze"},
        "legacy_feature_warning": "Legacy ADR includes health plus armor damage; current16 retained for reproduction only. New player and clean candidates exclude it.",
        "live_trading_enabled": False, "execution_pnl": None,
        "limitations": ["This report does not validate live in-round prediction or economic profitability.",
                        "More features do not guarantee predictive gain; candidates are compared on common samples.",
                        "No synchronized historical full orderbooks or measured game-feed receipt times are available here.",
                        "Repeated team/league/day effects and candidate multiplicity require independent future validation."]}
    return d, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_DIR / "map1")
    parser.add_argument("--players", type=Path, default=config.DATA_DIR / "player_features.parquet")
    parser.add_argument("--output", type=Path, help="Must not already exist; default is a new timestamped research folder")
    args = parser.parse_args()
    output = args.output or args.data / "research" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    if output.exists():
        parser.error("Output already exists; old experiments must not be overwritten")
    inputs = {"history": args.data / "history.parquet", "features": args.data / "features.parquet",
              "players": args.players}
    hashes = {key: file_hash(path) for key, path in inputs.items()}
    code_paths = {name: Path(__file__).with_name(name) for name in
                  ("map1_research.py", "map1_player_research.py", "map1_model.py", "map1_data.py", "map1_eligibility.py")}
    code_hashes = {name: file_hash(path) for name, path in code_paths.items()}
    history, features, players = (pd.read_parquet(inputs[key]) for key in ("history", "features", "players"))
    predictions, report = compare_models(history, features, players, progress=lambda msg: print(msg, flush=True))
    if hashes != {key: file_hash(path) for key, path in inputs.items()}:
        raise ValueError("research_input_changed_while_running")
    if code_hashes != {name: file_hash(path) for name, path in code_paths.items()}:
        raise ValueError("research_code_changed_while_running")
    report["inputs"] = {key: {"path": str(path.resolve()), "sha256": hashes[key]} for key, path in inputs.items()}
    report["code_hashes"] = code_hashes
    source_report = args.data / "validation.json"
    if source_report.exists():
        source = json.loads(source_report.read_text(encoding="utf-8"))
        report["upstream_audit_reference"] = {"path": str(source_report.resolve()),
                                             "sha256": file_hash(source_report), "audit": source.get("audit")}
    # Normal generated data output, isolated from production and prior cohorts.
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str((output / "report.json").resolve()),
                      "funnel": report["funnel"], "live_trading_enabled": False,
                      "promoted_model": None}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
