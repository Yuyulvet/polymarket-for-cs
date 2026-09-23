"""Apply exported 5E-compatible Map Winner models to strictly paired live rows.

This is a paper signal generator. It never places orders. Only nonterminal MQTT
states with bounded source lag and executable token quotes are accepted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .inplay_map_research import predict_variant_bundle
from .map1_data import map_name, terminal_score


def state_to_frame(row: dict, *, max_source_lag_seconds: float = 5.0) -> pd.DataFrame:
    if not row.get("decision_eligible"):
        raise ValueError("state_not_decision_eligible")
    if row.get("state_timing_quality") != "mqtt_push_receive_time":
        raise ValueError("state_not_mqtt_push")
    lag = row.get("state_source_lag_seconds")
    if lag is None or not np.isfinite(float(lag)) or not -2 <= float(lag) <= max_source_lag_seconds:
        raise ValueError("state_source_lag_rejected")
    features = row.get("features") or {}
    required = ("round_number", "focal_is_ct", "focal_score", "opponent_score",
                "score_diff", "focal_alive", "opponent_alive", "alive_diff", "hp_diff")
    if any(features.get(name) is None for name in required):
        raise ValueError("live_feature_missing")
    round_number = int(features["round_number"])
    focal_score, opponent_score = int(features["focal_score"]), int(features["opponent_score"])
    if round_number != focal_score + opponent_score + 1:
        raise ValueError("round_number_score_mismatch")
    if terminal_score(focal_score, opponent_score):
        raise ValueError("terminal_map_state")
    if min(int(features["focal_alive"]), int(features["opponent_alive"])) <= 0:
        raise ValueError("terminal_or_unverifiable_round_state")
    arena = map_name(features.get("map_name"))
    a_is_ct = 1.0 if features["focal_is_ct"] is True else -1.0
    record = {
        "map_name": arena,
        "map_side": arena + ("|A_CT" if a_is_ct == 1 else "|A_T"),
        "score_gap": float(features["score_diff"]),
        "score_gap_x_round": float(features["score_diff"]) * round_number,
        "a_is_ct": a_is_ct,
        "round_num": float(round_number),
        "alive_gap": float(features["alive_diff"]),
        "hp_gap": float(features["hp_diff"]),
    }
    if not np.isfinite(list(record.values())[2:]).all():
        raise ValueError("live_feature_nonfinite")
    return pd.DataFrame([record])


def predict_row(row: dict, artifact: dict, *, max_source_lag_seconds: float = 5.0) -> dict:
    if not artifact.get("paper_only") or artifact.get("promoted"):
        raise ValueError("model_artifact_not_paper_only")
    variants = artifact.get("variants") or {}
    if not {"fivee_score", "fivee_state"}.issubset(variants):
        raise ValueError("model_artifact_variants_missing")
    frame = state_to_frame(row, max_source_lag_seconds=max_source_lag_seconds)
    p_score = float(predict_variant_bundle(variants["fivee_score"], frame)[0])
    p_state = float(predict_variant_bundle(variants["fivee_state"], frame)[0])
    quote = row.get("quote_at_effective_decision") or {}
    bid, ask, mid = quote.get("bid"), quote.get("ask"), quote.get("mid")
    return {
        "schema_version": 1, "paper_only": True,
        "event_id": row.get("event_id"), "market_id": row.get("market_id"),
        "condition_id": row.get("condition_id"), "market": row.get("market"),
        "match_id": row.get("match_id"), "bout_num": row.get("bout_num"),
        "state_hash": row.get("state_hash"), "state_recv_utc": row.get("state_recv_utc"),
        "state_source_lag_seconds": row.get("state_source_lag_seconds"),
        "effective_decision_ts": row.get("effective_decision_ts"),
        "focal_team": row.get("focal_team"), "focal_outcome": row.get("focal_outcome"),
        "focal_token": row.get("focal_token"), "opponent_team": row.get("opponent_team"),
        "model_probability_score": p_score,
        "model_probability_state": p_state,
        "state_probability_increment": p_state - p_score,
        "executable_bid": bid, "executable_ask": ask, "market_mid": mid,
        "gross_model_minus_ask": None if ask is None else p_state - float(ask),
        "gross_bid_minus_model": None if bid is None else float(bid) - p_state,
        "model_input": frame.iloc[0].to_dict(),
        "trade_action": None,
        "reason": "research_probability_only_no_fee_or_fill_decision",
    }


def run(*, paired_path: Path, models_path: Path, output_dir: Path,
        max_source_lag_seconds: float = 5.0) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing_overwrite:{output_dir}")
    artifact = joblib.load(models_path)
    rows, rejected = [], Counter()
    scanned = 0
    with Path(paired_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            try:
                rows.append(predict_row(
                    json.loads(line), artifact,
                    max_source_lag_seconds=max_source_lag_seconds))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                rejected[str(exc)] += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_only": True, "live_trading_enabled": False,
        "paired_path": str(Path(paired_path).resolve()),
        "models_path": str(Path(models_path).resolve()),
        "model_research_version": artifact.get("research_version"),
        "model_trained_before": artifact.get("trained_before"),
        "max_source_lag_seconds": max_source_lag_seconds,
        "rows_scanned": scanned, "predictions": len(rows),
        "rejected": dict(sorted(rejected.items())),
        "profitability_evaluated": False,
        "limitations": [
            "Historical model event sampling does not match irregular 5E push sampling.",
            "Gross model-price gaps exclude fees, slippage, queue position and fill probability.",
            "No action is emitted until a fresh forward market-aligned acceptance set exists.",
        ],
        "artifacts": {"predictions": str(predictions_path)},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-source-lag", type=float, default=5.0)
    args = parser.parse_args(argv)
    report = run(paired_path=args.paired, models_path=args.models,
                 output_dir=args.output_dir,
                 max_source_lag_seconds=args.max_source_lag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
