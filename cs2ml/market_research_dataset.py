"""Build one-row-per-market-shock research datasets with explicit column roles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from .cs2_shock_annotation import annotate_shocks
from .market_features import build_market_features, load_realtime
from .market_shocks import detect_shocks, executable_drift
from .shock_taxonomy import classify_realtime, label_ex_post

SCHEMA_VERSION = 1
PARTITIONS = {"DISCOVERY", "VALIDATION", "FORWARD_PAPER"}

FEATURE_COLUMNS = {
    "shock_id", "event_id", "market_id", "condition_id", "market_type",
    "match_id", "series_id", "teams",
    "map_number", "decision_ts", "shock_window_seconds", "shock_magnitude",
    "shock_direction", "canonical_token", "canonical_outcome",
    "pre_shock_mid", "pre_shock_bid", "pre_shock_ask",
    "post_shock_bid", "post_shock_ask", "spread", "bid_size_l1",
    "ask_size_l1", "bid_depth_l5", "ask_depth_l5", "obi_1", "obi_5",
    "trade_imbalance", "volume", "trade_intensity",
    "microprice_mid_divergence", "realtime_shock_class",
    "pre_match_cs2_probability", "cs2_model_name", "cs2_model_version",
    "cs2_uncertainty", "cs2_prediction_timestamp",
    "cs2_information_age_seconds", "cs2_map", "cs2_team_identity",
    "roster_confidence", "tier", "history_coverage", "cs2_edge_before_shock",
    "shock_alignment",
}

EXECUTION_COLUMNS = {
    "entry_target_ts", "entry_quote_ts", "entry_bid", "entry_ask",
    "latency_seconds", "fee_status", "slippage_status",
}

METADATA_COLUMNS = {
    "schema_version", "paper_only", "promoted", "live_trading_enabled",
    "partition", "capture_qa_status", "capture_qa_exclusion_reasons",
    "capture_id", "annotation_available", "annotation_missing_reason",
    "taxonomy_version",
}


def _horizon_name(value) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number).replace(".", "p")


def _map_number(row: dict):
    if row.get("map_number") is not None:
        return int(row["map_number"])
    match = re.search(r"\bmap\s*(\d+)\b", str(row.get("market") or ""), re.I)
    return int(match.group(1)) if match else None


def _canonical_observations(observations: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in observations:
        shock_id = str(row.get("shock_id") or "")
        token = str(row.get("token") or "")
        if shock_id and token:
            grouped.setdefault(shock_id, {}).setdefault(token, []).append(row)
    result = {}
    for shock_id, tokens in grouped.items():
        canonical = min(tokens)
        teams = sorted({str(row.get("outcome")) for rows in tokens.values() for row in rows
                        if row.get("outcome") and str(row.get("outcome")).casefold()
                        not in {"yes", "no"}})
        selected = []
        for row in tokens[canonical]:
            value = dict(row)
            if not value.get("teams"):
                value["teams"] = teams
            selected.append(value)
        result[shock_id] = sorted(selected,
                                  key=lambda row: float(row["horizon_seconds"]))
    return result


def _annotation_index(annotations: list[dict] | None) -> dict[tuple[str, str], dict]:
    return {(str(row.get("shock_id")), str(row.get("token"))): row
            for row in (annotations or [])}


def _qa_for(row: dict, qa_reports: dict | None) -> tuple[str, list[str], str | None]:
    qa_reports = qa_reports or {}
    for key in (row.get("market_id"), row.get("capture_id"), row.get("event_id")):
        if key is not None and str(key) in qa_reports:
            report = qa_reports[str(key)]
            return (report.get("status", "UNKNOWN"),
                    list(report.get("exclusion_reasons") or []),
                    report.get("capture_id"))
    return "UNKNOWN", ["qa_report_not_available"], None


def build_dataset(observations: list[dict], *, annotations: list[dict] | None = None,
                  qa_reports: dict | None = None,
                  partition: str = "DISCOVERY") -> tuple[pd.DataFrame, dict]:
    if partition not in PARTITIONS:
        raise ValueError("invalid_dataset_partition")
    annotations_by_key = _annotation_index(annotations)
    rows = []
    future_columns = set()
    for shock_id, horizon_rows in sorted(_canonical_observations(observations).items()):
        first = horizon_rows[0]
        token = str(first["token"])
        window = int(float(first["shock_window_seconds"]))
        annotation = annotations_by_key.get((shock_id, token), {})
        taxonomy = classify_realtime(first)
        qa_status, qa_reasons, capture_id = _qa_for(first, qa_reports)
        row = {
            "schema_version": SCHEMA_VERSION, "paper_only": True,
            "promoted": False, "live_trading_enabled": False,
            "partition": partition, "shock_id": shock_id,
            "event_id": first.get("event_id"), "market_id": first.get("market_id"),
            "condition_id": first.get("condition_id"),
            "match_id": first.get("match_id"), "series_id": first.get("series_id"),
            "teams": first.get("teams"),
            "market_type": first.get("market_type"),
            "map_number": _map_number(first), "decision_ts": first.get("decision_ts"),
            "shock_window_seconds": first.get("shock_window_seconds"),
            "shock_magnitude": first.get("shock_magnitude"),
            "shock_direction": first.get("shock_direction"),
            "canonical_token": token, "canonical_outcome": first.get("outcome"),
            "pre_shock_mid": first.get("pre_shock_mid"),
            "pre_shock_bid": first.get("pre_shock_bid"),
            "pre_shock_ask": first.get("pre_shock_ask"),
            "post_shock_bid": first.get("post_shock_bid"),
            "post_shock_ask": first.get("post_shock_ask"),
            "spread": first.get("spread"), "bid_size_l1": first.get("bid_size_l1"),
            "ask_size_l1": first.get("ask_size_l1"),
            "bid_depth_l5": first.get("bid_depth_l5"),
            "ask_depth_l5": first.get("ask_depth_l5"),
            "obi_1": first.get("obi_1"), "obi_5": first.get("obi_5"),
            "trade_imbalance": first.get(f"trade_imbalance_{window}s"),
            "volume": first.get(f"volume_{window}s"),
            "trade_intensity": first.get(f"trades_per_second_{window}s"),
            "microprice_mid_divergence": first.get("microprice_mid_divergence"),
            "realtime_shock_class": taxonomy["realtime_class"],
            "taxonomy_version": taxonomy["taxonomy_version"],
            "entry_target_ts": first.get("entry_target_ts"),
            "entry_quote_ts": first.get("entry_quote_ts"),
            "entry_bid": first.get("entry_bid"), "entry_ask": first.get("entry_ask"),
            "latency_seconds": first.get("latency_seconds"),
            "fee_status": first.get("fee_status"),
            "slippage_status": first.get("slippage_status"),
            "capture_qa_status": qa_status,
            "capture_qa_exclusion_reasons": qa_reasons, "capture_id": capture_id,
            "annotation_available": bool(annotation.get("annotation_available")),
            "annotation_missing_reason": annotation.get("missing_reason"),
        }
        for key in ("pre_match_cs2_probability", "cs2_model_name", "cs2_model_version",
                    "cs2_uncertainty", "cs2_prediction_timestamp",
                    "cs2_information_age_seconds", "cs2_map", "cs2_team_identity",
                    "roster_confidence", "tier", "history_coverage",
                    "cs2_edge_before_shock", "shock_alignment"):
            row[key] = annotation.get(key)
        for observation in horizon_rows:
            suffix = _horizon_name(observation["horizon_seconds"])
            values = {
                f"h{suffix}_mid_drift": observation.get("raw_mid_drift"),
                f"h{suffix}_executable_pnl": observation.get("executable_pnl"),
                f"h{suffix}_label_available": bool(observation.get("label_valid")),
                f"h{suffix}_fill_flag": bool(observation.get("label_valid")),
                f"h{suffix}_exit_quote_ts": observation.get("exit_quote_ts"),
                f"h{suffix}_estimated_fees": observation.get("estimated_fees"),
                f"h{suffix}_estimated_slippage": observation.get("estimated_slippage"),
                f"h{suffix}_net_drift": observation.get("net_drift"),
                f"h{suffix}_ex_post_label": label_ex_post(observation)[
                    "ex_post_outcome_label"],
            }
            row.update(values)
            future_columns.update(values)
        rows.append(row)
    frame = pd.DataFrame(rows)
    roles = {
        "FEATURES_AVAILABLE_AT_DECISION": sorted(FEATURE_COLUMNS),
        "EXECUTION_INFORMATION": sorted(EXECUTION_COLUMNS),
        "FUTURE_LABELS": sorted(future_columns),
        "RESEARCH_METADATA_NOT_MODEL_INPUT": sorted(METADATA_COLUMNS),
    }
    assigned = [set(values) for values in roles.values()]
    for index, left in enumerate(assigned):
        for right in assigned[index + 1:]:
            if left & right:
                raise AssertionError(f"overlapping_column_roles:{sorted(left & right)}")
    if not frame.empty:
        missing_roles = set(frame.columns) - set().union(*assigned)
        if missing_roles:
            raise AssertionError(f"columns_without_role:{sorted(missing_roles)}")
    events = {str(value) for value in frame.get("event_id", []) if pd.notna(value)}
    matches = {str(value) for value in frame.get("match_id", []) if pd.notna(value)} or events
    series = {str(value) for value in frame.get("series_id", []) if pd.notna(value)} or matches
    maps = {(str(row.get("event_id")), str(row.get("market_id")), row.get("map_number"))
            for row in frame.to_dict("records") if row.get("market_type") == "Map Winner"}
    teams = set()
    for observation in observations:
        values = observation.get("teams")
        if isinstance(values, list):
            teams.update(str(value) for value in values if value)
        outcome = str(observation.get("outcome") or "")
        if outcome.casefold() not in {"", "yes", "no"}:
            teams.add(outcome)
    days = set()
    for value in frame.get("decision_ts", []):
        if pd.notna(value):
            days.add(datetime.fromtimestamp(float(value), timezone.utc).date().isoformat())
    qa_counts = (frame["capture_qa_status"].value_counts().to_dict()
                 if not frame.empty else {})
    metadata = {
        "schema_version": SCHEMA_VERSION, "dataset_unit": "one_market_level_shock",
        "canonical_token_rule": "lexicographically_smallest_token_id",
        "partition": partition, "partition_assigned_utc": datetime.now(timezone.utc).isoformat(),
        "paper_only": True, "promoted": False, "live_trading_enabled": False,
        "column_roles": roles, "rows": len(frame),
        "qa_policy": "PASS main; WARN sensitivity only; FAIL excluded with reason",
        "sample_counts": {
            "observations": len(observations), "market_level_shocks": len(frame),
            "maps": len(maps), "matches": len(matches), "series": len(series),
            "unique_teams": len(teams), "decision_days": len(days),
        },
        "qa_status_counts": qa_counts,
        "main_analysis_rows": int((frame.get("capture_qa_status") == "PASS").sum())
        if not frame.empty else 0,
    }
    return frame, metadata


def model_feature_columns(metadata: dict) -> list[str]:
    return list(metadata["column_roles"]["FEATURES_AVAILABLE_AT_DECISION"])


def future_label_columns(metadata: dict) -> list[str]:
    return list(metadata["column_roles"]["FUTURE_LABELS"])


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def write_dataset(frame: pd.DataFrame, metadata: dict, output_root: Path,
                  *, input_hashes: dict | None = None) -> Path:
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "market_shocks.jsonl"
    with data_path.open("x", encoding="utf-8") as handle:
        for row in frame.to_dict("records"):
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False,
                                    allow_nan=False) + "\n")
    protocol = Path(__file__).parents[1] / "docs" / "market-alpha-protocol-v1.md"
    metadata = {**metadata, "created_utc": datetime.now(timezone.utc).isoformat(),
                "inputs_sha256": input_hashes or {},
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "artifacts": {"dataset": str(data_path)}}
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _load_predictions(path: Path | None):
    if path is None:
        return []
    path = Path(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def load_qa_reports(paths: list[Path]) -> dict:
    result = {}
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        keys = {str(report.get("capture_id"))}
        metadata_path = path.parent / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            keys.update(str(metadata.get(key)) for key in
                        ("capture_id", "market_id", "event_id") if metadata.get(key) is not None)
        source = report.get("source_path")
        if source:
            sidecar = Path(source).with_suffix(".meta.json")
            if sidecar.is_file():
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                if metadata.get("event_id") is not None:
                    keys.add(str(metadata["event_id"]))
                for market in metadata.get("markets") or []:
                    if market.get("market_id") is not None:
                        keys.add(str(market["market_id"]))
        for key in keys - {"None"}:
            result[key] = report
    return result


def run(paths: list[Path], output_root: Path, *,
        prediction_path: Path | None = None, qa_paths: list[Path] | None = None,
        partition: str = "DISCOVERY", latency_seconds: float = 0.0,
        fee_bps: float | None = None,
        slippage_per_side: float | None = None) -> Path:
    paths = [Path(path) for path in paths]
    features = build_market_features(load_realtime(paths))
    shocks = detect_shocks(features)
    observations = executable_drift(
        features, shocks, latency_seconds=latency_seconds, fee_bps=fee_bps,
        slippage_per_side=slippage_per_side)
    annotations = annotate_shocks(shocks, _load_predictions(prediction_path))
    frame, metadata = build_dataset(
        observations, annotations=annotations,
        qa_reports=load_qa_reports(qa_paths or []), partition=partition)
    input_paths = [*paths, *(qa_paths or [])]
    if prediction_path is not None:
        input_paths.append(Path(prediction_path))
    hashes = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for path in input_paths}
    metadata["cs2_annotation"] = {
        "available": int(frame["annotation_available"].sum()) if not frame.empty else 0,
        "missing": int((~frame["annotation_available"]).sum()) if not frame.empty else 0,
        "prediction_source": str(prediction_path) if prediction_path else None,
    }
    return write_dataset(frame, metadata, output_root, input_hashes=hashes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/market_research/datasets"))
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--qa-report", action="append", type=Path, default=[])
    parser.add_argument("--partition", choices=sorted(PARTITIONS), default="DISCOVERY")
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--slippage-per-side", type=float, default=None)
    args = parser.parse_args(argv)
    print(run(args.paths, args.output_root, prediction_path=args.predictions,
              qa_paths=args.qa_report, partition=args.partition,
              latency_seconds=args.latency, fee_bps=args.fee_bps,
              slippage_per_side=args.slippage_per_side))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
