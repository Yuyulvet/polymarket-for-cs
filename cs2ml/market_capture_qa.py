"""Versioned quality assurance for normalized market capture JSONL files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).with_name("market_capture_qa_v1.json")
QA_SCHEMA_VERSION = 1


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _percentage(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _duplicate_key(row: dict) -> str:
    excluded = {"local_ts", "source_ts", "recv_utc", "received_at"}
    payload = {key: value for key, value in row.items() if key not in excluded}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_rows(path: Path) -> tuple[list[dict], int]:
    rows, malformed = [], 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                malformed += 1
    return rows, malformed


def _reconnects(capture_dir: Path) -> int:
    audit = capture_dir / "audit.jsonl"
    if not audit.is_file():
        return 0
    count = 0
    for line in audit.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += row.get("to") == "RECONNECTING"
    return count


def classify(metrics: dict, config: dict) -> tuple[str, list[str]]:
    failures, warnings = [], []
    fail, warn = config["fail"], config["warn"]
    if metrics["message_count"] < fail["minimum_message_count"]:
        failures.append("message_count_below_minimum")
    if fail["require_local_receive_monotonicity"] and not metrics[
            "local_receive_timestamp_monotonic"]:
        failures.append("local_receive_timestamp_not_monotonic")
    two_sided = metrics["valid_two_sided_quote_pct"]
    if two_sided is None or two_sided < fail["minimum_valid_two_sided_quote_pct"]:
        failures.append("valid_two_sided_quote_below_fail_threshold")
    if metrics["crossed_book_rate"] is not None and metrics[
            "crossed_book_rate"] > fail["maximum_crossed_book_rate"]:
        failures.append("crossed_book_rate_above_fail_threshold")
    if failures:
        return "FAIL", failures

    checks = (
        (two_sided is not None and two_sided < warn["minimum_valid_two_sided_quote_pct"],
         "valid_two_sided_quote_below_warn_threshold"),
        (metrics["valid_l1_depth_pct"] is None or metrics["valid_l1_depth_pct"]
         < warn["minimum_valid_l1_depth_pct"], "valid_l1_depth_below_warn_threshold"),
        (metrics["valid_l5_depth_pct"] is None or metrics["valid_l5_depth_pct"]
         < warn["minimum_valid_l5_depth_pct"], "valid_l5_depth_below_warn_threshold"),
        (metrics["duplicate_message_rate"] > warn["maximum_duplicate_message_rate"],
         "duplicate_message_rate_above_warn_threshold"),
        (metrics["stale_quote_rate"] is not None and metrics["stale_quote_rate"]
         > warn["maximum_stale_quote_rate"], "stale_quote_rate_above_warn_threshold"),
        (metrics["p99_quote_interval_seconds"] is not None and metrics[
         "p99_quote_interval_seconds"] > warn["maximum_p99_quote_interval_seconds"],
         "p99_quote_interval_above_warn_threshold"),
        (metrics["locked_book_rate"] is not None and metrics["locked_book_rate"]
         > warn["maximum_locked_book_rate"], "locked_book_rate_above_warn_threshold"),
    )
    warnings.extend(reason for condition, reason in checks if condition)
    return ("WARN", warnings) if warnings else ("PASS", [])


def analyze_capture(capture_dir: Path, *, config_path: Path = CONFIG_PATH) -> dict:
    capture_dir = Path(capture_dir)
    events_path = capture_dir if capture_dir.is_file() else capture_dir / "events.jsonl"
    audit_dir = capture_dir.parent if capture_dir.is_file() else capture_dir
    rows, malformed = _read_rows(events_path)
    config = load_config(config_path)
    counts = Counter(str(row.get("type") or "unknown") for row in rows)
    local_times = [_number(row.get("local_ts")) for row in rows]
    available_times = [value for value in local_times if value is not None]
    monotonic = (len(available_times) == len(rows)
                 and all(second >= first for first, second in
                         zip(available_times, available_times[1:])))

    quote_rows = []
    quote_times = defaultdict(list)
    valid_bid = valid_ask = two_sided = l1 = l5 = crossed = locked = 0
    for row, timestamp in zip(rows, local_times):
        bid, ask = _number(row.get("best_bid")), _number(row.get("best_ask"))
        if bid is None and ask is None:
            continue
        quote_rows.append(row)
        token = str(row.get("token") or "unknown")
        if timestamp is not None:
            quote_times[token].append(timestamp)
        valid_bid += bid is not None
        valid_ask += ask is not None
        if bid is not None and ask is not None:
            two_sided += 1
            crossed += bid > ask
            locked += bid == ask
        bid1, ask1 = _number(row.get("bid_size_l1")), _number(row.get("ask_size_l1"))
        bid5, ask5 = _number(row.get("bid_depth_l5")), _number(row.get("ask_depth_l5"))
        l1 += bid1 is not None and ask1 is not None and bid1 >= 0 and ask1 >= 0
        l5 += bid5 is not None and ask5 is not None and bid5 >= 0 and ask5 >= 0

    intervals = []
    for times in quote_times.values():
        intervals.extend(second - first for first, second in zip(times, times[1:])
                         if second >= first)
    stale = [value for value in intervals if value > config["stale_quote_seconds"]]
    trades = [row for row in rows if row.get("type") == "trade"]
    keys = [_duplicate_key(row) for row in rows]
    duplicates = len(keys) - len(set(keys))
    quote_count = len(quote_rows)
    metrics = {
        "capture_duration_seconds": (
            max(available_times) - min(available_times) if available_times else None),
        "message_count": len(rows), "malformed_line_count": malformed,
        "book_snapshot_count": counts["book"], "price_change_count": counts["pc"],
        "best_bid_ask_count": counts["best_bid_ask"], "trade_count": len(trades),
        "reconnect_count": _reconnects(audit_dir),
        "largest_missing_interval_seconds": max(intervals) if intervals else None,
        "median_quote_interval_seconds": _quantile(intervals, .5),
        "p95_quote_interval_seconds": _quantile(intervals, .95),
        "p99_quote_interval_seconds": _quantile(intervals, .99),
        "valid_best_bid_pct": _percentage(valid_bid, quote_count),
        "valid_best_ask_pct": _percentage(valid_ask, quote_count),
        "valid_two_sided_quote_pct": _percentage(two_sided, quote_count),
        "valid_l1_depth_pct": _percentage(l1, quote_count),
        "valid_l5_depth_pct": _percentage(l5, quote_count),
        "trade_side_availability_pct": _percentage(
            sum(bool(row.get("trade_side") or row.get("side")) for row in trades),
            len(trades)),
        "source_timestamp_availability_pct": _percentage(
            sum(_number(row.get("source_ts")) is not None for row in rows), len(rows)),
        "local_receive_timestamp_monotonic": monotonic,
        "duplicate_message_rate": duplicates / len(rows) if rows else 0.0,
        "crossed_book_rate": _percentage(crossed, two_sided),
        "locked_book_rate": _percentage(locked, two_sided),
        "stale_quote_rate": _percentage(len(stale), len(intervals)),
    }
    status, reasons = classify(metrics, config)
    input_sha = hashlib.sha256(events_path.read_bytes()).hexdigest()
    return {
        "schema_version": QA_SCHEMA_VERSION, "qa_config_version": config["config_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_only": True, "promoted": False, "live_trading_enabled": False,
        "capture_id": capture_dir.stem if capture_dir.is_file() else capture_dir.name,
        "status": status,
        "analysis_eligibility": config["analysis_policy"][status],
        "exclusion_reasons": reasons, "metrics": metrics,
        "thresholds": config, "events_sha256": input_sha,
        "qa_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qa_config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
    }


def write_qa_report(capture_dir: Path, *, config_path: Path = CONFIG_PATH) -> Path:
    capture_dir = Path(capture_dir)
    report = analyze_capture(capture_dir, config_path=config_path)
    path = capture_dir / "qa_report.json"
    if path.exists():
        raise FileExistsError(f"qa_report_already_exists:{path}")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
    return path


def write_file_qa_report(events_path: Path, output_root: Path,
                         *, config_path: Path = CONFIG_PATH) -> Path:
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = analyze_capture(Path(events_path), config_path=config_path)
    report["legacy_standalone_input"] = True
    report["source_path"] = str(Path(events_path))
    path = output / "qa_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Required for a standalone legacy JSONL file")
    args = parser.parse_args(argv)
    if args.capture_dir.is_file():
        if args.output_root is None:
            raise SystemExit("--output-root is required for a standalone JSONL file")
        result = write_file_qa_report(args.capture_dir, args.output_root,
                                      config_path=args.config)
    else:
        result = write_qa_report(args.capture_dir, config_path=args.config)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
