"""Paper-only executable alpha decay as a function of order latency."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .market_features import build_market_features, load_realtime
from .market_shocks import detect_shocks, executable_drift, summarize

SCHEMA_VERSION = 1
LATENCIES_SECONDS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
PRIMARY_WINDOW_SECONDS = 1.0
PRIMARY_THRESHOLD = 0.02
PRIMARY_HORIZON_SECONDS = 5.0
PRIMARY_MAX_QUOTE_WAIT_SECONDS = 1.0


def canonical_shocks(shocks: list[dict]) -> list[dict]:
    """One deterministic token view per market-level shock identity."""
    grouped: dict[str, list[dict]] = {}
    for shock in shocks:
        grouped.setdefault(str(shock["shock_id"]), []).append(shock)
    result = []
    for _, rows in sorted(grouped.items()):
        representative = dict(min(rows, key=lambda row: (
            str(row.get("token")), str(row.get("outcome")))))
        representative["teams"] = sorted({str(row.get("outcome")) for row in rows
                                           if row.get("outcome")
                                           and str(row.get("outcome")).casefold()
                                           not in {"yes", "no"}})
        result.append(representative)
    return result


def alpha_decay(features: pd.DataFrame, *,
                latencies: tuple[float, ...] = LATENCIES_SECONDS,
                shock_window: float = PRIMARY_WINDOW_SECONDS,
                threshold: float = PRIMARY_THRESHOLD,
                horizon: float = PRIMARY_HORIZON_SECONDS,
                max_quote_wait_seconds: float = PRIMARY_MAX_QUOTE_WAIT_SECONDS,
                fee_bps: float | None = None,
                slippage_per_side: float | None = None,
                bootstrap_draws: int = 2000) -> dict:
    if any(latency < 0 for latency in latencies):
        raise ValueError("latencies_must_be_nonnegative")
    detected = detect_shocks(features, shock_windows=(shock_window,),
                             minimum_threshold=threshold)
    shocks = canonical_shocks([row for row in detected
                               if row.get(f"passes_{threshold:.2f}", True)])
    curve = []
    for latency in latencies:
        observations = executable_drift(
            features, shocks, horizons=(horizon,), latency_seconds=latency,
            fee_bps=fee_bps, slippage_per_side=slippage_per_side,
            max_quote_wait_seconds=max_quote_wait_seconds)
        stats = summarize(observations, draws=bootstrap_draws)
        fillable = sum(bool(row.get("label_valid")) for row in observations)
        curve.append({
            "latency_seconds": latency,
            "available_opportunities": len(shocks),
            "fillable_opportunities": fillable,
            "no_fill_or_unavailable": len(shocks) - fillable,
            "number_of_observations": stats["number_of_observations"],
            "number_of_shocks": stats["number_of_shocks"],
            "number_of_maps": stats["number_of_maps"],
            "number_of_matches": stats["number_of_matches"],
            "number_of_series": stats["number_of_series"],
            "unique_teams": stats["unique_teams"],
            "decision_days": stats["decision_days"],
            "mean_executable_pnl": stats["mean_executable_pnl"],
            "median_executable_pnl": stats["median_executable_pnl"],
            "hit_rate": stats["hit_rate"], "profit_factor": stats["profit_factor"],
            "executable_pnl_95pct_clustered_ci":
                stats["executable_pnl_95pct_clustered_ci"],
            "clustered_ci_status": stats["clustered_ci_status"],
            "number_of_clusters": stats["number_of_clusters"],
            "mean_net_drift": stats["mean_net_drift"],
            "conclusion": stats["conclusion"],
        })
    return {
        "schema_version": SCHEMA_VERSION, "experiment": "executable_alpha_decay_v1",
        "paper_only": True, "promoted": False, "live_trading_enabled": False,
        "protocol": "market-alpha-protocol-v1",
        "shock_window_seconds": shock_window, "shock_threshold": threshold,
        "holding_horizon_seconds": horizon,
        "max_quote_wait_seconds": max_quote_wait_seconds,
        "fee_bps": fee_bps, "slippage_per_side": slippage_per_side,
        "latencies_seconds": list(latencies), "market_level_shocks": len(shocks),
        "curve": curve,
        "interpretation_policy": (
            "descriptive_only; do not classify infrastructure or residual alpha "
            "without adequate independent clusters"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(paths: list[Path], output_root: Path, *,
        qa_paths: list[Path] | None = None, **kwargs) -> Path:
    paths = [Path(path) for path in paths]
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    features = build_market_features(load_realtime(paths))
    report = alpha_decay(features, **kwargs)
    qa_reports = [json.loads(Path(path).read_text(encoding="utf-8"))
                  for path in (qa_paths or [])]
    qa_statuses = [report_.get("status", "UNKNOWN") for report_ in qa_reports]
    if not qa_statuses:
        analysis_scope = "diagnostic_only_qa_unknown"
    elif all(status == "PASS" for status in qa_statuses):
        analysis_scope = "main_analysis"
    elif any(status == "FAIL" for status in qa_statuses):
        analysis_scope = "excluded_fail_capture"
    else:
        analysis_scope = "sensitivity_analysis_only"
    report.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "DISCOVERY", "timing_basis": "local_receive_time",
        "inputs_sha256": {str(path): _sha256(path)
                          for path in [*paths, *(qa_paths or [])]},
        "code_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(
            Path(__file__).parents[1] / "docs" / "market-alpha-protocol-v1.md"),
        "capture_qa_statuses": qa_statuses,
        "analysis_scope": analysis_scope,
    })
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/market_research/alpha_decay"))
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--slippage-per-side", type=float, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--qa-report", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    print(run(args.paths, args.output_root, qa_paths=args.qa_report,
              fee_bps=args.fee_bps,
              slippage_per_side=args.slippage_per_side,
              bootstrap_draws=args.bootstrap_draws))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
