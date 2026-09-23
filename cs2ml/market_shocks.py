"""Experiment 0: executable residual drift after Polymarket price shocks.

Signals use only market features available at ``decision_ts``. Future quotes
are labels only. Long momentum pays the first ask at/after the decision and
receives the first bid at/after the horizon. Down momentum uses the symmetric
sell-at-bid / cover-at-ask cash flow. Missing fees or slippage remain unknown;
they are never silently replaced by zero.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .market_features import _json_safe, build_market_features, load_realtime

SCHEMA_VERSION = 1
SHOCK_WINDOWS_SECONDS = (1, 2, 5)
SHOCK_THRESHOLDS = (0.01, 0.02, 0.03, 0.05)
HORIZONS_SECONDS = (1, 2, 3, 5, 10, 30, 60)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _last_at_or_before(rows: list[dict], timestamp: float) -> dict | None:
    index = bisect_right(rows, timestamp,
                         key=lambda row: row["decision_ts"]) - 1
    return rows[index] if index >= 0 else None


def _first_at_or_after(rows: list[dict], timestamp: float,
                       max_wait_seconds: float | None) -> dict | None:
    index = bisect_left(rows, timestamp, key=lambda row: row["decision_ts"])
    if index >= len(rows):
        return None
    row = rows[index]
    if max_wait_seconds is not None and row["decision_ts"] - timestamp > max_wait_seconds:
        return None
    return row


def _magnitude_bucket(value: float) -> str:
    if value < .02:
        return "1-2c"
    if value < .03:
        return "2-3c"
    if value < .05:
        return "3-5c"
    return ">=5c"


def _probability_bucket(value) -> str:
    value = _finite(value)
    if value is None:
        return "unknown"
    lower = min(int(value / .2), 4) * .2
    return f"{lower:.1f}-{lower + .2:.1f}"


def _spread_bucket(value) -> str:
    value = _finite(value)
    if value is None:
        return "unknown"
    if value <= .01:
        return "<=1c"
    if value <= .02:
        return "1-2c"
    if value <= .05:
        return "2-5c"
    return ">5c"


def _liquidity_bucket(value) -> str:
    value = _finite(value)
    if value is None:
        return "unknown"
    if value < 10:
        return "<10"
    if value < 50:
        return "10-50"
    if value < 200:
        return "50-200"
    return ">=200"


def _signed_bucket(value) -> str:
    value = _finite(value)
    if value is None:
        return "unknown"
    if value < -1 / 3:
        return "negative"
    if value > 1 / 3:
        return "positive"
    return "neutral"


def _volume_bucket(value) -> str:
    value = _finite(value)
    if value is None:
        return "unknown"
    if value == 0:
        return "0"
    if value < 10:
        return "0-10"
    if value < 50:
        return "10-50"
    return ">=50"


def _market_type(row: dict) -> str:
    explicit = str(row.get("market_type") or "").strip()
    if explicit:
        return explicit
    name = str(row.get("market") or "").casefold()
    if "map" in name and "winner" in name:
        return "Map Winner"
    if ("match" in name or "series" in name) and "winner" in name:
        return "Match Winner"
    return "unknown"


def detect_shocks(features: pd.DataFrame, *,
                  shock_windows: tuple[float, ...] = SHOCK_WINDOWS_SECONDS,
                  minimum_threshold: float = min(SHOCK_THRESHOLDS),
                  cooldown_seconds: float | None = None) -> list[dict]:
    """Find distinct shocks using backward-only midpoint changes."""
    if minimum_threshold <= 0 or any(window <= 0 for window in shock_windows):
        raise ValueError("positive_shock_windows_and_threshold_required")
    if features.empty:
        return []
    records = features.to_dict("records")
    by_token: dict[str, list[dict]] = {}
    for row in records:
        token = str(row.get("token") or "")
        timestamp, mid = _finite(row.get("decision_ts")), _finite(row.get("mid"))
        if not token or timestamp is None or mid is None:
            continue
        normalized = dict(row)
        normalized["decision_ts"], normalized["mid"] = timestamp, mid
        by_token.setdefault(token, []).append(normalized)
    shocks = []
    for token, tape in by_token.items():
        tape.sort(key=lambda row: (row["decision_ts"], row.get("sequence", 0)))
        for window in shock_windows:
            last_shock = -math.inf
            cooldown = window if cooldown_seconds is None else cooldown_seconds
            for row in tape:
                timestamp = row["decision_ts"]
                prior = _last_at_or_before(tape, timestamp - window)
                if prior is None:
                    continue
                change = row["mid"] - prior["mid"]
                magnitude = abs(change)
                if magnitude + 1e-12 < minimum_threshold or timestamp - last_shock < cooldown:
                    continue
                last_shock = timestamp
                liquidity = None
                if _finite(row.get("bid_depth_l5")) is not None and _finite(
                        row.get("ask_depth_l5")) is not None:
                    liquidity = float(row["bid_depth_l5"]) + float(row["ask_depth_l5"])
                direction = "up" if change > 0 else "down"
                market_key = (f"event={row.get('event_id')}|market="
                              f"{row.get('market_id') or row.get('market')}")
                if row.get("event_id") is None and not (
                        row.get("market_id") or row.get("market")):
                    market_key = f"token={token}"
                event = {
                    **row,
                    "shock_id": f"{market_key}|window={window:g}|ts={timestamp:.9f}",
                    "token_shock_id": f"{token}:{window:g}:{timestamp:.9f}",
                    "shock_window_seconds": window, "shock_change": change,
                    "shock_magnitude": magnitude, "shock_direction": direction,
                    "shock_magnitude_bucket": _magnitude_bucket(magnitude),
                    "starting_probability_bucket": _probability_bucket(row["mid"]),
                    "spread_bucket": _spread_bucket(row.get("spread")),
                    "liquidity": liquidity,
                    "liquidity_bucket": _liquidity_bucket(liquidity),
                    "obi_bucket": _signed_bucket(row.get("obi_5")),
                    "trade_imbalance_bucket": _signed_bucket(
                        row.get(f"trade_imbalance_{int(window)}s")),
                    "volume_bucket": _volume_bucket(row.get(f"volume_{int(window)}s")),
                    "favorite_underdog": "favorite" if row["mid"] >= .5 else "underdog",
                    "market_type": _market_type(row),
                }
                for threshold in SHOCK_THRESHOLDS:
                    event[f"passes_{threshold:.2f}"] = magnitude + 1e-12 >= threshold
                shocks.append(event)
    shocks.sort(key=lambda row: (row["decision_ts"], row["token"],
                                 row["shock_window_seconds"]))
    return shocks


def executable_drift(features: pd.DataFrame, shocks: list[dict], *,
                     horizons: tuple[float, ...] = HORIZONS_SECONDS,
                     latency_seconds: float = 0.0,
                     fee_bps: float | None = None,
                     slippage_per_side: float | None = None,
                     max_quote_wait_seconds: float | None = 1.0) -> list[dict]:
    """Attach future labels using first executable quotes at/after targets."""
    if latency_seconds < 0 or (fee_bps is not None and fee_bps < 0):
        raise ValueError("nonnegative_latency_and_fee_required")
    if slippage_per_side is not None and slippage_per_side < 0:
        raise ValueError("nonnegative_slippage_required")
    if max_quote_wait_seconds is not None and max_quote_wait_seconds < 0:
        raise ValueError("nonnegative_quote_wait_required")
    quotes: dict[str, list[dict]] = {}
    for row in features.to_dict("records"):
        token = str(row.get("token") or "")
        timestamp = _finite(row.get("decision_ts"))
        bid, ask, mid = (_finite(row.get(key)) for key in
                         ("best_bid", "best_ask", "mid"))
        if token and timestamp is not None and bid is not None and ask is not None:
            quotes.setdefault(token, []).append({
                **row, "decision_ts": timestamp, "best_bid": bid,
                "best_ask": ask, "mid": mid if mid is not None else (bid + ask) / 2,
            })
    for tape in quotes.values():
        tape.sort(key=lambda row: (row["decision_ts"], row.get("sequence", 0)))

    observations = []
    for shock in shocks:
        tape = quotes.get(str(shock["token"])) or []
        decision_ts = float(shock["decision_ts"])
        entry_target = decision_ts + latency_seconds
        entry = _first_at_or_after(tape, entry_target, max_quote_wait_seconds)
        for horizon in horizons:
            exit_target = decision_ts + horizon
            exit_row = _first_at_or_after(tape, exit_target, max_quote_wait_seconds)
            valid = entry is not None and exit_row is not None and entry["decision_ts"] <= exit_row["decision_ts"]
            base = {
                **shock, "horizon_seconds": horizon,
                "latency_seconds": latency_seconds,
                "entry_target_ts": entry_target, "exit_target_ts": exit_target,
                "label_valid": valid,
                "entry_quote_ts": entry["decision_ts"] if entry else None,
                "exit_quote_ts": exit_row["decision_ts"] if exit_row else None,
                "fee_status": "known" if fee_bps is not None else "unknown",
                "slippage_status": (
                    "known" if slippage_per_side is not None else "unknown"),
            }
            if not valid:
                base.update({
                    "entry_bid": None, "entry_ask": None, "exit_bid": None,
                    "exit_ask": None, "raw_mid_drift": None,
                    "executable_pnl": None, "spread_cost": None,
                    "estimated_fees": None, "estimated_slippage": None,
                    "net_drift": None,
                })
                observations.append(base)
                continue
            direction = 1 if shock["shock_direction"] == "up" else -1
            raw_mid = direction * (exit_row["mid"] - entry["mid"])
            executable = (exit_row["best_bid"] - entry["best_ask"] if direction > 0
                          else entry["best_bid"] - exit_row["best_ask"])
            row_fee_bps = fee_bps
            if row_fee_bps is None:
                row_fee_bps = _finite(shock.get("taker_fee_bps"))
            estimated_fees = None
            if row_fee_bps is not None:
                traded_prices = ((entry["best_ask"], exit_row["best_bid"])
                                 if direction > 0
                                 else (entry["best_bid"], exit_row["best_ask"]))
                estimated_fees = sum(traded_prices) * row_fee_bps / 10_000.0
                base["fee_status"] = "known"
            estimated_slippage = (None if slippage_per_side is None
                                  else 2 * slippage_per_side)
            net = (executable - estimated_fees - estimated_slippage
                   if estimated_fees is not None and estimated_slippage is not None
                   else None)
            base.update({
                "entry_bid": entry["best_bid"], "entry_ask": entry["best_ask"],
                "exit_bid": exit_row["best_bid"], "exit_ask": exit_row["best_ask"],
                "raw_mid_drift": raw_mid, "executable_pnl": executable,
                "spread_cost": raw_mid - executable,
                "estimated_fees": estimated_fees,
                "estimated_slippage": estimated_slippage, "net_drift": net,
            })
            observations.append(base)
    return observations


def clustered_bootstrap(values, clusters, *, draws: int = 2000,
                        seed: int = 20260923) -> list[float] | None:
    pairs = [(float(value), str(cluster)) for value, cluster in zip(values, clusters)
             if _finite(value) is not None and cluster is not None]
    names = sorted({cluster for _, cluster in pairs})
    if len(names) < 2:
        return None
    grouped = {name: [value for value, cluster in pairs if cluster == name]
               for name in names}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        sample = rng.choice(names, size=len(names), replace=True)
        observations = [value for name in sample for value in grouped[str(name)]]
        estimates.append(float(np.mean(observations)))
    return [float(value) for value in np.quantile(estimates, [.025, .975])]


def _cluster(row: dict) -> str:
    for key in ("series_id", "match_id", "event_id"):
        value = row.get(key)
        if value is not None and str(value):
            return f"{key}:{value}"
    return f"token:{row.get('token')}"


def _teams(rows: list[dict]) -> set[str]:
    result = set()
    for row in rows:
        value = row.get("teams")
        if isinstance(value, list):
            result.update(str(item) for item in value if item)
        for key in ("team1", "team2", "focal_team", "opponent_team"):
            if row.get(key):
                result.add(str(row[key]))
        outcome = str(row.get("outcome") or "").strip()
        if (row.get("market_type") in {"Map Winner", "Match Winner"}
                and outcome.casefold() not in {"", "yes", "no"}):
            result.add(outcome)
    return result


def summarize(rows: list[dict], *, draws: int = 2000, seed: int = 20260923) -> dict:
    valid = [row for row in rows if row.get("label_valid")
             and _finite(row.get("executable_pnl")) is not None]
    pnl = [float(row["executable_pnl"]) for row in valid]
    net = [float(row["net_drift"]) for row in valid
           if _finite(row.get("net_drift")) is not None]
    positive = sum(value for value in pnl if value > 0)
    negative = -sum(value for value in pnl if value < 0)
    clusters = [_cluster(row) for row in valid]
    ci = clustered_bootstrap(pnl, clusters, draws=draws, seed=seed)
    unique_shocks = {row.get("shock_id") for row in rows}
    maps = {str(row.get("map_id") or row.get("bout_num")) for row in rows
            if row.get("map_id") is not None or row.get("bout_num") is not None}
    if not maps:
        maps = {f"{row.get('event_id')}:{row.get('market_id') or row.get('market')}"
                for row in rows if row.get("market_type") == "Map Winner"
                and (row.get("market_id") is not None or row.get("market"))}
    matches = {str(row.get("match_id")) for row in rows if row.get("match_id") is not None}
    series = {str(row.get("series_id")) for row in rows if row.get("series_id") is not None}
    days = {datetime.fromtimestamp(float(row["decision_ts"]), timezone.utc).date().isoformat()
            for row in rows if _finite(row.get("decision_ts")) is not None}
    if not series:
        series = matches or {str(row.get("event_id")) for row in rows
                             if row.get("event_id") is not None}
    if not matches:
        matches = {str(row.get("event_id")) for row in rows
                   if row.get("event_id") is not None}
    if any(row.get("net_drift") is None for row in valid):
        conclusion = "costs_unknown_no_net_edge_conclusion"
    else:
        net_ci = clustered_bootstrap(net, clusters, draws=draws, seed=seed)
        if net_ci is None or net_ci[0] <= 0 <= net_ci[1]:
            conclusion = "insufficient evidence of positive executable edge"
        elif net_ci[0] > 0:
            conclusion = "positive net drift observed; requires out-of-sample validation"
        else:
            conclusion = "net drift is negative in this cohort"
    return {
        "number_of_observations": len(valid), "number_of_shocks": len(unique_shocks),
        "number_of_maps": len(maps), "number_of_matches": len(matches),
        "number_of_series": len(series), "unique_teams": len(_teams(rows)),
        "decision_days": len(days), "cluster_unit": "series_or_match_or_event",
        "number_of_clusters": len(set(clusters)),
        "mean_executable_pnl": float(np.mean(pnl)) if pnl else None,
        "median_executable_pnl": float(np.median(pnl)) if pnl else None,
        "executable_pnl_95pct_clustered_ci": ci,
        "hit_rate": sum(value > 0 for value in pnl) / len(pnl) if pnl else None,
        "profit_factor": positive / negative if negative > 0 else None,
        "mean_raw_mid_drift": float(np.mean([row["raw_mid_drift"] for row in valid]))
        if valid else None,
        "mean_spread_cost": float(np.mean([row["spread_cost"] for row in valid]))
        if valid else None,
        "mean_estimated_fees": float(np.mean([row["estimated_fees"] for row in valid
                                               if row["estimated_fees"] is not None]))
        if any(row.get("estimated_fees") is not None for row in valid) else None,
        "mean_net_drift": float(np.mean(net)) if net else None,
        "conclusion": conclusion,
    }


STRATA = ("shock_direction", "shock_magnitude_bucket",
          "starting_probability_bucket", "spread_bucket", "liquidity_bucket",
          "obi_bucket", "trade_imbalance_bucket", "volume_bucket",
          "favorite_underdog", "market_type")


def build_report(observations: list[dict], *, draws: int = 2000,
                 seed: int = 20260923) -> dict:
    cohorts = []
    for window in SHOCK_WINDOWS_SECONDS:
        for threshold in SHOCK_THRESHOLDS:
            for horizon in HORIZONS_SECONDS:
                rows = [row for row in observations
                        if row["shock_window_seconds"] == window
                        and row.get(f"passes_{threshold:.2f}")
                        and row["horizon_seconds"] == horizon]
                summary = summarize(rows, draws=draws, seed=seed)
                strata = {}
                for field in STRATA:
                    values = sorted({str(row.get(field, "unknown")) for row in rows})
                    strata[field] = {
                        value: summarize([row for row in rows
                                          if str(row.get(field, "unknown")) == value],
                                         draws=draws, seed=seed)
                        for value in values
                    }
                cohorts.append({
                    "shock_window_seconds": window, "threshold": threshold,
                    "horizon_seconds": horizon, "summary": summary,
                    "strata": strata,
                })
    return {
        "schema_version": SCHEMA_VERSION, "paper_only": True, "promoted": False,
        "experiment": "price_shock_residual_drift",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timing_basis": "local_receive_time",
        "entry_rule": "first ask/bid at or after decision plus configured latency",
        "exit_rule": "first bid/ask at or after decision plus horizon",
        "fee_policy": "unknown stays unknown; never defaulted to zero",
        "slippage_policy": "unknown stays unknown; never defaulted to zero",
        "cohorts": cohorts,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(paths: list[Path], output_root: Path, *, latency_seconds: float = 0.0,
        fee_bps: float | None = None, slippage_per_side: float | None = None,
        max_quote_wait_seconds: float | None = 1.0, draws: int = 2000) -> Path:
    paths = [Path(path) for path in paths]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(output_root) / stamp
    output.mkdir(parents=True, exist_ok=False)
    features = build_market_features(load_realtime(paths))
    shocks = detect_shocks(features)
    observations = executable_drift(
        features, shocks, latency_seconds=latency_seconds, fee_bps=fee_bps,
        slippage_per_side=slippage_per_side,
        max_quote_wait_seconds=max_quote_wait_seconds)
    report = build_report(observations, draws=draws)
    report.update({
        "latency_seconds": latency_seconds, "configured_fee_bps": fee_bps,
        "configured_slippage_per_side": slippage_per_side,
        "max_quote_wait_seconds": max_quote_wait_seconds,
        "inputs_sha256": {str(path): _sha256(path) for path in paths},
        "code_sha256": {
            "market_shocks": _sha256(Path(__file__)),
            "market_features": _sha256(Path(__file__).with_name("market_features.py")),
        },
    })
    observations_path = output / "shock_observations.jsonl"
    with observations_path.open("x", encoding="utf-8") as handle:
        for row in observations:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False,
                                    allow_nan=False) + "\n")
    report["artifacts"] = {"observations": str(observations_path)}
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/market_research/shocks"))
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--slippage-per-side", type=float, default=None)
    parser.add_argument("--max-quote-wait", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args(argv)
    paths = args.paths or sorted(Path("data/realtime").glob("*.jsonl"))
    if not paths:
        raise SystemExit("no realtime JSONL inputs")
    print(run(paths, args.output_root, latency_seconds=args.latency,
              fee_bps=args.fee_bps, slippage_per_side=args.slippage_per_side,
              max_quote_wait_seconds=args.max_quote_wait,
              draws=args.bootstrap_draws))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
