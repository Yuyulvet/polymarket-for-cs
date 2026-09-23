"""Time-safe microstructure features from normalized realtime JSONL.

``local_ts`` (the recorder's local receive time) is the sole ordering clock.
Source timestamps are retained as evidence but never used to order observations.
Each emitted row is computed incrementally, so it can only depend on records at
or before that row's decision time.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

WINDOWS_SECONDS = (1, 3, 5, 10, 30)
RETURN_WINDOWS_SECONDS = (1, 3, 5)
SCHEMA_VERSION = 1


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _timestamp(row: dict) -> float | None:
    value = _number(row.get("local_ts"))
    if value is not None:
        return value
    # Compatibility for locally received raw captures. Never use source_ts.
    text = row.get("received_at") or row.get("recv_utc")
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _levels(value, *, reverse: bool) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for level in value:
        try:
            if isinstance(level, dict):
                price, size = level.get("price"), level.get("size")
            else:
                price, size = level[0], level[1]
            price, size = _number(price), _number(size)
        except (IndexError, TypeError):
            continue
        if price is not None and size is not None and size > 0:
            result.append({"price": price, "size": size})
    result.sort(key=lambda item: item["price"], reverse=reverse)
    return result[:5]


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _imbalance(bid, ask):
    if bid is None or ask is None:
        return None
    return _safe_ratio(bid - ask, bid + ask)


def _past(history: list[dict], timestamp: float) -> dict | None:
    """Last observation with local receive timestamp <= timestamp."""
    index = bisect_right(history, timestamp, key=lambda item: item["ts"]) - 1
    return history[index] if index >= 0 else None


def _realized_volatility(history: list[dict], now: float, window: float):
    if not history or history[-1].get("mid") is None:
        return None
    anchor = _past(history, now - window)
    if anchor is None or anchor.get("mid") is None:
        return None
    variance = history[-1]["cum_squared_mid_change"] - anchor["cum_squared_mid_change"]
    return math.sqrt(max(variance, 0.0))


def _trade_window(history: list[dict], now: float, window: float) -> dict:
    """Cumulative trade statistics for the half-open window (now-window, now]."""
    if not history:
        return {"buy": 0.0, "sell": 0.0, "volume": 0.0, "count": 0}
    end = history[-1]
    index = bisect_right(history, now - window,
                         key=lambda item: item["ts"]) - 1
    start = history[index] if index >= 0 else None
    return {key: end[key] - (start[key] if start else 0)
            for key in ("buy", "sell", "volume", "count")}


def _trade_side(side) -> int:
    value = str(side or "").strip().lower()
    if value in {"buy", "b", "bid"}:
        return 1
    if value in {"sell", "s", "ask"}:
        return -1
    return 0


def _update_quote(state: dict, row: dict, timestamp: float) -> bool:
    bids = _levels(row.get("bids_l5"), reverse=True)
    asks = _levels(row.get("asks_l5"), reverse=False)
    mappings = {
        "best_bid": _number(row.get("best_bid")),
        "best_ask": _number(row.get("best_ask")),
        "bid_size_l1": _number(row.get("bid_size_l1")),
        "ask_size_l1": _number(row.get("ask_size_l1")),
        "bid_depth_l5": _number(row.get("bid_depth_l5")),
        "ask_depth_l5": _number(row.get("ask_depth_l5")),
    }
    if bids:
        mappings["best_bid"] = bids[0]["price"]
        mappings["bid_size_l1"] = bids[0]["size"]
        mappings["bid_depth_l5"] = sum(level["size"] for level in bids)
    if asks:
        mappings["best_ask"] = asks[0]["price"]
        mappings["ask_size_l1"] = asks[0]["size"]
        mappings["ask_depth_l5"] = sum(level["size"] for level in asks)
    changed = False
    for key, value in mappings.items():
        if value is not None:
            state[key] = value
            changed = True
    if changed:
        state["quote_ts"] = timestamp
    return changed


def build_market_features(records: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Build one causal feature row per valid input record.

    Ties in ``local_ts`` preserve input order. A future suffix therefore cannot
    change any feature in an already processed prefix.
    """
    rows = records.to_dict("records") if isinstance(records, pd.DataFrame) else list(records)
    ordered = []
    for sequence, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        timestamp = _timestamp(row)
        token = str(row.get("token") or row.get("asset_id") or "")
        if timestamp is None or not token:
            continue
        ordered.append((timestamp, sequence, token, row))
    ordered.sort(key=lambda item: (item[0], item[1]))

    states: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    trades: dict[str, list[dict]] = {}
    output = []
    for timestamp, sequence, token, row in ordered:
        state = states.setdefault(token, {})
        history = histories.setdefault(token, [])
        tape = trades.setdefault(token, [])
        _update_quote(state, row, timestamp)

        trade_price = _number(row.get("trade_price"))
        trade_size = _number(row.get("trade_size"))
        trade_side = row.get("trade_side")
        if row.get("type") == "trade":
            trade_price = trade_price if trade_price is not None else _number(row.get("price"))
            trade_size = trade_size if trade_size is not None else _number(row.get("size"))
            trade_side = trade_side if trade_side is not None else row.get("side")
        if trade_price is not None and trade_size is not None and trade_size >= 0:
            side = _trade_side(trade_side)
            previous = tape[-1] if tape else {key: 0 for key in
                                              ("buy", "sell", "volume", "count")}
            tape.append({
                "ts": timestamp,
                "buy": previous["buy"] + (trade_size if side > 0 else 0),
                "sell": previous["sell"] + (trade_size if side < 0 else 0),
                "volume": previous["volume"] + trade_size,
                "count": previous["count"] + 1,
            })

        bid, ask = state.get("best_bid"), state.get("best_ask")
        bid1, ask1 = state.get("bid_size_l1"), state.get("ask_size_l1")
        bid5, ask5 = state.get("bid_depth_l5"), state.get("ask_depth_l5")
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        spread = ask - bid if bid is not None and ask is not None else None
        microprice = None
        if bid is not None and ask is not None and bid1 is not None and ask1 is not None:
            microprice = _safe_ratio(ask * bid1 + bid * ask1, bid1 + ask1)
        snapshot = {
            "ts": timestamp, "mid": mid, "spread": spread,
            "bid_size_l1": bid1, "ask_size_l1": ask1,
            "bid_depth_l5": bid5, "ask_depth_l5": ask5,
        }
        previous_mid = history[-1].get("mid") if history else None
        previous_cumulative = history[-1]["cum_squared_mid_change"] if history else 0.0
        squared_change = ((mid - previous_mid) ** 2
                          if mid is not None and previous_mid is not None else 0.0)
        snapshot["cum_squared_mid_change"] = previous_cumulative + squared_change
        history.append(snapshot)

        feature = {
            "schema_version": SCHEMA_VERSION, "paper_only": True,
            "decision_ts": timestamp, "local_ts": timestamp,
            "source_ts": _number(row.get("source_ts")), "sequence": sequence,
            "event_id": row.get("event_id"), "market_id": row.get("market_id"),
            "condition_id": row.get("condition_id"), "market": row.get("market"),
            "taker_fee_bps": _number(row.get("taker_fee_bps")),
            "market_type": row.get("market_type"), "match_id": row.get("match_id"),
            "series_id": row.get("series_id"), "map_id": row.get("map_id"),
            "teams": row.get("teams"), "token": token, "outcome": row.get("outcome"),
            "record_type": row.get("type"), "best_bid": bid, "best_ask": ask,
            "spread": spread, "mid": mid, "midpoint": mid, "microprice": microprice,
            "bid_size_l1": bid1, "ask_size_l1": ask1,
            "bid_depth_l5": bid5, "ask_depth_l5": ask5,
            "obi_1": _imbalance(bid1, ask1), "obi_5": _imbalance(bid5, ask5),
            "microprice_mid_divergence": (
                microprice - mid if microprice is not None and mid is not None else None),
            "quote_age_seconds": (
                timestamp - state["quote_ts"] if state.get("quote_ts") is not None else None),
            "trade_price": trade_price, "trade_size": trade_size,
            "trade_side": trade_side,
        }
        for window in WINDOWS_SECONDS:
            active = _trade_window(tape, timestamp, window)
            buy, sell = active["buy"], active["sell"]
            known_volume = buy + sell
            feature[f"trade_imbalance_{window}s"] = _safe_ratio(
                buy - sell, known_volume)
            feature[f"volume_{window}s"] = active["volume"]
            feature[f"trades_per_second_{window}s"] = active["count"] / window
            prior = _past(history, timestamp - window)
            feature[f"spread_change_{window}s"] = (
                spread - prior["spread"] if prior and spread is not None
                and prior.get("spread") is not None else None)
            for side in ("bid", "ask"):
                key = f"{side}_depth_l5"
                previous = prior.get(key) if prior else None
                current = snapshot.get(key)
                feature[f"{side}_depth_depletion_{window}s"] = (
                    _safe_ratio(previous - current, previous)
                    if previous is not None and current is not None else None)
        for window in RETURN_WINDOWS_SECONDS:
            prior = _past(history, timestamp - window)
            value = (mid - prior["mid"] if prior and mid is not None
                     and prior.get("mid") is not None else None)
            feature[f"return_{window}s"] = value
            feature[f"price_velocity_{window}s"] = (
                value / window if value is not None else None)
        for window in (5, 10, 30):
            feature[f"realized_volatility_{window}s"] = _realized_volatility(
                history, timestamp, window)
        p1, p2 = _past(history, timestamp - 1), _past(history, timestamp - 2)
        recent_velocity = feature.get("price_velocity_1s")
        previous_velocity = (p1["mid"] - p2["mid"] if p1 and p2
                             and p1.get("mid") is not None and p2.get("mid") is not None
                             else None)
        feature["price_velocity"] = recent_velocity
        feature["price_acceleration"] = (
            recent_velocity - previous_velocity
            if recent_velocity is not None and previous_velocity is not None else None)
        output.append(feature)
    return pd.DataFrame(output)


def load_realtime(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in paths:
        path = Path(path)
        metadata = {}
        metadata_path = path.with_suffix(".meta.json")
        if metadata_path.is_file():
            try:
                document = json.loads(metadata_path.read_text(encoding="utf-8"))
                for market in document.get("markets") or []:
                    fee = market.get("taker_fee_bps", market.get("taker_base_fee"))
                    for token in (market.get("tokens") or {}).values():
                        metadata[str(token)] = {
                            "event_id": document.get("event_id"),
                            "market_id": market.get("market_id"),
                            "condition_id": market.get("condition_id"),
                            "taker_fee_bps": fee,
                        }
            except (json.JSONDecodeError, OSError, TypeError):
                metadata = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    supplement = metadata.get(str(row.get("token") or ""), {})
                    for key, value in supplement.items():
                        if row.get(key) is None and value is not None:
                            row[key] = value
                    rows.append(row)
    return rows


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    try:
        return None if pd.isna(value) else value.item() if hasattr(value, "item") else value
    except (TypeError, ValueError):
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(paths: Iterable[Path], output_root: Path) -> Path:
    paths = [Path(path) for path in paths]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(output_root) / stamp
    output.mkdir(parents=True, exist_ok=False)
    frame = build_market_features(load_realtime(paths))
    records_path = output / "features.jsonl"
    with records_path.open("x", encoding="utf-8") as handle:
        for row in frame.to_dict("records"):
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False,
                                    allow_nan=False) + "\n")
    module_path = Path(__file__)
    report = {
        "schema_version": SCHEMA_VERSION, "paper_only": True, "promoted": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timing_basis": "local_receive_time", "rows": len(frame),
        "inputs_sha256": {str(path): _sha256(path) for path in paths},
        "code_sha256": _sha256(module_path),
        "artifacts": {"features": str(records_path)},
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/market_research/features"))
    args = parser.parse_args(argv)
    paths = args.paths or sorted(Path("data/realtime").glob("*.jsonl"))
    if not paths:
        raise SystemExit("no realtime JSONL inputs")
    print(run(paths, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
