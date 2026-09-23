"""Automated, paper-only CS2 Polymarket market discovery and capture lifecycle.

The collector uses Gamma event/market metadata for scheduling and closure. It
does not infer match state from a public game stream and never authenticates,
connects a wallet, or submits an order.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import time
from typing import Callable

from curl_cffi import requests as creq

from . import config
from .realtime_record import GAMMA, RealtimeRecorder

COLLECTOR_VERSION = "market_collector_v1"
CAPTURE_SCHEMA_VERSION = 1
DEFAULT_ROOT = config.DATA_DIR / "market_captures"

STATES = ("DISCOVERED", "SCHEDULED", "RECORDING", "RECONNECTING", "CLOSED",
          "FINALIZED", "FAILED", "INCOMPLETE")
TRANSITIONS = {
    None: {"DISCOVERED"},
    "DISCOVERED": {"SCHEDULED", "FAILED"},
    "SCHEDULED": {"RECORDING", "CLOSED", "FAILED", "INCOMPLETE"},
    "RECORDING": {"RECONNECTING", "CLOSED", "FAILED", "INCOMPLETE"},
    "RECONNECTING": {"RECORDING", "CLOSED", "FAILED", "INCOMPLETE"},
    "CLOSED": {"FINALIZED"},
    "FINALIZED": set(), "FAILED": set(), "INCOMPLETE": set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _array(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def _timestamp(value) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def is_cs2_event(event: dict) -> bool:
    fields = [event.get("title"), event.get("slug"), event.get("description")]
    for tag in event.get("tags") or []:
        if isinstance(tag, dict):
            fields.extend((tag.get("label"), tag.get("name"), tag.get("slug")))
        else:
            fields.append(tag)
    text = " ".join(str(value or "") for value in fields).casefold()
    patterns = (r"\bcs2\b", r"counter[ -]?strike(?: 2)?", r"\bcsgo\b")
    return any(re.search(pattern, text) for pattern in patterns)


def classify_market(name: str) -> tuple[str, int | None]:
    text = str(name or "").strip()
    folded = text.casefold()
    if re.search(r"\bmatch\s+winner\b|\bseries\s+winner\b", folded):
        return "Match Winner", None
    match = re.search(r"\bmap\s*(\d+)\s+winner\b", folded)
    if match:
        number = int(match.group(1))
        return f"Map {number} Winner", number
    return "Other", None


def capture_identity(event_id, market_id, token_ids) -> str:
    tokens = sorted(str(token) for token in token_ids)
    raw = json.dumps([str(event_id), str(market_id), tokens], separators=(",", ":"))
    suffix = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"event-{event_id}_market-{market_id}_{suffix}"


def market_spec(event: dict, market: dict) -> dict:
    outcomes, tokens = _array(market.get("outcomes")), _array(market.get("clobTokenIds"))
    if not outcomes or len(outcomes) != len(tokens) or len(set(map(str, tokens))) != len(tokens):
        raise ValueError("invalid_market_token_binding")
    name = market.get("groupItemTitle") or market.get("question") or ""
    market_type, map_number = classify_market(name)
    event_id, market_id = event.get("id"), market.get("id")
    if event_id is None or market_id is None:
        raise ValueError("event_and_market_id_required")
    identity = capture_identity(event_id, market_id, tokens)
    unknown = {}
    fields = {
        "condition_id": market.get("conditionId"),
        "event_scheduled_start": event.get("startDate") or market.get("startDate"),
        "seconds_delay": market.get("secondsDelay"),
        "fees_enabled": market.get("feesEnabled"),
        "fee_type": market.get("feeType"),
        "fee_schedule": market.get("feeSchedule"),
        "tick_size": market.get("orderPriceMinTickSize") or market.get("tickSize"),
        "min_order_size": market.get("orderMinSize") or market.get("minOrderSize"),
    }
    for key, value in fields.items():
        if value is None:
            unknown[key] = "not_present_in_market_metadata"
    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "paper_only": True, "promoted": False, "live_trading_enabled": False,
        "capture_id": identity, "event_id": str(event_id),
        "event_title": event.get("title"), "event_slug": event.get("slug"),
        "market_id": str(market_id), "condition_id": fields["condition_id"],
        "market_name": name, "market_type": market_type, "map_number": map_number,
        "outcomes": [str(item) for item in outcomes],
        "tokens": dict(zip(map(str, outcomes), map(str, tokens))),
        **fields,
        "capture_start": None, "capture_end": None,
        "timing_basis": "local_receive_time",
        "unknown_reasons": {
            **unknown, "capture_start": "set_in_manifest_when_recording_starts",
            "capture_end": "set_in_manifest_when_capture_terminates",
        },
    }


def extract_markets(event: dict, *, include_other: bool = True) -> list[dict]:
    if not is_cs2_event(event):
        return []
    result = []
    for market in event.get("markets") or []:
        try:
            spec = market_spec(event, market)
        except ValueError:
            continue
        if include_other or spec["market_type"] != "Other":
            result.append(spec)
    return result


def fetch_active_events(limit: int = 100) -> list[dict]:
    response = creq.get(f"{GAMMA}/events", params={
        "active": "true", "closed": "false", "limit": limit,
        "order": "startDate", "ascending": "true",
    }, impersonate="chrome", timeout=40)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else payload.get("events", [])


def discover_cs2_markets(fetch_events: Callable[[], list[dict]] = fetch_active_events,
                         *, now_ts: float | None = None,
                         lookahead_seconds: float = 48 * 3600) -> list[dict]:
    now_ts = time.time() if now_ts is None else now_ts
    candidates = []
    for event in fetch_events():
        if event.get("closed"):
            continue
        start = _timestamp(event.get("startDate"))
        if start is not None and start > now_ts + lookahead_seconds:
            continue
        candidates.extend(extract_markets(event))
    return sorted(candidates, key=lambda item: (
        _timestamp(item.get("event_scheduled_start")) or 0,
        item["event_id"], item["market_id"]))


class DuplicateCaptureError(RuntimeError):
    pass


class CaptureLifecycle:
    def __init__(self, root: Path, spec: dict, *, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.spec = dict(spec)
        self.clock = clock
        self.path = self.root / self.spec["capture_id"]
        self.state = None
        self.transitions: list[dict] = []

    def claim(self) -> "CaptureLifecycle":
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise DuplicateCaptureError(self.spec["capture_id"]) from exc
        (self.path / "metadata.json").write_text(
            json.dumps(self.spec, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.path / "capture.lock").write_text(
            json.dumps({"capture_id": self.spec["capture_id"], "claimed_at": utc_now()}),
            encoding="utf-8")
        self.transition("DISCOVERED", "capture_claimed")
        return self

    def transition(self, state: str, reason: str, **details) -> None:
        if state not in STATES or state not in TRANSITIONS[self.state]:
            raise ValueError(f"invalid_capture_transition:{self.state}->{state}")
        row = {"schema_version": 1, "capture_id": self.spec["capture_id"],
               "from": self.state, "to": state, "reason": reason,
               "local_ts": self.clock(), "recorded_utc": utc_now(), **details}
        with (self.path / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.state = state
        self.transitions.append(row)
        (self.path / "state.json").write_text(json.dumps(
            {"state": state, "updated_utc": row["recorded_utc"], "reason": reason},
            ensure_ascii=False, indent=2), encoding="utf-8")

    def finalize_manifest(self, *, counters: dict, capture_start: str | None,
                          capture_end: str | None, qa_report: str | None = None) -> dict:
        events = self.path / "events.jsonl"
        digest = None
        if events.is_file():
            sha = hashlib.sha256()
            with events.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    sha.update(chunk)
            digest = sha.hexdigest()
        manifest = {
            "schema_version": 1, "collector_version": COLLECTOR_VERSION,
            "paper_only": True, "promoted": False, "live_trading_enabled": False,
            "capture_id": self.spec["capture_id"], "final_state": self.state,
            "capture_start": capture_start, "capture_end": capture_end,
            "counters": counters, "events_sha256": digest,
            "metadata": "metadata.json", "audit_log": "audit.jsonl",
            "qa_report": qa_report, "transitions": self.transitions,
            "code_sha256": {
                "market_collector": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "realtime_record": hashlib.sha256(
                    Path(__file__).with_name("realtime_record.py").read_bytes()).hexdigest(),
            },
        }
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def _market_closed(event: dict, market_id: str) -> bool:
    if event.get("closed"):
        return True
    market = next((item for item in event.get("markets") or []
                   if str(item.get("id")) == str(market_id)), None)
    return bool(market and market.get("closed"))


def _default_fetch_event(event_id: str) -> dict:
    response = creq.get(f"{GAMMA}/events/{int(event_id)}",
                        impersonate="chrome", timeout=40)
    response.raise_for_status()
    return response.json()


def _default_record_chunk(lifecycle: CaptureLifecycle, seconds: float) -> dict:
    spec = lifecycle.spec
    recorder = RealtimeRecorder(
        {spec["market_name"]: spec["tokens"]}, lifecycle.path / "events.jsonl",
        event_id=spec["event_id"], market_metadata={spec["market_name"]: {
            "market_id": spec["market_id"], "condition_id": spec["condition_id"]}})
    return recorder.run(duration_hours=seconds / 3600.0)


def run_capture(spec: dict, root: Path = DEFAULT_ROOT, *, lead_seconds: float = 300,
                chunk_seconds: float = 30, max_duration_seconds: float = 8 * 3600,
                fetch_event: Callable[[str], dict] = _default_fetch_event,
                record_chunk: Callable[[CaptureLifecycle, float], dict] = _default_record_chunk,
                clock: Callable[[], float] = time.time,
                sleeper: Callable[[float], None] = time.sleep) -> dict:
    lifecycle = CaptureLifecycle(root, spec, clock=clock).claim()
    counters = {"chunks": 0, "events": 0, "messages": 0, "reconnects": 0}
    capture_start = capture_end = None
    lifecycle.transition("SCHEDULED", "metadata_schedule_accepted")
    scheduled = _timestamp(spec.get("event_scheduled_start"))
    start_at = clock() if scheduled is None else max(clock(), scheduled - lead_seconds)
    while clock() < start_at:
        sleeper(min(start_at - clock(), 30.0))
    try:
        event = fetch_event(spec["event_id"])
        if _market_closed(event, spec["market_id"]):
            lifecycle.transition("CLOSED", "metadata_closed_before_recording")
            lifecycle.transition("FINALIZED", "closed_capture_finalized")
        else:
            capture_start = datetime.fromtimestamp(clock(), timezone.utc).isoformat()
            lifecycle.transition("RECORDING", "scheduled_recording_started")
            deadline = clock() + max_duration_seconds
            while clock() < deadline:
                result = record_chunk(lifecycle, min(chunk_seconds, deadline - clock())) or {}
                counters["chunks"] += 1
                for key in ("events", "messages", "reconnects"):
                    counters[key] += int(result.get(key, 0) or 0)
                if result.get("reconnects"):
                    lifecycle.transition("RECONNECTING", "websocket_reconnect_observed",
                                         reconnects=result["reconnects"])
                    lifecycle.transition("RECORDING", "websocket_resubscribed")
                event = fetch_event(spec["event_id"])
                if _market_closed(event, spec["market_id"]):
                    lifecycle.transition("CLOSED", "metadata_market_closed")
                    lifecycle.transition("FINALIZED", "closed_capture_finalized")
                    break
            if lifecycle.state == "RECORDING":
                lifecycle.transition("INCOMPLETE", "maximum_capture_duration_reached")
    except Exception as exc:
        terminal = "INCOMPLETE" if counters["chunks"] or (lifecycle.path / "events.jsonl").exists() else "FAILED"
        lifecycle.transition(terminal, "capture_exception", error_type=type(exc).__name__)
    capture_end = datetime.fromtimestamp(clock(), timezone.utc).isoformat()

    qa_path = None
    events_path = lifecycle.path / "events.jsonl"
    if events_path.is_file():
        from .market_capture_qa import write_qa_report
        try:
            qa_path = str(write_qa_report(lifecycle.path))
        except Exception as exc:
            counters["qa_error_type"] = type(exc).__name__
    return lifecycle.finalize_manifest(
        counters=counters, capture_start=capture_start, capture_end=capture_end,
        qa_report=qa_path)


def collect_discovered(root: Path = DEFAULT_ROOT, *, max_workers: int = 4,
                       fetch_events: Callable[[], list[dict]] = fetch_active_events,
                       **capture_kwargs) -> dict:
    specs = discover_cs2_markets(fetch_events)
    results, duplicates = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_capture, spec, root, **capture_kwargs): spec
                   for spec in specs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except DuplicateCaptureError:
                duplicates.append(futures[future]["capture_id"])
    return {"discovered": len(specs), "captures": results,
            "duplicates_skipped": sorted(duplicates), "paper_only": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--lookahead-hours", type=float, default=48)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    specs = discover_cs2_markets(
        fetch_active_events, lookahead_seconds=args.lookahead_hours * 3600)
    if args.discover_only:
        print(json.dumps(specs, ensure_ascii=False, indent=2))
        return 0
    result = collect_discovered(args.root, max_workers=args.max_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
