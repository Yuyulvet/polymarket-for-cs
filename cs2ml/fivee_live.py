"""Record the public 5EPlay CS2 event log with auditable receive timing.

Full snapshots are deduplicated locally. Initial and post-outage snapshots are
marked as backfill and excluded from strict executable backtests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from curl_cffi import requests

BASE = "https://esports-data.5eplaycdn.com"
LOG_PATH = "/v1/api/csgo/match/{mid}/event/log"
SNAPSHOT_LIMIT = 500


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def entry_key(entry: dict) -> str:
    """Stable full-entry key; prefixes of log_info can collide."""
    raw = json.dumps(
        [entry.get("update_version"), entry.get("bout_id"), entry.get("log_info")],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def source_version(entry: dict) -> int | None:
    try:
        value = int(entry.get("update_version"))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def ordered_unique_entries(entries: Iterable[dict], seen: set[str]) -> list[dict]:
    fresh: list[tuple[int, int, dict]] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        key = entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        version = source_version(entry)
        fresh.append((version if version is not None else 2**63 - 1, position, entry))
    fresh.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in fresh]


def load_seen(paths: Iterable[Path]) -> set[str]:
    """Load event identities from schema-v1 or schema-v2 sessions."""
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid_jsonl:{path}:{line_number}") from exc
                entry = row.get("entry") if isinstance(row, dict) else None
                if isinstance(entry, dict):
                    seen.add(entry_key(entry))
    return seen


class FiveERecorder:
    """Bounded single-match recorder with persistent retry and gap markers."""

    def __init__(
        self, match_id: str, out_path: Path, *, duration_seconds: float,
        poll_seconds: float = 15.0, request_timeout: float = 20.0,
        max_backoff_seconds: float = 60.0, max_consecutive_errors: int = 0,
        initial_seen: set[str] | None = None, request_get: Callable | None = None,
        wall_clock: Callable[[], str] = utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not str(match_id).strip():
            raise ValueError("match_id_required")
        values = (("duration_seconds", duration_seconds), ("poll_seconds", poll_seconds),
                  ("request_timeout", request_timeout),
                  ("max_backoff_seconds", max_backoff_seconds))
        for name, value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name}_must_be_positive")
        if isinstance(max_consecutive_errors, bool) or max_consecutive_errors < 0:
            raise ValueError("max_consecutive_errors_must_be_nonnegative")
        self.match_id, self.out_path = str(match_id), Path(out_path)
        self.duration_seconds, self.poll_seconds = float(duration_seconds), float(poll_seconds)
        self.request_timeout = float(request_timeout)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.seen, self.resumed = set(initial_seen or ()), bool(initial_seen)
        self.request_get = request_get or requests.get
        self.wall_clock, self.monotonic_clock = wall_clock, monotonic_clock
        self.sleeper = sleeper
        self.url = BASE + LOG_PATH.format(mid=self.match_id)
        self.records_written = self.events_written = self.poll_index = 0

    def _write(self, handle, record_type: str, **fields) -> None:
        row = {"schema_version": 2, "record_type": record_type,
               "recv_utc": self.wall_clock(), "recv_mono": self.monotonic_clock(), **fields}
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        self.records_written += 1

    def _sleep(self, seconds: float, deadline: float) -> None:
        remaining = deadline - self.monotonic_clock()
        if remaining > 0:
            self.sleeper(min(seconds, remaining))

    def run(self) -> int:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = self.monotonic_clock() + self.duration_seconds
        errors, outage_start, backoff, first_success = 0, None, 1.0, True
        with self.out_path.open("a", encoding="utf-8") as handle:
            self._write(handle, "session_start", match_id=self.match_id,
                        resumed=self.resumed, preloaded_event_keys=len(self.seen))
            while self.monotonic_clock() < deadline:
                self.poll_index += 1
                try:
                    response = self.request_get(
                        self.url, params={"update_version": "0", "limit": SNAPSHOT_LIMIT},
                        impersonate="chrome", timeout=self.request_timeout)
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict) or not payload.get("success"):
                        message = payload.get("message") if isinstance(payload, dict) else None
                        raise RuntimeError(f"api_not_success:{message!r}")
                    snapshot = (payload.get("data") or {}).get("list") or []
                    if not isinstance(snapshot, list):
                        raise RuntimeError("api_list_not_array")
                    recovering, initial = errors > 0, first_success
                    fresh = ordered_unique_entries(snapshot, self.seen)
                    latest = max((source_version(e) or -1 for e in snapshot), default=-1)
                    truncated = len(snapshot) >= SNAPSHOT_LIMIT
                    if recovering:
                        self._write(handle, "source_recovered", poll_index=self.poll_index,
                                    outage_seconds=max(0.0, self.monotonic_clock()-outage_start),
                                    consecutive_errors=errors, recovered_new_events=len(fresh),
                                    snapshot_may_be_truncated=truncated)
                    quality = ("initial_or_resume_snapshot" if initial else
                               "recovery_backfill" if recovering else "normal_poll_receive_time")
                    for entry in fresh:
                        self._write(handle, "event", poll_index=self.poll_index,
                                    initial_snapshot=initial, recovery_snapshot=recovering,
                                    decision_eligible=not initial and not recovering,
                                    timing_quality=quality,
                                    source_update_version=source_version(entry), entry=entry)
                        self.events_written += 1
                    self._write(handle, "poll_ok", poll_index=self.poll_index,
                                snapshot_events=len(snapshot), new_events=len(fresh),
                                latest_source_version=None if latest < 0 else latest,
                                snapshot_may_be_truncated=truncated)
                    if fresh:
                        print(f"[{self.wall_clock()}] +{len(fresh)} events "
                              f"(total {self.events_written}, poll {self.poll_index})", flush=True)
                    first_success, errors, outage_start, backoff = False, 0, None, 1.0
                    self._sleep(self.poll_seconds, deadline)
                except Exception as exc:
                    now = self.monotonic_clock()
                    errors += 1
                    outage_start = now if outage_start is None else outage_start
                    delay = min(backoff, self.max_backoff_seconds)
                    self._write(handle, "source_error", poll_index=self.poll_index,
                                error_type=type(exc).__name__, error=str(exc)[:500],
                                consecutive_errors=errors, retry_in_seconds=delay)
                    print(f"[{self.wall_clock()}] source error #{errors}: "
                          f"{str(exc)[:160]}; retry in {delay:g}s", flush=True)
                    if self.max_consecutive_errors and errors >= self.max_consecutive_errors:
                        self._write(handle, "session_end", exit_reason="max_consecutive_errors",
                                    events_written=self.events_written, polls=self.poll_index)
                        return 2
                    self._sleep(delay, deadline)
                    backoff = min(backoff * 2.0, self.max_backoff_seconds)
            self._write(handle, "session_end", exit_reason="deadline",
                        events_written=self.events_written, polls=self.poll_index,
                        ended_during_outage=errors > 0)
        print(f"[{self.wall_clock()}] done, {self.events_written} events -> {self.out_path}", flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=60.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=0,
                        help="0 means keep reconnecting until the time limit")
    parser.add_argument("--resume-from", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", default="data/fivee")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"eventlog_{args.match_id}_{stamp}.jsonl"
    meta_path = out_dir / f"eventlog_{args.match_id}_{stamp}.meta.json"
    initial_seen = load_seen(args.resume_from)
    meta = {
        "schema_version": 2, "match_id": args.match_id,
        "source": "esports-data.5eplaycdn.com",
        "timing_basis": "local_receive_time; source_update_version_preserved_separately",
        "strict_backtest_policy": "exclude_initial_and_recovery_snapshots",
        "poll_seconds": args.poll_seconds, "snapshot_limit": SNAPSHOT_LIMIT,
        "max_consecutive_errors": args.max_consecutive_errors,
        "max_backoff_seconds": args.max_backoff_seconds,
        "resume_from": [str(path) for path in args.resume_from],
        "preloaded_event_keys": len(initial_seen), "started_utc": utc_now(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder = FiveERecorder(
        args.match_id, out_path, duration_seconds=args.hours * 3600.0,
        poll_seconds=args.poll_seconds, request_timeout=args.request_timeout,
        max_backoff_seconds=args.max_backoff_seconds,
        max_consecutive_errors=args.max_consecutive_errors, initial_seen=initial_seen)
    return recorder.run()


if __name__ == "__main__":
    raise SystemExit(main())
