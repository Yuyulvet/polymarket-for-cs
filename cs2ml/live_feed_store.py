"""Bounded local raw-receipt journal; no networking, parsing or trading.

The receipt timestamp means entry into this recorder, not source publication
time or the operating system's packet-arrival time. Payload text is inert data.
Unknown source timestamps stay null. Repeated arrivals are deliberately kept.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Callable


class RecordingLimitError(RuntimeError):
    """A record would exceed the configured record count or total byte budget."""


def _json_bytes(value) -> bytes:
    # Strict JSON rejects NaN/Infinity and unserializable Python objects.
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _utc_receipt(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("receipt_clock_requires_timezone_aware_datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class RawFeedRecorder:
    """Create a fresh session and append strict-JSON arrivals without deduplication.

    max_bytes includes metadata.json plus events.jsonl, using actual UTF-8 byte
    counts. Limits are checked before writing; they never truncate a payload.
    Writes are flushed, not fsynced. An I/O failure closes the recorder; a partial
    last record may remain for inspection and is never silently resumed.

    Optional clocks support deterministic tests. The caller supplies source
    timestamps as received; no source time, unit, timezone or offset is guessed.
    """

    def __init__(self, directory: Path, metadata: dict, max_records=10000,
                 max_bytes=16 * 1024 * 1024, *, wall_clock=None, monotonic_clock=None):
        for name, value in (("max_records", max_records), ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}_must_be_positive_integer")
        if not isinstance(metadata, dict):
            raise TypeError("metadata_must_be_dict")
        for key in ("source", "match_url"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(f"metadata_requires_{key}")
        match_id = metadata.get("hltv_match_id")
        if (isinstance(match_id, bool) or not isinstance(match_id, (str, int))
                or not str(match_id).isdigit() or int(match_id) <= 0):
            raise ValueError("metadata_requires_positive_hltv_match_id")
        self.directory = Path(directory)
        self.max_records, self.max_bytes = max_records, max_bytes
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or time.monotonic_ns
        self._lock = threading.Lock()
        self._closed = False
        self._records_written = 0
        self._last_monotonic = None
        manifest = {"schema_version": 1, "session_started_at": _utc_receipt(self._wall_clock),
                    "metadata": metadata,
                    "limits": {"max_records": max_records, "max_bytes": max_bytes,
                               "byte_budget_scope": "metadata_and_events_utf8"},
                    "receipt_time_basis": "recorder_append_not_source_or_network_packet_time"}
        encoded = _json_bytes(manifest)
        if len(encoded) > max_bytes:
            raise RecordingLimitError("max_bytes_exceeded_by_metadata")
        # Existing directories, files and symlinks are never reused, even empty.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata_path = self.directory / "metadata.json"
        self.events_path = self.directory / "events.jsonl"
        with self.metadata_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
        self._handle = self.events_path.open("xb")
        self._bytes_written = len(encoded)

    @property
    def records_written(self):
        return self._records_written

    @property
    def bytes_written(self):
        return self._bytes_written

    @property
    def closed(self):
        return self._closed

    def append(self, event_type, payload, source_timestamp=None):
        """Record one arrival/status/error; return the detached persisted record."""
        with self._lock:
            if self._closed:
                raise ValueError("recorder_is_closed")
            if self._records_written >= self.max_records:
                raise RecordingLimitError("max_records_reached")
            received_at = _utc_receipt(self._wall_clock)
            monotonic_ns = self._monotonic_clock()
            if (isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int)
                    or monotonic_ns < 0
                    or (self._last_monotonic is not None and monotonic_ns < self._last_monotonic)):
                raise ValueError("invalid_or_decreasing_monotonic_clock")
            if not isinstance(event_type, str) or not event_type.strip():
                raise ValueError("event_type_must_be_nonempty_string")
            record = {"sequence": self._records_written + 1, "event_type": event_type,
                      "received_at": received_at, "monotonic_ns": monotonic_ns,
                      "source_timestamp": source_timestamp, "payload": payload}
            encoded = _json_bytes(record)
            if self._bytes_written + len(encoded) > self.max_bytes:
                raise RecordingLimitError("max_bytes_reached")
            # Return a deep JSON copy, so caller mutation cannot alter the receipt.
            detached = json.loads(encoded)
            try:
                written = self._handle.write(encoded)
                if written != len(encoded):
                    raise OSError("incomplete_feed_record_write")
                self._handle.flush()
            except OSError:
                self._closed = True
                try:
                    self._handle.close()
                except OSError:
                    pass  # Preserve the original write/flush failure.
                raise
            self._records_written += 1
            self._bytes_written += len(encoded)
            self._last_monotonic = monotonic_ns
            return detached

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._handle.close()

    def __enter__(self):
        if self._closed:
            raise ValueError("recorder_is_closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
