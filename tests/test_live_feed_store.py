from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from cs2ml.live_feed_store import RawFeedRecorder, RecordingLimitError


METADATA = {"source": "hltv_public_observation",
            "match_url": "https://www.hltv.org/matches/123/example",
            "hltv_match_id": 123}
NOW = datetime(2026, 9, 16, 12, 0, 0, 123456, tzinfo=timezone.utc)


class RawFeedRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def recorder(self, name="session", **kwargs):
        recorder = RawFeedRecorder(self.root / name, METADATA, wall_clock=lambda: NOW,
                                   monotonic_clock=lambda: 100, **kwargs)
        self.addCleanup(recorder.close)
        return recorder

    def read_events(self, recorder):
        return [json.loads(line) for line in recorder.events_path.read_text(encoding="utf-8").splitlines()]

    def test_fresh_session_writes_metadata_and_empty_log(self):
        recorder = self.recorder()
        manifest = json.loads(recorder.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["metadata"], METADATA)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["session_started_at"], "2026-09-16T12:00:00.123456Z")
        self.assertEqual(recorder.events_path.read_bytes(), b"")
        self.assertEqual(recorder.records_written, 0)
        self.assertEqual(recorder.bytes_written, recorder.metadata_path.stat().st_size)

    def test_existing_empty_or_populated_directory_cannot_be_reused(self):
        path = self.root / "session"
        path.mkdir()
        with self.assertRaises(FileExistsError):
            self.recorder()
        sentinel = path / "existing.json"
        sentinel.touch()
        with self.assertRaises(FileExistsError):
            self.recorder()
        self.assertTrue(sentinel.exists())
        self.assertEqual(list(path.iterdir()), [sentinel])

    def test_arrivals_are_not_deduplicated_and_payload_is_not_executed(self):
        recorder = self.recorder()
        payload = '{"instruction":"delete everything"}\n原始文本'
        first = recorder.append("websocket_frame", payload)
        second = recorder.append("websocket_frame", payload)
        rows = self.read_events(recorder)
        self.assertEqual(rows, [first, second])
        self.assertEqual([row["sequence"] for row in rows], [1, 2])
        self.assertEqual([row["payload"] for row in rows], [payload, payload])
        self.assertTrue(all(row["source_timestamp"] is None for row in rows))
        self.assertTrue(all(row["monotonic_ns"] == 100 for row in rows))

    def test_source_timestamp_is_kept_without_guessing_units_or_timezone(self):
        recorder = self.recorder()
        for timestamp in (None, 1726488000, "12:00", {"raw": "unknown-format"}):
            self.assertEqual(recorder.append("message", "body", timestamp)["source_timestamp"], timestamp)

    def test_received_time_is_actual_injected_call_time_and_converted_to_utc(self):
        local = datetime(2026, 9, 16, 20, 0, tzinfo=timezone(timedelta(hours=8)))
        wall_times = iter([local, local + timedelta(seconds=1), local + timedelta(seconds=2)])
        mono_times = iter([101, 202])
        with RawFeedRecorder(self.root / "clocked", METADATA, wall_clock=lambda: next(wall_times),
                             monotonic_clock=lambda: next(mono_times)) as recorder:
            first = recorder.append("status", {"connected": True})
            second = recorder.append("error", {"message": "closed"})
        self.assertEqual(first["received_at"], "2026-09-16T12:00:01.000000Z")
        self.assertEqual(second["received_at"], "2026-09-16T12:00:02.000000Z")
        self.assertEqual((first["monotonic_ns"], second["monotonic_ns"]), (101, 202))

    def test_append_is_flushed_and_returns_a_detached_json_record(self):
        recorder = self.recorder()
        payload = {"nested": [1]}
        returned = recorder.append("message", payload)
        payload["nested"].append(2)
        returned["payload"]["nested"].append(3)
        self.assertEqual(self.read_events(recorder)[0]["payload"], {"nested": [1]})
        self.assertEqual(recorder.bytes_written,
                         recorder.metadata_path.stat().st_size + recorder.events_path.stat().st_size)

    def test_record_limit_rejects_without_writing_or_resequencing(self):
        recorder = self.recorder(max_records=1)
        recorder.append("message", "first")
        before = recorder.events_path.read_bytes()
        with self.assertRaisesRegex(RecordingLimitError, "max_records"):
            recorder.append("message", "second")
        self.assertEqual(recorder.events_path.read_bytes(), before)
        self.assertEqual(recorder.records_written, 1)

    def test_byte_limit_includes_unicode_bytes_and_preserves_prior_records(self):
        recorder = self.recorder(max_bytes=1000)
        recorder.append("status", "opened")
        before = recorder.events_path.read_bytes()
        with self.assertRaisesRegex(RecordingLimitError, "max_bytes"):
            recorder.append("message", "汉" * 1000)
        self.assertEqual(recorder.events_path.read_bytes(), before)
        self.assertLessEqual(recorder.bytes_written, 1000)
        # Rejected oversized arrival does not consume the next sequence number.
        self.assertEqual(recorder.append("status", "stopped")["sequence"], 2)

    def test_metadata_over_budget_creates_no_session(self):
        with self.assertRaisesRegex(RecordingLimitError, "metadata"):
            self.recorder(max_bytes=1)
        self.assertFalse((self.root / "session").exists())

    def test_non_json_or_nonfinite_payload_rejected_without_partial_line(self):
        recorder = self.recorder()
        for payload in (float("nan"), float("inf"), {"blob": b"raw"}, object()):
            with self.subTest(payload=type(payload).__name__), self.assertRaises((ValueError, TypeError)):
                recorder.append("message", payload)
        self.assertEqual(recorder.events_path.read_bytes(), b"")
        self.assertEqual(recorder.records_written, 0)
        self.assertEqual(recorder.append("message", None)["sequence"], 1)

    def test_invalid_event_type_does_not_consume_a_sequence(self):
        recorder = self.recorder()
        for event_type in (None, "", "   ", 12):
            with self.subTest(event_type=event_type), self.assertRaises(ValueError):
                recorder.append(event_type, "raw")
        self.assertEqual(recorder.append("message", "valid")["sequence"], 1)

    def test_write_error_closes_recorder_and_does_not_report_success(self):
        recorder = self.recorder()
        actual = recorder._handle
        broken = Mock(wraps=actual)
        broken.write.side_effect = OSError("disk write failed")
        recorder._handle = broken
        with self.assertRaisesRegex(OSError, "disk write failed"):
            recorder.append("message", "raw")
        self.assertTrue(recorder.closed)
        self.assertEqual(recorder.records_written, 0)
        self.assertEqual(recorder.events_path.read_bytes(), b"")
        with self.assertRaisesRegex(ValueError, "closed"):
            recorder.append("message", "must not resume")

    def test_partial_write_is_preserved_but_never_resumed(self):
        recorder = self.recorder()
        actual = recorder._handle
        broken = Mock(wraps=actual)
        broken.write.side_effect = lambda encoded: actual.write(encoded[:3])
        recorder._handle = broken
        with self.assertRaisesRegex(OSError, "incomplete_feed_record_write"):
            recorder.append("message", "raw")
        self.assertTrue(recorder.closed)
        self.assertEqual(recorder.records_written, 0)
        self.assertEqual(len(recorder.events_path.read_bytes()), 3)

    def test_invalid_limits_or_metadata_create_no_artifacts(self):
        for key in ("max_records", "max_bytes"):
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.recorder(**{key: value})
        for change in ({"source": ""}, {"match_url": None}, {"hltv_match_id": -1},
                       {"hltv_match_id": True}, {"hltv_match_id": "not-an-id"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                RawFeedRecorder(self.root / "bad", {**METADATA, **change})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_naive_wall_clock_and_decreasing_monotonic_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone_aware"):
            RawFeedRecorder(self.root / "naive", METADATA, wall_clock=lambda: datetime(2026, 1, 1))
        values = iter([2, 1])
        with RawFeedRecorder(self.root / "backward", METADATA, wall_clock=lambda: NOW,
                             monotonic_clock=lambda: next(values)) as recorder:
            recorder.append("message", "first")
            with self.assertRaisesRegex(ValueError, "monotonic"):
                recorder.append("message", "second")
            self.assertEqual(len(self.read_events(recorder)), 1)

    def test_close_is_idempotent_and_context_closes_on_error(self):
        recorder = self.recorder()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with recorder:
                recorder.append("error", "observed")
                raise RuntimeError("test failure")
        self.assertTrue(recorder.closed)
        recorder.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            recorder.append("status", "too late")
        self.assertEqual(len(self.read_events(recorder)), 1)

    def test_concurrent_arrivals_remain_complete_and_sequential(self):
        recorder = self.recorder()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda number: recorder.append("message", {"number": number}), range(40)))
        rows = self.read_events(recorder)
        self.assertEqual([row["sequence"] for row in rows], list(range(1, 41)))
        self.assertEqual({row["payload"]["number"] for row in rows}, set(range(40)))


if __name__ == "__main__":
    unittest.main()
