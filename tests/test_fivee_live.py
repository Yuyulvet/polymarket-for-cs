from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.fivee_live import FiveERecorder, entry_key, load_seen, ordered_unique_entries


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.start = datetime(2026, 9, 17, 12, tzinfo=timezone.utc)

    def monotonic(self):
        return self.value

    def wall(self):
        return (self.start + timedelta(seconds=self.value)).isoformat()

    def sleep(self, seconds):
        self.value += seconds


class Response:
    def __init__(self, entries):
        self.entries = entries

    def raise_for_status(self):
        return None

    def json(self):
        return {"success": True, "data": {"list": self.entries}}


def event(version, info="event"):
    return {"update_version": str(version), "bout_id": "match_1", "log_info": info}


class FiveEIdentityTests(unittest.TestCase):
    def test_full_log_info_is_part_of_identity_and_entries_are_source_sorted(self):
        first = event(2, "same prefix " + "a" * 80)
        second = event(1, "same prefix " + "b" * 80)
        self.assertNotEqual(entry_key(first), entry_key(second))
        seen = set()
        self.assertEqual(ordered_unique_entries([first, second, first], seen), [second, first])
        self.assertEqual(ordered_unique_entries([second], seen), [])

    def test_resume_loader_accepts_old_and_new_envelopes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            rows = [{"recv_utc": "old", "entry": event(1)},
                    {"schema_version": 2, "record_type": "event", "entry": event(2)}]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            self.assertEqual(load_seen([path]), {entry_key(event(1)), entry_key(event(2))})


class FiveEReconnectTests(unittest.TestCase):
    def test_outage_reconnects_and_backfill_is_not_decision_eligible(self):
        clock = FakeClock()
        calls = iter([
            Response([event(2), event(1)]),
            OSError("temporary disconnect"),
            Response([event(3), event(2), event(1)]),
            Response([event(4), event(3), event(2), event(1)]),
        ])

        def get(*args, **kwargs):
            result = next(calls)
            if isinstance(result, Exception):
                raise result
            return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = FiveERecorder(
                "match", path, duration_seconds=3.5, poll_seconds=1,
                request_get=get, wall_clock=clock.wall,
                monotonic_clock=clock.monotonic, sleeper=clock.sleep,
            )
            self.assertEqual(recorder.run(), 0)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        types = [row["record_type"] for row in rows]
        self.assertIn("source_error", types)
        self.assertIn("source_recovered", types)
        events = [row for row in rows if row["record_type"] == "event"]
        self.assertEqual([row["source_update_version"] for row in events], [1, 2, 3, 4])
        self.assertEqual([row["decision_eligible"] for row in events], [False, False, False, True])
        self.assertTrue(events[2]["recovery_snapshot"])
        recovered = next(row for row in rows if row["record_type"] == "source_recovered")
        self.assertEqual(recovered["recovered_new_events"], 1)
        self.assertEqual(rows[-1]["record_type"], "session_end")

    def test_optional_error_limit_is_explicit_not_the_default(self):
        clock = FakeClock()

        def broken(*args, **kwargs):
            raise OSError("offline")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = FiveERecorder(
                "match", path, duration_seconds=10, max_consecutive_errors=2,
                request_get=broken, wall_clock=clock.wall,
                monotonic_clock=clock.monotonic, sleeper=clock.sleep,
            )
            self.assertEqual(recorder.run(), 2)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["exit_reason"], "max_consecutive_errors")


if __name__ == "__main__":
    unittest.main()
