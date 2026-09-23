from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.market_collector import (CaptureLifecycle, DuplicateCaptureError,
                                    capture_identity, classify_market,
                                    discover_cs2_markets, extract_markets,
                                    market_spec, run_capture)


def event(closed=False):
    return {"id": "100", "title": "Alpha vs Beta - CS2", "slug": "alpha-beta-cs2",
            "startDate": "2026-09-23T00:00:00Z", "closed": closed,
            "markets": [{"id": "200", "conditionId": "0xabc",
                         "groupItemTitle": "Map 2 Winner",
                         "outcomes": '["Alpha", "Beta"]',
                         "clobTokenIds": '["t1", "t2"]', "closed": closed,
                         "secondsDelay": 1, "feesEnabled": True}]}


class Clock:
    def __init__(self, value=1_800_000_000.0):
        self.value = value

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class MarketCollectorTests(unittest.TestCase):
    def test_market_classification_and_discovery(self):
        self.assertEqual(classify_market("Match Winner"), ("Match Winner", None))
        self.assertEqual(classify_market("Map 3 Winner"), ("Map 3 Winner", 3))
        self.assertEqual(classify_market("Total Maps"), ("Other", None))
        specs = discover_cs2_markets(lambda: [event()], now_ts=1_700_000_000,
                                     lookahead_seconds=200_000_000)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["map_number"], 2)
        self.assertTrue(specs[0]["paper_only"])

    def test_capture_identity_is_order_independent(self):
        self.assertEqual(capture_identity("1", "2", ["b", "a"]),
                         capture_identity("1", "2", ["a", "b"]))

    def test_duplicate_capture_is_rejected_atomically(self):
        spec = market_spec(event(), event()["markets"][0])
        with tempfile.TemporaryDirectory() as directory:
            first = CaptureLifecycle(Path(directory), spec).claim()
            self.assertEqual(first.state, "DISCOVERED")
            with self.assertRaises(DuplicateCaptureError):
                CaptureLifecycle(Path(directory), spec).claim()

    def test_lifecycle_records_reconnect_then_closes(self):
        spec = market_spec(event(), event()["markets"][0])
        clock = Clock()
        calls = {"fetch": 0}

        def fetch(_):
            calls["fetch"] += 1
            return event(closed=calls["fetch"] >= 2)

        def record(lifecycle, seconds):
            clock.value += min(seconds, 1)
            (lifecycle.path / "events.jsonl").write_text(json.dumps({
                "type": "book", "local_ts": clock.value, "source_ts": clock.value,
                "token": "t1", "best_bid": .4, "best_ask": .42,
                "bid_size_l1": 10, "ask_size_l1": 10,
                "bid_depth_l5": 50, "ask_depth_l5": 50}) + "\n", encoding="utf-8")
            return {"events": 2, "messages": 1, "reconnects": 1}

        with tempfile.TemporaryDirectory() as directory:
            manifest = run_capture(spec, Path(directory), lead_seconds=0,
                                   chunk_seconds=1, max_duration_seconds=10,
                                   fetch_event=fetch, record_chunk=record,
                                   clock=clock, sleeper=clock.sleep)
            audit = [json.loads(line) for line in (
                Path(directory) / spec["capture_id"] / "audit.jsonl").read_text().splitlines()]
        self.assertEqual(manifest["final_state"], "FINALIZED")
        self.assertEqual([row["to"] for row in audit],
                         ["DISCOVERED", "SCHEDULED", "RECORDING", "RECONNECTING",
                          "RECORDING", "CLOSED", "FINALIZED"])
        self.assertEqual(manifest["counters"]["reconnects"], 1)
        self.assertTrue(manifest["qa_report"])

    def test_non_cs2_event_is_ignored(self):
        fixture = event(); fixture["title"] = "Election winner"; fixture["slug"] = "election"
        self.assertEqual(extract_markets(fixture), [])

    def test_capture_exception_is_explicit_failed_state(self):
        spec = market_spec(event(), event()["markets"][0])
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            manifest = run_capture(
                spec, Path(directory), lead_seconds=0, max_duration_seconds=1,
                fetch_event=lambda _: event(),
                record_chunk=lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
                clock=clock, sleeper=clock.sleep)
        self.assertEqual(manifest["final_state"], "FAILED")
        self.assertEqual(manifest["transitions"][-1]["reason"], "capture_exception")


if __name__ == "__main__":
    unittest.main()
