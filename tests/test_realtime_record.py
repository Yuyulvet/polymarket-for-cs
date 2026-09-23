from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from cs2ml.realtime_record import RealtimeRecorder, tokens_from_event


class RealtimeRecorderTests(unittest.TestCase):
    def test_top_normal_bids_and_asks(self):
        levels = [["0.40", "2"], ["0.55", "3"], ["0.50", "4"]]
        self.assertEqual(RealtimeRecorder._top(levels, "bids"), 0.55)
        self.assertEqual(RealtimeRecorder._top(levels, "asks"), 0.40)

    def test_top_empty_levels(self):
        self.assertIsNone(RealtimeRecorder._top([], "bids"))
        self.assertIsNone(RealtimeRecorder._top(None, "asks"))

    def test_top_skips_malformed_levels(self):
        levels = [None, [], ["bad", "2"], {"size": "3"}, ["0.42", "4"]]
        self.assertEqual(RealtimeRecorder._top(levels, "bids"), 0.42)

    def test_top_supports_dict_levels(self):
        levels = [{"price": "0.61", "size": "8"},
                  {"price": "0.59", "size": "9"}]
        self.assertEqual(RealtimeRecorder._top(levels, "bids"), 0.61)
        self.assertEqual(RealtimeRecorder._top(levels, "asks"), 0.59)

    def test_top_unexpected_payload(self):
        self.assertIsNone(RealtimeRecorder._top({"price": "0.5"}, "bids"))
        self.assertIsNone(RealtimeRecorder._top([{"price": "0.5"}], "unknown"))

    def test_tokens_preserve_outcome_order(self):
        event = {"markets": [{
            "groupItemTitle": "Map 2 Winner",
            "outcomes": '["Alpha", "Beta"]',
            "clobTokenIds": '["token-a", "token-b"]',
        }]}
        self.assertEqual(tokens_from_event(event), {
            "Map 2 Winner": {"Alpha": "token-a", "Beta": "token-b"},
        })

    def test_rows_include_event_and_receive_and_source_times(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.jsonl"
            recorder = RealtimeRecorder(
                {"Map 2 Winner": {"Alpha": "token-a"}}, path, event_id="123")
            recorder._handle({
                "timestamp": "1789655276412",
                "price_changes": [{
                    "asset_id": "token-a", "best_bid": "0.4", "best_ask": "0.5",
                    "price": "0.45", "size": "10", "side": "BUY",
                }],
            })
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["event_id"], "123")
        self.assertAlmostEqual(row["source_ts"], 1789655276.412)
        self.assertIn("recv_utc", row)
        self.assertEqual(row["best_bid"], 0.4)
        self.assertEqual(row["best_ask"], 0.5)

    def test_book_snapshot_records_depth_and_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.jsonl"
            recorder = RealtimeRecorder(
                {"Map 2 Winner": {"Alpha": "token-a"}}, path, event_id="123",
                market_metadata={"Map 2 Winner": {
                    "market_id": "456", "condition_id": "0xabc"}})
            recorder._handle({
                "asset_id": "token-a", "timestamp": 1000,
                "bids": [["0.40", "2"], ["0.45", "3"], ["bad", "9"]],
                "asks": [{"price": "0.55", "size": "4"},
                         {"price": "0.60", "size": "5"}],
            })
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["best_bid"], 0.45)
        self.assertEqual(row["best_ask"], 0.55)
        self.assertAlmostEqual(row["spread"], 0.10)
        self.assertEqual(row["bid_size_l1"], 3.0)
        self.assertEqual(row["ask_size_l1"], 4.0)
        self.assertEqual(row["bid_depth_l5"], 5.0)
        self.assertEqual(row["ask_depth_l5"], 9.0)
        self.assertEqual(row["bids_l5"][0], {"price": 0.45, "size": 3.0})
        self.assertEqual(row["market_id"], "456")
        self.assertEqual(row["condition_id"], "0xabc")

    def test_array_payload_records_books_with_one_receive_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.jsonl"
            recorder = RealtimeRecorder(
                {"Map 1 Winner": {"Alpha": "a", "Beta": "b"}}, path)
            raw = json.dumps([
                {"asset_id": "a", "bids": [[.4, 1]], "asks": [[.5, 2]]},
                {"asset_id": "b", "bids": [[.5, 2]], "asks": [[.6, 1]]},
            ])
            with patch("cs2ml.realtime_record.time.time", return_value=123.5):
                recorder._on_message(None, raw)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["local_ts"] for row in rows}, {123.5})

    def test_best_bid_ask_event_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.jsonl"
            recorder = RealtimeRecorder({"Match Winner": {"Alpha": "a"}}, path)
            recorder._handle({"asset_id": "a", "best_bid": ".48",
                              "best_ask": ".51", "timestamp": "1000"})
            row = json.loads(path.read_text())
        self.assertEqual(row["type"], "best_bid_ask")
        self.assertAlmostEqual(row["spread"], .03)

    def test_run_actively_closes_healthy_socket_at_deadline(self):
        class HealthySocket:
            def __init__(self):
                self.closed = threading.Event()

            def close(self):
                self.closed.set()

            def run_forever(self, **_):
                self.closed.wait(timeout=1)

        with tempfile.TemporaryDirectory() as directory:
            recorder = RealtimeRecorder(
                {"Match Winner": {"Alpha": "a"}},
                Path(directory) / "quotes.jsonl")
            socket = HealthySocket()
            with patch.object(recorder, "_connect", return_value=socket):
                result = recorder.run(duration_hours=0.00001)
        self.assertTrue(socket.closed.is_set())
        self.assertEqual(result["reconnects"], 0)


if __name__ == "__main__":
    unittest.main()
