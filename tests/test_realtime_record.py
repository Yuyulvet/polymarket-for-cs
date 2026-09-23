from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.realtime_record import RealtimeRecorder, tokens_from_event


class RealtimeRecorderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
