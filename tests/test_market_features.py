from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from cs2ml.market_features import build_market_features, load_realtime, run


def quote(ts, bid, ask, bid_size=10, ask_size=10):
    return {"type": "book", "local_ts": ts, "token": "t", "market": "Map 1 Winner",
            "best_bid": bid, "best_ask": ask,
            "bid_size_l1": bid_size, "ask_size_l1": ask_size,
            "bid_depth_l5": bid_size * 2, "ask_depth_l5": ask_size * 2}


class MarketFeatureTests(unittest.TestCase):
    def test_book_features_and_windows(self):
        frame = build_market_features([
            quote(0, .40, .42, 30, 10),
            {"type": "trade", "local_ts": .5, "token": "t",
             "price": .42, "size": 6, "side": "BUY"},
            quote(1, .42, .44, 10, 30),
        ])
        row = frame.iloc[-1]
        self.assertAlmostEqual(row["mid"], .43)
        self.assertAlmostEqual(row["obi_1"], -.5)
        self.assertAlmostEqual(row["obi_5"], -.5)
        self.assertAlmostEqual(row["microprice"], .425)
        self.assertAlmostEqual(row["return_1s"], .02)
        self.assertEqual(row["volume_1s"], 6)
        self.assertEqual(row["trade_imbalance_1s"], 1)

    def test_trade_imbalance_is_causal_and_windowed(self):
        frame = build_market_features([
            quote(0, .4, .42),
            {"type": "trade", "local_ts": 1, "token": "t",
             "price": .42, "size": 5, "side": "BUY"},
            {"type": "trade", "local_ts": 2, "token": "t",
             "price": .41, "size": 3, "side": "SELL"},
        ])
        self.assertEqual(frame.iloc[1]["trade_imbalance_1s"], 1)
        self.assertEqual(frame.iloc[2]["trade_imbalance_1s"], -1)
        self.assertAlmostEqual(frame.iloc[2]["trade_imbalance_3s"], (5 - 3) / 8)

    def test_future_suffix_cannot_change_past_features(self):
        prefix = [quote(0, .4, .42), quote(1, .41, .43)]
        before = build_market_features(prefix)
        after = build_market_features(prefix + [
            quote(2, .8, .82, 999, 1),
            {"type": "trade", "local_ts": 3, "token": "t",
             "price": .82, "size": 1000, "side": "BUY"},
        ])
        pd.testing.assert_frame_equal(
            before.fillna("missing").reset_index(drop=True),
            after.iloc[:len(before)].fillna("missing").reset_index(drop=True),
            check_dtype=False)

    def test_source_timestamp_never_controls_order(self):
        rows = [
            {**quote(10, .4, .42), "source_ts": 9999},
            {**quote(11, .5, .52), "source_ts": 1},
        ]
        frame = build_market_features(rows)
        self.assertAlmostEqual(frame.iloc[-1]["return_1s"], .10)
        self.assertEqual(frame.iloc[-1]["source_ts"], 1)

    def test_sidecar_metadata_and_timestamped_output_are_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "event.jsonl"
            source.write_text(json.dumps(quote(1, .4, .42)) + "\n", encoding="utf-8")
            source.with_suffix(".meta.json").write_text(json.dumps({
                "event_id": "e1", "markets": [{"market_id": "m1",
                "condition_id": "c1", "tokens": {"Alpha": "t"},
                "taker_fee_bps": 25}]}), encoding="utf-8")
            loaded = load_realtime([source])
            self.assertEqual(loaded[0]["market_id"], "m1")
            self.assertEqual(loaded[0]["taker_fee_bps"], 25)
            output = run([source], root / "out")
            saved = json.loads((output / "features.jsonl").read_text())
        self.assertIsNone(saved["trade_price"])
        self.assertEqual(saved["event_id"], "e1")


if __name__ == "__main__":
    unittest.main()
