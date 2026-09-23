from __future__ import annotations

import unittest

from cs2ml.alpha_decay import alpha_decay
from cs2ml.market_features import build_market_features


def quote(ts, mid):
    return {"type": "book", "local_ts": ts, "token": "t", "event_id": "e",
            "market_id": "m", "market": "Map 1 Winner", "outcome": "Alpha",
            "best_bid": mid - .01, "best_ask": mid + .01,
            "bid_size_l1": 10, "ask_size_l1": 10,
            "bid_depth_l5": 50, "ask_depth_l5": 50}


class AlphaDecayTests(unittest.TestCase):
    def test_latency_uses_first_quote_at_or_after_target(self):
        features = build_market_features([
            quote(0, .50), quote(1, .53), quote(1.1, .55), quote(6, .58),
        ])
        report = alpha_decay(features, latencies=(0, .1), threshold=.02,
                             horizon=5, max_quote_wait_seconds=.01,
                             bootstrap_draws=10)
        zero, delayed = report["curve"]
        self.assertEqual(zero["fillable_opportunities"], 1)
        self.assertEqual(delayed["fillable_opportunities"], 1)
        self.assertGreater(zero["mean_executable_pnl"], delayed["mean_executable_pnl"])

    def test_no_backward_or_nearest_future_fill(self):
        features = build_market_features([quote(0, .50), quote(1, .53), quote(6, .58)])
        report = alpha_decay(features, latencies=(.1,), threshold=.02, horizon=5,
                             max_quote_wait_seconds=.2, bootstrap_draws=10)
        point = report["curve"][0]
        self.assertEqual(point["available_opportunities"], 2)
        self.assertEqual(point["fillable_opportunities"], 0)
        self.assertIsNone(point["mean_executable_pnl"])


if __name__ == "__main__":
    unittest.main()
