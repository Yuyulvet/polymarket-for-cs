from __future__ import annotations

import unittest

from cs2ml.market_features import build_market_features
from cs2ml.market_shocks import (clustered_bootstrap, detect_shocks,
                                 executable_drift, summarize)


def quote(ts, mid, *, token="t", event="e1", spread=.02):
    return {"type": "book", "local_ts": ts, "token": token,
            "event_id": event, "match_id": event, "series_id": event,
            "market": "Map 1 Winner", "best_bid": mid - spread / 2,
            "best_ask": mid + spread / 2, "bid_size_l1": 10,
            "ask_size_l1": 10, "bid_depth_l5": 50, "ask_depth_l5": 50}


class ShockStudyTests(unittest.TestCase):
    def test_up_shock_uses_ask_entry_and_bid_exit(self):
        features = build_market_features([
            quote(0, .50), quote(1, .52), quote(2, .54),
        ])
        shocks = detect_shocks(features, shock_windows=(1,), cooldown_seconds=1)
        shock = next(row for row in shocks if row["decision_ts"] == 1)
        rows = executable_drift(features, [shock], horizons=(1,),
                                max_quote_wait_seconds=0)
        row = rows[0]
        self.assertAlmostEqual(row["entry_ask"], .53)
        self.assertAlmostEqual(row["exit_bid"], .53)
        self.assertAlmostEqual(row["raw_mid_drift"], .02)
        self.assertAlmostEqual(row["executable_pnl"], 0)
        self.assertAlmostEqual(row["spread_cost"], .02)
        self.assertIsNone(row["estimated_fees"])
        self.assertIsNone(row["net_drift"])

    def test_down_shock_uses_bid_entry_and_ask_cover(self):
        features = build_market_features([
            quote(0, .50), quote(1, .47), quote(2, .45),
        ])
        shock = next(row for row in detect_shocks(
            features, shock_windows=(1,), cooldown_seconds=1)
                     if row["decision_ts"] == 1)
        row = executable_drift(features, [shock], horizons=(1,),
                               max_quote_wait_seconds=0)[0]
        self.assertAlmostEqual(row["entry_bid"], .46)
        self.assertAlmostEqual(row["exit_ask"], .46)
        self.assertAlmostEqual(row["executable_pnl"], 0)
        self.assertAlmostEqual(row["raw_mid_drift"], .02)

    def test_known_costs_produce_net_drift(self):
        features = build_market_features([quote(0, .50), quote(1, .52), quote(2, .55)])
        shock = next(row for row in detect_shocks(features, shock_windows=(1,))
                     if row["decision_ts"] == 1)
        row = executable_drift(features, [shock], horizons=(1,), fee_bps=100,
                               slippage_per_side=.001,
                               max_quote_wait_seconds=0)[0]
        expected_fee = (.53 + .54) * .01
        self.assertAlmostEqual(row["estimated_fees"], expected_fee)
        self.assertAlmostEqual(row["net_drift"], .54 - .53 - expected_fee - .002)

    def test_detection_is_unchanged_by_future_suffix(self):
        prefix = build_market_features([quote(0, .50), quote(1, .52)])
        full = build_market_features([quote(0, .50), quote(1, .52), quote(2, .90)])
        first = detect_shocks(prefix, shock_windows=(1,))
        second = [row for row in detect_shocks(full, shock_windows=(1,))
                  if row["decision_ts"] <= 1]
        self.assertEqual([(row["shock_id"], row["shock_change"]) for row in first],
                         [(row["shock_id"], row["shock_change"]) for row in second])

    def test_clustered_counts_do_not_treat_ticks_as_matches(self):
        rows = []
        for event, pnl in (("m1", .01), ("m2", -.01)):
            for index in range(5):
                rows.append({"label_valid": True, "executable_pnl": pnl,
                             "net_drift": None, "raw_mid_drift": pnl,
                             "spread_cost": 0, "estimated_fees": None,
                             "shock_id": f"{event}:{index}", "event_id": event,
                             "match_id": event, "series_id": event,
                             "decision_ts": index})
        report = summarize(rows, draws=100)
        self.assertEqual(report["number_of_observations"], 10)
        self.assertEqual(report["number_of_matches"], 2)
        self.assertEqual(report["number_of_series"], 2)
        self.assertEqual(report["number_of_clusters"], 2)
        self.assertIsNone(report["executable_pnl_95pct_clustered_ci"])
        self.assertEqual(report["clustered_ci_status"],
                         "insufficient_clusters_minimum_5")
        self.assertEqual(report["conclusion"], "costs_unknown_no_net_edge_conclusion")

    def test_one_cluster_cannot_fabricate_confidence_interval(self):
        self.assertIsNone(clustered_bootstrap([.1, .2], ["one", "one"], draws=10))

    def test_invalid_cost_or_latency_configuration_fails_closed(self):
        features = build_market_features([quote(0, .5), quote(1, .52)])
        with self.assertRaises(ValueError):
            executable_drift(features, [], latency_seconds=-.1)
        with self.assertRaises(ValueError):
            executable_drift(features, [], fee_bps=-1)
        with self.assertRaises(ValueError):
            executable_drift(features, [], slippage_per_side=-.01)

    def test_complementary_tokens_share_one_market_shock_id(self):
        features = build_market_features([
            quote(0, .5, token="yes"), quote(0, .5, token="no"),
            quote(1, .52, token="yes"), quote(1, .48, token="no"),
        ])
        shocks = [row for row in detect_shocks(features, shock_windows=(1,))
                  if row["decision_ts"] == 1]
        self.assertEqual(len(shocks), 2)
        self.assertEqual(len({row["shock_id"] for row in shocks}), 1)
        self.assertEqual(len({row["token_shock_id"] for row in shocks}), 2)


if __name__ == "__main__":
    unittest.main()
