from __future__ import annotations

import unittest

import pandas as pd

from cs2ml.market_research_dataset import (build_dataset, future_label_columns,
                                           model_feature_columns)


def observation(token, outcome, horizon, pnl):
    return {
        "shock_id": "market-shock", "token_shock_id": f"{token}-shock",
        "token": token, "outcome": outcome, "event_id": "e", "market_id": "m",
        "condition_id": "c", "market": "Map 1 Winner", "market_type": "Map Winner",
        "decision_ts": 100, "shock_window_seconds": 1, "shock_magnitude": .03,
        "shock_direction": "up" if token == "a" else "down",
        "pre_shock_mid": .5, "pre_shock_bid": .49, "pre_shock_ask": .51,
        "post_shock_bid": .52, "post_shock_ask": .54, "spread": .02,
        "bid_size_l1": 10, "ask_size_l1": 10, "bid_depth_l5": 50,
        "ask_depth_l5": 50, "obi_1": 0, "obi_5": 0,
        "trade_imbalance_1s": .5, "volume_1s": 20,
        "trades_per_second_1s": 3, "microprice_mid_divergence": .001,
        "pre_shock_obi_1": 0, "pre_shock_obi_5": 0,
        "bid_depth_depletion_1s": .1, "ask_depth_depletion_1s": .3,
        "entry_target_ts": 100, "entry_quote_ts": 100,
        "entry_bid": .52, "entry_ask": .54, "latency_seconds": 0,
        "fee_status": "unknown", "slippage_status": "unknown",
        "horizon_seconds": horizon, "label_valid": True,
        "raw_mid_drift": pnl + .02, "executable_pnl": pnl,
        "exit_quote_ts": 100 + horizon, "estimated_fees": None,
        "estimated_slippage": None, "net_drift": None,
    }


class MarketResearchDatasetTests(unittest.TestCase):
    def test_yes_no_views_deduplicate_to_one_market_shock(self):
        observations = [observation(token, outcome, horizon, pnl)
                        for token, outcome in (("a", "Alpha"), ("b", "Beta"))
                        for horizon, pnl in ((1, -.01), (5, .01))]
        annotations = [{"shock_id": "market-shock", "token": "a",
                        "annotation_available": False, "missing_reason": "missing"}]
        frame, metadata = build_dataset(observations, annotations=annotations,
                                        qa_reports={"m": {"status": "WARN",
                                        "exclusion_reasons": ["depth"]}})
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].canonical_token, "a")
        self.assertEqual(frame.iloc[0].capture_qa_status, "WARN")
        self.assertIsNone(frame.iloc[0].h5_estimated_fees)
        self.assertIsNone(frame.iloc[0].h5_estimated_slippage)
        self.assertEqual(metadata["dataset_unit"], "one_market_level_shock")

    def test_column_roles_are_disjoint_and_future_changes_do_not_change_features(self):
        original = [observation("a", "Alpha", 5, .01)]
        changed = [observation("a", "Alpha", 5, -.50)]
        first, meta = build_dataset(original)
        second, _ = build_dataset(changed)
        features = model_feature_columns(meta)
        labels = future_label_columns(meta)
        self.assertFalse(set(features) & set(labels))
        pd.testing.assert_series_equal(first.iloc[0][features], second.iloc[0][features])
        self.assertNotEqual(first.iloc[0].h5_executable_pnl,
                            second.iloc[0].h5_executable_pnl)


if __name__ == "__main__":
    unittest.main()
