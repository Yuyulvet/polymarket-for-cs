from __future__ import annotations

import unittest

from cs2ml.shock_taxonomy import classify_realtime, label_ex_post


def base():
    return {"shock_direction": "up", "shock_window_seconds": 1,
            "spread": .04, "liquidity": 20, "volume_1s": 2,
            "trades_per_second_1s": 1, "trade_imbalance_1s": 1,
            "ask_depth_depletion_1s": .5,
            "pre_shock_bid": .4, "pre_shock_ask": .42,
            "post_shock_bid": .43, "post_shock_ask": .45}


class ShockTaxonomyTests(unittest.TestCase):
    def test_thin_book_classification_is_realtime_only(self):
        first = {**base(), "raw_mid_drift": .9, "executable_pnl": .8}
        second = {**base(), "raw_mid_drift": -.9, "executable_pnl": -.8}
        self.assertEqual(classify_realtime(first), classify_realtime(second))
        self.assertEqual(classify_realtime(first)["realtime_class"],
                         "TYPE_2_THIN_BOOK_JUMP")

    def test_reversal_is_ex_post_label_only(self):
        label = label_ex_post({"label_valid": True, "raw_mid_drift": -.01})
        self.assertEqual(label["ex_post_outcome_label"],
                         "TYPE_4_TRANSIENT_OR_REVERSAL")
        self.assertTrue(label["label_uses_future_information"])


if __name__ == "__main__":
    unittest.main()
