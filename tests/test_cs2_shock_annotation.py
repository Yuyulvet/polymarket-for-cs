from __future__ import annotations

import unittest

from cs2ml.cs2_shock_annotation import annotate_shocks


class CS2ShockAnnotationTests(unittest.TestCase):
    def test_latest_as_of_prediction_wins_and_future_is_excluded(self):
        shock = {"shock_id": "s", "token": "t", "event_id": "e",
                 "outcome": "Alpha", "decision_ts": 100, "pre_shock_mid": .50,
                 "shock_change": .03}
        predictions = [
            {"event_id": "e", "outcome": "Alpha", "prediction_timestamp": 90,
             "probability": .60, "model_name": "fundamental", "model_version": "v1"},
            {"event_id": "e", "outcome": "Alpha", "prediction_timestamp": 110,
             "probability": .99, "model_name": "future"},
        ]
        result = annotate_shocks([shock], predictions)[0]
        self.assertEqual(result["pre_match_cs2_probability"], .60)
        self.assertEqual(result["cs2_information_age_seconds"], 10)
        self.assertAlmostEqual(result["cs2_edge_before_shock"], .10)
        self.assertEqual(result["shock_alignment"], "aligned")

    def test_no_identity_or_asof_match_stays_null(self):
        result = annotate_shocks(
            [{"shock_id": "s", "token": "t", "event_id": "e",
              "decision_ts": 100}],
            [{"event_id": "other", "prediction_timestamp": 90,
              "probability": .7}])[0]
        self.assertFalse(result["annotation_available"])
        self.assertIsNone(result["pre_match_cs2_probability"])
        self.assertTrue(result["missing_reason"])


if __name__ == "__main__":
    unittest.main()
