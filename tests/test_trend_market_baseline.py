"""Phase-3 market-only baseline tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cs2ml import trend_market_baseline as tb
from cs2ml import trend_dataset as td
from tests.test_trend_dataset import _write_session, _spec


def _two_session_rows(root: Path):
    session_a = _write_session(root / "a")
    session_b = _write_session(root / "b")
    spec_a, spec_b = _spec(), _spec()
    spec_a["session_id"] = "sess-a"
    spec_b["session_id"] = "sess-b"
    frame_a, _ = td.build_rows(session_a, spec_a)
    frame_b, _ = td.build_rows(session_b, spec_b)
    return frame_a, frame_b


class InsufficientSessionsTests(unittest.TestCase):
    def test_single_session_reports_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_a, _ = _two_session_rows(Path(tmp))
            out = Path(tmp) / "one.parquet"
            frame_a.to_parquet(out, index=False)
            report = tb.evaluate(tb.load_rows([out]))
        self.assertTrue(report["insufficient_sessions"])
        self.assertEqual(report["splits"], [])
        self.assertEqual(tb.aggregate(report), {})


class ForwardEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        frame_a, frame_b = _two_session_rows(root)
        self.path_a = root / "a.parquet"
        self.path_b = root / "b.parquet"
        frame_a.to_parquet(self.path_a, index=False)
        frame_b.to_parquet(self.path_b, index=False)
        self.df = tb.load_rows([self.path_a, self.path_b])
        self.report = tb.evaluate(self.df)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_sessions_one_forward_split(self):
        self.assertFalse(self.report["insufficient_sessions"])
        self.assertEqual(self.report["n_sessions"], 2)
        self.assertEqual(len(self.report["splits"]), 1)
        split = self.report["splits"][0]
        self.assertEqual(split["train_sessions"], ["sess-a"])
        self.assertEqual(split["test_session"], "sess-b")

    def test_all_models_scored_on_all_horizons(self):
        split = self.report["splits"][0]
        for horizon in ("h15", "h30", "h60", "h120"):
            for model in tb.MODELS:
                metrics = split["horizons"][horizon][model]
                self.assertIn("brier", metrics)
                self.assertIn("net_median", metrics)
                self.assertTrue(0.0 <= metrics["brier"] <= 1.0)
                self.assertEqual(metrics["n_test"], 8)

    def test_constant_classifier_beats_random(self):
        split = self.report["splits"][0]
        for horizon in ("h15", "h30", "h60", "h120"):
            brier = split["horizons"][horizon]["constant"]["brier"]
            self.assertLessEqual(brier, 0.25 + 1e-9)  # coin-flip ceiling

    def test_random_walk_uses_current_price(self):
        split = self.report["splits"][0]
        for horizon in ("h15", "h30", "h60", "h120"):
            m = split["horizons"][horizon]["random_walk"]
            self.assertTrue(0.0 < min(m["calibration"][0]["mean_pred"],
                                      m["calibration"][-1]["mean_pred"]))

    def test_net_metrics_present_for_every_model(self):
        split = self.report["splits"][0]
        for model in tb.MODELS:
            m = split["horizons"]["h15"][model]
            self.assertIn("net_mean", m)
            self.assertIn("net_q05", m)
            self.assertLess(m["net_q05"], m["net_median"] + 1e-9)

    def test_aggregation_matches_split(self):
        agg = tb.aggregate(self.report)
        self.assertIn("h15", agg)
        self.assertAlmostEqual(agg["h15"]["constant"]["brier_mean"],
                               self.report["splits"][0]["horizons"]["h15"]["constant"]["brier"])

    def test_market_features_only(self):
        for column in tb.MARKET_FEATURES:
            self.assertIn(column, self.df.columns)
        self.assertFalse(any(c.startswith("game_") for c in tb.MARKET_FEATURES))


class MetricsTests(unittest.TestCase):
    def test_brier_and_logloss_perfect_and_worst(self):
        import numpy as np
        y = np.array([0.0, 1.0, 1.0, 0.0])
        perfect = y.copy()
        self.assertAlmostEqual(tb.brier_score(y, perfect), 0.0)
        self.assertAlmostEqual(tb.logloss_score(y, perfect), 0.0, delta=2e-4)
        wrong = 1 - y
        self.assertAlmostEqual(tb.brier_score(y, wrong), 1.0)
        flipped = tb.logloss_score(y, np.clip(wrong, 1e-4, 1 - 1e-4))
        self.assertGreater(flipped, 5.0)

    def test_calibration_bins_cover_support(self):
        import numpy as np
        y = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        p = np.linspace(0.05, 0.95, 10)
        bins = tb.calibration_bins(y, p, bins=10)
        self.assertEqual(sum(b["n"] for b in bins), 10)
        for b in bins:
            self.assertGreaterEqual(b["mean_pred"], b["bin"][0] - 1e-9)


if __name__ == "__main__":
    unittest.main()
