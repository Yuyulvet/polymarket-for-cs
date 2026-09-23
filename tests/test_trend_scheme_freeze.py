"""Phase-6 scheme comparison, freeze integrity, frozen replay tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cs2ml import trend_scheme_freeze as sf
from cs2ml import trend_dataset as td
from tests.test_trend_dataset import _write_session, _spec


def _rows(root: Path, n=3) -> pd.DataFrame:
    frames = []
    for i in range(n):
        session = _write_session(root / f"s{i}")
        spec = _spec()
        spec["session_id"] = f"sess-{i}"
        frame, _ = td.build_rows(session, spec)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class CandidateSchemeTests(unittest.TestCase):
    def test_unknown_fair_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown_fair_source"):
            sf.CandidateScheme(name="x", fair_source="oracle")

    def test_scheme_hash_stable_and_sensitive(self):
        a = sf.CandidateScheme(name="a")
        self.assertEqual(a.scheme_hash(), sf.CandidateScheme(name="a").scheme_hash())
        self.assertNotEqual(a.scheme_hash(),
                            sf.CandidateScheme(name="a", min_edge=0.05).scheme_hash())

    def test_default_candidates_cover_sources(self):
        sources = {c.fair_source for c in sf.default_candidates()}
        self.assertEqual(sources, set(sf.FAIR_SOURCES))


class WalkForwardForecastTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.df = _rows(Path(self.tmp.name), n=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_session_has_no_prediction(self):
        from cs2ml.trend_market_baseline import session_order
        scheme = sf.CandidateScheme(name="wf", fair_source="wf_ridge_exit")
        pred = sf.walk_forward_exit_forecast(self.df, scheme)
        df = self.df.copy()
        df["session_start"] = df.groupby("session_id")["info_mono"].transform("min")
        first_session = session_order(df)[0]
        is_first = (self.df["session_id"] == first_session).to_numpy()
        self.assertTrue(np.isnan(pred[is_first]).all())
        self.assertFalse(np.isnan(pred[~is_first]).any())

    def test_predictions_are_finite_prices(self):
        scheme = sf.CandidateScheme(name="wf", fair_source="wf_ridge_exit")
        pred = sf.walk_forward_exit_forecast(self.df, scheme)
        valid = pred[np.isfinite(pred)]
        self.assertTrue(((valid >= 0.0) & (valid <= 1.0)).all())


class SelectionAndFreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.df = _rows(self.root, n=3)
        self.team1 = pd.Series(self.df["outcome"] == "Yes", index=self.df.index)

    def tearDown(self):
        self.tmp.cleanup()

    def _candidates(self):
        return [sf.CandidateScheme(name="control", fair_source="market_prior"),
                sf.CandidateScheme(name="game_h30", fair_source="game_heuristic",
                                   horizon_seconds=30),
                sf.CandidateScheme(name="game_h60", fair_source="game_heuristic",
                                   horizon_seconds=60)]

    def test_select_respects_floor_and_pre_registered_criterion(self):
        selection = sf.select_scheme(self.df, self._candidates(),
                                     min_trades_floor=1,
                                     outcome_is_team1=self.team1)
        self.assertIsNotNone(selection["winner"])
        names = {c["name"] for c in selection["candidates"]}
        self.assertEqual(names, {"control", "game_h30", "game_h60"})
        # control (market prior) must have zero trades -> ineligible at floor>=1
        control = next(c for c in selection["candidates"] if c["name"] == "control")
        self.assertFalse(control["eligible"])
        self.assertEqual(control["n_trades"], 0)
        self.assertIn("max net_mean", selection["criterion"])

    def test_floor_exclusion(self):
        selection = sf.select_scheme(self.df, self._candidates(),
                                     min_trades_floor=10_000,
                                     outcome_is_team1=self.team1)
        self.assertIsNone(selection["winner"])
        self.assertIn("no_candidate_meets_trade_floor",
                      selection["no_eligible_reason"])

    def test_freeze_roundtrip_and_tamper_detection(self):
        selection = sf.select_scheme(self.df, self._candidates(),
                                     min_trades_floor=1,
                                     outcome_is_team1=self.team1)
        path = self.root / "frozen_scheme.json"
        payload = sf.freeze_scheme(selection, path)
        loaded = sf.load_frozen(path)
        self.assertEqual(loaded["sha256"], payload["sha256"])
        # 篡改任何一个字段都必须被发现
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["selection"]["winner"]["scheme"]["min_edge"] = 0.99
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen_scheme_tampered"):
            sf.load_frozen(path)

    def test_cannot_freeze_without_winner(self):
        selection = {"winner": None}
        with self.assertRaisesRegex(ValueError, "cannot_freeze_without_winner"):
            sf.freeze_scheme(selection, self.root / "x.json")

    def test_run_frozen_matches_winner_candidate(self):
        selection = sf.select_scheme(self.df, self._candidates(),
                                     min_trades_floor=1,
                                     outcome_is_team1=self.team1)
        path = self.root / "frozen_scheme.json"
        sf.freeze_scheme(selection, path)
        frozen = sf.load_frozen(path)
        frozen_result = sf.run_frozen(self.df, frozen, self.team1)
        winner_scheme = sf.CandidateScheme(**frozen["selection"]["winner"]["scheme"])
        direct = sf.run_candidate(self.df, winner_scheme, self.team1)
        pd.testing.assert_frame_equal(frozen_result["ledger"], direct["ledger"])
        self.assertEqual(frozen_result["report"]["n_trades"],
                         direct["report"]["n_trades"])

    def test_wf_candidate_runs_end_to_end(self):
        scheme = sf.CandidateScheme(name="wf", fair_source="wf_ridge_exit")
        result = sf.run_candidate(self.df, scheme)
        self.assertIn("n_trades", result["report"])


if __name__ == "__main__":
    unittest.main()
