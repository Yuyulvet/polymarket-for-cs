"""Phase-4 mechanism incremental tests (same-sample market/game/mixed)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cs2ml import trend_game_increment as gi
from cs2ml import trend_dataset as td
from tests.test_trend_dataset import _write_session, _spec


def _two_sessions(root: Path):
    frames = []
    for name in ("sess-a", "sess-b"):
        session = _write_session(root / name)
        spec = _spec()
        spec["session_id"] = name
        frame, _ = td.build_rows(session, spec)
        frames.append(frame)
    return frames


class EncodingTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = _two_sessions(Path(tmp))[0]
        self.df, self.fs = gi.build_feature_sets(frame)

    def test_categorical_codes(self):
        pistol = self.df[self.df["decision_kind"] == "post_pistol_freeze"]
        # fixture: team1 first_half_side CT -> +1
        self.assertEqual(set(pistol["game_side_t1_code"]), {1.0})
        rnd_end = self.df[self.df["decision_kind"] == "round_end_economy_update"]
        # fixture: CT (team1) wins every recorded round -> +1
        self.assertEqual(set(rnd_end["game_pistol_winner_code"].unique()), {1.0})

    def test_engineered_gaps(self):
        pistol = self.df[self.df["decision_kind"] == "post_pistol_freeze"].iloc[0]
        self.assertEqual(pistol["game_alive_gap"], 5 - 4)
        self.assertEqual(pistol["game_hp_gap"], 500 - 320)
        self.assertEqual(pistol["game_money_gap"], 20000 - 9000)
        self.assertAlmostEqual(pistol["game_money_share_t1"], 20000 / 29000)

    def test_prematch_rows_marked_absent(self):
        prematch = self.df[self.df["decision_kind"] == "prematch_prior_dislocation"]
        self.assertEqual(set(prematch["game_state_present"]), {0.0})
        live = self.df[self.df["decision_kind"] != "prematch_prior_dislocation"]
        self.assertEqual(set(live["game_state_present"]), {1.0})

    def test_no_duplicate_columns(self):
        self.assertEqual(len(self.df.columns), len(set(self.df.columns)))

    def test_feature_set_disjointness(self):
        market, game = set(self.fs["market_only"]), set(self.fs["game_only"])
        self.assertFalse(market & game)
        self.assertEqual(self.fs["market_plus_game"],
                         self.fs["market_only"] + self.fs["game_only"])


class MechanismEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        frame_a, frame_b = _two_sessions(root)
        self.report = gi.evaluate_mechanisms(
            __import__("pandas").concat([frame_a, frame_b], ignore_index=True))

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_mechanisms_present(self):
        self.assertEqual(set(self.report["mechanisms"]),
                         set(gi.MECHANISMS))

    def test_mechanism_slices_partition_rows(self):
        total = sum(m["n_rows"] for m in self.report["mechanisms"].values())
        # per session: 2 prematch + 2 pistol + 4 round_end rows (2 tokens each)
        self.assertEqual(total, 2 * 8)

    def test_same_samples_across_feature_sets(self):
        for name, mech in self.report["mechanisms"].items():
            if mech.get("insufficient_sessions"):
                continue
            for split in mech["splits"]:
                for horizon, fs_results in split["horizons"].items():
                    counts = {fs: next(iter(m.values()))["n_test"]
                              for fs, m in fs_results.items()}
                    self.assertEqual(len(set(counts.values())), 1,
                                     f"{name} {horizon} sample mismatch: {counts}")

    def test_deltas_direction_and_shape(self):
        mech = self.report["mechanisms"]["4A_post_pistol_freeze"]
        deltas = mech["deltas"]
        self.assertIn("h15", deltas)
        self.assertIn("game_only", deltas["h15"])
        self.assertIn("market_plus_game", deltas["h15"])
        for fs in ("game_only", "market_plus_game"):
            for model in ("logistic", "xgboost"):
                entry = deltas["h15"][fs].get(model)
                if entry is None:
                    continue
                for key in entry:
                    self.assertTrue(key.startswith("d_"))

    def test_market_only_delta_is_zero(self):
        # delta of market_only against itself must never appear / be zero
        for mech in self.report["mechanisms"].values():
            for horizon, fs_results in mech.get("deltas", {}).items():
                self.assertNotIn("market_only", fs_results)

    def test_4c_flags_missing_prior_feature(self):
        mech = self.report["mechanisms"]["4C_prematch_prior_dislocation"]
        self.assertIn("no_prematch_prior_feature_yet", mech["prior_feature_status"])

    def test_insufficient_sessions_carries_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = _two_sessions(Path(tmp))[0]
            report = gi.evaluate_mechanisms(frame)
        for mech in report["mechanisms"].values():
            self.assertTrue(mech["insufficient_sessions"])
            self.assertEqual(mech["splits"], [])
            self.assertEqual(mech["deltas"], {})


class RandomWalkFallbackTests(unittest.TestCase):
    def test_random_walk_without_price_columns(self):
        import pandas as pd
        from cs2ml import trend_market_baseline as tb
        X = pd.DataFrame({"game_score_t1": [1.0, 2.0]})
        p = tb.fit_predict("clf", "random_walk", X, np.array([0, 1]), X)
        self.assertTrue((p == 0.5).all())
        r = tb.fit_predict("reg", "random_walk", X, np.array([0.5, 0.6]), X)
        self.assertTrue((r == 0.0).all())


if __name__ == "__main__":
    unittest.main()
