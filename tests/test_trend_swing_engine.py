"""Phase-5 swing engine tests: rule gating, risk caps, paper ledger."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cs2ml import trend_swing_engine as se
from cs2ml import trend_dataset as td
from tests.test_trend_dataset import _write_session, _spec


def _rows(root: Path, n_sessions=2) -> pd.DataFrame:
    frames = []
    for i in range(n_sessions):
        session = _write_session(root / f"s{i}")
        spec = _spec()
        spec["session_id"] = f"sess-{i}"
        frame, _ = td.build_rows(session, spec)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class RuleValidationTests(unittest.TestCase):
    def test_horizon_must_be_protocol_horizon(self):
        with self.assertRaisesRegex(ValueError, "horizon_not_in_protocol"):
            se.SwingRule(horizon_seconds=45)

    def test_param_ranges(self):
        with self.assertRaises(ValueError):
            se.SwingRule(min_edge=1.5)
        with self.assertRaises(ValueError):
            se.SwingRule(max_entry_ask=0.0)

    def test_rule_hash_stable_and_sensitive(self):
        a = se.SwingRule().rule_hash()
        self.assertEqual(a, se.SwingRule().rule_hash())
        self.assertNotEqual(a, se.SwingRule(min_edge=0.05).rule_hash())


class ReplayGatingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.df = _rows(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _replay(self, rule=None, p=None, cash=0.0):
        rule = rule or se.SwingRule()
        if p is None:
            p = np.full(len(self.df), 0.95)
        return se.replay(self.df, rule, p, initial_cash=cash)

    def test_bout_exclusivity(self):
        # default rule: first eligible trade per (session, bout) wins,
        # everything else in that bout is skipped
        result = self._replay()
        report = result["report"]
        self.assertEqual(report["n_trades"], 2)  # one per session
        self.assertEqual(report["skip_reasons"].get("bout_position_open"), 10)
        self.assertEqual(report["skip_reasons"].get("kind_not_allowed"), 4)
        kinds = set(result["ledger"]["decision_kind"])
        self.assertEqual(kinds, {"post_pistol_freeze"})

    def test_market_prior_never_trades(self):
        # no-edge control: fair=mid < ask, edge negative -> zero trades
        result = se.replay(self.df, se.SwingRule(), se.market_prior(self.df))
        self.assertEqual(result["report"]["n_trades"], 0)
        self.assertEqual(result["report"]["skip_reasons"].get("edge_below_threshold"), 12)

    def test_edge_threshold_gating(self):
        p = self.df["entry_ask"].astype(float).to_numpy()  # edge == 0
        result = self._replay(p=p)
        self.assertEqual(result["report"]["n_trades"], 0)
        self.assertEqual(result["report"]["skip_reasons"].get("edge_below_threshold"), 12)

    def test_entry_ask_cap(self):
        rule = se.SwingRule(max_entry_ask=0.5)
        result = self._replay(rule=rule)
        # only No-token rows (ask ~0.37) can pass; bout exclusivity leaves 1/session
        self.assertEqual(result["report"]["n_trades"], 2)
        self.assertTrue((result["ledger"]["entry_ask"] <= 0.5).all())
        self.assertEqual(set(result["ledger"]["outcome"]), {"No"})

    def test_spread_cap(self):
        rule = se.SwingRule(max_spread=0.001)
        result = self._replay(rule=rule)
        self.assertEqual(result["report"]["n_trades"], 0)
        self.assertEqual(result["report"]["skip_reasons"].get("spread_cap"), 12)

    def test_session_cap_without_bout_lock(self):
        rule = se.SwingRule(one_position_per_bout=False, max_open_per_session=2)
        result = self._replay(rule=rule)
        self.assertEqual(result["report"]["n_trades"], 4)  # 2 per session
        self.assertEqual(result["report"]["skip_reasons"].get("session_position_cap"), 8)

    def test_horizon_selection_matches_labels(self):
        rule = se.SwingRule(horizon_seconds=15, max_entry_ask=1.0)
        result = self._replay(rule=rule)
        ledger = result["ledger"]
        for col, expected in (("exit_bid", "exit_bid_h15"), ("net", "net_h15")):
            merged = ledger.merge(
                self.df[["session_id", "exec_mono", "outcome", expected]],
                on=["session_id", "exec_mono", "outcome"])
            self.assertTrue(np.allclose(merged[col], merged[expected]))

    def test_cash_accounting(self):
        result = self._replay(cash=10.0)
        nets = result["ledger"]["net"].sum()
        self.assertAlmostEqual(result["report"]["cash_final"], 10.0 + nets)
        self.assertAlmostEqual(result["report"]["net_total"], nets)

    def test_ci_and_stats_present(self):
        rule = se.SwingRule(one_position_per_bout=False, max_open_per_session=12)
        result = self._replay(rule=rule)
        report = result["report"]
        self.assertIsNotNone(report["ci95_lower_bound"])
        self.assertGreaterEqual(report["win_rate"], 0.0)
        self.assertIn("post_pistol_freeze", report["per_mechanism"])
        self.assertEqual(report["mode"], "paper_replay_only_no_orders")

    def test_empty_input_safe(self):
        result = se.replay(self.df.iloc[0:0], se.SwingRule(), np.array([]))
        self.assertEqual(result["report"]["n_trades"], 0)
        self.assertIsNone(result["report"]["net_mean"])

    def test_p_fair_length_guard(self):
        with self.assertRaisesRegex(ValueError, "p_fair_length_mismatch"):
            se.replay(self.df, se.SwingRule(), np.array([0.9]))

    def test_determinism(self):
        a = self._replay(rule=se.SwingRule(one_position_per_bout=False,
                                           max_open_per_session=12))["ledger"]
        b = self._replay(rule=se.SwingRule(one_position_per_bout=False,
                                           max_open_per_session=12))["ledger"]
        pd.testing.assert_frame_equal(a, b)


class GameHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.df = _rows(Path(self.tmp.name), n_sessions=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_symmetry_team_perspective(self):
        team1 = pd.Series(True, index=self.df.index)
        team2 = pd.Series(False, index=self.df.index)
        p1 = se.game_heuristic(self.df, team1)
        p2 = se.game_heuristic(self.df, team2)
        np.testing.assert_allclose(p1 + p2, 1.0, atol=1e-9)

    def test_pistol_row_is_team1_favored(self):
        pistol = self.df["decision_kind"] == "post_pistol_freeze"
        team1 = pd.Series(True, index=self.df.index)
        p = se.game_heuristic(self.df, team1)
        # fixture: team1 leads 1:0 with money edge -> fair > 0.5
        self.assertTrue((p[pistol.to_numpy()] > 0.5).all())

    def test_prematch_rows_are_neutral(self):
        prematch = (self.df["decision_kind"] == "prematch_prior_dislocation").to_numpy()
        team1 = pd.Series(True, index=self.df.index)
        p = se.game_heuristic(self.df, team1)
        np.testing.assert_allclose(p[prematch], 0.5, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
