from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cs2ml.forward_sim import (_actual_winner, _fit_round_lookup, _fit_transition,
                              _prepare_dataset, _simulate, attach_history_times,
                              evaluate_forward, normalize_map_rounds)
from cs2ml.map1_data import first_ct_is_ct, roster_key


A = roster_key([str(76561198000000000 + i) for i in range(5)])
B = roster_key([str(76561198000000000 + i) for i in range(5, 10)])


def rounds(winners, path="/series/map1.dem", match="series", legacy=True):
    rows = []
    for i, winner in enumerate(winners, 1):
        actual_ct = first_ct_is_ct(i)
        stored_ct = i <= 12 if legacy else actual_ct
        rows.append({"demo_path": path, "match_id": match, "map_name": "de_mirage", "round_num": i,
                     "ct_roster": A if stored_ct else B, "t_roster": B if stored_ct else A,
                     "ct_tier": "eco" if stored_ct else "full", "t_tier": "full" if stored_ct else "eco",
                     "ct_equip": 800 if stored_ct else 5000, "t_equip": 5000 if stored_ct else 800,
                     "winner_side": "CT" if (winner == "a") == actual_ct else "T",
                     "winner_roster": "wrong_legacy_label", "label_ct_win": -1})
    return pd.DataFrame(rows)


def transitions():
    return {"normal": {(phase, tier, won): {tier: 1.} for phase in ("regulation", "overtime")
                       for tier in ("eco", "force", "full") for won in (0, 1)},
            "resets": {phase: {("eco", "full"): 1.} for phase in ("regulation", "overtime")}}


def probabilities(p=.5):
    return {("de_mirage", rc, ct, t): p for rc in ("pistol", "post_pistol", "regular", "overtime")
            for ct in ("eco", "force", "full") for t in ("eco", "force", "full")}


class SequenceRng:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        return next(self.values)

    def choice(self, n, p):
        return int(np.argmax(p))


class MapPreparationTests(unittest.TestCase):
    def test_series_maps_are_isolated_and_winners_rebuilt(self):
        data, audit = _prepare_dataset(pd.concat([rounds(["a"] * 13),
                                                  rounds(["b"] * 13, "/series/map2.dem")]))
        self.assertEqual(audit["usable_maps"], 2)
        self.assertEqual(_actual_winner(data[data.demo_path.str.endswith("map1.dem")]), A)
        self.assertEqual(_actual_winner(data[data.demo_path.str.endswith("map2.dem")]), B)
        self.assertTrue(data.label_ct_win.isin([0, 1]).all())

    def test_overtime_equipment_and_structure_follow_roster_not_legacy_side(self):
        source = rounds(["a", "b"] * 12 + ["a"] * 4)
        source["ct_hp"], source["t_hp"] = 20., 450.
        source["ct_first"], source["t_first"] = 0, 1
        source["first_kill_side"], source["rwaff"] = "T", 0
        source["hp_gap"] = -430.
        d = normalize_map_rounds(source)
        last = d.iloc[27]
        self.assertEqual((last.ct_roster, last.t_roster), (A, B))
        self.assertEqual((last.ct_tier, last.t_tier), ("eco", "full"))
        self.assertEqual((last.ct_equip, last.t_equip), (800, 5000))
        self.assertEqual((last.ct_hp, last.t_hp, last.hp_gap), (450, 20, 430))
        self.assertEqual((last.ct_first, last.t_first, last.first_kill_side, last.rwaff), (1, 0, "CT", 1))
        self.assertEqual((last.score0, last.score1), (16, 12))
        self.assertTrue(last.legacy_side_repaired)
        self.assertEqual(_actual_winner(d), A)
        again = normalize_map_rounds(d)
        self.assertEqual(again.iloc[-1].hp_gap, 430)

    def test_repeated_overtime_winner_and_side_switch(self):
        d = normalize_map_rounds(rounds(["a", "b"] * 15 + ["b"] * 4))
        self.assertEqual((d.iloc[-1].score0, d.iloc[-1].score1), (15, 19))
        self.assertEqual(d.loc[d.round_num.eq(31), "ct_roster"].iloc[0], A)
        self.assertEqual(d.loc[d.round_num.eq(34), "ct_roster"].iloc[0], B)
        self.assertEqual(_actual_winner(d), B)

    def test_bad_maps_are_rejected_not_partially_salvaged(self):
        good = rounds(["a"] * 13)
        variants = [good.iloc[:-1], good.drop(index=3), pd.concat([good, good.iloc[-1:]]),
                    rounds(["a"] * 14)]
        missing = good.copy()
        missing.loc[2, "ct_tier"] = None
        variants.append(missing)
        for bad in variants:
            with self.subTest(length=len(bad)):
                d, audit = _prepare_dataset(bad)
                self.assertTrue(d.empty)
                self.assertEqual(sum(audit["excluded_maps"].values()), 1)

    def test_map_id_is_accepted_without_demo_path(self):
        d, audit = _prepare_dataset(rounds(["a"] * 13).rename(columns={"demo_path": "map_id"}))
        self.assertEqual(len(d), 13)
        self.assertEqual(audit["usable_maps"], 1)

    def test_transition_does_not_cross_maps_or_half_resets(self):
        first = rounds(["a"] * 13)
        second = rounds(["b"] * 13, "/series/map2.dem")
        first.loc[first.round_num.eq(13), ["ct_tier", "t_tier"]] = ["force", "force"]
        trans = _fit_transition(pd.concat([first, second]))
        for distribution in trans["normal"].values():
            self.assertNotIn("force", distribution)
        self.assertEqual(trans["normal"][("regulation", "eco", 1)], {"eco": 1.})
        self.assertEqual(trans["normal"][("regulation", "full", 0)], {"full": 1.})
        self.assertEqual(trans["resets"]["regulation"], {("force", "force"): .5, ("full", "eco"): .5})

    def test_lookup_does_not_invent_overtime_training(self):
        d = normalize_map_rounds(rounds(["a"] * 13))
        lookup = _fit_round_lookup(d)
        self.assertTrue(lookup)
        self.assertFalse(any(key[1] == "overtime" for key in lookup))


class SimulationTests(unittest.TestCase):
    def test_regulation_win_does_not_require_future_economy(self):
        self.assertEqual(_simulate(probabilities(0), {}, "de_mirage", 23, 12, 10,
                                   "eco", "full", 1, np.random.default_rng(1)), 1.)

    def test_12_12_requires_four_overtime_wins_and_ot_half_switch(self):
        rng = SequenceRng([.9, .9, .9, .1])
        self.assertEqual(_simulate(probabilities(), transitions(), "de_mirage", 25, 12, 12,
                                   "eco", "full", 1, rng), 1.)
        self.assertEqual(rng.calls, 4)

    def test_repeated_overtime_retains_side_between_blocks(self):
        rng = SequenceRng([.9, .9, .9, .9, .9, .9, .1, .1, .1, .9])
        self.assertEqual(_simulate(probabilities(), transitions(), "de_mirage", 25, 12, 12,
                                   "eco", "full", 1, rng), 1.)
        self.assertEqual(rng.calls, 10)

    def test_observed_buy_at_round_13_is_not_overwritten(self):
        lookup = {("de_mirage", "pistol", "full", "force"): 0.}
        self.assertEqual(_simulate(lookup, {}, "de_mirage", 13, 12, 0,
                                   "force", "full", 1, SequenceRng([.9])), 1.)

    def test_next_half_uses_training_reset_distribution(self):
        lookup = {("de_mirage", "regular", "eco", "full"): 1.,
                  ("de_mirage", "pistol", "force", "full"): 0.}
        trans = {"resets": {"regulation": {("force", "full"): 1.}}}
        self.assertEqual(_simulate(lookup, trans, "de_mirage", 12, 11, 0,
                                   "eco", "full", 1, SequenceRng([.1, .9])), 1.)

    def test_missing_probabilities_transitions_and_reset_fail_closed(self):
        cases = [({}, transitions(), 23, 12, 10, "missing_round_probability"),
                 (probabilities(), {}, 1, 0, 0, "missing_equipment_transition"),
                 (probabilities(), {}, 12, 10, 1, "missing_equipment_transition")]
        for lookup, trans, rnd, a, b, error in cases:
            with self.subTest(error=error, rnd=rnd), self.assertRaisesRegex(ValueError, error):
                _simulate(lookup, trans, "de_mirage", rnd, a, b, "eco", "full", 1,
                          np.random.default_rng(0))

    def test_cap_raises_instead_of_assigning_unfinished_paths_a_loss(self):
        with self.assertRaisesRegex(ValueError, "simulation_round_cap_reached"):
            _simulate(probabilities(), transitions(), "de_mirage", 25, 12, 12,
                      "eco", "full", 1, SequenceRng([.9]), max_rounds=25)

    def test_invalid_or_terminal_start_is_rejected(self):
        for rnd, a, b in [(14, 13, 0), (26, 14, 11), (2, 0, 0), (1, -1, 1), (13.5, 6, 6)]:
            with self.subTest(state=(rnd, a, b)), self.assertRaises(ValueError):
                _simulate(probabilities(), transitions(), "de_mirage", rnd, a, b,
                          "eco", "full", 1, np.random.default_rng(0))

    def test_nonfinite_probability_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid_round_probability"):
            _simulate(probabilities(np.nan), transitions(), "de_mirage", 1, 0, 0,
                      "eco", "full", 1, np.random.default_rng(0))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.concat([rounds(["a"] * 13, f"/{s}/map{m}.dem", s)
                                for s in ("one", "two", "three") for m in (1, 2)], ignore_index=True)
        self.meta = pd.DataFrame([{"map_id": f"/{s}/map{m}.dem", "match_id": s,
                                   "start_at": pd.Timestamp(f"2026-01-0{i}T10:00Z"),
                                   "available_at": pd.Timestamp(f"2026-01-0{i}T13:00Z")}
                                  for i, s in enumerate(("one", "two", "three"), 1) for m in (1, 2)])

    def test_evaluation_counts_maps_and_uses_only_completed_earlier_series(self):
        with patch("cs2ml.forward_sim.build_round_dataset", return_value=self.frame), \
                patch("cs2ml.forward_sim._fit_round_lookup", return_value={}) as fit, \
                patch("cs2ml.forward_sim._fit_transition", return_value={}), \
                patch("cs2ml.forward_sim._simulate", return_value=.6), patch("builtins.print"):
            result = evaluate_forward(1, self.meta, min_train_series=1)
        self.assertEqual(result["n"][1], 4)
        self.assertEqual(result["audit"]["usable_maps"], 6)
        self.assertEqual(len(fit.call_args_list), 2)
        self.assertEqual(set(fit.call_args_list[0].args[0].match_id), {"one"})
        self.assertEqual(set(fit.call_args_list[1].args[0].match_id), {"one", "two"})
        self.assertTrue(all(f["train_max_available_at"] < f["cutoff"] for f in result["folds"]))

    def test_simultaneous_series_and_unavailable_results_do_not_train_each_other(self):
        self.meta.loc[self.meta.match_id.eq("one"), "available_at"] = pd.Timestamp("2026-01-02T10:00Z")
        self.meta.loc[self.meta.match_id.eq("three"), "start_at"] = pd.Timestamp("2026-01-02T10:00Z")
        with patch("cs2ml.forward_sim.build_round_dataset", return_value=self.frame), \
                patch("cs2ml.forward_sim._fit_round_lookup") as fit, patch("builtins.print"):
            result = evaluate_forward(1, self.meta, min_train_series=1)
        fit.assert_not_called()
        self.assertEqual(result["n"][1], 0)

    def test_missing_or_ambiguous_metadata_never_falls_back(self):
        d, _ = _prepare_dataset(self.frame)
        with self.assertRaisesRegex(ValueError, "ambiguous_map_metadata"):
            attach_history_times(d, pd.concat([self.meta, self.meta.iloc[:1]]))
        with patch("cs2ml.forward_sim.build_round_dataset", return_value=self.frame):
            report = evaluate_forward(1, pd.DataFrame())
        self.assertIn("availability_metadata_required", report["error"])


if __name__ == "__main__":
    unittest.main()
