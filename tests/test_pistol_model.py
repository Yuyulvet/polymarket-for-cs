from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from cs2ml.map1_data import features_at, roster_key
from cs2ml.pistol_model import (
    CONTEXT,
    _indexed_roster_features,
    _player_history_index,
    _player_rate,
    _roster_history_index,
    extract_pistol_events,
    fit_model,
    forward_splits,
    load_event_manifest,
    matchup_slugs,
    model_definitions,
    training_weights,
)


A = [str(76561198010000000 + i) for i in range(1, 6)]
B = [str(76561198020000000 + i) for i in range(1, 6)]
RA, RB = roster_key(A), roster_key(B)
T = pd.Timestamp("2026-08-01T12:00:00Z")
MANIFEST = Path(__file__).parents[1] / "cs2ml" / "starladder_fall_2026.json"


def synthetic_inputs(extra_rounds: int = 0):
    map_id = "/demos/mouz-vs-nrg-m1-mirage.dem"
    winners = [RA, RB] * 6 + [RA] + [RB] * extra_rounds
    records = []
    score_a = score_b = 0
    for round_num, winner in enumerate(winners, 1):
        first_half = round_num <= 12
        ct, t = (RA, RB) if first_half else (RB, RA)
        winner_side = "CT" if winner == ct else "T"
        records.append({
            "demo_path": map_id,
            "map_name": "de_mirage",
            "round_num": round_num,
            "winner_side": winner_side,
            "first_kill_side": winner_side,
            "outcome_type": "stomp" if round_num % 3 == 0 else "normal",
            "ct_roster": ct,
            "t_roster": t,
            "roster_a": RA,
            "roster_b": RB,
            "ct_is_a": ct == RA,
            "score_a_before": score_a,
            "score_b_before": score_b,
        })
        if winner == RA:
            score_a += 1
        else:
            score_b += 1
    history = pd.DataFrame([{
        "map_id": map_id,
        "match_id": "series-1",
        "map_name": "de_mirage",
        "roster_a": RA,
        "roster_b": RB,
        "start_at": T,
        "available_at": T + pd.Timedelta(hours=3),
    }])
    labels = pd.DataFrame([{
        "map_id": map_id,
        "complete": True,
        "history_check": "matched",
    }])
    return pd.DataFrame(records), history, labels


class PistolModelTests(unittest.TestCase):
    def test_player_index_normalizes_timestamp_units_and_excludes_equal_time(self):
        events = pd.DataFrame({
            "player": [A[0], A[0]], "won": [1, 0], "is_ct": [True, False],
            "map_name": ["de_mirage", "de_dust2"],
            "available_at": pd.Series([T-pd.Timedelta(days=10), T],
                                      dtype="datetime64[us, UTC]"),
        })
        rate, count = _player_rate(_player_history_index(events), [A[0]], T)
        weight = 2 ** (-10 / 90)
        self.assertAlmostEqual(count, weight, places=12)
        self.assertAlmostEqual(rate, (weight + 1) / (weight + 2), places=12)
        side_rate, side_count = _player_rate(
            _player_history_index(events), [A[0]], T, arena="de_mirage", is_ct=True)
        self.assertAlmostEqual(side_count, weight, places=12)
        self.assertAlmostEqual(side_rate, rate, places=12)

    def test_indexed_roster_features_are_identical_to_reference(self):
        rows = []
        for i, (left, right, arena, won) in enumerate((
                (RA, RB, "de_mirage", 1), (RB, RA, "de_dust2", 0),
                (RA, RB, "de_mirage", 0))):
            row = {
                "match_id": f"past-{i}", "map_name": arena, "roster_a": left,
                "roster_b": right, "start_at": T - pd.Timedelta(days=20-i),
                "available_at": T - pd.Timedelta(days=19-i), "y": won,
                "rounds": 24+i,
            }
            for side, scale in (("a", 1.0), ("b", .9)):
                values = {"round_wins": 13, "pistol_wins": 1,
                          "ct_wins": 7, "ct_rounds": 12,
                          "kills": 80, "deaths": 70, "damage": 6200,
                          "rounds_played": 100, "opening_kills": 12, "trade_kills": 15}
                row.update({f"{name}_{side}": value * scale for name, value in values.items()})
            rows.append(row)
        history = pd.DataFrame(rows)
        expected, expected_coverage = features_at(
            history, RA, RB, "de_mirage", T, exclude_match="current")
        actual, actual_coverage = _indexed_roster_features(
            _roster_history_index(history), RA, RB, "de_mirage", T, "current")
        self.assertEqual(expected_coverage, actual_coverage)
        for name, value in expected.items():
            self.assertAlmostEqual(value, actual[name], places=12, msg=name)

    def test_manifest_and_matchup_identity(self):
        manifest = load_event_manifest(MANIFEST)
        self.assertEqual(matchup_slugs("natus-vincere-vs-spirit-m1-mirage.dem"),
                         ("natus-vincere", "spirit"))
        magic = next(team for team in manifest["teams"] if team["name"] == "magic")
        self.assertIn("eksiver", magic["players"])
        self.assertNotIn("AW", magic["players"])

    def test_round_13_context_uses_only_completed_first_half(self):
        manifest = load_event_manifest(MANIFEST)
        base = synthetic_inputs()
        extended = synthetic_inputs(extra_rounds=7)
        first, _ = extract_pistol_events(*base, manifest)
        second, _ = extract_pistol_events(*extended, manifest)
        columns = ["y", "a_is_ct", *CONTEXT]
        pd.testing.assert_series_equal(
            first[first.pistol_round.eq(13)].iloc[0][columns],
            second[second.pistol_round.eq(13)].iloc[0][columns],
            check_names=False,
        )
        self.assertEqual(len(first), 2)
        self.assertTrue(first.target_event_team.all())

    def test_incomplete_round_ledger_is_quarantined(self):
        manifest = load_event_manifest(MANIFEST)
        rounds, history, labels = synthetic_inputs()
        rounds = rounds[rounds.round_num.ne(8)]
        with self.assertRaisesRegex(ValueError, "empty_or_duplicate_pistol_targets"):
            extract_pistol_events(rounds, history, labels, manifest)

    def test_model_design_has_no_collinear_map_side_indicators(self):
        definitions = model_definitions(["side_de_mirage", "side_de_dust2"], 13)
        self.assertEqual(definitions["side_only"], ["a_is_ct"])
        for features in definitions.values():
            self.assertFalse(any(name.startswith("side_de_") for name in features))

    def test_mirrored_fit_produces_complement_probabilities(self):
        train = pd.DataFrame({
            "f": [-1.0, -.4, .3, 1.2],
            "y": [0, 0, 1, 1],
            "match_id": ["a", "b", "c", "d"],
            "start_at": pd.date_range("2026-01-01", periods=4, tz="UTC"),
            "target_event_team": [False] * 4,
        })
        model = fit_model(train, ["f"], pd.Timestamp("2026-02-01T00:00:00Z"))
        probabilities = model.predict_proba(np.array([[.8], [-.8]]))[:, 1]
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)

    def test_series_weight_is_equalized_and_target_recent_is_prioritized(self):
        train = pd.DataFrame({
            "match_id": ["many", "many", "single", "target"],
            "start_at": [T] * 4,
            "target_event_team": [False, False, False, True],
        })
        weights = training_weights(train, T + pd.Timedelta(days=1))
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]), places=12)
        self.assertAlmostEqual(float(weights[3] / weights[2]), 3.0, places=12)

    def test_forward_splits_exclude_future_labels_and_test_series(self):
        starts = pd.date_range("2026-01-01", periods=12, tz="UTC")
        data = pd.DataFrame({
            "start_at": starts,
            "available_at": starts + pd.Timedelta(hours=3),
            "match_id": [f"m{i // 2}" for i in range(12)],
        })
        for train_index, test_index, test_start in forward_splits(data, n_blocks=3):
            train, test = data.iloc[train_index], data.iloc[test_index]
            self.assertTrue(train.available_at.lt(test_start).all())
            self.assertFalse(set(train.match_id) & set(test.match_id))


if __name__ == "__main__":
    unittest.main()
