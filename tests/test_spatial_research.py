from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cs2ml.spatial_research import (extract_one, features, paired_delta, snapshot_features)
from cs2ml.inround_model import forward_splits


def fixture():
    players = pd.DataFrame([{
        "steamid": str(1000+i), "team_name": "CT" if i < 5 else "TERRORIST",
        "X": 100.*i, "Y": float(i % 3)*200, "Z": 0., "health": 100,
        "armor_value": 100, "current_equip_value": 4000., "is_alive": True,
        "last_place_name": "BombsiteA" if i < 5 else "TopofMid",
        "active_weapon_name": "AK-47", "tick": 1060,
    } for i in range(10)])
    row = {"round_num": 1, "ct_roster": ",".join(str(1000+i) for i in range(5)),
           "t_roster": ",".join(str(1000+i) for i in range(5, 10)),
           "ct_is_a": True, "score_a_before": 0, "score_b_before": 0,
           "round_start_tick": 100, "round_end_tick": 2500, "winner_side": "CT"}
    return players, row


class SpatialResearchTests(unittest.TestCase):
    def test_outcome_fields_do_not_change_features(self):
        players, row = fixture()
        expected = snapshot_features(players, row, 15)
        row.update(winner_side="T", first_kill_side="T", bomb_planted=True,
                   round_end_tick=99999, final_margin=5, future_attack_site="B")
        self.assertEqual(expected, snapshot_features(players, row, 15))

    def test_player_permutation_does_not_change_features(self):
        players, row = fixture()
        self.assertEqual(snapshot_features(players, row, 15),
                         snapshot_features(players.sample(frac=1, random_state=4), row, 15))

    def test_same_macro_different_deployment(self):
        players, row = fixture()
        original = snapshot_features(players, row, 15)
        players.loc[players.team_name.eq("CT"), "X"] += 2500
        players.loc[players.team_name.eq("CT"), "last_place_name"] = "BombsiteB"
        changed = snapshot_features(players, row, 15)
        self.assertEqual(original[0], changed[0])
        self.assertNotEqual(original[2], changed[2])

    def test_reject_damage_missing_players_region_and_identity(self):
        players, row = fixture()
        bad = players.copy()
        bad.loc[0, "health"] = 99
        with self.assertRaisesRegex(ValueError, "already_damaged"):
            snapshot_features(bad, row, 15)
        with self.assertRaisesRegex(ValueError, "ten_players"):
            snapshot_features(players.iloc[:-1], row, 15)
        bad = players.copy()
        bad.loc[0, "last_place_name"] = None
        with self.assertRaisesRegex(ValueError, "missing_region"):
            snapshot_features(bad, row, 15)
        bad = players.copy()
        bad.loc[0, "steamid"] = "9999"
        with self.assertRaisesRegex(ValueError, "roster_mismatch"):
            snapshot_features(bad, row, 15)

    def test_feature_ablation_excludes_targets(self):
        players, row = fixture()
        macro, coarse, deployment = snapshot_features(players, row, 15)
        records = [{"macro": macro, "coarse": coarse, "deployment": deployment,
                    "label_ct_win": 1, "label_ct_map_win": 0}]
        base = features(records, "macro")[0]
        rich = features(records, "regional_deployment")[0]
        self.assertLess(set(base), set(rich))
        self.assertNotIn("label_ct_win", rich)
        self.assertNotIn("label_ct_map_win", rich)

    def test_fixed_checkpoints_exclude_past_plant_not_future(self):
        players, row = fixture()
        snapshots = pd.concat([players.assign(tick=1060), players.assign(tick=2020)])
        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "sample.dem"
            demo.touch()
            with patch("cs2ml.spatial_research.DemoParser") as parser:
                parser.return_value.parse_ticks.return_value = snapshots
                result = extract_one((str(demo), [row], row["ct_roster"], [1500], tmp))
            self.assertEqual([r["elapsed"] for r in result["rows"]], [15])
            self.assertEqual(result["rejected"], {"already_planted": 1, "round_already_ended": 1})

    def test_forward_series_do_not_overlap_or_use_unavailable_labels(self):
        start = pd.Timestamp("2026-01-01", tz="UTC")
        data = pd.DataFrame([{"match_id": str(i), "start_at": start+pd.Timedelta(days=i),
            "available_at": start+pd.Timedelta(days=i+2)} for i in range(30) for _ in range(2)])
        folds = list(forward_splits(data, n_splits=4, min_train_series=2))
        self.assertTrue(folds)
        for train, test in folds:
            tr, te = data.iloc[train], data.iloc[test]
            self.assertFalse(set(tr.match_id) & set(te.match_id))
            self.assertLess(tr.available_at.max(), te.start_at.min())

    def test_uncertainty_is_clustered_by_series(self):
        d = pd.DataFrame({"match_id": ["a", "a", "b"], "demo_path": ["x", "x", "y"],
                          "round_num": [1, 1, 1], "label_ct_win": [1, 1, 0]})
        result = paired_delta(d, np.array([.5, .5, .5]), np.array([.8, .8, .2]), "label_ct_win")
        self.assertEqual(result["series"], 2)
        self.assertAlmostEqual(result["candidate_minus_baseline_brier"], -.21)


if __name__ == "__main__":
    unittest.main()
