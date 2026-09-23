from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cs2ml.map1_data import roster_key
from cs2ml.map1_player_research import PLAYER_FEATURES, build_player_features


A = [str(76561198000000001 + i) for i in range(5)]
B = [str(76561198000000006 + i) for i in range(5)]
RA, RB = roster_key(A), roster_key(B)
T = pd.Timestamp("2026-09-01T12:00:00Z")


def fixture(mid="past", available=None, arena="de_mirage"):
    path = f"d:/demos/{mid}-m1-mirage.dem"
    history = pd.DataFrame([{"map_id": path, "match_id": mid, "map_name": arena,
                             "available_at": T - pd.Timedelta(days=1) if available is None else available,
                             "roster_a": RA, "roster_b": RB, "y": 1, "rounds": 20}])
    players = []
    for side, roster in (("a", RA), ("b", RB)):
        for sid in roster.split(","):
            players.append({"demo_path": path, "steamid": sid, "roster_key": roster,
                            "map_name": arena, "kills": 20 if side == "a" else 10,
                            "deaths": 10 if side == "a" else 20, "rounds_played": 20,
                            "opening_kills": 3 if side == "a" else 1,
                            "opening_deaths": 1 if side == "a" else 3,
                            "headshots": 10 if side == "a" else 3,
                            "trade_kills": 5 if side == "a" else 1})
    target = pd.DataFrame([{"match_id": "target", "map_id": "d:/demos/target.dem",
                           "roster_a": RA, "roster_b": RB, "map_name": "de_mirage", "decision_at": T}], index=[72])
    return history, pd.DataFrame(players), target


class PlayerResearchTests(unittest.TestCase):
    def test_features_and_audit_are_finite_and_preserve_index(self):
        h, p, t = fixture()
        features, audit = build_player_features(h, p, t)
        self.assertEqual(features.index.tolist(), [72])
        self.assertEqual(len(PLAYER_FEATURES), 12)
        self.assertTrue(np.isfinite(features[PLAYER_FEATURES].to_numpy()).all())
        self.assertGreater(features.iloc[0].player_all_kd_gap, 0)
        self.assertGreater(features.iloc[0].player_all_win_gap, 0)
        self.assertLess(features.iloc[0].player_all_opening_death_gap, 0)
        self.assertEqual(features.iloc[0].player_all_known_a, 5)
        self.assertEqual(features.iloc[0].player_all_min_maps_a, 1)
        self.assertEqual(audit["accepted_player_rows"], 10)
        self.assertFalse(any("damage" in name or "adr" in name for name in PLAYER_FEATURES))

    def test_swap_negates_every_feature(self):
        h, p, t = fixture()
        normal, _ = build_player_features(h, p, t)
        t["roster_a"], t["roster_b"] = RB, RA
        flipped, _ = build_player_features(h, p, t)
        np.testing.assert_allclose(normal[PLAYER_FEATURES], -flipped[PLAYER_FEATURES], atol=1e-14)

    def test_future_equal_time_and_current_series_cannot_change_features(self):
        h, p, t = fixture()
        expected, _ = build_player_features(h, p, t)
        for mid, at in (("future", T + pd.Timedelta(days=5)), ("equal", T),
                        ("target", T - pd.Timedelta(days=2))):
            more_h, more_p, _ = fixture(mid, at)
            h, p = pd.concat([h, more_h]), pd.concat([p, more_p])
        actual, _ = build_player_features(h, p, t)
        pd.testing.assert_frame_equal(expected, actual)

    def test_same_map_is_excluded_even_if_series_is_mislabelled(self):
        h, p, t = fixture()
        t["map_id"] = h.iloc[0].map_id
        result, _ = build_player_features(h, p, t)
        np.testing.assert_array_equal(result[PLAYER_FEATURES], 0.)
        self.assertEqual(result.iloc[0].player_all_known_a, 0)

    def test_known_players_transfer_to_new_roster_and_unknown_is_reported(self):
        h, p, t = fixture()
        substitute = "76561198999999999"
        t["roster_a"] = roster_key([*A[:4], substitute])
        result, _ = build_player_features(h, p, t)
        row = result.iloc[0]
        self.assertEqual(row.player_all_known_a, 4)
        self.assertEqual(row.player_all_min_maps_a, 0)
        self.assertEqual(row.player_all_mean_maps_a, .8)
        self.assertEqual(row.player_all_unknown_a, [substitute])
        self.assertGreater(row.player_all_kd_gap, 0)

    def test_zero_history_is_finite_and_symmetric(self):
        h, p, t = fixture(available=T)
        result, _ = build_player_features(h, p, t)
        np.testing.assert_array_equal(result[PLAYER_FEATURES], 0.)
        self.assertEqual(result.iloc[0].player_all_unknown_a, A)
        self.assertIsNone(result.iloc[0].player_max_history_available_at)

    def test_other_map_only_changes_all_scope(self):
        h, p, t = fixture(arena="de_nuke")
        result, _ = build_player_features(h, p, t)
        self.assertGreater(result.iloc[0].player_all_kd_gap, 0)
        np.testing.assert_array_equal(result[[n for n in PLAYER_FEATURES if "_map_" in n]], 0.)
        self.assertEqual(result.iloc[0].player_map_known_a, 0)

    def test_duplicate_player_discards_whole_map(self):
        h, p, t = fixture()
        p = pd.concat([p, p.iloc[[0]]])
        result, audit = build_player_features(h, p, t)
        self.assertEqual(audit["accepted_player_rows"], 0)
        self.assertEqual(audit["excluded_player_rows"]["duplicate_player_map"], 11)
        np.testing.assert_array_equal(result[PLAYER_FEATURES], 0.)

    def test_untrusted_player_rows_require_clean_map_membership(self):
        h, p, t = fixture()
        extra = p.iloc[[0]].copy()
        extra["demo_path"] = "d:/demos/unclean.dem"
        p.loc[0, "roster_key"] = RB
        p.loc[1, "steamid"] = "76561198999999999"
        result, audit = build_player_features(h, pd.concat([p, extra]), t)
        self.assertEqual(audit["accepted_player_rows"], 8)
        self.assertEqual(audit["excluded_player_rows"]["roster_or_player_identity_mismatch"], 2)
        self.assertEqual(audit["excluded_player_rows"]["map_not_in_clean_history"], 1)
        self.assertEqual(result.iloc[0].player_all_known_a, 3)

    def test_bad_rounds_and_counts_are_not_silently_used(self):
        h, p, t = fixture()
        p.loc[0, "rounds_played"] = 21
        p.loc[1, "headshots"] = 999
        p.loc[2, "kills"] = -1
        _, audit = build_player_features(h, p, t)
        self.assertEqual(audit["accepted_player_rows"], 7)
        self.assertEqual(audit["excluded_player_rows"]["round_count_mismatch"], 1)
        self.assertEqual(audit["excluded_player_rows"]["inconsistent_player_counts"], 1)
        self.assertEqual(audit["excluded_player_rows"]["invalid_player_counts"], 1)

    def test_raw_timestamps_are_never_used_as_authority(self):
        h, p, t = fixture(available=T + pd.Timedelta(days=1))
        p["available_at"] = T - pd.Timedelta(days=100)
        result, _ = build_player_features(h, p, t)
        np.testing.assert_array_equal(result[PLAYER_FEATURES], 0.)

    def test_duplicate_clean_map_raises(self):
        h, p, t = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate clean-history"):
            build_player_features(pd.concat([h, h]), p, t)

    def test_naive_timestamp_and_overlapping_target_raise(self):
        h, p, t = fixture()
        t["decision_at"] = pd.Timestamp("2026-09-01")
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            build_player_features(h, p, t)
        t["decision_at"] = T
        t["roster_b"] = RA
        with self.assertRaisesRegex(ValueError, "Overlapping target"):
            build_player_features(h, p, t)

    def test_normalizes_paths_and_preserves_duplicate_target_index(self):
        h, p, t = fixture()
        p["demo_path"] = p.demo_path.str.replace("/", "\\", regex=False).str.upper()
        result, audit = build_player_features(h, p, pd.concat([t, t]))
        self.assertEqual(result.index.tolist(), [72, 72])
        self.assertEqual(audit["accepted_player_rows"], 10)
        pd.testing.assert_series_equal(result.iloc[0], result.iloc[1])

    def test_empty_targets_and_player_input(self):
        h, p, t = fixture()
        result, audit = build_player_features(h, p.iloc[:0], t)
        np.testing.assert_array_equal(result[PLAYER_FEATURES], 0.)
        self.assertEqual(audit["accepted_player_rows"], 0)
        result, audit = build_player_features(h, p, t.iloc[:0])
        self.assertTrue(result.empty)
        self.assertTrue(set(PLAYER_FEATURES).issubset(result.columns))


if __name__ == "__main__":
    unittest.main()
