from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from cs2ml.midround import (CACHE_NAME, CACHE_VERSION, _auc, _validate_cache, build_midround,
                            compute_midround, round_windows, states_from_events,
                            temporal_series_splits)


def snapshots(tick=2020, flipped=False, dead=()):
    return pd.DataFrame([{"tick": tick, "steamid": str(i),
                          "team_name": "CT" if (i <= 5) != flipped else "TERRORIST",
                          "is_alive": i not in dead} for i in range(1, 11)])


def kills(*items):
    return pd.DataFrame(items, columns=["tick", "attacker_steamid", "user_steamid"])


def windows():
    return round_windows(pd.DataFrame({"tick": [100]}),
                         pd.DataFrame({"round": [1], "tick": [3000], "winner": ["CT"]}))


def extract(events=None, snap=None, win=None):
    return states_from_events(Path("series/map1.dem"), windows() if win is None else win,
                              kills() if events is None else events,
                              snapshots() if snap is None else snap)


class MidroundBoundaryTests(unittest.TestCase):
    def test_parser_empty_event_lists_are_not_errors(self):
        parser = Mock()
        parser.parse_event.side_effect = lambda name: {
            "round_freeze_end": pd.DataFrame({"tick": [100]}),
            "round_end": pd.DataFrame({"round": [1], "tick": [3000], "winner": ["CT"]}),
        }.get(name, [])
        parser.parse_ticks.return_value = snapshots()
        with patch("cs2ml.midround.DemoParser", return_value=parser):
            result = compute_midround(Path("series/map1.dem"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].first_kill_side, "none")

    def test_missing_round_events_as_lists_return_no_states(self):
        parser = Mock()
        parser.parse_event.return_value = []
        with patch("cs2ml.midround.DemoParser", return_value=parser):
            self.assertIsNone(compute_midround(Path("series/map1.dem")))
        parser.parse_ticks.assert_not_called()

    def test_future_first_kill_is_not_visible(self):
        result = extract(kills((2021, "1", "6")))
        self.assertEqual(result.iloc[0].first_kill_side, "none")
        self.assertEqual((result.iloc[0].ct_alive, result.iloc[0].t_alive), (5, 5))

    def test_exact_cutoff_is_inclusive_but_pre_freeze_death_is_not(self):
        result = extract(kills((99, "6", "1"), (2020, "1", "6")), snapshots(dead=(6,)))
        self.assertEqual(result.iloc[0].first_kill_side, "CT")
        self.assertEqual(result.iloc[0].t_alive, 4)

    def test_ended_at_or_before_cutoff_and_unknown_end_are_excluded(self):
        freeze = pd.DataFrame({"tick": [100]})
        for end in (2019, 2020):
            with self.subTest(end=end):
                self.assertTrue(round_windows(freeze, pd.DataFrame({"round": [1], "tick": [end], "winner": ["CT"]})).empty)
        self.assertTrue(round_windows(freeze, pd.DataFrame(columns=["round", "tick", "winner"])).empty)

    def test_end_must_belong_to_same_round_interval(self):
        freeze = pd.DataFrame({"tick": [100, 4000]})
        ends = pd.DataFrame({"round": [1, 2], "tick": [5000, 7000], "winner": ["CT", "T"]})
        self.assertEqual(round_windows(freeze, ends).round_num.tolist(), [2])

    def test_bomb_terminal_event_excludes_stale_round_end_window(self):
        freeze = pd.DataFrame({"tick": [100]})
        ends = pd.DataFrame({"round": [1], "tick": [3000], "winner": ["CT"]})
        for terminal_tick in (1500, 2020):
            with self.subTest(terminal_tick=terminal_tick):
                self.assertTrue(round_windows(freeze, ends, terminal_ticks=[terminal_tick]).empty)
        self.assertEqual(len(round_windows(freeze, ends, terminal_ticks=[99, 2021])), 1)

    def test_duplicate_freeze_or_round_end_fails_closed(self):
        freeze = pd.DataFrame({"tick": [100]})
        ends = pd.DataFrame({"round": [1], "tick": [3000], "winner": ["CT"]})
        with self.assertRaisesRegex(ValueError, "duplicate_freeze_ticks"):
            round_windows(pd.concat([freeze, freeze]), ends)
        with self.assertRaisesRegex(ValueError, "ambiguous_round_end_identity"):
            round_windows(freeze, pd.concat([ends, ends]))

    def test_nonfinite_or_fractional_ticks_are_rejected(self):
        for value in (np.nan, 2.5, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                round_windows(pd.DataFrame({"tick": [value]}), pd.DataFrame(columns=["round", "tick", "winner"]))

    def test_overtime_uses_actual_snapshot_sides(self):
        win = windows()
        win["round_num"] = 28
        result = extract(kills((150, "1", "6")), snapshots(flipped=True, dead=(6,)), win)
        self.assertEqual(result.iloc[0].first_kill_side, "T")
        self.assertEqual((result.iloc[0].ct_alive, result.iloc[0].t_alive), (4, 5))

    def test_world_death_and_teamkill_do_not_become_enemy_opening(self):
        events = kills((150, None, "1"), (200, "2", "3"), (250, "6", "4"))
        result = extract(events, snapshots(dead=(1, 3, 4)))
        self.assertEqual(result.iloc[0].first_kill_side, "T")
        self.assertEqual((result.iloc[0].ct_deaths30, result.iloc[0].ct_alive), (3, 2))

    def test_same_tick_opposite_openings_are_order_independent(self):
        events = kills((150, "1", "6"), (150, "7", "2"))
        normal = extract(events, snapshots(dead=(2, 6)))
        reverse = extract(events.iloc[::-1], snapshots(dead=(2, 6)))
        pd.testing.assert_frame_equal(normal, reverse)
        self.assertEqual(normal.iloc[0].first_kill_side, "simultaneous")

    def test_elimination_is_excluded_even_before_delayed_round_end(self):
        for dead in ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10)):
            with self.subTest(dead=dead):
                self.assertTrue(extract(snap=snapshots(dead=dead)).empty)

    def test_missing_duplicate_or_nonexact_player_snapshot_is_excluded(self):
        incomplete = snapshots().iloc[:-1]
        duplicate = snapshots()
        duplicate.loc[9, "steamid"] = "1"
        late = snapshots(tick=2021)
        missing_alive = snapshots()
        missing_alive["is_alive"] = missing_alive.is_alive.astype(object)
        missing_alive.loc[0, "is_alive"] = None
        for snap in (incomplete, duplicate, late, missing_alive):
            with self.subTest(rows=len(snap)):
                self.assertTrue(extract(snap=snap).empty)

    def test_compute_requests_only_active_round_cutoffs(self):
        parser = Mock()
        events = {"round_freeze_end": pd.DataFrame({"tick": [100, 4000]}),
                  "round_end": pd.DataFrame({"round": [1, 2], "tick": [2020, 7000], "winner": ["CT", "T"]}),
                  "bomb_defused": pd.DataFrame(), "bomb_exploded": pd.DataFrame(),
                  "player_death": kills((7001, "1", "6"))}
        parser.parse_event.side_effect = events.__getitem__
        parser.parse_ticks.return_value = snapshots(tick=5920)
        with patch("cs2ml.midround.DemoParser", return_value=parser):
            result = compute_midround(Path("series/map.dem"))
        parser.parse_ticks.assert_called_once_with(["team_name", "is_alive"], ticks=[5920])
        self.assertEqual(result.round_num.tolist(), [2])
        self.assertEqual(result.iloc[0].first_kill_side, "none")


class MidroundCacheTests(unittest.TestCase):
    def test_old_cache_is_not_reused_and_missing_safe_cache_does_not_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract().to_parquet(root / "midround.parquet", index=False)
            with patch("cs2ml.midround.config.DATA_DIR", root), patch("cs2ml.midround.discover_demos") as discover:
                with self.assertRaisesRegex(FileNotFoundError, "explicitly rebuild"):
                    build_midround()
            discover.assert_not_called()

    def test_rebuild_cannot_overwrite_previous_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / CACHE_NAME
            extract().to_parquet(cache, index=False)
            original = cache.read_bytes()
            with self.assertRaises(FileExistsError):
                build_midround(rebuild=True, cache_path=cache)
            self.assertEqual(cache.read_bytes(), original)

    def test_versionless_or_closed_snapshot_cache_is_rejected(self):
        for broken in (extract().drop(columns="schema_version"),
                       extract().assign(schema_version=1),
                       extract().assign(round_end_tick=2020)):
            with self.assertRaisesRegex(ValueError, "rebuild_required"):
                _validate_cache(broken)

    def test_invalid_cached_state_values_are_rejected(self):
        for column, value in (("ct_alive", 6), ("t_alive", 0), ("ct_alive", 2.5),
                              ("ct_deaths30", 1), ("first_kill_side", "future_CT"),
                              ("freeze_tick", np.nan), ("round_num", 0), ("round_end_tick", np.inf)):
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                _validate_cache(extract().assign(**{column: value}))

    def test_explicit_empty_demo_list_does_not_discover_or_write(self):
        with patch("cs2ml.midround.discover_demos") as discover:
            self.assertTrue(build_midround([]).empty)
        discover.assert_not_called()

    def test_explicit_rebuild_creates_new_version_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "midround.parquet"
            extract().drop(columns="schema_version").to_parquet(old, index=False)
            before = old.read_bytes()
            with patch("cs2ml.midround.config.DATA_DIR", root), \
                 patch("cs2ml.midround.discover_demos", return_value=[Path("series/map1.dem")]), \
                 patch("cs2ml.midround.compute_midround", return_value=extract()):
                created = build_midround(rebuild=True)
                loaded = build_midround()
            self.assertTrue(created.schema_version.eq(CACHE_VERSION).all())
            pd.testing.assert_frame_equal(created, loaded)
            self.assertEqual(old.read_bytes(), before)


class MidroundTemporalTests(unittest.TestCase):
    def fixture(self):
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        rows = []
        for i in range(5):
            for arena in ("map1", "map2"):
                rows.append({"match_id": f"series{i}", "demo_path": arena,
                             "start_at": start + pd.Timedelta(days=i),
                             "available_at": start + pd.Timedelta(days=i, hours=4)})
        return pd.DataFrame(rows)

    def test_splits_are_whole_series_and_results_strictly_past(self):
        data = self.fixture()
        for train, test in temporal_series_splits(data):
            self.assertFalse(set(data.iloc[train].match_id) & set(data.iloc[test].match_id))
            self.assertLess(data.iloc[train].available_at.max(), data.iloc[test].start_at.min())
            self.assertEqual(len(test), 2)

    def test_tied_starts_and_equal_availability_do_not_enter_training(self):
        data = self.fixture()
        data.loc[data.match_id.eq("series2"), "start_at"] = data.loc[6, "start_at"]
        data.loc[data.match_id.eq("series2"), "available_at"] = data.loc[6, "available_at"]
        data.loc[data.match_id.eq("series1"), "available_at"] = data.loc[6, "start_at"]
        splits = list(temporal_series_splits(data, min_train_series=1))
        train, test = next((tr, te) for tr, te in splits if "series2" in set(data.iloc[te].match_id))
        self.assertEqual(set(data.iloc[test].match_id), {"series2", "series3"})
        self.assertEqual(set(data.iloc[train].match_id), {"series0"})

    def test_missing_or_inconsistent_timing_has_no_random_cv_fallback(self):
        data = self.fixture()
        with self.assertRaisesRegex(ValueError, "no_group_kfold_fallback"):
            list(temporal_series_splits(data.drop(columns="available_at")))
        data.loc[1, "start_at"] += pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "inconsistent_series_timing"):
            list(temporal_series_splits(data))

    def test_appending_future_series_does_not_change_earlier_splits(self):
        data = self.fixture()
        before = list(temporal_series_splits(data.iloc[:8]))
        after = list(temporal_series_splits(data))
        self.assertEqual(len(after), len(before) + 1)
        for expected, actual in zip(before, after):
            np.testing.assert_array_equal(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])

    def test_auc_disallows_outcome_or_future_tick_features(self):
        for feature in ("round_end_tick", "label_ct_win"):
            with self.assertRaisesRegex(ValueError, "unapproved_midround_model_feature"):
                _auc(self.fixture(), [feature])

    def test_temporal_auc_scores_only_after_trainable_history(self):
        frame = self.fixture()
        frame["label_ct_win"] = np.arange(len(frame)) % 2
        frame["ct_alive"] = 3 + frame.label_ct_win
        self.assertGreater(_auc(frame, ["ct_alive"]), .99)


if __name__ == "__main__":
    unittest.main()
