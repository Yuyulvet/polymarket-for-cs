from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from cs2ml.inplay_map_research import (ALL_NUMERIC, BASE, CATEGORICAL, FIVEE_BASE,
                                      FIVEE_STATE, FORBIDDEN_FEATURES,
                                      VARIANTS, _fit_predict, _metrics, evaluate_frame,
                                      predict_variant_bundle,
                                      forward_splits, hierarchical_weights, mirror_features,
                                      paired_series_increment, prepare_map_events, run)
from cs2ml.map1_data import first_ct_is_ct, path_key, roster_key


A, B = roster_key(["1", "2", "3", "4", "5"]), roster_key(["6", "7", "8", "9", "10"])
T = pd.Timestamp("2026-01-01T12:00:00Z")


def fixture(n_series=8, maps_per_series=2):
    events, rounds, labels, history = [], [], [], []
    for series in range(n_series):
        for arena in range(maps_per_series):
            mid = f"series-{series}"
            path = f"d:/demos/{mid}/game-m{arena+1}-mirage.dem"
            a_wins = (series + arena) % 2 == 0
            winner = A if a_wins else B
            start = T + pd.Timedelta(days=series)
            history.append({"map_id": path, "match_id": mid, "start_at": start,
                            "available_at": start + pd.Timedelta(hours=3), "y": int(a_wins)})
            labels.append({"demo_path": path, "map_id": path, "roster_a": A, "roster_b": B,
                           "winner_roster": winner, "score_a": 13 if a_wins else 0,
                           "score_b": 0 if a_wins else 13, "n_rounds": 13, "complete": True,
                           "history_check": "matched"})
            for number in range(1, 14):
                ct, tt = (A, B) if first_ct_is_ct(number) else (B, A)
                freeze = number * 10000
                rounds.append({"demo_path": path, "map_name": "de_mirage", "round_num": number,
                               "ct_roster": ct, "t_roster": tt, "roster_a": A, "roster_b": B,
                               "ct_is_a": ct == A, "score_a_before": number-1 if a_wins else 0,
                               "score_b_before": 0 if a_wins else number-1, "winner_roster": winner,
                               "winner_side": "CT" if winner == ct else "T", "round_start_tick": freeze,
                               "round_end_tick": freeze+4000, "ct_equip0": 4000.+number*10,
                               "t_equip0": 3500.+number*10})
                for offset, kind in ((0, "round_start"), (100, "kill")):
                    ct_alive = 5 if not offset else 4 + number % 2
                    t_alive = 5 if not offset else 5 - number % 2
                    events.append({"demo_path": path, "map_name": "de_mirage", "match_id": mid,
                                   "round_num": number, "tick": freeze+offset, "event_tick": freeze+offset,
                                   "state_tick": freeze+offset+1, "event_type": kind, "schema_version": 2,
                                   "observation_basis": "first_full_tick_after_event_batch",
                                   "ct_roster": ct, "t_roster": tt, "ct_alive": ct_alive, "t_alive": t_alive,
                                   "ct_hp": ct_alive*90., "t_hp": t_alive*80., "bomb_state": 0,
                                   "weapon": "ak47" if offset else None,
                                   "ct_contact": 600., "t_contact": 700., "ct_spread": 800.,
                                   "t_spread": 900., "ct_aiming": 100., "t_aiming": 200.,
                                   # Deliberately present but never accepted as features.
                                   "label_ct_win": int(winner == ct), "future_score": 999})
    return tuple(pd.DataFrame(rows) for rows in (events, rounds, labels, history))


class MapEventPreparationTests(unittest.TestCase):
    def setUp(self):
        self.events, self.rounds, self.labels, self.history = fixture(1, 1)

    def prepare(self, events=None, rounds=None, labels=None, history=None):
        return prepare_map_events(self.events if events is None else events,
                                  self.rounds if rounds is None else rounds,
                                  self.labels if labels is None else labels,
                                  self.history if history is None else history)

    def test_target_is_new_complete_map_winner_and_current_round_result_is_not_feature(self):
        history = self.history.assign(y=0)
        data, audit = self.prepare(history=history)
        self.assertEqual(len(data), 26)
        self.assertTrue(data.y.eq(1).all())
        self.assertNotIn("label_ct_win", data)
        self.assertNotIn("future_score", data)
        self.assertNotIn("round_end_tick", data)
        self.assertEqual(audit["retained_maps"], 1)
        for columns in VARIANTS.values():
            self.assertFalse(set(columns) & FORBIDDEN_FEATURES)

    def test_pre_round_scores_and_actual_sides_survive_halftime(self):
        data, _ = self.prepare()
        early = data[data.round_num.eq(1)].iloc[0]
        late = data[data.round_num.eq(13)].iloc[0]
        self.assertEqual((early.score_gap, early.a_is_ct), (0, 1))
        self.assertEqual((late.score_gap, late.a_is_ct), (12, -1))
        self.assertEqual(late.starting_equip_gap, -500.)
        self.assertAlmostEqual(early.elapsed_seconds, 1/64.)

    def test_incomplete_or_history_quarantined_maps_are_excluded_wholesale(self):
        for change in ({"complete": False}, {"history_check": "missing_chronology"},
                       {"history_check": "new_ledger_disagrees_with_history"}):
            with self.subTest(change=change):
                data, audit = self.prepare(labels=self.labels.assign(**change))
                self.assertTrue(data.empty)
                self.assertEqual(sum(audit["excluded_maps"].values()), 1)
                self.assertEqual(audit["excluded_event_rows"]["map_ineligible"], 26)

    def test_partial_round_ledger_cannot_be_used_to_recreate_map_label(self):
        data, audit = self.prepare(rounds=self.rounds.iloc[:-1])
        self.assertTrue(data.empty)
        self.assertIn("incomplete_or_duplicate_map_round_history", audit["excluded_maps"])

    def test_corrupted_prior_score_winner_or_side_schedule_excludes_map(self):
        for column, value in (("score_a_before", 99), ("winner_roster", B),
                              ("ct_roster", B), ("ct_is_a", False)):
            rounds = self.rounds.copy()
            rounds.loc[0, column] = value
            with self.subTest(column=column):
                data, audit = self.prepare(rounds=rounds)
                self.assertTrue(data.empty)
                self.assertEqual(sum(audit["excluded_maps"].values()), 1)

    def test_duplicate_map_labels_or_event_state_keys_are_errors(self):
        with self.assertRaisesRegex(ValueError, "map_label_identity"):
            self.prepare(labels=pd.concat([self.labels, self.labels]))
        with self.assertRaisesRegex(ValueError, "duplicate_event_state_identity"):
            self.prepare(events=pd.concat([self.events, self.events.iloc[[0]]]))

    def test_missing_history_check_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "history_check"):
            self.prepare(labels=self.labels.drop(columns="history_check"))

    def test_terminal_state_excluded_but_postplant_t_wipe_retained(self):
        events = self.events.copy()
        events.loc[1, ["t_alive", "t_hp", "bomb_state"]] = [0, 0, 1]
        data, _ = self.prepare(events=events)
        self.assertIn(self.events.loc[1, "state_tick"], data.state_tick.tolist())
        self.assertEqual(data.loc[data.state_tick.eq(10101), "bomb_advantage"].iloc[0], -1)
        events.loc[1, "bomb_state"] = 0
        data, audit = self.prepare(events=events)
        self.assertNotIn(10101, data.state_tick.tolist())
        self.assertEqual(audit["excluded_event_rows"]["t_eliminated_before_plant"], 1)

    def test_state_timing_boundary_and_current_roster_mismatches_fail_closed(self):
        for changes in ({"state_tick": 10100}, {"schema_version": 1},
                        {"ct_roster": B}, {"ct_hp": 9999}, {"bomb_state": 2}):
            events = self.events.copy()
            for key, value in changes.items():
                events.loc[1, key] = value
            with self.subTest(changes=changes):
                data, audit = self.prepare(events=events)
                self.assertEqual(len(data), 25)
                self.assertEqual(sum(audit["excluded_event_rows"].values()), 1)

    def test_later_state_mutations_do_not_change_earlier_features(self):
        before, _ = self.prepare()
        events = self.events.copy()
        events.loc[events.round_num.gt(5), "ct_hp"] = events.loc[events.round_num.gt(5), "ct_alive"] * 20
        after, _ = self.prepare(events=events)
        pd.testing.assert_frame_equal(before[before.round_num.le(5)], after[after.round_num.le(5)])

    def test_unmapped_chronology_is_counted_not_guessed(self):
        history = self.history.copy()
        history["map_id"] = "d:/unrelated.dem"
        data, audit = self.prepare(history=history)
        self.assertTrue(data.empty)
        self.assertEqual(audit["chronology"]["rows_missing_clean_chronology"], 26)


class WeightingAndTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = prepare_map_events(*fixture())

    def test_weights_equalize_series_maps_rounds_and_events(self):
        rows = pd.DataFrame([
            {"match_id": "s1", "map_id": "m1", "round_num": 1},
            {"match_id": "s1", "map_id": "m1", "round_num": 1},
            {"match_id": "s1", "map_id": "m1", "round_num": 2},
            {"match_id": "s1", "map_id": "m2", "round_num": 1},
            {"match_id": "s2", "map_id": "m3", "round_num": 1},
        ])
        weights = hierarchical_weights(rows)
        np.testing.assert_allclose(weights, [.125, .125, .25, .5, 1.])
        np.testing.assert_allclose(pd.Series(weights).groupby(rows.match_id).sum(), [1., 1.])

    def test_whole_series_chronological_blocks_use_only_available_past_labels(self):
        data = self.data.copy()
        data.loc[data.match_id.eq("series-0"), "available_at"] = T + pd.Timedelta(days=100)
        splits = list(forward_splits(data, n_folds=3, min_train_series=2))
        self.assertTrue(splits)
        for train, test in splits:
            tr, te = data.iloc[train], data.iloc[test]
            self.assertNotIn("series-0", set(tr.match_id))
            self.assertFalse(set(tr.match_id) & set(te.match_id))
            self.assertLess(tr.available_at.max(), te.start_at.min())
            for mid in te.match_id.unique():
                self.assertEqual(len(te[te.match_id.eq(mid)]), len(data[data.match_id.eq(mid)]))

    def test_equal_time_series_stay_together_and_strict_equal_availability_is_excluded(self):
        data = self.data.copy()
        data.loc[data.match_id.eq("series-5"), "start_at"] = T + pd.Timedelta(days=4)
        data.loc[data.match_id.eq("series-0"), "available_at"] = T + pd.Timedelta(days=3)
        for train, test in forward_splits(data, n_folds=3, min_train_series=1):
            selected = set(data.iloc[test].match_id)
            self.assertEqual("series-4" in selected, "series-5" in selected)
            if data.iloc[test].start_at.min() == T + pd.Timedelta(days=3):
                self.assertNotIn("series-0", set(data.iloc[train].match_id))

    def test_fit_outputs_exact_roster_swap_complements(self):
        train = self.data[self.data.match_id.isin(["series-0", "series-1"])]
        test = self.data[self.data.match_id.eq("series-2")]
        normal = _fit_predict(train, test, BASE, CATEGORICAL["score_side"])
        opposite = _fit_predict(train, mirror_features(test), BASE, CATEGORICAL["score_side"])
        np.testing.assert_allclose(normal + opposite, 1., atol=1e-12)

    def test_fivee_variants_use_only_verified_live_fields(self):
        self.assertEqual(VARIANTS["fivee_score"], FIVEE_BASE)
        self.assertEqual(VARIANTS["fivee_state"], FIVEE_STATE)
        self.assertNotIn("elapsed_seconds", FIVEE_STATE)
        self.assertNotIn("starting_equip_gap", FIVEE_STATE)
        self.assertNotIn("bomb_advantage", FIVEE_STATE)

    def test_future_training_data_cannot_enter_preprocessor_or_earlier_predictions(self):
        changed = self.data.copy()
        future = changed.match_id.isin(["series-6", "series-7"])
        changed.loc[future, "y"] = 1 - changed.loc[future, "y"]
        first, before = evaluate_frame(self.data, n_folds=2, min_train_series=2)
        second, _ = evaluate_frame(changed, n_folds=2, min_train_series=2)
        columns = ["p_" + name for name in VARIANTS]
        pd.testing.assert_frame_equal(first[columns], second[columns])
        self.assertEqual(before["status"], "diagnostic_only")

    def test_same_event_cohort_fixed_layers_and_no_promotion_claim(self):
        predictions, report = evaluate_frame(self.data, n_folds=2, min_train_series=2)
        self.assertGreater(report["scored_series"], 0)
        self.assertEqual({model["event_rows"] for model in report["models"].values()}, {report["scored_event_rows"]})
        self.assertEqual(report["models"]["constant_50"]["brier"], .25)
        self.assertIsNone(report["promoted_model"])
        self.assertIsNone(report["execution_pnl"])
        self.assertFalse(report["live_trading_enabled"])
        self.assertEqual(report["target"], "canonical_roster_a_map_winner")
        for name in VARIANTS:
            self.assertTrue(predictions["p_" + name].notna().equals(predictions.p_constant_50.notna()))
        increments = report["paired_loss_increment_candidate_minus_reference"]
        self.assertIn("fivee_state_minus_fivee_score", increments)
        self.assertIn("fivee_state_minus_score_side", increments)
        for fold in report["folds"]:
            self.assertEqual(fold["series_overlap"], 0)
            self.assertFalse(set(fold["train_series_ids"]) & set(fold["test_series_ids"]))

    def test_nonfinite_state_is_dropped_from_every_candidate_cohort(self):
        data = self.data.copy()
        data.loc[0, "hp_gap"] = np.nan
        predictions, report = evaluate_frame(data, n_folds=2, min_train_series=2)
        self.assertEqual(report["excluded_nonfinite_or_invalid_target"], 1)
        self.assertEqual(len(predictions), len(data)-1)

    def test_calibration_counts_all_boundaries_once_using_hierarchical_weights(self):
        data = self.data.iloc[:11].copy()
        p = np.arange(11) / 10
        metrics = _metrics(data, p)
        self.assertEqual(sum(row["event_rows"] for row in metrics["calibration"]), 11)
        self.assertAlmostEqual(sum(row["weight"] for row in metrics["calibration"]), 1.)
        self.assertEqual([row["event_rows"] for row in metrics["calibration"]], [1]*9+[2])

    def test_insufficient_history_has_no_fabricated_scores(self):
        _, report = evaluate_frame(self.data, min_train_series=100)
        self.assertEqual(report["status"], "insufficient_past_series")
        self.assertEqual(report["scored_event_rows"], 0)
        self.assertNotIn("brier", report["models"]["score_side"])

    def test_paired_bootstrap_resamples_series_not_duplicate_event_rows(self):
        data = self.data.copy()
        data["candidate"] = np.where(data.y.eq(1), .7, .3)
        data["reference"] = .5
        first = paired_series_increment(data, "candidate", "reference", draws=100)
        repeated = pd.concat([data, data], ignore_index=True)
        second = paired_series_increment(repeated, "candidate", "reference", draws=100)
        self.assertEqual(first, second)
        self.assertEqual(first["series_resampling"]["series_n"], 8)
        self.assertEqual(first["series_resampling"]["resampling_unit"], "whole_series")
        self.assertAlmostEqual(first["brier"], -.16)
        self.assertIsNotNone(first["series_resampling"]["brier_interval"])

    def test_one_series_does_not_fabricate_a_bootstrap_interval(self):
        data = self.data[self.data.match_id.eq("series-0")].assign(candidate=.5, reference=.5)
        report = paired_series_increment(data, "candidate", "reference")
        self.assertEqual(report["brier"], 0.)
        self.assertIsNone(report["series_resampling"]["brier_interval"])


class ResearchArtifactTests(unittest.TestCase):
    def test_existing_output_is_refused_before_reading(self):
        with tempfile.TemporaryDirectory() as directory, patch("cs2ml.inplay_map_research.pd.read_parquet") as read:
            with self.assertRaisesRegex(FileExistsError, "refusing_overwrite"):
                run(Path(directory) / "inputs", directory)
            read.assert_not_called()

    def test_temporary_artifact_roundtrip_records_inputs_code_and_diagnostic_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            e, r, labels, history = fixture(4)
            for name, frame in (("inround_events", e), ("round_states", r), ("map_labels", labels), ("history", history)):
                frame.to_parquet(inputs / f"{name}.parquet", index=False)
            output = root / "research"
            report = run(inputs, output, inputs / "history.parquet", n_folds=2, min_train_series=1)
            saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["version"], report["version"])
            self.assertEqual(len(saved["inputs"]), 4)
            self.assertEqual(len(saved["code_hashes"]), 3)
            self.assertTrue((output / "predictions.parquet").is_file())
            self.assertTrue((output / "models.joblib").is_file())
            artifact = joblib.load(output / "models.joblib")
            self.assertTrue(artifact["paper_only"])
            self.assertFalse(artifact["promoted"])
            data, _ = prepare_map_events(e, r, labels, history)
            sample = data.iloc[:3]
            normal = predict_variant_bundle(artifact["variants"]["fivee_state"], sample)
            opposite = predict_variant_bundle(
                artifact["variants"]["fivee_state"], mirror_features(sample))
            np.testing.assert_allclose(normal + opposite, 1., atol=1e-12)
            self.assertIsNone(saved["promoted_model"])


if __name__ == "__main__":
    unittest.main()
