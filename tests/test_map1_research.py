from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cs2ml.map1_data import FEATURES, build_features, roster_key
from cs2ml.map1_eligibility import roster_history_eligibility
from cs2ml.map1_model import fit_model, probability_metrics, walk_forward
from cs2ml.map1_player_research import PLAYER_FEATURES, build_player_features
from cs2ml.map1_research import (audit_inputs, cohort_metrics, compare_models,
                                 elo_predictions, main, model_definitions)
from test_map1 import RA, RB, T, history_row


RC = roster_key([str(76561198000000011 + i) for i in range(5)])


def research_fixture(n=12, sparse_last=False):
    history = pd.DataFrame([
        history_row(f"research-{i:03d}", T + pd.Timedelta(days=i),
                    "a" if i % 2 else "b")
        for i in range(n)
    ])
    if sparse_last:
        history.loc[n - 1, "roster_a"] = RC
    features = build_features(history)
    players = []
    for row in history.to_dict("records"):
        for side in ("a", "b"):
            for sid in row[f"roster_{side}"].split(","):
                players.append({
                    "demo_path": row["map_id"], "match_id": row["match_id"],
                    "map_name": row["map_name"], "steamid": sid,
                    "roster_key": row[f"roster_{side}"],
                    "kills": 12 if side == "a" else 8,
                    "deaths": 8 if side == "a" else 12,
                    "rounds_played": row["rounds"], "opening_kills": 2,
                    "opening_deaths": 2, "trade_kills": 3, "headshots": 4,
                })
    return history, features, pd.DataFrame(players)


class ResearchAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history, cls.features, _ = research_fixture(5)

    def test_valid_audit_is_explicitly_internal_and_does_not_mutate_inputs(self):
        h, f = self.history.copy(deep=True), self.features.copy(deep=True)
        report = audit_inputs(h, f)
        self.assertEqual((report["clean_maps"], report["map1_targets"]), (5, 5))
        self.assertEqual(report["lead_seconds"], 60)
        self.assertTrue(report["cached_features_recomputed"])
        self.assertEqual(report["independent_external_result_audit"], "not_performed")
        pd.testing.assert_frame_equal(h, self.history)
        pd.testing.assert_frame_equal(f, self.features)

    def test_empty_inputs_rejected(self):
        for h, f in ((self.history.iloc[:0], self.features),
                     (self.history, self.features.iloc[:0])):
            with self.subTest(empty="history" if h.empty else "features"):
                with self.assertRaisesRegex(ValueError, "empty_research_inputs"):
                    audit_inputs(h, f)

    def test_duplicate_map_and_duplicate_series_map_identity_rejected(self):
        for different_path in (False, True):
            h = pd.concat([self.history, self.history.iloc[[0]]], ignore_index=True)
            if different_path:
                h.loc[len(h) - 1, "map_id"] = "/another/path.dem"
            with self.subTest(different_path=different_path):
                with self.assertRaisesRegex(ValueError, "duplicate_history_map_identity"):
                    audit_inputs(h, self.features)

    def test_duplicate_targets_and_non_map1_rejected(self):
        duplicated = pd.concat([self.features, self.features.iloc[[0]]], ignore_index=True)
        not_map1 = self.features.copy()
        not_map1.loc[0, "map_no"] = 2
        for features in (duplicated, not_map1):
            with self.assertRaisesRegex(ValueError, "expected_one_map1_per_series"):
                audit_inputs(self.history, features)

    def test_missing_map1_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "cached_features_missing_or_extra_map1"):
            audit_inputs(self.history, self.features.iloc[1:])

    def test_cached_identity_and_label_mismatch_rejected(self):
        for key, value in (("y", 1 - self.features.iloc[0].y),
                           ("map_name", "de_nuke"), ("roster_a", RC),
                           ("match_id", "wrong-series")):
            f = self.features.copy()
            f.loc[0, key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"cached_target_identity_mismatch:{key}"):
                    audit_inputs(self.history, f)

    def test_score_label_round_total_and_terminal_state_mismatch_rejected(self):
        for key, value in (("y", 1 - self.history.iloc[0].y),
                           ("rounds", 99), ("round_wins_b", 12)):
            h = self.history.copy()
            h.loc[0, key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "score_label_or_terminal_state_mismatch"):
                    audit_inputs(h, self.features)

    def test_invalid_fractional_negative_and_nonfinite_score_rejected(self):
        for score in (-1., .5, np.nan, np.inf):
            h = self.history.copy()
            h["round_wins_a"] = h.round_wins_a.astype(float)
            h.loc[0, "round_wins_a"] = score
            with self.subTest(score=score):
                with self.assertRaisesRegex(ValueError, "invalid_map_score"):
                    audit_inputs(h, self.features)

    def test_overlapping_or_noncanonical_rosters_rejected(self):
        for roster in (RB, ",".join(reversed(RA.split(",")))):
            h = self.history.copy()
            h.loc[0, "roster_a"] = roster
            with self.assertRaisesRegex(ValueError, "invalid_history_roster_identity"):
                audit_inputs(h, self.features)

    def test_result_must_be_available_after_start(self):
        h = self.history.copy()
        h.loc[0, "available_at"] = h.loc[0, "start_at"]
        with self.assertRaisesRegex(ValueError, "history_result_not_after_start"):
            audit_inputs(h, self.features)

    def test_cached_start_and_availability_must_match_history(self):
        for key in ("start_at", "available_at"):
            f = self.features.copy()
            f.loc[0, key] += pd.Timedelta(seconds=1)
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"cached_target_time_mismatch:{key}"):
                    audit_inputs(self.history, f)

    def test_post_start_cutoff_and_future_feature_history_rejected(self):
        f = self.features.copy()
        f.loc[0, "decision_at"] = f.loc[0, "start_at"] + pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "invalid_prediction_cutoff"):
            audit_inputs(self.history, f)
        f = self.features.copy()
        f.loc[0, "max_history_available_at"] = f.loc[0, "decision_at"].isoformat()
        with self.assertRaisesRegex(ValueError, "future_feature_history"):
            audit_inputs(self.history, f)

    def test_nonfinite_features_and_mixed_leads_rejected(self):
        bad_feature = self.features.copy()
        bad_feature.loc[0, FEATURES[0]] = np.nan
        mixed_lead = self.features.copy()
        mixed_lead.loc[0, "decision_at"] -= pd.Timedelta(seconds=1)
        for features in (bad_feature, mixed_lead):
            with self.assertRaisesRegex(ValueError, "invalid_feature_values_or_mixed_lead_times"):
                audit_inputs(self.history, features)

    def test_rebuild_detects_feature_and_coverage_tampering(self):
        for column in (FEATURES[0], "history_all_a", "history_map_b"):
            f = self.features.copy()
            f.loc[4, column] += 1
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, f"feature_cache_rebuild_mismatch:{column}"):
                    audit_inputs(self.history, f)

    def test_skipped_cache_rebuild_is_explicit_in_report(self):
        report = audit_inputs(self.history, self.features, verify_features=False)
        self.assertFalse(report["cached_features_recomputed"])


class HistoryEligibilityTests(unittest.TestCase):
    def test_shared_minimum_applies_to_both_sides(self):
        for a, b, expected in ((3, 3, True), (5, 3, True), (3, 2, False), (2, 20, False)):
            with self.subTest(a=a, b=b):
                result = roster_history_eligibility({"history_all_a": a, "history_all_b": b})
                self.assertEqual(result["eligible"], expected)
                self.assertEqual(result["reason"], "eligible" if expected else "insufficient_roster_history")
        self.assertTrue(roster_history_eligibility({"history_all_a": 0, "history_all_b": 0}, 0)["eligible"])

    def test_invalid_or_missing_coverage_fails_closed(self):
        invalid = [{}, {"history_all_a": 5}]
        invalid += [{"history_all_a": value, "history_all_b": 4}
                    for value in (np.nan, np.inf, -1, 3.1, None, "unknown")]
        for coverage in invalid:
            with self.subTest(coverage=coverage):
                self.assertEqual(roster_history_eligibility(coverage),
                                 {"eligible": False, "reason": "invalid_roster_history_coverage"})

    def test_invalid_minimum_rejected(self):
        for minimum in (-1, 2.5, True, "3", None):
            with self.subTest(minimum=minimum):
                with self.assertRaisesRegex(ValueError, "invalid_minimum_roster_history"):
                    roster_history_eligibility({"history_all_a": 5, "history_all_b": 5}, minimum)


class EloBaselineTests(unittest.TestCase):
    def test_roster_swap_is_complement_and_index_is_preserved(self):
        h = pd.DataFrame([history_row("past", T - pd.Timedelta(days=2))])
        targets = pd.DataFrame([
            {"match_id": "target", "decision_at": T, "roster_a": RA, "roster_b": RB},
            {"match_id": "target", "decision_at": T, "roster_a": RB, "roster_b": RA},
        ], index=[42, 8])
        predictions = elo_predictions(h, targets)
        self.assertEqual(predictions.index.tolist(), [42, 8])
        self.assertGreater(predictions.iloc[0], .5)
        self.assertAlmostEqual(predictions.sum(), 1.)

    def test_future_equal_time_and_same_series_results_do_not_change_prior(self):
        h = pd.DataFrame([history_row("past", T - pd.Timedelta(days=2))])
        target = pd.DataFrame([{"match_id": "target", "decision_at": T,
                                "roster_a": RA, "roster_b": RB}])
        expected = elo_predictions(h, target)
        extra = []
        for mid, available in (("future", T + pd.Timedelta(days=1)),
                               ("equal", T), ("target", T - pd.Timedelta(days=1))):
            row = history_row(mid, T - pd.Timedelta(days=3), "b")
            row["available_at"] = available
            extra.append(row)
        result = elo_predictions(pd.concat([h, pd.DataFrame(extra)], ignore_index=True), target)
        pd.testing.assert_series_equal(result, expected)

    def test_tied_result_batches_are_independent_of_row_order(self):
        first = history_row("first", T - pd.Timedelta(days=1))
        second = history_row("second", T - pd.Timedelta(days=1))
        second.update(roster_a=RB, roster_b=RC)
        h = pd.DataFrame([first, second])
        target = pd.DataFrame([{"match_id": "target", "decision_at": T,
                                "roster_a": RA, "roster_b": RC}])
        normal = elo_predictions(h, target)
        reverse = elo_predictions(h.iloc[::-1], target)
        pd.testing.assert_series_equal(normal, reverse)
        self.assertAlmostEqual(normal.iloc[0], 1 / (1 + 10 ** (-20 / 400)))

    def test_invalid_elo_parameters_rejected(self):
        h, f, _ = research_fixture(2)
        for parameters in ({"k": -1}, {"scale": 0}, {"k": np.nan}, {"scale": np.inf}):
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(ValueError, "invalid_elo_parameters"):
                    elo_predictions(h, f, **parameters)


class GenericModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.features, _ = research_fixture(12)

    def test_default_columns_equal_explicit_legacy_columns(self):
        normal = walk_forward(self.features, min_train=2)
        explicit = walk_forward(self.features, min_train=2, feature_columns=FEATURES)
        pd.testing.assert_frame_equal(normal, explicit)

    def test_custom_model_has_complement_symmetry(self):
        columns = ["gap_one", "gap_two"]
        train = pd.DataFrame({"gap_one": [-2., -1., .2, .5, 1.2, 3.],
                              "gap_two": [1., -.5, .8, -1., 0., 2.],
                              "y": [0, 0, 1, 0, 1, 1]})
        model = fit_model(train, columns)
        x = train[columns].to_numpy()
        np.testing.assert_allclose(model.predict_proba(x)[:, 1] + model.predict_proba(-x)[:, 1],
                                   1., rtol=0, atol=1e-12)

    def test_fit_rejects_empty_duplicate_and_nonfinite_columns(self):
        for columns in ([], [FEATURES[0], FEATURES[0]]):
            with self.assertRaisesRegex(ValueError, "distinct feature columns"):
                fit_model(self.features, columns)
        invalid = self.features.copy()
        invalid.loc[0, FEATURES[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "Nonfinite features"):
            fit_model(invalid)

    def test_unused_target_result_column_cannot_enter_model(self):
        original = fit_model(self.features)
        added = self.features.assign(future_result_label=self.features.y * 999.)
        candidate = fit_model(added)
        x = self.features[FEATURES].to_numpy()
        np.testing.assert_allclose(original.predict_proba(x), candidate.predict_proba(x))

    def test_appending_future_rows_cannot_change_earlier_development_predictions(self):
        _, f, _ = research_fixture(16)
        before = walk_forward(self.features, min_train=2).set_index("map_id")
        after = walk_forward(f, min_train=2).set_index("map_id")
        # Appending data changes the fractional tail boundary. Compare only rows
        # that remain development rows in both runs, never two different freezes.
        common = before.index[before.partition.eq("development") & before.p_model.notna()]
        self.assertTrue(after.loc[common, "partition"].eq("development").all())
        np.testing.assert_allclose(before.loc[common, "p_model"], after.loc[common, "p_model"], atol=1e-12)
        np.testing.assert_array_equal(before.loc[common, "n_train"], after.loc[common, "n_train"])

    def test_calibration_assigns_boundaries_once_and_reports_all_samples(self):
        p = np.array([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
        report = probability_metrics(np.arange(len(p)) % 2, p)
        bins = report["calibration"]
        self.assertEqual(report["n"], 11)
        self.assertEqual(sum(row["n"] for row in bins), report["n"])
        self.assertEqual([row["low"] for row in bins], [i / 10 for i in range(10)])
        self.assertEqual([row["n"] for row in bins], [1] * 9 + [2])

    def test_calibration_filters_missing_predictions_without_recounting(self):
        report = probability_metrics([0, 1, 0, 1], [.3, np.nan, .7, np.inf])
        self.assertEqual(report["n"], 2)
        self.assertEqual(sum(row["n"] for row in report["calibration"]), 2)


class ComparisonIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history, cls.features, cls.players = research_fixture(12, sparse_last=True)
        cls.predictions, cls.report = compare_models(cls.history, cls.features, cls.players, min_train=2)

    def test_fixed_candidates_use_same_rows_and_sparse_rosters_are_visible(self):
        self.assertEqual(self.report["funnel"],
                         {"clean_maps": 12, "map1_targets": 12,
                          "history_eligible": 8, "common_predictions": 10})
        development = self.report["partitions"]["development"]
        tail = self.report["partitions"]["diagnostic_tail"]
        self.assertEqual((development["target_rows"], development["history_eligible_rows"]), (9, 6))
        self.assertEqual((development["all_common"]["n"], development["history_eligible_common"]["n"]), (7, 6))
        self.assertEqual((tail["target_rows"], tail["history_eligible_rows"]), (3, 2))
        self.assertEqual((tail["all_common"]["n"], tail["history_eligible_common"]["n"]), (3, 2))
        for partition in self.report["partitions"].values():
            for key in ("all_common", "history_eligible_common"):
                cohort = partition[key]
                self.assertEqual({m["n"] for m in cohort["metrics"].values()}, {cohort["n"]})

    def test_diagnostic_report_does_not_claim_untouched_test_or_promote(self):
        self.assertFalse(self.report["live_trading_enabled"])
        self.assertIsNone(self.report["execution_pnl"])
        self.assertEqual(self.report["validation_design"]["tail_status"],
                         "previously_inspected_diagnostic_not_untouched_test")
        self.assertEqual(self.report["validation_design"]["candidate_selection"],
                         "none_no_automatic_promotion_or_tuning")
        self.assertNotIn("holdout", self.predictions.partition.unique())
        self.assertNotIn("promoted_model", self.report)
        tail = self.predictions[self.predictions.partition.eq("diagnostic_tail")]
        self.assertEqual(tail.n_train.nunique(), 1)

    def test_feature_rows_and_predictions_remain_target_aligned(self):
        pd.testing.assert_series_equal(self.predictions.map_id, self.features.map_id)
        expected, _ = build_player_features(self.history, self.players, self.features)
        pd.testing.assert_frame_equal(self.predictions[PLAYER_FEATURES], expected[PLAYER_FEATURES])
        for name in model_definitions():
            self.assertTrue(self.predictions[f"p_{name}"].iloc[2:].notna().all())

    def test_new_candidates_exclude_legacy_armor_damage_feature(self):
        definitions = model_definitions()
        self.assertEqual(definitions["current_roster16"], FEATURES)
        for name, columns in definitions.items():
            if name != "current_roster16":
                self.assertFalse(any("adr" in key or "damage" in key for key in columns))
            self.assertEqual(len(columns), len(set(columns)))

    def test_common_sample_drops_missing_model_row_from_all_baselines(self):
        frame = self.predictions.iloc[2:5].copy()
        frame.loc[frame.index[0], "p_current_roster16"] = np.nan
        columns = {"constant_0_5": "p_constant_0_5", "roster_elo": "p_roster_elo",
                   "current_roster16": "p_current_roster16"}
        report = cohort_metrics(frame, columns)
        self.assertEqual(report["n"], 2)
        self.assertEqual({m["n"] for m in report["metrics"].values()}, {2})
        self.assertEqual(report["paired_brier_difference_model_minus_reference"]
                         ["constant_0_5"]["constant_0_5"], 0.)

    def test_reordered_player_output_is_rejected(self):
        values, audit = build_player_features(self.history, self.players, self.features)
        with patch("cs2ml.map1_research.build_player_features", return_value=(values.iloc[::-1], audit)):
            with self.assertRaisesRegex(ValueError, "player_features_changed_target_alignment"):
                compare_models(self.history, self.features, self.players, min_train=2)

    def test_reordered_prediction_output_is_rejected(self):
        def reverse_predictions(*args, **kwargs):
            return walk_forward(*args, **kwargs).iloc[::-1]
        with patch("cs2ml.map1_research.walk_forward", side_effect=reverse_predictions):
            with self.assertRaisesRegex(ValueError, "prediction_target_alignment_changed"):
                compare_models(self.history, self.features, self.players, min_train=2)


class ResearchCommandTests(unittest.TestCase):
    def test_existing_output_is_refused_before_reading_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stderr = io.StringIO()
            with patch("sys.argv", ["map1_research", "--output", str(output)]), \
                 patch("cs2ml.map1_research.pd.read_parquet") as read, \
                 patch("cs2ml.map1_research.compare_models") as compare, \
                 redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exited:
                    main()
            self.assertEqual(exited.exception.code, 2)
            self.assertIn("must not be overwritten", stderr.getvalue())
            read.assert_not_called()
            compare.assert_not_called()
            self.assertEqual(list(output.iterdir()), [])

    def test_source_or_code_change_during_research_aborts_before_artifact_creation(self):
        for filename, reason in (("history.parquet", "research_input_changed_while_running"),
                                 ("map1_model.py", "research_code_changed_while_running")):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "new-experiment"
                seen = {}

                def changing_hash(path):
                    key = str(path)
                    seen[key] = seen.get(key, 0) + 1
                    return key + ("changed" if Path(path).name == filename and seen[key] > 1 else "")

                with patch("sys.argv", ["map1_research", "--data", directory, "--output", str(output)]), \
                     patch("cs2ml.map1_research.file_hash", side_effect=changing_hash), \
                     patch("cs2ml.map1_research.pd.read_parquet", return_value=pd.DataFrame()) as read, \
                     patch("cs2ml.map1_research.compare_models", return_value=(pd.DataFrame(), {})) as compare:
                    with self.assertRaisesRegex(ValueError, reason):
                        main()
                self.assertEqual(read.call_count, 3)
                compare.assert_called_once()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
