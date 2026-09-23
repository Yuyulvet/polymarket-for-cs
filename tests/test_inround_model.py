from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cs2ml.inround import (_round_of, add_round_identity_scores, build, completion_audit,
                          compute_inround, identity_problem, observation_tick, snapshot_problem, valid_snapshot)
from cs2ml.map1_data import first_ct_is_ct, roster_key
from cs2ml.inround_model import (FEATURE_STEPS, NUMERIC_FEATURES, attach_chronology,
                                evaluate_frame, forward_splits, hierarchical_weights,
                                prepare_events, terminal_reason)


T = pd.Timestamp("2026-09-01T12:00:00Z")


def fixture():
    events = []
    # Last T killed after plant is deliberately retained; CT can still fail defuse.
    for tick, ct, tt, bomb, kind in [(100, 5, 5, 0, "round_start"), (200, 3, 3, 0, "kill"),
                                     (300, 3, 1, 1, "bomb_planted"), (400, 3, 0, 1, "kill"),
                                     (450, 0, 0, 1, "kill"), (500, 3, 0, 2, "bomb_defused"),
                                     (600, 3, 0, 2, "kill")]:
        events.append({"demo_path": "D:/a.dem", "round_num": 1, "tick": tick,
                       "event_tick": tick, "state_tick": tick+1, "event_type": kind,
                       "map_name": "de_mirage", "ct_alive": ct, "t_alive": tt,
                       "ct_hp": ct*80., "t_hp": tt*90., "bomb_state": bomb,
                       "schema_version": 2, "observation_basis": "first_full_tick_after_event_batch"})
    rounds = pd.DataFrame([{"demo_path": "D:/a.dem", "round_num": 1, "winner_side": "CT",
                            "ct_equip0": 4000., "t_equip0": 3500.,
                            "round_start_tick": 100, "round_end_tick": 501,
                            "round_len_ticks": 401, "final_margin": 3,
                            "outcome_type": "stomp", "bomb_defused": True}])
    return pd.DataFrame(events), rounds


class TerminalAndTimingTests(unittest.TestCase):
    def test_postplant_t_wipe_is_not_terminal(self):
        self.assertIsNone(terminal_reason(3, 0, 1, "kill"))
        self.assertIsNotNone(terminal_reason(3, 0, 0, "kill"))
        self.assertIsNotNone(terminal_reason(0, 3, 1, "kill"))
        self.assertIsNotNone(terminal_reason(1, 1, 2))
        self.assertIsNotNone(terminal_reason(1, 1, 3))
        self.assertIsNotNone(terminal_reason(2, 2, 0, "round_end"))

    def test_sanitizer_keeps_only_undecided_active_states(self):
        events, rounds = fixture()
        clean, audit = prepare_events(events, rounds)
        self.assertEqual(clean.tick.tolist(), [100, 200, 300, 400])
        self.assertEqual(audit["verified_event_rows"], 4)
        self.assertEqual(audit["retained_rounds"], 1)
        self.assertEqual(audit["excluded_rows"]["ct_eliminated"], 1)
        self.assertNotIn("outcome_type", clean)
        self.assertNotIn("final_margin", clean)
        self.assertNotIn("bomb_defused", clean)
        for _, categorical, numeric in FEATURE_STEPS:
            self.assertFalse(set(categorical+numeric) & {"round_end_tick", "label_ct_win", "winner_side"})

    def test_terminal_bomb_state_cannot_hide_behind_kill_type(self):
        events, rounds = fixture()
        events.loc[1, "bomb_state"] = 3
        clean, audit = prepare_events(events, rounds)
        self.assertNotIn(200, clean.tick.tolist())
        self.assertEqual(audit["excluded_rows"]["terminal_bomb_or_round_event"], 1)

    def test_legacy_requires_explicit_opt_in_and_never_claims_verified(self):
        events, rounds = fixture()
        events = events.drop(columns=["event_tick", "state_tick", "schema_version", "observation_basis"])
        rounds = rounds.drop(columns=["round_start_tick", "round_end_tick"])
        clean, audit = prepare_events(events, rounds)
        self.assertTrue(clean.empty)
        self.assertEqual(audit["verified_event_rows"], 0)
        clean, audit = prepare_events(events, rounds, allow_legacy_timing=True)
        self.assertEqual(clean.tick.tolist(), [100, 200, 300, 400])
        self.assertFalse(clean.timing_verified.any())

    def test_state_at_round_end_is_excluded_even_when_event_was_earlier(self):
        events, rounds = fixture()
        events.loc[1, ["tick", "event_tick", "state_tick"]] = [500, 500, 501]
        events = events.drop(index=5)
        clean, audit = prepare_events(events, rounds)
        self.assertNotIn(500, clean.tick.tolist())
        self.assertGreater(audit["excluded_rows"]["outside_active_round"], 0)

    def test_v2_cannot_backdate_state_tick(self):
        events, rounds = fixture()
        events.loc[1, "state_tick"] = 200
        clean, audit = prepare_events(events, rounds)
        self.assertNotIn(200, clean.tick.tolist())
        self.assertEqual(audit["excluded_rows"]["unverified_observation_timing"], 1)

    def test_overlapping_round_boundaries_are_rejected(self):
        e, r = fixture()
        next_round = r.copy()
        next_round["round_num"] = 2
        next_round["round_start_tick"] = 350
        next_round["round_end_tick"] = 700
        clean, audit = prepare_events(e, pd.concat([r, next_round]))
        self.assertTrue(clean.empty)
        self.assertEqual(audit["excluded_rows"]["invalid_round_boundary"], len(e))

    def test_duplicate_labels_or_tick_batches_raise(self):
        e, r = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate round labels"):
            prepare_events(e, pd.concat([r, r]))
        with self.assertRaisesRegex(ValueError, "Duplicate tick batches"):
            prepare_events(pd.concat([e, e.iloc[[0]]]), r)

    def test_invalid_labels_and_invalid_alive_values_fail_closed(self):
        e, r = fixture()
        r["winner_side"] = "unknown"
        clean, _ = prepare_events(e, r)
        self.assertTrue(clean.empty)
        e, r = fixture()
        e.loc[1, "ct_alive"] = np.nan
        clean, _ = prepare_events(e, r)
        self.assertNotIn(200, clean.tick.tolist())

    def test_warmup_and_observation_tick_helpers(self):
        self.assertEqual(_round_of(99, [100, 600]), 0)
        self.assertEqual(_round_of(100, [100, 600]), 1)
        self.assertEqual(observation_tick(100), 101)

    def test_invalid_hp_and_equipment_are_not_imputed_into_real_states(self):
        e, r = fixture()
        e.loc[1, "ct_hp"] = 301
        clean, audit = prepare_events(e, r)
        self.assertNotIn(200, clean.tick.tolist())
        self.assertEqual(audit["excluded_rows"]["invalid_hp_or_equipment"], 1)
        r["ct_equip0"] = np.nan
        clean, _ = prepare_events(e, r)
        self.assertTrue(clean.empty)

    def test_build_protects_legacy_directory_and_existing_outputs(self):
        from cs2ml import config
        with self.assertRaisesRegex(ValueError, "legacy artifacts"):
            build(output_dir=config.DATA_DIR)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "inround_events.parquet").touch()
            with self.assertRaises(FileExistsError):
                build(output_dir=path)


class ChronologyTests(unittest.TestCase):
    def data(self):
        rows = []
        for number in range(8):
            start = T + pd.Timedelta(days=number)
            for arena in ("one", "two"):
                for rnd in (1, 2):
                    rows.append({"demo_path": f"d:/series{number}/{arena}.dem", "round_num": rnd,
                                 "match_id": str(number), "start_at": start,
                                 "available_at": start+pd.Timedelta(hours=3),
                                 "event_type": "kill", "label_ct_win": rnd % 2,
                                 "map_name": "de_mirage", "round_class": "normal",
                                 "ct_tier6": "rifle", "t_tier6": "rifle",
                                 **{feature: float(rnd % 2) for feature in NUMERIC_FEATURES}})
        return pd.DataFrame(rows)

    def test_forward_splits_never_train_future_or_same_series(self):
        data = self.data()
        data.loc[data.match_id.eq("0"), "available_at"] = T + pd.Timedelta(days=5)
        splits = list(forward_splits(data, n_splits=3, min_train_series=1))
        self.assertTrue(splits)
        for train, test in splits:
            tr, te = data.iloc[train], data.iloc[test]
            self.assertFalse(set(tr.match_id) & set(te.match_id))
            self.assertLess(tr.available_at.max(), te.start_at.min())
            for mid in te.match_id.unique():
                self.assertEqual(len(te[te.match_id.eq(mid)]), len(data[data.match_id.eq(mid)]))

    def test_strict_equality_in_availability_is_not_training(self):
        data = self.data()
        data.loc[data.match_id.eq("0"), "available_at"] = T + pd.Timedelta(days=2)
        train, test = next(forward_splits(data, n_splits=3, min_train_series=1))
        self.assertNotIn("0", data.iloc[train].match_id.tolist())
        self.assertEqual(data.iloc[test].start_at.min(), T + pd.Timedelta(days=2))

    def test_tied_start_series_stay_together(self):
        data = self.data()
        data.loc[data.match_id.eq("3"), "start_at"] = T + pd.Timedelta(days=2)
        for _, test in forward_splits(data, n_splits=3, min_train_series=1):
            selected = set(data.iloc[test].match_id)
            self.assertEqual("2" in selected, "3" in selected)

    def test_equal_series_and_round_weights_despite_duplicate_events(self):
        data = self.data()
        data = pd.concat([data, data.iloc[[0]*100]], ignore_index=True)
        weights = hierarchical_weights(data)
        totals = pd.Series(weights).groupby(data.match_id).sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.)
        first = data.match_id.eq("0") & data.demo_path.eq("d:/series0/one.dem") & data.round_num.eq(1)
        self.assertAlmostEqual(weights[first].sum(), .25)

    def test_chronology_uses_latest_series_availability_and_rejects_unmapped(self):
        e, r = fixture()
        e, _ = prepare_events(e, r)
        meta = pd.DataFrame([{"map_id": "d:/a.dem", "match_id": "a", "start_at": T,
                              "available_at": T+pd.Timedelta(hours=1)},
                             {"map_id": "d:/b.dem", "match_id": "a", "start_at": T,
                              "available_at": T+pd.Timedelta(hours=3)}])
        data, audit = attach_chronology(e, meta)
        self.assertTrue(data.available_at.eq(T+pd.Timedelta(hours=3)).all())
        self.assertEqual(audit["retained_series"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate chronology"):
            attach_chronology(e, pd.concat([meta, meta]))
        e["match_id"] = "wrong-series"
        with self.assertRaisesRegex(ValueError, "identity conflicts"):
            attach_chronology(e, meta)

    def test_evaluation_reports_common_cohort_and_no_iid_claim(self):
        report = evaluate_frame(self.data(), n_splits=3, min_train_series=1)
        self.assertEqual(report["status"], "exploratory_only")
        self.assertLess(report["scored_series"], report["scored_rounds"])
        self.assertEqual(report["models"]["constant_50"]["brier"], .25)
        for model in report["models"].values():
            self.assertEqual(model["event_rows"], report["scored_event_rows"])
        for fold in report["folds"]:
            self.assertEqual(fold["series_overlap"], 0)
        self.assertTrue(evaluate_frame(self.data().iloc[:0])["status"].startswith("insufficient"))


class ExtractorTimingTests(unittest.TestCase):
    def test_duplicate_freeze_and_crossing_round_end_fail_closed(self):
        for ticks, expected in [([100, 100], "Duplicate freeze"), ([100, 300], "overlaps")]:
            def event(name):
                if name == "round_freeze_end":
                    return pd.DataFrame({"tick": ticks})
                if name == "round_end":
                    return pd.DataFrame({"tick": [400], "round": [1], "winner": ["CT"]})
                return pd.DataFrame(columns=["tick"])
            with patch("cs2ml.inround.DemoParser") as factory:
                factory.return_value.parse_header.return_value = {"map_name": "de_mirage"}
                factory.return_value.parse_event.side_effect = event
                with self.assertRaisesRegex(ValueError, expected):
                    compute_inround("d:/fixture.dem")

    def test_incomplete_or_ambiguous_snapshot_is_rejected(self):
        frame = pd.DataFrame([{"steamid": str(i+1), "team_name": "CT" if i < 5 else "T",
                               "is_alive": True, "health": 100., "current_equip_value": 4000.}
                              for i in range(10)])
        self.assertTrue(valid_snapshot(frame))
        self.assertFalse(valid_snapshot(frame.iloc[:9]))
        frame.loc[0, "steamid"] = frame.loc[1, "steamid"]
        self.assertFalse(valid_snapshot(frame))
        frame.loc[0, "steamid"] = "1"
        frame["is_alive"] = frame.is_alive.astype(object)
        frame.loc[0, "is_alive"] = np.nan
        self.assertFalse(valid_snapshot(frame))

    def test_tick_plus_one_bomb_event_is_known_at_state_observation(self):
        class FakeParser:
            def __init__(self, path):
                pass

            def parse_header(self):
                return {"map_name": "de_mirage"}

            def parse_event(self, name):
                if name == "round_freeze_end":
                    return pd.DataFrame({"tick": [100]})
                if name == "round_end":
                    return pd.DataFrame({"tick": [500], "round": [1], "winner": ["CT"], "reason": ["defuse"]})
                if name == "player_death":
                    return pd.DataFrame([{"tick": 200, "attacker_steamid": "1", "user_steamid": "6",
                                          "weapon": "ak47", "headshot": False, "thrusmoke": False,
                                          "assistedflash": False}])
                if name == "bomb_planted":
                    return pd.DataFrame({"tick": [201]})
                return pd.DataFrame(columns=["tick"])

            def parse_ticks(self, fields, ticks):
                self.assert_post_event = ticks
                return pd.DataFrame([{"tick": tick, "team_name": "CT" if i < 5 else "TERRORIST",
                                      "steamid": str(i+1), "is_alive": i != 5 or tick < 200,
                                      "health": 100., "current_equip_value": 4000.,
                                      "X": float(i), "Y": float(i), "yaw": 0.}
                                     for tick in ticks for i in range(10)])

        with patch("cs2ml.inround.DemoParser", FakeParser):
            events, rounds = compute_inround("d:/fixture.dem")
        row = events.loc[events.event_tick.eq(200)].iloc[0]
        self.assertEqual(row.state_tick, 201)
        self.assertEqual(row.bomb_state, 1)
        self.assertTrue(events.state_tick.gt(events.event_tick).all())
        self.assertTrue(events.state_tick.lt(events.round_end_tick).all())
        clean, audit = prepare_events(events, rounds)
        self.assertEqual(audit["verified_event_rows"], len(clean))


RA = roster_key([str(100+i) for i in range(5)])
RB = roster_key([str(200+i) for i in range(5)])


def ledger(winners):
    rows = []
    for number, winner in enumerate(winners, 1):
        ct, tt = (RA, RB) if first_ct_is_ct(number) else (RB, RA)
        rows.append({"round_num": number, "ct_roster": ct, "t_roster": tt,
                     "winner_side": "CT" if winner == ct else "T"})
    return pd.DataFrame(rows)


def ledger_parser(winners=None, *, bad_features=False, missing_initial_round=None,
                  all_identity_missing=False, extra_unresolved=False, changed_roster_round=None,
                  end_snapshot_side_switch=False):
    winners = [RA]*13 if winners is None else winners

    class Parser:
        def __init__(self, path):
            pass

        def parse_header(self):
            return {"map_name": "de_mirage"}

        def parse_event(self, name):
            if name == "round_freeze_end":
                return pd.DataFrame({"tick": [100+1000*i for i in range(len(winners))]})
            if name == "round_end":
                ends = ledger(winners)
                rows = [{"tick": 600+1000*i, "round": i+1, "winner": row.winner_side}
                        for i, row in enumerate(ends.itertuples(index=False))]
                if extra_unresolved:
                    rows.append({"tick": 600+1000*len(winners), "round": len(winners)+1, "winner": None})
                return pd.DataFrame(rows)
            return pd.DataFrame(columns=["tick"])

        def parse_ticks(self, fields, ticks):
            rows = []
            for tick in ticks:
                number = min(len(winners), max(1, (tick-100)//1000+1))
                for player, sid in enumerate((RA+","+RB).split(",")):
                    if all_identity_missing and player == 0:
                        continue
                    if number == missing_initial_round and tick == 101+1000*(number-1) and player == 0:
                        continue
                    side = "CT" if ((player < 5) == first_ct_is_ct(number)) else "TERRORIST"
                    if end_snapshot_side_switch and (tick-601) % 1000 == 0:
                        side = "TERRORIST" if side == "CT" else "CT"
                    actual_sid = "999" if player == 0 and number == changed_roster_round else sid
                    rows.append({"tick": tick, "team_name": side, "steamid": actual_sid,
                                 "is_alive": True, "health": np.nan if bad_features else 100.,
                                 "current_equip_value": 4000., "X": float(player), "Y": 1., "yaw": 0.})
            return pd.DataFrame(rows)

    return Parser


class MapCompletionAuditTests(unittest.TestCase):
    def test_regulation_label_and_past_only_scores_follow_direct_rosters(self):
        rounds = add_round_identity_scores(ledger([RA]*13))
        self.assertEqual(rounds.score_a_before.tolist(), list(range(13)))
        self.assertEqual(rounds.score_b_before.tolist(), [0]*13)
        self.assertFalse(rounds.iloc[-1].ct_is_a)
        self.assertEqual(rounds.iloc[-1].winner_side, "T")
        audit = completion_audit(rounds, 13)
        self.assertEqual((audit["score_a"], audit["score_b"]), (13, 0))
        self.assertEqual(audit["winner_roster"], RA)
        self.assertTrue(audit["complete"])
        self.assertFalse(audit["overtime"])

    def test_current_and_future_results_cannot_change_prior_scores(self):
        original = ledger([RA]*13)
        altered = original.copy()
        altered.loc[5:, "winner_side"] = altered.loc[5:, "winner_side"].map({"CT": "T", "T": "CT"})
        a, b = add_round_identity_scores(original), add_round_identity_scores(altered)
        pd.testing.assert_frame_equal(a.loc[:5, ["score_a_before", "score_b_before"]],
                                      b.loc[:5, ["score_a_before", "score_b_before"]])

    def test_repeated_overtime_uses_observed_rosters_and_mr3_terminal(self):
        winners = [RA, RB]*12 + [RA, RB]*3 + [RA]*4
        rounds = add_round_identity_scores(ledger(winners))
        result = completion_audit(rounds, len(winners))
        self.assertTrue(result["complete"])
        self.assertEqual((result["score_a"], result["score_b"]), (19, 15))
        self.assertEqual(result["n_overtime_rounds"], 10)
        self.assertTrue(result["overtime"])

    def test_incomplete_and_early_terminal_do_not_create_map_labels(self):
        for winners, reason in (([RA]*12, "incomplete_or_nonstandard_final_score"),
                                ([RA]*13+[RB], "rounds_after_terminal_score")):
            result = completion_audit(ledger(winners), len(winners))
            self.assertFalse(result["complete"])
            self.assertIsNone(result["winner_roster"])
            self.assertEqual(result["reason_code"], reason)

    def test_missing_round_cannot_be_filled_by_final_score(self):
        rounds = add_round_identity_scores(ledger([RA]*13).drop(index=4))
        self.assertTrue(rounds.loc[rounds.round_num.ge(6), "score_a_before"].isna().all())
        audit = completion_audit(rounds, 13)
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["reason_code"], "incomplete_round_sequence")

    def test_side_schedule_and_roster_change_reject_map_label(self):
        rounds = ledger([RA]*13)
        rounds.loc[12, ["ct_roster", "t_roster", "winner_side"]] = [RA, RB, "CT"]
        self.assertEqual(completion_audit(rounds, 13)["reason_code"], "nonstandard_side_schedule")
        rounds.loc[12, "ct_roster"] = roster_key(["999", "101", "102", "103", "104"])
        self.assertEqual(completion_audit(rounds, 13)["reason_code"], "roster_changed_within_map")


class StructuredExtractionAuditTests(unittest.TestCase):
    def test_valid_demo_has_direct_identity_complete_label_and_json_audit(self):
        report = {}
        with patch("cs2ml.inround.DemoParser", ledger_parser()):
            events, rounds = compute_inround("d:/fixture.dem", audit=report)
        self.assertEqual(report["status"], "accepted")
        self.assertTrue(report["map_completion"]["complete"])
        self.assertEqual(len(events), 13)
        self.assertEqual(len(rounds), 13)
        self.assertEqual(events.iloc[0].ct_roster, RA)
        self.assertEqual(events.iloc[-1].ct_roster, RB)
        self.assertEqual(events.iloc[-1].score_a_before, 12)
        self.assertEqual(rounds.iloc[-1].winner_roster, RA)
        json.dumps(report, allow_nan=False)

    def test_missing_feature_data_does_not_erase_identity_ledger(self):
        report = {}
        with patch("cs2ml.inround.DemoParser", ledger_parser(bad_features=True)):
            events, rounds = compute_inround("d:/fixture.dem", audit=report)
        self.assertTrue(events.empty)
        self.assertEqual(len(rounds), 13)
        self.assertTrue(report["map_completion"]["complete"])
        self.assertTrue(rounds.ct_equip0.isna().all())
        self.assertEqual(report["excluded_states_by_reason"]["snapshot_invalid_hp"], 13)
        prepared, _ = prepare_events(events, rounds)
        self.assertTrue(prepared.empty)

    def test_post_end_side_switch_invalidates_summary_not_initial_roster_label(self):
        report = {}
        with patch("cs2ml.inround.DemoParser", ledger_parser(end_snapshot_side_switch=True)):
            _, rounds = compute_inround("d:/fixture.dem", audit=report)
        self.assertTrue(report["map_completion"]["complete"])
        self.assertEqual(report["map_completion"]["winner_roster"], RA)
        self.assertTrue(rounds.final_margin.isna().all())
        self.assertTrue(rounds.outcome_type.eq("unknown").all())
        self.assertEqual(len(report["invalid_round_end_identities"]), 13)

    def test_missing_initial_identity_retains_other_rounds_but_no_map_label(self):
        report = {}
        with patch("cs2ml.inround.DemoParser", ledger_parser(missing_initial_round=5)):
            events, rounds = compute_inround("d:/fixture.dem", audit=report)
        self.assertEqual(len(rounds), 12)
        self.assertNotIn(5, rounds.round_num.tolist())
        self.assertFalse(report["map_completion"]["complete"])
        self.assertEqual(report["invalid_round_starts"]["5"], "snapshot_not_ten_players")

    def test_unresolved_raw_end_cannot_hide_behind_retained_terminal_score(self):
        report = {}
        with patch("cs2ml.inround.DemoParser", ledger_parser(extra_unresolved=True)):
            _, rounds = compute_inround("d:/fixture.dem", audit=report)
        self.assertEqual(len(rounds), 13)
        self.assertEqual(report["map_completion"]["score_a"], 13)
        self.assertFalse(report["map_completion"]["complete"])
        self.assertIsNone(report["map_completion"]["winner_roster"])
        self.assertEqual(report["map_completion"]["reason_code"], "unresolved_raw_round_end_winners")

    def test_identity_change_has_structured_error_and_preserves_exception_contract(self):
        report = {}
        with patch("cs2ml.inround.DemoParser", ledger_parser(changed_roster_round=7)):
            with self.assertRaisesRegex(ValueError, "Roster composition changed"):
                compute_inround("d:/fixture.dem", audit=report)
        self.assertEqual(report["reason_code"], "roster_changed_within_map")
        self.assertEqual(report["status"], "rejected")
        json.dumps(report, allow_nan=False)

    def test_none_result_has_structured_reason_and_bounded_exclusions(self):
        report = {}
        with patch("cs2ml.inround.DemoParser", ledger_parser([RA]*30, all_identity_missing=True)):
            result = compute_inround("d:/fixture.dem", audit=report)
        self.assertIsNone(result)
        self.assertEqual(report["reason_code"], "no_valid_event_states")
        self.assertEqual(report["excluded_states_by_reason"]["snapshot_not_ten_players"], 30)
        self.assertEqual(len(report["excluded_state_examples"]), 20)

    def test_parser_failure_and_missing_source_have_diagnostics(self):
        report = {}
        with patch("cs2ml.inround.DemoParser", side_effect=RuntimeError("broken fixture")):
            with self.assertRaises(RuntimeError):
                compute_inround("d:/fixture.dem", audit=report)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["reason_code"], "parser_or_extraction_error")
        with patch("cs2ml.inround.DemoParser") as factory:
            factory.return_value.parse_header.return_value = {"map_name": "de_mirage"}
            factory.return_value.parse_event.return_value = pd.DataFrame()
            self.assertIsNone(compute_inround("d:/fixture.dem", audit=report))
        self.assertEqual(report["reason_code"], "missing_freeze_events")
        self.assertNotIn("exception", report)


if __name__ == "__main__":
    unittest.main()
