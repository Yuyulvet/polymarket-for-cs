"""Offline rebuild contracts; synthetic files are never parsed as demos."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from cs2ml import inplay_rebuild as rebuild
from cs2ml.map1_data import first_ct_is_ct, path_key


A, B = "1,2,3,4,5", "10,6,7,8,9"


def ledger(path, winners=None):
    winners = winners if winners is not None else [A] * 12 + [B] * 9 + [A]
    scores = {A: 0, B: 0}
    rows = []
    for number, winner in enumerate(winners, 1):
        ct, tt = (A, B) if first_ct_is_ct(number) else (B, A)
        start = number * 3000
        rows.append({"schema_version": 2, "demo_path": str(path), "map_name": "de_mirage",
                     "round_num": number, "ct_roster": ct, "t_roster": tt,
                     "ct_is_a": ct == A, "winner_side": "CT" if winner == ct else "T",
                     "score_a_before": scores[A], "score_b_before": scores[B],
                     "ct_equip0": 5000., "t_equip0": 4500.,
                     "round_start_tick": start, "round_end_tick": start + 2500})
        scores[winner] += 1
    completion = {"complete": True, "n_rounds": len(rows), "roster_a": A, "roster_b": B,
                  "score_a": scores[A], "score_b": scores[B],
                  "winner_roster": A if scores[A] > scores[B] else B}
    return pd.DataFrame(rows), completion


def raw_event(path, number=1):
    tick = number * 3000 + 10
    return pd.DataFrame([{"schema_version": 2, "demo_path": str(path), "map_name": "de_mirage",
                          "round_num": number, "tick": tick, "event_tick": tick,
                          "state_tick": tick + 1, "event_type": "kill",
                          "observation_basis": "first_full_tick_after_event_batch",
                          "ct_alive": 5, "t_alive": 4, "ct_hp": 450., "t_hp": 350.,
                          "bomb_state": 0}])


def chronology(path, rounds=None, completion=None):
    if rounds is None:
        rounds, completion = ledger(path)
    return {"map_id": path_key(path), "match_id": Path(path).parent.name,
            "map_name": "de_mirage", "roster_a": A, "roster_b": B,
            "round_wins_a": completion["score_a"], "round_wins_b": completion["score_b"],
            "rounds": len(rounds), "y": int(completion["winner_roster"] == A),
            "start_at": pd.Timestamp("2026-01-01T10:00Z"),
            "available_at": pd.Timestamp("2026-01-01T13:00Z")}


def midround(path):
    return pd.DataFrame([{"schema_version": 2, "demo_path": str(path),
                          "match_id": Path(path).parent.name, "round_num": 1,
                          "freeze_tick": 3000, "cutoff_tick": 4920, "round_end_tick": 5500,
                          "seconds": 30, "tick_rate": 64, "ct_alive": 5, "t_alive": 4,
                          "ct_deaths30": 0, "t_deaths30": 1, "first_kill_side": "CT"}])


def fake_extract(path, audit):
    rounds, completion = ledger(path)
    audit.update(status="accepted", map_completion=completion)
    return raw_event(path), rounds


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("D:/synthetic/series/map1.dem")
        self.rounds, self.complete = ledger(self.path)

    def test_complete_label_uses_stable_rosters_and_not_last_side(self):
        result = rebuild.validate_map_label(self.rounds, self.complete)
        self.assertEqual((result["score_a"], result["score_b"], result["winner_roster"]), (13, 9, A))
        self.assertEqual(result["n_rounds"], 22)
        self.assertEqual(result["match_id"], "series")
        self.assertEqual(self.rounds.iloc[-1].ct_roster, B)
        self.assertEqual(self.rounds.iloc[-1].score_a_before, 12)

    def test_repeated_overtime_uses_actual_current_side(self):
        rounds, completion = ledger(self.path, [A, B] * 15 + [B] * 4)
        result = rebuild.validate_map_label(rounds, completion)
        self.assertEqual((result["score_a"], result["score_b"], result["n_overtime_rounds"]), (15, 19, 10))
        self.assertEqual(result["winner_roster"], B)
        self.assertTrue(result["overtime"])

    def test_noncontiguous_or_disagreeing_round_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "noncontiguous_round_ledger"):
            rebuild.validate_map_label(self.rounds.drop(index=2), self.complete)
        completion = {**self.complete, "n_rounds": 21}
        with self.assertRaisesRegex(ValueError, "raw_and_retained_round_count_disagree"):
            rebuild.validate_map_label(self.rounds, completion)

    def test_future_score_and_wrong_side_features_are_rejected(self):
        for key, value in (("score_a_before", 13), ("score_b_before", 9), ("ct_is_a", False)):
            bad = self.rounds.copy()
            bad.loc[0, key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "invalid_past_only_score_or_side"):
                rebuild.validate_map_label(bad, self.complete)

    def test_first_terminal_must_be_last_and_tied_overtime_is_not_complete(self):
        for winners, reason in (([A] * 14, "rounds_after_terminal_score"),
                                ([A, B] * 12, "incomplete_or_nonstandard_map"),
                                ([A, B] * 15, "incomplete_or_nonstandard_map")):
            rounds, completion = ledger(self.path, winners)
            with self.subTest(reason=reason, n=len(winners)), self.assertRaisesRegex(ValueError, reason):
                rebuild.validate_map_label(rounds, completion)

    def test_roster_changes_mixed_maps_and_unknown_winners_are_rejected(self):
        for key, value, reason in (("ct_roster", "1,2,3,4,99", "roster_changed"),
                                   ("demo_path", "D:/synthetic/series/map2.dem", "mixed_map_identity"),
                                   ("winner_side", "UNKNOWN", "unknown_round_winner")):
            bad = self.rounds.copy()
            bad.loc[1, key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, reason):
                rebuild.validate_map_label(bad, self.complete)

    def test_incomplete_or_conflicting_extractor_declaration_is_rejected(self):
        for completion in ({**self.complete, "complete": False, "reason_code": "missing_round"},
                           {**self.complete, "winner_roster": B}):
            with self.subTest(completion=completion), self.assertRaises(ValueError):
                rebuild.validate_map_label(self.rounds, completion)

    def test_nonstandard_side_schedule_is_independently_rejected(self):
        bad = self.rounds.copy()
        # Keep scores/winner identity internally consistent while lying about the
        # mandatory halftime swap; completion.complete must not override this.
        mask = bad.round_num.gt(12)
        bad.loc[mask, "ct_roster"], bad.loc[mask, "t_roster"] = A, B
        bad.loc[mask, "ct_is_a"] = True
        bad.loc[mask, "winner_side"] = bad.loc[mask, "winner_side"].map({"CT": "T", "T": "CT"})
        with self.assertRaises(ValueError):
            rebuild.validate_map_label(bad, self.complete)


class HistoryTests(unittest.TestCase):
    def test_matched_missing_and_conflicting_history_have_distinct_states(self):
        path = Path("D:/synthetic/series/map1.dem")
        rounds, completion = ledger(path)
        label = rebuild.validate_map_label(rounds, completion)
        hist = chronology(path)
        self.assertEqual(rebuild.compare_history(label, hist), "matched")
        self.assertEqual(rebuild.compare_history(label, None), "missing_chronology")
        for field, value in (("round_wins_a", 12), ("rounds", 21), ("map_name", "de_nuke"),
                             ("match_id", "wrong-series"), ("roster_a", B)):
            with self.subTest(field=field):
                self.assertEqual(rebuild.compare_history(label, {**hist, field: value}),
                                 "new_ledger_disagrees_with_history")

    def test_missing_invalid_or_naive_chronology_never_counts_as_matched(self):
        path = Path("D:/synthetic/series/map1.dem")
        rounds, completion = ledger(path)
        label = rebuild.validate_map_label(rounds, completion)
        hist = chronology(path)
        bad_history = [{k: v for k, v in hist.items() if k not in {"start_at", "available_at"}},
                       {**hist, "available_at": hist["start_at"]},
                       {**hist, "start_at": pd.Timestamp("2026-01-01")},
                       {**hist, "start_at": "not a timestamp"}]
        for bad in bad_history:
            with self.subTest(history=bad):
                self.assertNotEqual(rebuild.compare_history(label, bad), "matched")


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_path = self.root / "series" / "map1.dem"
        self.source_path.parent.mkdir()
        self.source_path.write_bytes(b"synthetic demo placeholder; must not parse")
        self.source = rebuild.fingerprint(self.source_path)
        self.folder = self.root / "shard"
        self.history = chronology(self.source_path)

    def build(self, *, extract=fake_extract, mid_extract=midround, history=True):
        return rebuild.rebuild_one(self.source, self.folder, self.history if history else None,
                                   extract=extract, mid_extract=mid_extract)

    def test_sparse_events_do_not_shorten_full_map_label(self):
        result = self.build()
        self.assertEqual(result["status"], "map_complete")
        self.assertTrue(result["map_eligible"])
        self.assertEqual(result["artifacts"]["inround_events"]["rows"], 1)
        self.assertEqual(result["artifacts"]["round_states"]["rows"], 22)
        labels = pd.read_parquet(self.folder / "map_labels.parquet")
        self.assertEqual((labels.iloc[0].n_rounds, labels.iloc[0].score_a), (22, 13))
        self.assertEqual(self.source_path.read_bytes(), b"synthetic demo placeholder; must not parse")
        self.assertEqual(rebuild._verify_shard(self.folder, self.source)["status"], "map_complete")
        # Terminal-only observations are excluded without discarding the complete
        # independent round ledger or making labels depend on sparse features.
        def terminal_extract(path, audit):
            events, rounds = fake_extract(path, audit)
            events["event_type"], events["bomb_state"] = "bomb_defused", 2
            return events, rounds
        self.folder = self.root / "no-accepted-events"
        terminal = self.build(extract=terminal_extract)
        self.assertEqual(terminal["status"], "map_complete")
        self.assertTrue(terminal["map_eligible"])
        self.assertNotIn("inround_events", terminal["artifacts"])
        self.assertEqual(terminal["artifacts"]["round_states"]["rows"], 22)
        self.assertEqual(terminal["errors"]["events"], "no_nonterminal_valid_events")
        self.assertEqual(pd.read_parquet(self.folder / "map_labels.parquet").iloc[0].n_rounds, 22)

    def test_partial_round_data_cannot_become_complete_map_label(self):
        def extract(path, audit):
            rounds, complete = ledger(path)
            audit["map_completion"] = complete
            return raw_event(path), rounds.iloc[:1]
        result = self.build(extract=extract)
        self.assertFalse(result["map_eligible"])
        self.assertEqual(result["status"], "round_data_only")
        self.assertIn("raw_and_retained_round_count_disagree", result["errors"]["map_label"])
        self.assertFalse((self.folder / "map_labels.parquet").exists())

    def test_history_disagreement_quarantines_complete_raw_evidence(self):
        self.history["round_wins_a"] = 12
        result = self.build()
        self.assertEqual(result["status"], "map_complete")
        self.assertEqual(result["chronology"], "new_ledger_disagrees_with_history")
        self.assertFalse(result["map_eligible"])
        self.assertTrue((self.folder / "map_labels.parquet").is_file())

    def test_source_change_before_parse_is_audited_without_calling_extractors(self):
        self.source_path.write_bytes(b"changed input")
        extract, mid = Mock(), Mock()
        result = self.build(extract=extract, mid_extract=mid)
        self.assertEqual(result["errors"]["source"], "source_changed_before_parse")
        extract.assert_not_called()
        mid.assert_not_called()
        self.assertFalse(result["map_eligible"])
        self.assertTrue((self.folder / "audit.json").is_file())

    def test_source_change_during_parse_disables_map_eligibility(self):
        def extract(path, audit):
            parsed = fake_extract(path, audit)
            Path(path).write_bytes(b"changed input during extraction")
            return parsed
        result = self.build(extract=extract)
        self.assertEqual(result["status"], "source_changed")
        self.assertFalse(result["map_eligible"])
        self.assertEqual(result["errors"]["source"], "source_changed_during_parse")
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            rebuild._verify_shard(self.folder, self.source)

    def test_extractor_failure_is_explicit_and_independent_midround_data_survives(self):
        result = self.build(extract=Mock(side_effect=ValueError("synthetic parse failure")))
        self.assertEqual(result["status"], "excluded")
        self.assertIn("synthetic parse failure", result["errors"]["events"])
        self.assertIn("midround_v2", result["artifacts"])
        self.assertFalse(result["map_eligible"])

    def test_existing_output_and_json_are_never_overwritten(self):
        self.folder.mkdir()
        marker = self.folder / "legacy.txt"
        marker.write_text("keep unchanged", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep unchanged")
        output = self.root / "existing.json"
        rebuild._json(output, {"original": True})
        with self.assertRaises(FileExistsError):
            rebuild._json(output, {"original": False})
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"original": True})

    def test_resume_rejects_artifact_tampering_and_incomplete_shards(self):
        self.build()
        artifact = self.folder / "map_labels.parquet"
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "artifact integrity failure"):
            rebuild._verify_shard(self.folder, self.source)
        empty = self.root / "empty-shard"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "Incomplete shard"):
            rebuild._verify_shard(empty, self.source)

    def interrupted_run(self):
        output = self.root / "run"
        history_path = self.root / "history.parquet"
        pd.DataFrame([self.history]).to_parquet(history_path, index=False)
        original = rebuild.rebuild_one
        def injected(source, folder, history):
            return original(source, folder, history, extract=fake_extract, mid_extract=midround)
        with patch.object(rebuild, "rebuild_one", side_effect=injected), \
                patch.object(rebuild, "aggregate", side_effect=RuntimeError("simulated interruption")), \
                patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                rebuild.run([self.source_path], output, history_path)
        return output, history_path

    def test_resume_reuses_verified_shard_without_reparse_and_completed_run_is_protected(self):
        output, history_path = self.interrupted_run()
        with patch.object(rebuild, "rebuild_one") as extract, patch("builtins.print"):
            result = rebuild.run([self.source_path], output, history_path, resume=True)
        extract.assert_not_called()
        self.assertEqual(result["map_eligible_with_chronology"], 1)
        original = (output / "map_labels.parquet").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "Completed rebuild"):
            rebuild.run([self.source_path], output, history_path, resume=True)
        with self.assertRaises(FileExistsError):
            rebuild.run([self.source_path], output, history_path)
        self.assertEqual((output / "map_labels.parquet").read_bytes(), original)

    def test_resume_rejects_history_or_manifest_code_tampering(self):
        output, history_path = self.interrupted_run()
        manifest_path = output / "manifest.json"
        original_manifest = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(original_manifest)
        manifest["code_sha256"]["inplay_rebuild.py"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest/code/history differs"):
            rebuild.run([self.source_path], output, history_path, resume=True)
        manifest_path.write_text(original_manifest, encoding="utf-8")
        changed_history = pd.read_parquet(history_path)
        changed_history["available_at"] += pd.Timedelta(hours=1)
        changed_history.to_parquet(history_path, index=False)
        with self.assertRaisesRegex(ValueError, "manifest/code/history differs"):
            rebuild.run([self.source_path], output, history_path, resume=True)
        self.assertFalse((output / "audit.json").exists())


if __name__ == "__main__":
    unittest.main()
