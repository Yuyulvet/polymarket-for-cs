from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.fivee_state import FiveEStateRecorder, state_hash, summarize_payload


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.start = datetime(2026, 9, 17, 12, tzinfo=timezone.utc)

    def monotonic(self):
        return self.value

    def wall(self):
        return (self.start + timedelta(seconds=self.value)).isoformat()

    def sleep(self, seconds):
        self.value += seconds


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def player(name, money, hp=100, weapon="ak47"):
    return {
        "id": f"player_{name}", "name": name, "money": str(money), "hp": str(hp),
        "weapon": weapon, "weapon_logo": f"https://img/{weapon}.png",
        "helmet": "1", "kevlar": "1", "has_defusekit": "2", "c4": "2",
        "kill": "2", "death": "1", "assist": "0",
    }


def payload(score=11, money=2500):
    return {
        "success": True,
        "data": {"match": {
            "mc_info": {
                "id": "match", "t1_info": {"id": "a", "disp_name": "Alpha"},
                "t2_info": {"id": "b", "disp_name": "Beta"},
            },
            "global_state": {"status": "1", "t1_score": "0", "t2_score": "0"},
            "bouts_state": [{
                "status": "1", "bout_num": "2", "map_name": "Dust2",
                "curr_round_num": "14", "round_start_time": "123", "game_time": "55",
                "curr_bout_stage": "sh", "bomb_planted": "2",
                "t1_stats": {"fh_role": "T", "all_score": str(score),
                             "fh_score": "2", "sh_score": "1", "ot_score": "0"},
                "t2_stats": {"fh_role": "CT", "all_score": "2",
                             "fh_score": "2", "sh_score": "0", "ot_score": "0"},
                "t1_pr_stats": [player("one", money), player("two", 1000, 0, "")],
                "t2_pr_stats": [player("three", 2000), player("four", 3000)],
            }],
        }},
    }


class FiveEStateSummaryTests(unittest.TestCase):
    def test_live_economy_is_normalized_without_claiming_full_inventory(self):
        summary, raw = summarize_payload(payload(), "match")
        bout = summary["live_bouts"][0]
        self.assertEqual(bout["map_name"], "Dust2")
        self.assertEqual(bout["round_number"], 14)
        self.assertEqual(bout["source_round_number"], 14)
        self.assertTrue(bout["source_round_number_matches_score"])
        self.assertEqual(bout["team1"]["current_side"], "CT")
        self.assertEqual(bout["team1"]["economy"]["money_sum"], 3500)
        self.assertEqual(bout["team1"]["economy"]["hp_sum"], 100)
        self.assertEqual(bout["team1"]["economy"]["alive_count"], 1)
        self.assertIn("display_weapon", bout["team1"]["players"][0])
        self.assertNotIn("equipment_value", bout["team1"]["economy"])
        self.assertEqual(raw["mc_info"]["id"], "match")

    def test_state_hash_changes_with_actionable_state(self):
        first, _ = summarize_payload(payload(score=3), "match")
        second, _ = summarize_payload(payload(score=12), "match")
        self.assertNotEqual(state_hash(first), state_hash(second))

    def test_match_identity_is_checked(self):
        with self.assertRaisesRegex(ValueError, "match_id_mismatch"):
            summarize_payload(payload(), "another_match")


class FiveEStateRecorderTests(unittest.TestCase):
    def test_initial_is_ineligible_and_only_changes_are_written(self):
        clock = FakeClock()
        calls = iter([Response(payload()), Response(payload()), Response(payload(score=12))])

        def get(*args, **kwargs):
            return next(calls)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.jsonl"
            recorder = FiveEStateRecorder(
                "match", path, duration_seconds=2.5, poll_seconds=1,
                request_get=get, wall_clock=clock.wall,
                monotonic_clock=clock.monotonic, sleeper=clock.sleep,
            )
            self.assertEqual(recorder.run(), 0)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        snapshots = [row for row in rows if row["record_type"] == "state_snapshot"]
        self.assertEqual(len(snapshots), 2)
        self.assertFalse(snapshots[0]["decision_eligible"])
        self.assertTrue(snapshots[0]["initial_snapshot"])
        self.assertTrue(snapshots[1]["decision_eligible"])
        self.assertEqual(snapshots[1]["summary"]["live_bouts"][0]["team1"]["score"], 12)

    def test_recovery_change_is_marked_ineligible(self):
        clock = FakeClock()
        calls = iter([Response(payload()), OSError("offline"), Response(payload(score=12))])

        def get(*args, **kwargs):
            result = next(calls)
            if isinstance(result, Exception):
                raise result
            return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.jsonl"
            recorder = FiveEStateRecorder(
                "match", path, duration_seconds=3, poll_seconds=1,
                request_get=get, wall_clock=clock.wall,
                monotonic_clock=clock.monotonic, sleeper=clock.sleep,
            )
            self.assertEqual(recorder.run(), 0)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        snapshots = [row for row in rows if row["record_type"] == "state_snapshot"]
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(snapshots[1]["recovery_snapshot"])
        self.assertFalse(snapshots[1]["decision_eligible"])
        self.assertIn("source_recovered", [row["record_type"] for row in rows])

    def test_source_round_mismatch_is_retained_but_not_used(self):
        raw = payload(score=3)
        summary, _ = summarize_payload(raw, "match")
        bout = summary["live_bouts"][0]
        self.assertEqual(bout["completed_rounds"], 5)
        self.assertEqual(bout["round_number"], 6)
        self.assertEqual(bout["source_round_number"], 14)
        self.assertFalse(bout["source_round_number_matches_score"])


if __name__ == "__main__":
    unittest.main()
