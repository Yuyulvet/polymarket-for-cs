from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


MODULE = Path(__file__).parents[1] / ".scratch" / "stream_lag.py"
SPEC = importlib.util.spec_from_file_location("stream_lag_under_test", MODULE)
stream_lag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stream_lag)


class StreamLagTests(unittest.TestCase):
    def test_visual_timestamps_are_epoch_seconds_and_frames_can_be_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binding = {"event_id": "123", "team_a": "A", "team_b": "B",
                       "map_number": 2, "market": "Map 2 Winner"}
            (root / "metadata.json").write_text(json.dumps({
                "schema_version": 2, "binding_status": "confirmed", "binding": binding
            }), encoding="utf-8")
            (root / "binding_confirmation.json").write_text(json.dumps({
                "binding": binding, "display_left_is": "team_a"
            }), encoding="utf-8")
            path = root / "verified_score_events.jsonl"
            rows = [
                {"type": "verified_score_transition", "after_frame": 1,
                 "received_at": "2026-09-16T16:34:57.800000+00:00",
                 "scoring_team": "A", "eligible_for_lag_analysis": True, **binding},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = stream_lag.load_visual(folder)
        self.assertEqual(result.frame.tolist(), [1])
        self.assertGreater(result.iloc[0].t0, 1_700_000_000)
        self.assertAlmostEqual(result.iloc[0].t0, pd.Timestamp(rows[0]["received_at"]).timestamp())

    def test_cross_match_verified_row_is_rejected_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binding = {"event_id": "123", "team_a": "A", "team_b": "B",
                       "map_number": 2, "market": "Map 2 Winner"}
            (root / "metadata.json").write_text(json.dumps({
                "schema_version": 2, "binding_status": "confirmed", "binding": binding
            }), encoding="utf-8")
            (root / "binding_confirmation.json").write_text(json.dumps({
                "binding": binding, "display_left_is": "team_a"
            }), encoding="utf-8")
            row = {"type": "verified_score_transition", "after_frame": 2,
                   "received_at": "2026-09-16T16:35:00+00:00",
                   "scoring_team": "B", "eligible_for_lag_analysis": True,
                   **{**binding, "event_id": "999"}}
            (root / "verified_score_events.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cross_match"):
                stream_lag.load_visual(folder)

    def test_multiple_separated_reprices_are_retained_and_binary_twins_collapse(self):
        rows = []
        # Two stable step changes, mirrored by the binary outcomes.
        for second in range(0, 42, 2):
            a = .5 if second < 10 else (.7 if second < 28 else .4)
            for outcome, mid in (("A", a), ("B", 1-a)):
                rows.append({"ts": 1000. + second, "market": "Map 2 Winner",
                             "outcome": outcome, "mid": mid})
        result = stream_lag.price_jumps(pd.DataFrame(rows), .08)
        self.assertGreaterEqual(len(result), 2)
        self.assertTrue(result.outcomes.eq(2).all())
        self.assertTrue(set(result.positive_outcome).issubset({"A", "B"}))
        self.assertGreater(result.t1.max() - result.t1.min(), 10)

    def test_pairing_is_one_to_one(self):
        visual = pd.DataFrame({"t0": [100., 110.], "frame": [1, 2], "frac": [.2, .2]})
        jumps = pd.DataFrame({"t1": [101., 111., 112.], "market": ["m"]*3,
                              "delta": [.1, .1, .1]})
        result = stream_lag.pair_one_to_one(visual, jumps, 5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.frame.nunique(), 2)
        self.assertEqual(result.t1.nunique(), 2)

    def test_pairing_requires_price_move_toward_round_winner(self):
        visual = pd.DataFrame({"t0": [100.], "frame": [1],
                               "scoring_team": ["A"]})
        jumps = pd.DataFrame({"t1": [101., 102.], "market": ["m", "m"],
                              "delta": [.1, .1], "positive_outcome": ["B", "A"]})
        result = stream_lag.pair_one_to_one(visual, jumps, 5)
        self.assertEqual(result.t1.tolist(), [102.])


if __name__ == "__main__":
    unittest.main()
