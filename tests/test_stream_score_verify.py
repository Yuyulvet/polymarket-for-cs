from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.stream_score_verify import confirm_binding, session_status, verify_score_transition
from cs2ml.stream_score_watch import validate_binding


BINDING = {"event_id": "123", "team_a": "Lavked", "team_b": "Honvéd",
           "map_number": 2, "market": "Map 2 Winner"}


def write_session(folder: Path) -> None:
    (folder / "metadata.json").write_text(json.dumps({
        "schema_version": 2, "binding_status": "pending_manual_confirmation",
        "binding": BINDING, "eligible_for_inference": False,
    }), encoding="utf-8")
    rows = [
        {"type": "calibration_frame", "frame": 1},
        {"type": "candidate_visual_change", "before_frame": 10, "after_frame": 11,
         "before_received_at": "2026-09-17T00:00:00+00:00",
         "received_at": "2026-09-17T00:00:01+00:00",
         "before_path": "before11.jpg", "after_path": "after11.jpg", **BINDING},
        {"type": "candidate_visual_change", "before_frame": 11, "after_frame": 12,
         "before_received_at": "2026-09-17T00:00:01+00:00",
         "received_at": "2026-09-17T00:00:02+00:00",
         "before_path": "before12.jpg", "after_path": "after12.jpg", **BINDING},
    ]
    (folder / "events.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")


class StreamScoreVerifyTests(unittest.TestCase):
    def test_binding_validation_rejects_ambiguous_identity(self):
        self.assertEqual(validate_binding("123", "A", "B", 2)["market"], "Map 2 Winner")
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_binding("123", "A", "a", 2)
        with self.assertRaisesRegex(ValueError, "numeric"):
            validate_binding("abc", "A", "B", 2)

    def test_reversed_broadcast_order_maps_scores_to_market_teams(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_session(folder)
            confirmation = confirm_binding(folder, "Honved", "Lavked", 2, 1)
            self.assertEqual(confirmation["display_left_is"], "team_b")
            first = verify_score_transition(folder, 11, 4, 10, 5, 10)
            self.assertEqual((first["after_score_a"], first["after_score_b"]), (10, 5))
            self.assertEqual(first["scoring_team"], "Honvéd")
            second = verify_score_transition(folder, 12, 5, 10, 5, 11)
            self.assertEqual((second["after_score_a"], second["after_score_b"]), (11, 5))
            self.assertEqual(second["scoring_team"], "Lavked")
            rows = (folder / "verified_score_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            status = session_status(folder)
            self.assertEqual((status["candidate_count"], status["verified_count"]), (2, 2))
            self.assertTrue(all(row["verified"] for row in status["candidates"]))

    def test_identity_mismatch_quarantines_session(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_session(folder)
            with self.assertRaisesRegex(ValueError, "identity"):
                confirm_binding(folder, "Nexus", "Ex-Trustee", 2, 1)
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["binding_status"], "quarantined_identity_mismatch")
            self.assertTrue((folder / "quarantine.json").exists())

    def test_unknown_evidence_frame_is_correctable_and_does_not_quarantine(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_session(folder)
            with self.assertRaisesRegex(ValueError, "unknown_evidence_frame"):
                confirm_binding(folder, "Lavked", "Honvéd", 2, 999)
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["binding_status"], "pending_manual_confirmation")
            self.assertFalse((folder / "quarantine.json").exists())

    def test_non_single_or_discontinuous_score_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_session(folder)
            confirm_binding(folder, "Lavked", "Honvéd", 2, 1)
            with self.assertRaisesRegex(ValueError, "increase_by_one"):
                verify_score_transition(folder, 11, 10, 4, 12, 4)
            verify_score_transition(folder, 11, 10, 4, 11, 4)
            with self.assertRaisesRegex(ValueError, "not_continuous"):
                verify_score_transition(folder, 12, 12, 4, 13, 4)


if __name__ == "__main__":
    unittest.main()
