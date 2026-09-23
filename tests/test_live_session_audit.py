from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.live_session_audit import audit_session
from cs2ml.trend_protocol import PROTOCOL_ID


BASE = datetime(2026, 9, 19, tzinfo=timezone.utc)


def stamp(seconds):
    return (BASE + timedelta(seconds=seconds)).isoformat()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def event_row(seconds, event_type, detail):
    log = {"type": str(event_type), "round_start": {"round_num": ""},
           "round_end": {"ct_score": "", "t_score": "", "winner": ""}}
    log[{1: "round_start", 2: "round_end"}[event_type]].update(detail)
    return {"record_type": "event", "recv_utc": stamp(seconds),
            "decision_eligible": True,
            "entry": {"bout_num": "1", "map_name": "Mirage",
                      "log_info": json.dumps(log)}}


class LiveSessionAuditTests(unittest.TestCase):
    def build(self, root, *, complete=True):
        write_jsonl(root/"events.jsonl", [
            {"record_type": "session_start", "recv_utc": stamp(0), "match_id": "match"},
            event_row(10, 1, {"round_num": "1"}),
            event_row(590, 2, {"ct_score": "13", "t_score": "8", "winner": "CT"}),
            *([{"record_type": "session_end", "recv_utc": stamp(600)}] if complete else []),
        ])
        write_jsonl(root/"states.jsonl", [
            {"record_type": "session_start", "recv_utc": stamp(0), "match_id": "match"},
            {"record_type": "state_snapshot", "recv_utc": stamp(20),
             "decision_eligible": True, "source_transport": "mqtt",
             "timing_quality": "mqtt_push_receive_time"},
            {"record_type": "session_end", "recv_utc": stamp(600)},
        ])
        tokens = {"A": "ta", "B": "tb"}
        (root/"meta.json").write_text(json.dumps({
            "binding": {"event_id": "123", "market_id": "9", "outcome_tokens": tokens},
            "markets": [{"market_id": "10", "tokens": {"X": "tx", "Y": "ty"}}]}))
        raw = json.dumps([{"asset_id": token, "bids": [], "asks": []}
                          for token in tokens.values()])
        write_jsonl(root/"market.jsonl", [
            {"type": "market_raw", "received_at": stamp(5), "raw": raw},
            {"type": "market_raw", "received_at": stamp(595), "raw": raw},
        ])
        (root/"summary.json").write_text(json.dumps({"exit_reason": "deadline"}))
        return {"schema_version": 1, "protocol_id": PROTOCOL_ID,
                "session_id": "complete-session", "match_id": "match",
                "event_id": "123", "teams": ["A", "B"],
                "maps": [{"bout_num": 1, "map_name": "de_mirage",
                          "market_id": "9", "market_name": "Map 1 Winner"}],
                "sources": {"fivee_events": ["events.jsonl"],
                            "fivee_states": ["states.jsonl"],
                            "market": {"format": "raw_depth", "events": "market.jsonl",
                                       "metadata": "meta.json", "summary": "summary.json"}}}

    def test_complete_three_source_session_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root)
            report = audit_session(value, root)
            self.assertTrue(report["eligible_for_trend_dataset"])
            self.assertGreaterEqual(report["three_source_overlap_seconds"], 300)
            self.assertEqual(report["sources"]["market"]["expected_tokens"], ["ta", "tb"])

    def test_interrupted_or_top_only_session_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root, complete=False)
            value["sources"]["fivee_states"] = []
            value["sources"]["market"] = {
                "format": "normalized_top", "events": "market.jsonl", "metadata": "meta.json"}
            report = audit_session(value, root)
            self.assertFalse(report["eligible_for_trend_dataset"])
            self.assertIn("fivee_event_segment_missing_session_end", report["blocking_reasons"])
            self.assertIn("fivee_mqtt_state_source_missing", report["blocking_reasons"])
            self.assertIn("market_full_depth_missing", report["blocking_reasons"])

    def test_v2_accepts_operator_restart_within_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root, complete=False)
            # simulate a killed first segment + resumed second segment (same file)
            write_jsonl(root/"events.jsonl", [
                {"record_type": "session_start", "recv_utc": stamp(0), "match_id": "match"},
                event_row(10, 1, {"round_num": "1"}),
                {"record_type": "session_start", "recv_utc": stamp(40), "match_id": "match"},
                event_row(590, 2, {"ct_score": "13", "t_score": "8", "winner": "CT"}),
                {"record_type": "session_end", "recv_utc": stamp(600)},
            ])
            report = audit_session(value, root, audit_version=2)
            self.assertNotIn("fivee_event_segment_missing_session_end",
                             report["blocking_reasons"])
            self.assertEqual(report["restarts_accepted"], 1)
            # v1 counted segments per path, so it never saw intra-file restarts
            # at all — that blindness is exactly why R1 exists.
            self.assertEqual(report["v1_blocking_reasons"], [])

    def test_v2_rejects_restart_with_too_large_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root, complete=False)
            write_jsonl(root/"events.jsonl", [
                {"record_type": "session_start", "recv_utc": stamp(0), "match_id": "match"},
                event_row(10, 1, {"round_num": "1"}),
                {"record_type": "session_start", "recv_utc": stamp(4000), "match_id": "match"},
                event_row(4010, 2, {"ct_score": "13", "t_score": "8", "winner": "CT"}),
                {"record_type": "session_end", "recv_utc": stamp(4020)},
            ])
            report = audit_session(value, root, audit_version=2)
            self.assertIn("fivee_event_segment_missing_session_end",
                          report["blocking_reasons"])

    def test_v2_market_post_match_clean_close_is_normal_when_tape_covers_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root)
            (root/"summary.json").write_text(json.dumps({"exit_reason": "connection_closed"}))
            report = audit_session(value, root, audit_version=2)
            # market tape ends at 595s >= event end 590s: coverage complete
            self.assertNotIn("market_capture_missing_normal_end", report["blocking_reasons"])
            self.assertTrue(report["sources"]["market"]["normal_end_v2"])
            self.assertIn("market_capture_missing_normal_end",
                          report["v1_blocking_reasons"])

    def test_v2_r2_uses_last_match_event_not_recorder_deadline(self):
        """Semifinal scenario: recorder polls on until the hours-deadline
        (session_end far after the final round), market socket cleanly closes
        minutes after the last match event. Tape completeness must be judged
        against the last match event, not the deadline row."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root)
            (root/"summary.json").write_text(json.dumps({"exit_reason": "connection_closed"}))
            write_jsonl(root/"events.jsonl", [
                {"record_type": "session_start", "recv_utc": stamp(0), "match_id": "match"},
                event_row(10, 1, {"round_num": "1"}),
                event_row(590, 2, {"ct_score": "13", "t_score": "8", "winner": "CT"}),
                # recorder keeps polling for hours after the match ends
                {"record_type": "poll_ok", "recv_utc": stamp(3600)},
                {"record_type": "session_end", "recv_utc": stamp(3600)},
            ])
            write_jsonl(root/"market.jsonl", [
                {"type": "market_raw", "received_at": stamp(5), "raw":
                 json.dumps([{"asset_id": "ta", "bids": [], "asks": []}])},
                # market goes quiet and the server closes the socket ~8 min
                # after the last match event at 590s
                {"type": "market_raw", "received_at": stamp(1070), "raw":
                 json.dumps([{"asset_id": "ta", "bids": [], "asks": []}])},
            ])
            report = audit_session(value, root, audit_version=2)
            self.assertNotIn("market_capture_missing_normal_end",
                             report["blocking_reasons"])
            self.assertTrue(report["sources"]["market"]["normal_end_v2"])

    def test_v2_market_mid_match_close_still_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root)
            (root/"summary.json").write_text(json.dumps({"exit_reason": "connection_closed"}))
            # truncate the market tape so it ends long before the event stream
            write_jsonl(root/"market.jsonl", [
                {"type": "market_raw", "received_at": stamp(5), "raw":
                 json.dumps([{"asset_id": "ta", "bids": [], "asks": []}])},
            ])
            report = audit_session(value, root, audit_version=2)
            self.assertIn("market_capture_missing_normal_end", report["blocking_reasons"])
            self.assertFalse(report["sources"]["market"]["normal_end_v2"])

    def test_v2_per_bout_status_and_no_complete_bout_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root)
            # bout 1 starts at round 4 (source-side late log); only round 4+ observed
            write_jsonl(root/"events.jsonl", [
                {"record_type": "session_start", "recv_utc": stamp(0), "match_id": "match"},
                event_row(10, 1, {"round_num": "4"}),
                event_row(590, 2, {"ct_score": "13", "t_score": "8", "winner": "CT"}),
                {"record_type": "session_end", "recv_utc": stamp(600)},
            ])
            report = audit_session(value, root, audit_version=2)
            self.assertEqual(report["bout_status"], {"1": "incomplete_no_round_one"})
            self.assertEqual(report["incomplete_bouts"], [1])
            self.assertFalse(report["eligible_for_trend_dataset"])
            self.assertIn("no_complete_bout", report["blocking_reasons"])
            # v1 name preserved for comparison
            self.assertIn("capture_did_not_observe_round_one_live",
                          report["v1_blocking_reasons"])

    def test_v2_session_with_one_complete_bout_passes_bout_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = self.build(root)
            value["maps"] = [
                {"bout_num": 1, "map_name": "de_anubis", "market_id": "9",
                 "market_name": "Map 1 Winner"},
                {"bout_num": 2, "map_name": "de_inferno", "market_id": "9",
                 "market_name": "Map 2 Winner"},
            ]
            write_jsonl(root/"events.jsonl", [
                {"record_type": "session_start", "recv_utc": stamp(0), "match_id": "match"},
                # bout 1: log starts at round 4 (incomplete), map Anubis
                {**event_row(10, 1, {"round_num": "4"}),
                 "entry": {"bout_num": "1", "map_name": "Anubis", "log_info":
                           json.dumps({"type": "1", "round_start": {"round_num": "4"},
                                       "round_end": {}})}},
                event_row(590, 2, {"ct_score": "13", "t_score": "8", "winner": "CT"}),
                # bout 2: fully observed on Inferno (map_name varies by row)
                {**event_row(595, 1, {"round_num": "1"}),
                 "entry": {"bout_num": "2", "map_name": "Inferno", "log_info":
                           json.dumps({"type": "1", "round_start": {"round_num": "1"},
                                       "round_end": {}})}},
                {**event_row(598, 2, {"ct_score": "13", "t_score": "6", "winner": "CT"}),
                 "entry": {"bout_num": "2", "map_name": "Inferno", "log_info":
                           json.dumps({"type": "2", "round_start": {},
                                       "round_end": {"ct_score": "13", "t_score": "6",
                                                     "winner": "CT"}})}},
                {"record_type": "session_end", "recv_utc": stamp(600)},
            ])
            report = audit_session(value, root, audit_version=2)
            self.assertTrue(report["eligible_for_trend_dataset"], report["blocking_reasons"])
            self.assertEqual(report["bout_status"]["1"], "incomplete_no_round_one")
            self.assertEqual(report["bout_status"]["2"], "complete")
            self.assertEqual(report["incomplete_bouts"], [1])


if __name__ == "__main__":
    unittest.main()
