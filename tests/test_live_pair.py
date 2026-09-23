from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.live_pair import load_binding, load_quotes, load_states, pair_states


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def team(name, team_id, score, money, hp, weapon):
    return {
        "name": name, "id": team_id, "current_side": "CT" if name == "Alpha" else "T",
        "score": score,
        "players": [{"display_weapon": weapon}],
        "economy": {"money_sum": money, "hp_sum": hp, "alive_count": 5,
                    "helmet_count": 4, "kevlar_count": 5, "defuse_kit_count": 1,
                    "display_weapon_known_players": 1},
    }


class LivePairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.meta_path, self.state_path, self.quote_path = (
            root / "market.meta.json", root / "state.jsonl", root / "quotes.jsonl")
        self.meta = {
            "schema_version": 2, "event_id": "123", "timing_basis": "local_receive_time",
            "markets": [{"market": "Map 2 Winner", "market_id": "456",
                         "condition_id": "0xabc", "outcomes": ["Alpha", "Beta"],
                         "tokens": {"Alpha": "ta", "Beta": "tb"},
                         "seconds_delay": 1}],
        }
        self.meta_path.write_text(json.dumps(self.meta), encoding="utf-8")
        bouts = [{"bout_num": 2, "map_name": "Dust2", "round_number": 10,
                  "stage": "fh", "bomb_state": 2,
                  "team1": team("Alpha", "a", 6, 12000, 500, "ak47"),
                  "team2": team("Beta", "b", 3, 5000, 400, "awp")}]
        base = {"schema_version": 1, "record_type": "state_snapshot",
                "timing_quality": "mqtt_push_receive_time",
                "source_lag_seconds": 1.0,
                "summary": {"match_id": "match", "live_bouts": bouts}}
        write_jsonl(self.state_path, [
            {**base, "state_hash": "initial", "recv_utc": "1970-01-01T00:01:39Z",
             "decision_eligible": False},
            {**base, "state_hash": "eligible", "recv_utc": "1970-01-01T00:01:40Z",
             "decision_eligible": True},
        ])
        quote_rows = []
        for ts, a_bid, a_ask in [(99.5, .59, .61), (100.5, .60, .62),
                                 (101.0, .61, .63), (102.0, .62, .64), (106.0, .66, .68)]:
            for outcome, token, bid, ask in (
                ("Alpha", "ta", a_bid, a_ask),
                ("Beta", "tb", 1-a_ask, 1-a_bid),
            ):
                quote_rows.append({"event_id": "123", "market": "Map 2 Winner",
                                   "outcome": outcome, "token": token,
                                   "local_ts": ts, "source_ts": ts+.01,
                                   "best_bid": bid, "best_ask": ask})
        write_jsonl(self.quote_path, quote_rows)

    def tearDown(self):
        self.temp.cleanup()

    def test_strict_pairing_excludes_initial_and_mirrors_features(self):
        meta, binding = load_binding(self.meta_path, event_id="123", market="Map 2 Winner")
        states, audit = load_states([self.state_path], bout_num=2)
        quotes, _ = load_quotes(self.quote_path, meta, binding)
        rows, pair_audit = pair_states(states, quotes, meta, binding, horizons=(1, 5))
        self.assertEqual(audit["ineligible"], 1)
        self.assertEqual(pair_audit["paired_states"], 1)
        self.assertEqual(len(rows), 2)
        alpha = next(row for row in rows if row["focal_outcome"] == "Alpha")
        beta = next(row for row in rows if row["focal_outcome"] == "Beta")
        self.assertEqual(alpha["features"]["score_diff"], 3)
        self.assertEqual(beta["features"]["score_diff"], -3)
        self.assertEqual(alpha["features"]["money_diff"], 7000)
        self.assertEqual(alpha["state_source_lag_seconds"], 1.0)
        self.assertEqual(alpha["features"]["rifle_count_diff"], 1)
        self.assertEqual(alpha["effective_decision_ts"], 101.0)
        self.assertEqual(alpha["quote_at_effective_decision"]["ask"], .63)
        self.assertEqual(alpha["quote_at_effective_decision"]["quote_age_seconds"], 0)
        self.assertEqual(alpha["first_quote_after_effective_decision"]["quote_wait_seconds"], 0)
        self.assertIsNone(alpha["first_quote_after_effective_decision"]["quote_age_seconds"])
        self.assertEqual(alpha["evaluation_only"]["5"]["ask"], .68)

    def test_non_monotonic_state_is_quarantined(self):
        base = {"schema_version": 1, "record_type": "state_snapshot",
                "decision_eligible": True,
                "timing_quality": "mqtt_push_receive_time",
                "source_lag_seconds": 1.0}
        def snapshot(digest, timestamp, round_number, alpha_score, beta_score):
            bout = {"bout_id": "bout", "bout_num": 2,
                    "round_number": round_number,
                    "team1": team("Alpha", "a", alpha_score, 1000, 500, "ak47"),
                    "team2": team("Beta", "b", beta_score, 1000, 500, "awp")}
            return {**base, "state_hash": digest, "recv_utc": timestamp,
                    "summary": {"match_id": "match", "live_bouts": [bout]}}
        write_jsonl(self.state_path, [
            snapshot("a", "1970-01-01T00:01:40Z", 20, 12, 8),
            snapshot("stale", "1970-01-01T00:01:41Z", 21, 11, 8),
            snapshot("b", "1970-01-01T00:01:42Z", 22, 12, 9),
        ])
        states, audit = load_states([self.state_path], bout_num=2)
        self.assertEqual([state["state_hash"] for state in states], ["a", "b"])
        self.assertEqual(audit["non_monotonic"], 1)

    def test_unchanged_top_is_coalesced_to_latest_receipt(self):
        meta, binding = load_binding(self.meta_path, market="Map 2 Winner")
        write_jsonl(self.quote_path, [
            {"event_id": "123", "market": "Map 2 Winner", "outcome": "Alpha",
             "token": "ta", "local_ts": 100, "best_bid": .4, "best_ask": .5},
            {"event_id": "123", "market": "Map 2 Winner", "outcome": "Alpha",
             "token": "ta", "local_ts": 101, "best_bid": .4, "best_ask": .5},
        ])
        quotes, audit = load_quotes(self.quote_path, meta, binding)
        self.assertEqual(len(quotes["Alpha"].quotes), 1)
        self.assertEqual(quotes["Alpha"].quotes[0].ts, 101)
        self.assertEqual(audit["coalesced_unchanged_top"], 1)

    def test_http_poll_state_is_excluded_by_default(self):
        row = json.loads(self.state_path.read_text(encoding="utf-8").splitlines()[1])
        row["timing_quality"] = "normal_poll_receive_time"
        write_jsonl(self.state_path, [row])
        states, audit = load_states([self.state_path], bout_num=2)
        self.assertEqual(states, [])
        self.assertEqual(audit["disallowed_timing_quality"], 1)
        diagnostic, _ = load_states(
            [self.state_path], bout_num=2, allowed_timing_qualities=None,
            max_source_lag_seconds=None)
        self.assertEqual(len(diagnostic), 1)

    def test_stale_mqtt_state_is_excluded(self):
        row = json.loads(self.state_path.read_text(encoding="utf-8").splitlines()[1])
        row["source_lag_seconds"] = 8.0
        write_jsonl(self.state_path, [row])
        states, audit = load_states([self.state_path], bout_num=2)
        self.assertEqual(states, [])
        self.assertEqual(audit["source_lag_rejected"], 1)

    def test_wrong_token_is_rejected(self):
        meta, binding = load_binding(self.meta_path, market="Map 2 Winner")
        write_jsonl(self.quote_path, [{"event_id": "123", "market": "Map 2 Winner",
                                      "outcome": "Alpha", "token": "wrong",
                                      "local_ts": 100, "best_bid": .4, "best_ask": .5}])
        _, audit = load_quotes(self.quote_path, meta, binding)
        self.assertEqual(audit["identity_mismatch"], 1)

    def test_cross_event_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "event_id_mismatch"):
            load_binding(self.meta_path, event_id="999", market="Map 2 Winner")


if __name__ == "__main__":
    unittest.main()
