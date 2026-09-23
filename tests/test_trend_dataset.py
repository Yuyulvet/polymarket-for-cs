"""Phase-2 dataset builder tests: synthetic three-source session + spot checks."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cs2ml import trend_dataset as td

BASE = 100000.0
TOK_YES = "111"
TOK_NO = "222"


def _market_row(mono: float, messages: list[dict]) -> str:
    return json.dumps({"type": "market_raw",
                       "received_monotonic_seconds": mono,
                       "raw": json.dumps(messages)})


def _book_msg(token: str, bid: float, ask: float, bid_size=100.0,
              ask_size=100.0) -> dict:
    return {"event_type": "book", "asset_id": token,
            "bids": [{"price": str(bid), "size": str(bid_size)}],
            "asks": [{"price": str(ask), "size": str(ask_size)}]}


def _pc_msg(token: str, price: float, size: float, side: str) -> dict:
    return {"event_type": "price_change",
            "price_changes": [{"asset_id": token, "price": str(price),
                               "size": str(size), "side": side}]}


def _trade_msg(token: str, price: float, size: float, side: str) -> dict:
    return {"event_type": "last_trade_price", "asset_id": token,
            "price": str(price), "size": str(size), "side": side}


def _fivee_event(mono: float, log: dict, *, bout_num=1, eligible=True,
                 initial=False, recovering=False) -> str:
    return json.dumps({"schema_version": 2, "record_type": "event",
                       "recv_utc": "2026-09-19T00:00:00+00:00",
                       "recv_mono": mono,
                       "initial_snapshot": initial,
                       "recovery_snapshot": recovering,
                       "decision_eligible": eligible,
                       "entry": {"bout_num": str(bout_num),
                                 "log_info": json.dumps(log)}})


def _fivee_state(mono: float, summary: dict, *, eligible=True,
                 transport="mqtt", quality="mqtt_push_receive_time") -> str:
    return json.dumps({"record_type": "state_snapshot", "recv_mono": mono,
                       "decision_eligible": eligible,
                       "source_transport": transport,
                       "timing_quality": quality, "summary": summary})


def _summary(score1=1, score2=0, round_number=2) -> dict:
    return {"live_bouts": [{
        "bout_num": 1, "map_name": "de_ancient", "round_number": round_number,
        "bomb_state": 0,
        "team1": {"score": score1, "first_half_side": "CT",
                  "current_side": "CT",
                  "economy": {"alive_count": 5, "hp_sum": 500,
                              "money_sum": 20000}},
        "team2": {"score": score2, "first_half_side": "T",
                  "current_side": "T",
                  "economy": {"alive_count": 4, "hp_sum": 320,
                              "money_sum": 9000}}},
    ]}


def _spec() -> dict:
    return {"schema_version": 1, "session_id": "test-session",
            "match_id": "csgo_mc_1", "event_id": "7001",
            "teams": ["A", "B"],
            "maps": [{"bout_num": 1, "map_name": "de_ancient",
                      "market_id": "1001", "market_name": "Map 1 Winner"}]}


def _write_session(root: Path) -> Path:
    session = root / "session"
    (session / "market").mkdir(parents=True)
    # ---- market raw: t=0 book; t=20 pc yes bid up; t=50 book ask up
    market_rows = [
        _market_row(BASE + 0, [_book_msg(TOK_YES, 0.60, 0.62),
                               _book_msg(TOK_NO, 0.38, 0.40)]),
        _market_row(BASE + 5, [_trade_msg(TOK_YES, 0.61, 10.0, "BUY")]),
        _market_row(BASE + 20, [_book_msg(TOK_YES, 0.63, 0.64),
                                _book_msg(TOK_NO, 0.36, 0.37)]),
        _market_row(BASE + 50, [_book_msg(TOK_YES, 0.66, 0.68),
                                _book_msg(TOK_NO, 0.32, 0.34)]),
        _market_row(BASE + 200, [_book_msg(TOK_YES, 0.70, 0.72),
                                 _book_msg(TOK_NO, 0.28, 0.30)]),
        _market_row(BASE + 400, [_book_msg(TOK_YES, 0.74, 0.76),
                                 _book_msg(TOK_NO, 0.24, 0.26)]),
    ]
    (session / "market" / "events.jsonl").write_text(
        "\n".join(market_rows) + "\n", encoding="utf-8")
    metadata = {"mode": "read_only_raw_evidence_multi",
                "markets": [{"market_id": "1001",
                             "condition_id": "0xabc",
                             "market_name": "Map 1 Winner",
                             "outcomes": ["Yes", "No"],
                             "tokens": {"Yes": TOK_YES, "No": TOK_NO},
                             "seconds_delay": 1, "taker_base_fee": 0}]}
    (session / "market" / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8")
    # ---- fivee events: initial snapshot t=1 (ineligible); pistol t=10;
    #      round_end t=30; non-trigger type t=40; recovery t=45; round_end t=60
    events = [
        _fivee_event(BASE + 1, {"type": "6"}, initial=True, eligible=False),
        _fivee_event(BASE + 10, {"type": "1",
                                 "round_start": {"round_num": 2}}),
        _fivee_event(BASE + 30, {"type": "2",
                                 "round_end": {"ct_score": 1, "t_score": 0}}),
        _fivee_event(BASE + 40, {"type": "7"}),
        _fivee_event(BASE + 45, {"type": "1",
                                 "round_start": {"round_num": 3}},
                     eligible=False, recovering=True),
        _fivee_event(BASE + 60, {"type": "2",
                                 "round_end": {"ct_score": 2, "t_score": 0}}),
    ]
    (session / "fivee_events.jsonl").write_text(
        "\n".join(events) + "\n", encoding="utf-8")
    # ---- fivee states: strict snapshot before pistol; a recovery row after
    states = [
        _fivee_state(BASE + 9, _summary(score1=1, score2=0, round_number=2)),
        _fivee_state(BASE + 31, _summary(score1=2, score2=0, round_number=3)),
        _fivee_state(BASE + 35, _summary(), eligible=False,
                     quality="recovery_backfill"),
    ]
    (session / "fivee_states.jsonl").write_text(
        "\n".join(states) + "\n", encoding="utf-8")
    return session


class BookRebuilderTests(unittest.TestCase):
    def test_snapshot_change_and_trade(self):
        reb = td.MarketRebuilder({TOK_YES})
        reb.feed_row(1.0, json.dumps([_book_msg(TOK_YES, 0.60, 0.62)]))
        reb.feed_row(2.0, json.dumps([_pc_msg(TOK_YES, 0.61, 50.0, "BUY")]))
        reb.feed_row(3.0, json.dumps([_pc_msg(TOK_YES, 0.60, 0.0, "BUY")]))
        reb.feed_row(4.0, json.dumps([_trade_msg(TOK_YES, 0.615, 5.0, "BUY")]))
        tape = reb.tapes[TOK_YES]
        self.assertEqual(tape.states[0][1:3], (0.60, 0.62))
        self.assertEqual(tape.states[1][1:3], (0.61, 0.62))  # BUY adds bid
        self.assertEqual(tape.states[2][1:3], (0.61, 0.62))  # remove 0.60 no-op
        self.assertEqual(len(tape.trades), 1)
        self.assertEqual(tape.trades[0][1], 0.615)

    def test_unknown_token_ignored(self):
        reb = td.MarketRebuilder({TOK_YES})
        reb.feed_row(1.0, json.dumps([_book_msg("999", 0.5, 0.6)]))
        self.assertEqual(reb.applied.get("book_snapshot", 0), 0)


class TriggerTests(unittest.TestCase):
    def test_collect_triggers_eligible_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            rows = [
                _fivee_event(1.0, {"type": "6"}, initial=True, eligible=False),
                _fivee_event(2.0, {"type": "1", "round_start": {"round_num": 2}}),
                _fivee_event(3.0, {"type": "2", "round_end": {"ct_score": 1,
                                                              "t_score": 0}}),
                _fivee_event(4.0, {"type": "1", "round_start": {"round_num": 3}}),
            ]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            triggers, stats = td.collect_triggers(path)
        kinds = [(t.kind, t.round_num) for t in triggers]
        self.assertEqual(kinds, [("post_pistol_freeze", 2),
                                 ("round_end_economy_update", 1)])


class BuildRowsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.session = _write_session(self.root)
        self.frame, self.report = td.build_rows(self.session, _spec())

    def tearDown(self):
        self.tmp.cleanup()

    def test_trigger_count_and_long_format(self):
        # prematch + pistol + round_end x2 = 4 triggers x 2 tokens = 8 rows
        self.assertEqual(len(self.frame), 8)
        kinds = set(self.frame["decision_kind"])
        self.assertEqual(kinds, {"prematch_prior_dislocation",
                                 "post_pistol_freeze",
                                 "round_end_economy_update"})

    def test_exec_time_includes_delay(self):
        pistol = self.frame[self.frame["decision_kind"] == "post_pistol_freeze"]
        # info_mono = BASE+10, seconds_delay = 1, latency = 0
        self.assertTrue((pistol["exec_mono"] == BASE + 11).all())

    def test_entry_and_exit_are_executable_quotes(self):
        prematch = self.frame[(self.frame["decision_kind"] == "prematch_prior_dislocation")
                              & (self.frame["outcome"] == "Yes")].iloc[0]
        # t_exec = first_book(BASE+0) + delay(1) = BASE+1, so the first book
        # at BASE+0 is NOT executable; entry comes from the BASE+20 book.
        self.assertEqual(prematch["entry_ask"], 0.64)
        self.assertEqual(prematch["exit_bid_h15"], 0.63)  # book at BASE+20
        self.assertEqual(prematch["exit_bid_h30"], 0.66)  # book at BASE+50
        self.assertEqual(prematch["exit_bid_h60"], 0.70)  # book at BASE+200
        self.assertEqual(prematch["exit_bid_h120"], 0.70)  # BASE+121 < 400
        self.assertAlmostEqual(prematch["net_h15"], 0.63 - 0.64)
        self.assertAlmostEqual(prematch["net_h120"], 0.70 - 0.64)

    def test_exit_bid_comes_from_forward_scan(self):
        # every exit bid equals some observed bid at/after exec+horizon
        for horizon in (15, 30, 60, 120):
            col = f"exit_bid_h{horizon}"
            valid = self.frame[self.frame[f"label_valid_h{horizon}"]]
            self.assertTrue(((valid[col] >= 0.24) & (valid[col] <= 0.76)).all(),
                            f"horizon {horizon}")

    def test_state_join_backward_only(self):
        pistol = self.frame[(self.frame["decision_kind"] == "post_pistol_freeze")
                            & (self.frame["outcome"] == "Yes")].iloc[0]
        # snapshot at BASE+9 (1:0, round 2) must win over BASE+31 (2:0)
        self.assertEqual(pistol["game_score_t1"], 1)
        self.assertEqual(pistol["game_round_num"], 2)
        rnd_end = self.frame[(self.frame["decision_kind"] == "round_end_economy_update")
                             & (self.frame["outcome"] == "Yes")]
        # t=30 trigger joins t=9 state (1:0); t=60 trigger joins t=31 state (2:0)
        self.assertEqual(sorted(rnd_end["game_score_t1"]), [1, 2])

    def test_recovery_rows_do_not_join(self):
        # the t=35 recovery state row must be invisible to every trigger
        non_prematch = self.frame[self.frame["decision_kind"] != "prematch_prior_dislocation"]
        self.assertEqual(set(non_prematch["game_alive_t1"]), {5})

    def test_pistol_winner_from_prior_round_end(self):
        rnd_end = self.frame[(self.frame["decision_kind"] == "round_end_economy_update")
                             & (self.frame["outcome"] == "Yes")]
        # a round_end trigger knows its own result at info_mono:
        # t=30 (round 1 end, ct_up) -> pistol winner t1, streak_t2=1;
        # t=60 (round 2 end, ct_up) -> pistol t1, streak_t2=2
        self.assertEqual(sorted(rnd_end["game_pistol_winner_side"].fillna("NONE")),
                         ["t1", "t1"])
        self.assertEqual(sorted(rnd_end["game_loss_streak_t2"]), [1, 2])

    def test_report_hashes_and_protocol(self):
        self.assertEqual(self.report["protocol_id"], "cs2_swing_trend_v1")
        self.assertTrue(self.report["protocol_sha256"])
        self.assertTrue(self.report["inputs_sha256"]["market_events"])
        self.assertEqual(self.report["trigger_stats"]["trigger_post_pistol_freeze"], 1)
        self.assertEqual(self.report["trigger_stats"]["trigger_round_end_economy_update"], 2)
        self.assertEqual(self.report["trigger_stats"]["ineligible_event_skipped"], 2)

    def test_rows_are_parquet_serializable(self):
        out = self.root / "out"
        out.mkdir()
        self.frame.to_parquet(out / "rows.parquet", index=False)
        reread = pd.read_parquet(out / "rows.parquet")
        self.assertEqual(len(reread), 8)


class FeeLookupTests(unittest.TestCase):
    def test_clob_lookup_used_when_metadata_missing(self):
        calls = []
        def fetch(condition_id):
            calls.append(condition_id)
            return {"seconds_delay": 2, "taker_base_fee": 250}
        params = td.fee_params_for_market({"condition_id": "0xabc",
                                           "seconds_delay": None,
                                           "taker_fee_bps": None},
                                          fetch_clob=fetch)
        self.assertEqual(params["seconds_delay"], 2)
        self.assertEqual(params["taker_fee_bps"], 250)
        self.assertEqual(params["param_source"], "clob_api")
        self.assertEqual(calls, ["0xabc"])

    def test_metadata_wins_and_fee_math(self):
        params = td.fee_params_for_market({"condition_id": "0xabc",
                                           "seconds_delay": 1,
                                           "taker_fee_bps": 1000})
        self.assertEqual(params["param_source"], "capture_metadata")
        with tempfile.TemporaryDirectory() as tmp:
            session = _write_session(Path(tmp))
            spec = _spec()
            frame, _ = td.build_rows(session, spec)
        prematch = frame[(frame["decision_kind"] == "prematch_prior_dislocation")
                         & (frame["outcome"] == "Yes")].iloc[0]
        # fee 0 in metadata -> net equals raw difference
        self.assertAlmostEqual(prematch["net_h15"], 0.63 - 0.64)


class NormalizedFormatRejectionTests(unittest.TestCase):
    def test_normalized_top_format_rejected_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "legacy" / "session"
            (session / "market").mkdir(parents=True)
            metadata = {"mode": "normalized_top",
                        "markets": [{"market_id": "1001", "tokens": {}}]}
            (session / "market" / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8")
            spec = _spec()
            spec["sources"] = {
                "fivee_events": [], "fivee_states": [],
                "market": {"format": "normalized_top",
                           "events": "legacy/session/market/events.jsonl",
                           "metadata": "legacy/session/market/metadata.json"}}
            (session / "market" / "events.jsonl").write_text("", encoding="utf-8")
            frame, report = td.build_rows(session, spec, root=root)
        self.assertEqual(len(frame), 0)
        self.assertEqual(report["drop_reasons"],
                         {"market_format_not_time_joinable": 1})


class SpotCheckTests(unittest.TestCase):
    """验收：随机抽查一行，能从原始记录手工复现。"""

    def test_manual_row_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = _write_session(Path(tmp))
            frame, _ = td.build_rows(session, _spec())
            # 手工重放 tape，复现 prematch Yes 行
            reb = td.MarketRebuilder({TOK_YES, TOK_NO})
            with (session / "market" / "events.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    reb.feed_row(float(row["received_monotonic_seconds"]),
                                 row["raw"])
            tape = reb.tapes[TOK_YES]
            t_exec = BASE + 0 + 1  # first_book + delay
            ask, _ = td._first_at_or_after(tape.states, t_exec, 0, 2)
            bid15, _ = td._first_at_or_after(tape.states, t_exec + 15, 0, 1)
            row = frame[(frame["decision_kind"] == "prematch_prior_dislocation")
                        & (frame["outcome"] == "Yes")].iloc[0]
            self.assertEqual(row["entry_ask"], ask)
            self.assertEqual(row["exit_bid_h15"], bid15)
            self.assertAlmostEqual(row["net_h15"], bid15 - ask)


if __name__ == "__main__":
    unittest.main()
