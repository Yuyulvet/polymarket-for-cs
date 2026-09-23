from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from cs2ml.map1_data import build_features
from cs2ml.map1_desk import PaperDesk
from cs2ml.map1_market import bind_market
from cs2ml.map1_paper import replay
from cs2ml.map1_sources import binary_resolution, describe_event, discover
from cs2ml.map1_store import DeskStore
from cs2ml.map1_web import make_server
from test_map1 import CONDITION, T, context, event, history_row, resolution, snapshot


def cs_event():
    e = event()
    e.update(title="Counter-Strike: Alpha Academy vs Beta (BO3)", slug="cs2-alpha-beta-2026-09-01")
    return e


def closed_evidence():
    e = cs_event()
    e["closed"] = True
    e["markets"][0].update(closed=True, umaResolutionStatus="resolved", outcomePrices='["1","0"]')
    c = {"condition_id": CONDITION, "closed": True,
         "tokens": [{"token_id": "11", "winner": True, "price": .01},
                    {"token_id": "22", "winner": False, "price": .99}]}
    return e, c


class SourceTests(unittest.TestCase):
    def test_cs2_map1_only_and_start_filter(self):
        self.assertEqual(describe_event(cs_event(), T)["status"], "awaiting_confirmation")
        self.assertEqual(describe_event(cs_event(), T+pd.Timedelta(hours=2))["reason"], "prematch_window_closed")
        e = cs_event(); e["slug"] = "valorant-a-b"; e["title"] = "Valorant: A vs B"
        self.assertEqual(describe_event(e, T)["reason"], "not_a_cs2_match")
        e = cs_event(); e["markets"][0]["sportsMarketType"] = "moneyline"
        self.assertEqual(describe_event(e, T)["reason"], "missing_or_ambiguous_map1_market")

    def test_ended_flag_blocks_even_if_schedule_is_in_future(self):
        e = cs_event(); e["ended"] = True
        self.assertEqual(describe_event(e, T)["reason"], "prematch_window_closed")
        with self.assertRaisesRegex(ValueError, "event_not_prematch"):
            bind_market(e, context(), T)

    def test_pagination_dedup_and_truncation(self):
        calls = []
        e2 = cs_event(); e2["id"] = "124"
        def get(url, params=None):
            calls.append((url, params))
            if url.endswith("/sports"):
                return [{"sport": "cs2", "primaryTagId": 100780}]
            if params.get("after_cursor") == "page2":
                return {"events": [cs_event(), e2]}
            return {"events": [cs_event()], "next_cursor": "page2"}
        result = discover(T.isoformat(), get, max_pages=2, page_size=2)
        self.assertEqual([e["id"] for e in result["events"]], ["123", "124"])
        self.assertFalse(result["truncated"])
        self.assertEqual(calls[1][1]["ascending"], "false")
        self.assertEqual(calls[2][1]["after_cursor"], "page2")
        self.assertTrue(discover(T.isoformat(), get, max_pages=1)["truncated"])

    def test_repeated_cursor_is_failure_not_success(self):
        def get(url, params=None):
            return [{"sport": "cs2", "primaryTagId": 1}] if url.endswith("sports") else {"events": [], "next_cursor": "same"}
        with self.assertRaisesRegex(ValueError, "repeated_discovery_cursor"):
            discover(T.isoformat(), get)

    def test_binary_settlement_uses_winner_not_price(self):
        e, c = closed_evidence()
        result = binary_resolution(e, c, bind_market(event(), context(), T), T+pd.Timedelta(hours=4))
        self.assertEqual(result["payouts"], {"11": 1., "22": 0.})
        self.assertEqual(result["resolved_at"], (T+pd.Timedelta(hours=4)).isoformat())
        self.assertIn("evidence", result)

    def test_unresolved_price_one_is_not_settlement(self):
        e, c = closed_evidence(); e["markets"][0]["umaResolutionStatus"] = "proposed"
        self.assertIsNone(binary_resolution(e, c, bind_market(event(), context(), T), T))

    def test_ambiguous_void_and_identity_conflict_fail_closed(self):
        e, c = closed_evidence()
        b = bind_market(event(), context(), T)
        c["tokens"][0]["winner"] = False
        with self.assertRaisesRegex(ValueError, "requires_review"):
            binary_resolution(e, c, b, T)
        c["condition_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "condition_mismatch"):
            binary_resolution(e, c, b, T)
        e["id"] = "999"
        with self.assertRaisesRegex(ValueError, "event_mismatch"):
            binary_resolution(e, c, b, T)


class PersistenceTests(unittest.TestCase):
    def test_restart_dedup_and_explicit_resolution_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "desk.sqlite"
            store = DeskStore(path)
            for s in [snapshot(), snapshot(), snapshot(T+pd.Timedelta(seconds=2))]:
                store.add_snapshot(s)
            self.assertTrue(store.add_resolution(resolution()))
            self.assertFalse(store.add_resolution(resolution()))
            restarted = DeskStore(path)
            self.assertEqual(len(restarted.snapshots()), 2)
            r = replay(restarted.snapshots(), restarted.resolutions())
            self.assertAlmostEqual(r["realized_pnl"], 5.88)
            with self.assertRaisesRegex(ValueError, "conflicting_resolution"):
                restarted.add_resolution(resolution(0))

    def test_context_lock_and_protocol_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            store = DeskStore(Path(folder) / "desk.sqlite")
            store.pin_protocol({"shares": 10})
            store.pin_protocol({"shares": 10})
            store.save_context(context(), bind_market(event(), context(), T), T.isoformat())
            with self.assertRaisesRegex(ValueError, "protocol_changed"):
                store.pin_protocol({"shares": 20})
            store.add_snapshot(snapshot())
            with self.assertRaisesRegex(ValueError, "context_locked"):
                store.save_context(context(), bind_market(event(), context(), T), T.isoformat())

    def test_empty_cohort_allows_protocol_update(self):
        with tempfile.TemporaryDirectory() as folder:
            store = DeskStore(Path(folder) / "desk.sqlite")
            store.pin_protocol({"shares": 10})
            store.pin_protocol({"shares": 20})
            self.assertFalse(store.snapshots())

    def test_same_timestamp_different_data_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as folder:
            store = DeskStore(Path(folder) / "desk.sqlite")
            store.add_snapshot(snapshot())
            with self.assertRaisesRegex(ValueError, "conflicting_observation"):
                store.add_snapshot(snapshot(p=.1))

    def test_clock_expires_pending_during_disconnect(self):
        r = replay([snapshot()], as_of=T+pd.Timedelta(seconds=30))
        self.assertEqual(r["pending_count"], 0)
        self.assertEqual(r["cash"], 1000)
        self.assertEqual(r["ledger"][-1]["reason"], "missing_timely_post_delay_book")

    def test_pre_delay_book_cannot_fill_after_delay(self):
        later = snapshot(T+pd.Timedelta(seconds=2))
        later["books"] = snapshot()["books"]
        r = replay([snapshot(), later])
        self.assertEqual(r["cash"], 1000)
        self.assertEqual(r["ledger"][-1]["reason"], "book_predates_eligibility")


class DeskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        history = pd.DataFrame([history_row(f"h{i}", T-pd.Timedelta(days=20-i), "a" if i%2 else "b") for i in range(10)])
        history.to_parquet(self.root / "history.parquet")
        build_features(history).to_parquet(self.root / "features.parquet")
        self.desk = PaperDesk(self.root)

    def tearDown(self):
        self.desk.close()
        self.temp.cleanup()

    def confirm(self):
        with patch("cs2ml.map1_desk.now", return_value=T.isoformat()), patch("cs2ml.map1_desk.fetch_event", return_value=cs_event()):
            return self.desk.run("confirm", context())

    def test_confirmation_receipt_not_backdated_and_no_auto_start(self):
        self.assertFalse(self.desk.running)
        self.confirm()
        saved = self.desk.store.contexts()["123"]["context"]
        self.assertEqual(saved["map_known_at"], T.isoformat())
        self.assertEqual(saved["declared_map_known_at"], context()["map_known_at"])
        self.assertFalse(self.desk.state()["live_trading_enabled"])

    def test_failed_confirmation_does_not_persist(self):
        c = context(); c["team_a"]["roster"] = ["1"]
        with patch("cs2ml.map1_desk.now", return_value=T.isoformat()), patch("cs2ml.map1_desk.fetch_event") as network:
            result = self.desk.run("confirm", c)
        network.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertFalse(self.desk.store.contexts())

    def test_discovery_persists_excluded_without_creating_context(self):
        e2 = cs_event(); e2["id"] = "456"; e2["live"] = True
        result = {"events": [cs_event(), e2], "truncated": True, "pages": 1, "sport_metadata": {"sport": "cs2"}}
        with patch("cs2ml.map1_desk.discover", return_value=result), patch("cs2ml.map1_desk.now", return_value=T.isoformat()):
            self.desk.run("discover")
            state = self.desk.state()
        self.assertEqual(len(state["events"]), 2)
        self.assertFalse(self.desk.store.contexts())
        self.assertEqual(state["runs"][0]["result"]["warning"], "discovery_page_limit_reached")

    def test_capture_fill_and_automatic_settlement_end_to_end(self):
        self.confirm()
        for at in [T, T+pd.Timedelta(seconds=2)]:
            with patch("cs2ml.map1_desk.now", return_value=at.isoformat()), patch("cs2ml.map1_desk.capture_once", return_value=snapshot(at)):
                self.desk.run("capture")
        with patch("cs2ml.map1_desk.now", return_value=(T+pd.Timedelta(minutes=5)).isoformat()):
            state = self.desk.state()
        self.assertEqual(len(state["report"]["open_positions"]), 1)
        self.assertEqual(state["report"]["open_positions"][0]["event_id"], "123")
        e, clob = closed_evidence()
        with patch("cs2ml.map1_desk.now", return_value=(T+pd.Timedelta(hours=4)).isoformat()), \
             patch("cs2ml.map1_desk.fetch_event", return_value=e), patch("cs2ml.map1_desk.public_get", return_value=clob):
            result = self.desk.run("settle")
            self.assertEqual(result["settled"], ["123"])
            self.assertEqual(self.desk.run("settle")["settled"], [])
            report = self.desk.report()
        self.assertAlmostEqual(report["realized_pnl"], 5.88)
        self.assertEqual(report["settled_trade_count"], 1)

    def test_missing_context_and_past_matches_never_capture(self):
        with patch("cs2ml.map1_desk.capture_once") as capture:
            self.assertEqual(self.desk.capture_round()["captured"], 0)
            self.confirm()
            with patch("cs2ml.map1_desk.now", return_value=(T+pd.Timedelta(hours=2)).isoformat()):
                self.assertEqual(self.desk.capture_round()["captured"], 0)
            capture.assert_not_called()

    def test_ambiguous_resolution_visible_and_does_not_credit_cash(self):
        self.confirm()
        e, clob = closed_evidence(); clob["tokens"][0]["winner"] = False
        with patch("cs2ml.map1_desk.now", return_value=(T+pd.Timedelta(hours=4)).isoformat()), \
             patch("cs2ml.map1_desk.fetch_event", return_value=e), patch("cs2ml.map1_desk.public_get", return_value=clob):
            self.desk.run("settle")
            state = self.desk.state()
        self.assertFalse(self.desk.store.resolutions())
        self.assertEqual(state["report"]["cash"], 1000)
        self.assertEqual(state["events"][0]["settlement_check"]["status"], "requires_review")
        self.assertTrue(state["warnings"])

    def test_restart_preserves_context_and_is_paused(self):
        self.confirm()
        self.desk.close()
        self.desk = PaperDesk(self.root)
        self.assertFalse(self.desk.running)
        self.assertEqual(self.desk.store.contexts()["123"]["confirmed_at"], T.isoformat())

    def test_cycle_discovery_failure_does_not_block_other_tasks(self):
        self.desk.running = True
        with patch.object(self.desk, "discover_once", side_effect=RuntimeError("offline")), \
             patch.object(self.desk, "capture_round", return_value={"captured": 1}) as capture, \
             patch.object(self.desk, "settle_once", return_value={"settled": []}) as settle:
            result = self.desk.run("cycle")
        self.desk.running = False
        capture.assert_called_once(); settle.assert_called_once()
        self.assertEqual(result["discovery"]["status"], "error")
        self.assertEqual(result["capture"]["captured"], 1)

    def test_no_overlapping_background_operations(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_discovery():
            entered.set(); release.wait(5); return {}
        with patch.object(self.desk, "discover_once", side_effect=blocked_discovery):
            self.desk.submit("discover")
            self.assertTrue(entered.wait(3))
            try:
                with self.assertRaisesRegex(ValueError, "another_operation"):
                    self.desk.submit("capture")
            finally:
                release.set()
            self.desk.future.result(timeout=3)


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.desk = PaperDesk(Path(self.temp.name))
        self.server = make_server(self.desk, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.desk.close(); self.temp.cleanup()

    def request(self, path, body=None, headers=None):
        req = Request(self.base+path, data=json.dumps(body).encode() if body is not None else None,
                      headers=headers or {})
        return urlopen(req, timeout=5)

    def test_serves_paused_state_and_page_security_headers(self):
        with self.request("/api/state") as r:
            state = json.load(r)
        self.assertEqual(state["mode"], "paper_only")
        self.assertFalse(state["running"])
        self.assertTrue(state["csrf_token"])
        self.assertIsNotNone(state["data_error"])
        with self.request("/") as r:
            self.assertIn("frame-ancestors 'none'", r.headers["Content-Security-Policy"])
            self.assertIn("PAPER ONLY", r.read().decode())

    def test_external_host_and_cross_origin_mutations_rejected(self):
        for path, body, headers in [("/api/state", None, {"Host": "evil.example"}),
                                    ("/api/discover", {}, {"Content-Type": "application/json", "Origin": "https://evil.example"})]:
            with self.assertRaises(HTTPError) as error:
                self.request(path, body, headers)
            self.assertEqual(error.exception.code, 403)

    def test_same_origin_action_and_no_order_route(self):
        with self.request("/api/state") as r:
            token = json.load(r)["csrf_token"]
        headers = {"Origin": self.base, "X-CSRF-Token": token, "Content-Type": "application/json"}
        with self.request("/api/stop", {}, headers) as r:
            self.assertFalse(json.load(r)["running"])
        with self.assertRaises(HTTPError) as error:
            self.request("/api/orders", {}, headers)
        self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError) as error:
            self.request("/api/start", {}, headers)
        self.assertEqual(error.exception.code, 400)  # Missing model artifacts.

    def test_export_does_not_mutate_or_fabricate_results(self):
        with self.request("/api/export/report") as r:
            self.assertIn("attachment", r.headers["Content-Disposition"])
            data = json.load(r)
        self.assertEqual(data["cash"], 1000)
        self.assertEqual(data["realized_pnl"], 0)
        self.assertEqual(data["observed_events"], 0)
        self.assertEqual(data["live_readiness"], "NOT_AUTHORIZED_NOT_VALIDATED")

    def test_confirmation_through_http_is_audited(self):
        with self.request("/api/state") as r:
            token = json.load(r)["csrf_token"]
        headers = {"Origin": self.base, "X-CSRF-Token": token, "Content-Type": "application/json"}
        with patch("cs2ml.map1_desk.now", return_value=T.isoformat()), patch("cs2ml.map1_desk.fetch_event", return_value=cs_event()):
            with self.request("/api/confirm", context(), headers) as r:
                self.assertEqual(r.status, 202)
            result = self.desk.future.result(timeout=3)
        self.assertEqual(result["event_id"], "123")
        self.assertEqual(self.desk.store.contexts()["123"]["confirmed_at"], T.isoformat())


if __name__ == "__main__":
    unittest.main()
