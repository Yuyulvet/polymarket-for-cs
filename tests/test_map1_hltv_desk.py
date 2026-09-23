"""Offline tests of the reviewed HLTV-to-paper boundary, never live scraping."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from cs2ml.map1_desk import PaperDesk
from cs2ml.map1_hltv_desk import HltvReview, MAX_EVIDENCE_AGE, steam_id
from cs2ml.map1_store import DeskStore
from cs2ml.map1_web import make_server
from test_map1 import A, B, T, context, snapshot
from test_map1_desk import cs_event


MATCH_URL = "https://www.hltv.org/matches/2398020/alpha-academy-vs-beta-test"


def parsed_match():
    return {
        "match_id": "2398020", "event_name": "Offline test league",
        "scheduled_start_at": context()["scheduled_start_at"],
        "map_name": "de_mirage", "status": "prematch", "issues": [],
        "teams": [
            {"hltv_team_id": "101", "name": "Alpha Academy", "players": [
                {"hltv_player_id": str(i), "name": f"player{i}"} for i in range(1, 6)]},
            {"hltv_team_id": "102", "name": "Beta", "players": [
                {"hltv_player_id": str(i), "name": f"player{i}"} for i in range(6, 11)]},
        ],
    }


def identity_payload(evidence_id, event_id="123"):
    return {"event_id": event_id, "evidence_id": evidence_id, "verified": True,
            "mappings": [{"hltv_player_id": str(i), "steamid": sid,
                          "source": f"fixture:independently-verified-player-{i}"}
                         for i, sid in enumerate(A + B, 1)]}


@contextmanager
def fixed_clock(at=T):
    with patch("cs2ml.map1_desk.now", return_value=at.isoformat()), \
         patch("cs2ml.map1_hltv_desk.now", return_value=at.isoformat()):
        yield


@contextmanager
def source_page(parsed=None, error=None):
    """Mock only the public fetching/parsing boundary, not the review checks."""
    with patch("cs2ml.map1_hltv_desk.fetch_match_page", side_effect=error,
               return_value={"html": "<html>offline fixture</html>",
                             "received_at": "1999-01-01T00:00:00Z"}) as fetch, \
         patch("cs2ml.map1_hltv_desk.parse_match_page",
               return_value=copy.deepcopy(parsed if parsed is not None else parsed_match())):
        yield fetch


class HltvReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = DeskStore(self.root / "map1" / "desk.sqlite")
        self.review = HltvReview(self.store, self.root / "map1")

    def tearDown(self):
        self.temp.cleanup()

    def imported(self, parsed=None, at=T, event=None, error=None):
        self.review.last_request = None
        with fixed_clock(at), source_page(parsed, error):
            return self.review.import_match(event or cs_event(), MATCH_URL)

    def mapped(self, parsed=None):
        view = self.imported(parsed)
        with fixed_clock():
            self.review.save_identities(identity_payload(view["evidence_id"]))
        return view

    def test_import_is_evidence_only_and_receipt_cannot_be_backdated(self):
        view = self.imported()
        self.assertFalse(view["ready"])
        self.assertIn("hltv_player_identities_unverified", view["issues"])
        self.assertFalse(self.store.contexts())
        self.assertFalse(self.store.snapshots())
        saved = self.store.hltv_evidence(view["evidence_id"])
        self.assertEqual(saved["received_at"], T.isoformat())
        self.assertEqual(saved["html"], "<html>offline fixture</html>")
        self.assertEqual(saved["sha256"], hashlib.sha256(saved["html"].encode()).hexdigest())

    def test_exact_nickname_hints_are_not_verified_and_ambiguity_is_preserved(self):
        pd.DataFrame([{"name": "PLAYER1", "steamid": A[0]},
                      {"name": "player1", "steamid": B[0]},
                      {"name": "player10-longer", "steamid": B[-1]}]).to_csv(
                          self.root / "player_hlvt_all.csv", index=False)
        self.review = HltvReview(self.store, self.root / "map1")
        view = self.imported()
        first = view["teams"][0]["players"][0]
        self.assertEqual({p["steamid"] for p in first["candidates"]}, {A[0], B[0]})
        self.assertTrue(all(p["verified"] is False for p in first["candidates"]))
        self.assertFalse(first["verified"])
        self.assertIsNone(first["steamid"])
        self.assertEqual(view["teams"][1]["players"][-1]["candidates"], [])
        self.assertFalse(self.store.hltv_identities())

    def test_identity_attestation_source_and_valid_individual_steamid_required(self):
        eid = self.imported()["evidence_id"]
        cases = []
        missing_attestation = identity_payload(eid); missing_attestation["verified"] = "true"
        cases.append((missing_attestation, "attestation_required"))
        missing_source = identity_payload(eid); missing_source["mappings"][0]["source"] = " "
        cases.append((missing_source, "missing_evidence"))
        unknown_player = identity_payload(eid); unknown_player["mappings"][0]["hltv_player_id"] = "9999"
        cases.append((unknown_player, "invalid_identity"))
        duplicate = identity_payload(eid); duplicate["mappings"][1] = duplicate["mappings"][0]
        cases.append((duplicate, "invalid_identity"))
        invalid_sid = identity_payload(eid); invalid_sid["mappings"][0]["steamid"] = "1"
        cases.append((invalid_sid, "invalid_individual_steamid64"))
        for payload, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.review.save_identities(payload)
            self.assertFalse(self.store.hltv_identities())
        for sid in ["", "76561197960265727", "76561202255233024", "１２３", "steam:123"]:
            with self.subTest(sid=sid), self.assertRaises(ValueError):
                steam_id(sid)

    def test_parquet_alias_hints_keep_demo_evidence_and_deduplicate_csv_fallback(self):
        pd.DataFrame([
            {"name": "player1", "steamid": A[0], "demo_path": "fixture-old.dem"},
            {"name": "old-player1-alias", "steamid": A[0], "demo_path": "fixture-older.dem"},
            {"name": "PLAYER1", "steamid": A[0], "demo_path": "fixture-new.dem"},
        ]).to_parquet(self.root / "player_features.parquet")
        pd.DataFrame([{"name": "player1", "steamid": A[0]},
                      {"name": "player2", "steamid": A[1]}]).to_csv(
                          self.root / "player_hlvt_all.csv", index=False)
        self.review = HltvReview(self.store, self.root / "map1")
        view = self.imported()
        first, second = view["teams"][0]["players"][:2]
        self.assertEqual(len(first["candidates"]), 1)
        self.assertEqual(first["candidates"][0]["steamid"], A[0])
        self.assertIn("player_features.parquet :: fixture-old.dem", first["candidates"][0]["source"])
        self.assertEqual(self.review.candidates["old-player1-alias"][0]["steamid"], A[0])
        self.assertIn("fixture-older.dem", self.review.candidates["old-player1-alias"][0]["source"])
        self.assertEqual(second["candidates"][0]["steamid"], A[1])
        self.assertIn("player_hlvt_all.csv", second["candidates"][0]["source"])
        self.assertFalse(first["verified"])
        self.assertFalse(second["verified"])
        self.assertFalse(self.store.hltv_identities())

    def test_identity_conflicts_are_atomic_one_to_one_and_restart_safe(self):
        eid = self.imported()["evidence_id"]
        initial = identity_payload(eid); initial["mappings"] = initial["mappings"][:1]
        with fixed_clock():
            self.review.save_identities(initial)
        conflict = identity_payload(eid)
        conflict["mappings"] = [conflict["mappings"][1], conflict["mappings"][0]]
        conflict["mappings"][1]["steamid"] = B[0]
        with self.assertRaisesRegex(ValueError, "identity_conflict"):
            self.review.save_identities(conflict)
        self.assertEqual(set(self.store.hltv_identities()), {"1"})  # No partial insert of player 2.
        conflict = identity_payload(eid); conflict["mappings"] = conflict["mappings"][1:2]
        conflict["mappings"][0]["steamid"] = A[0]
        with self.assertRaisesRegex(ValueError, "identity_conflict"):
            self.review.save_identities(conflict)
        original = self.store.hltv_identities()["1"]
        with fixed_clock(T + pd.Timedelta(minutes=1)):
            self.review.save_identities(initial)
        restarted = DeskStore(self.store.path)
        self.assertEqual(restarted.hltv_identities(), {"1": original})
        self.assertEqual(restarted.hltv_evidence(eid), self.store.hltv_evidence(eid))

    def test_latest_evidence_is_required_and_cannot_cross_events(self):
        old = self.imported()["evidence_id"]
        fresh = self.imported(at=T + pd.Timedelta(seconds=1))["evidence_id"]
        with self.assertRaisesRegex(ValueError, "superseded"):
            self.review.current("123", old)
        with self.assertRaisesRegex(ValueError, "event_mismatch"):
            self.review.current("456", fresh)
        with self.assertRaisesRegex(ValueError, "superseded"):
            self.review.save_identities(identity_payload(old))

    def test_human_match_review_required_and_confirmation_uses_server_clock(self):
        view = self.mapped()
        submitted = {"event_id": "123", "evidence_id": view["evidence_id"],
                     "reviewed": False, "map_known_at": "1999-01-01T00:00:00Z",
                     "roster_known_at": "1999-01-01T00:00:00Z"}
        at = (T + pd.Timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "match_review_required"):
            self.review.confirmed_context(submitted, cs_event(), at)
        submitted["reviewed"] = True
        result = self.review.confirmed_context(submitted, cs_event(), at)
        self.assertEqual(result["map_known_at"], at)
        self.assertEqual(result["roster_known_at"], at)
        self.assertEqual(result["source_received_at"], T.isoformat())
        self.assertEqual(result["team_a"]["roster"], A)
        self.assertFalse(self.store.contexts())  # Only the desk can persist a bound context.

    def test_map_lineup_match_schedule_and_age_fail_closed(self):
        evidence_id = self.mapped()["evidence_id"]
        original = self.store.hltv_evidence(evidence_id)
        cases = []
        missing_map = copy.deepcopy(original); missing_map["parsed"]["map_name"] = None
        cases.append((missing_map, "hltv_map1_not_announced"))
        for status in ("live", "finished", "unknown"):
            changed = copy.deepcopy(original); changed["parsed"]["status"] = status
            cases.append((changed, "hltv_match_not_prematch"))
        wrong_team = copy.deepcopy(original); wrong_team["parsed"]["teams"][0]["name"] = "Alpha"
        cases.append((wrong_team, "hltv_team_identity_mismatch"))
        wrong_start = copy.deepcopy(original)
        wrong_start["parsed"]["scheduled_start_at"] = (T + pd.Timedelta(hours=2)).isoformat()
        cases.append((wrong_start, "hltv_schedule_mismatch"))
        incomplete = copy.deepcopy(original); incomplete["parsed"]["teams"][0]["players"].pop()
        cases.append((incomplete, "hltv_incomplete_lineup"))
        duplicate = copy.deepcopy(original)
        duplicate["parsed"]["teams"][1]["players"][0] = duplicate["parsed"]["teams"][0]["players"][0]
        cases.append((duplicate, "hltv_incomplete_or_duplicate_players"))
        parse_issue = copy.deepcopy(original); parse_issue["parsed"]["issues"] = ["access_challenge"]
        cases.append((parse_issue, "access_challenge"))
        for evidence, reason in cases:
            with self.subTest(reason=reason):
                preview = self.review.preview(evidence, cs_event(), T)
                self.assertFalse(preview["ready"])
                self.assertIn(reason, preview["issues"])
        for at in (T - pd.Timedelta(seconds=1), T + pd.Timedelta(seconds=MAX_EVIDENCE_AGE + 1)):
            self.assertIn("hltv_evidence_stale_or_future", self.review.preview(original, cs_event(), at)["issues"])
        live_event = cs_event(); live_event["live"] = True
        self.assertIn("hltv_market_not_prematch", self.review.preview(original, live_event, T)["issues"])

    def test_same_hltv_match_cannot_bind_second_market_event(self):
        view = self.mapped()
        c = self.review.confirmed_context({"event_id": "123", "evidence_id": view["evidence_id"],
                                           "reviewed": True}, cs_event(), T.isoformat())
        self.store.save_context(c, {}, T.isoformat())
        second = cs_event(); second["id"] = "456"
        other = self.imported(event=second)
        self.assertFalse(other["ready"])
        self.assertIn("hltv_match_already_bound_to_another_event", other["issues"])

    def test_failure_is_audited_without_identity_saving_or_network_retry(self):
        view = self.imported(error=RuntimeError("HTTP 403 access restricted"))
        saved = self.store.hltv_evidence(view["evidence_id"])
        self.assertIsNone(saved["html"])
        self.assertIn("HTTP 403", saved["error"])
        self.assertIn("hltv_source_unavailable_manual_retry_required", view["issues"])
        with self.assertRaisesRegex(ValueError, "valid_source_required"):
            self.review.save_identities(identity_payload(view["evidence_id"]))
        with fixed_clock(), source_page() as fetch, self.assertRaisesRegex(ValueError, "rate_limit"):
            self.review.import_match(cs_event(), MATCH_URL)
        fetch.assert_not_called()


class HltvDeskIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "map1"
        self.desk = PaperDesk(self.root)
        # Model computation is mocked separately; evidence/context checks are real.
        self.desk.data_error = None
        self.desk.history = self.desk.features = pd.DataFrame()

    def tearDown(self):
        self.desk.close()
        self.temp.cleanup()

    def imported(self, at=T, parsed=None, event=None, error=None):
        self.desk.hltv.last_request = None
        e = event or cs_event()
        with fixed_clock(at), patch("cs2ml.map1_desk.fetch_event", return_value=e), source_page(parsed, error):
            return self.desk.run("hltv_import", {"event_id": e["id"], "url": MATCH_URL,
                                                  "received_at": "1999-01-01T00:00:00Z"})

    def confirmed(self):
        imported = self.imported()
        with fixed_clock():
            saved = self.desk.run("hltv_identities", identity_payload(imported["evidence_id"]))
        self.assertEqual(saved["identities_saved"], 10)
        with fixed_clock(), patch("cs2ml.map1_desk.fetch_event", return_value=cs_event()):
            confirmed = self.desk.run("hltv_confirm", {"event_id": "123", "evidence_id": imported["evidence_id"],
                                                       "reviewed": True, "confirmed_at": "1999-01-01T00:00:00Z"})
        self.assertEqual(confirmed["event_id"], "123")
        return self.desk.store.contexts()["123"]["context"]

    def test_import_and_identity_review_never_auto_confirm_or_start(self):
        imported = self.imported()
        self.assertFalse(imported["ready_for_review"])
        with fixed_clock():
            self.desk.run("hltv_identities", identity_payload(imported["evidence_id"]))
            state = self.desk.state()
        self.assertTrue(state["events"][0]["hltv"]["ready"])
        self.assertFalse(state["events"][0]["confirmed"])
        self.assertFalse(state["running"])
        self.assertFalse(state["live_trading_enabled"])
        self.assertFalse(self.desk.store.snapshots())
        self.assertFalse(self.desk.store.contexts())

    def test_invalid_url_rejected_before_any_network(self):
        for url in ("http://127.0.0.1/matches/123/x", "https://evil.example/matches/123/x",
                    "https://www.hltv.org/player/123/x", "https://www.hltv.org@evil.example/matches/123/x"):
            with self.subTest(url=url), fixed_clock(), patch("cs2ml.map1_desk.fetch_event") as market, \
                 patch("cs2ml.map1_hltv_desk.fetch_match_page") as source:
                result = self.desk.run("hltv_import", {"event_id": "123", "url": url})
                self.assertEqual(result["status"], "error")
                market.assert_not_called(); source.assert_not_called()

    def test_fetched_event_id_must_match_requested_event_before_import_or_confirmation(self):
        imported = self.imported()
        with fixed_clock():
            self.desk.run("hltv_identities", identity_payload(imported["evidence_id"]))
        wrong = cs_event(); wrong["id"] = "456"  # Same names and schedule are not sufficient identity.
        self.desk.hltv.last_request = None
        with fixed_clock(), patch("cs2ml.map1_desk.fetch_event", return_value=wrong), source_page() as source:
            result = self.desk.run("hltv_import", {"event_id": "123", "url": MATCH_URL})
            self.assertEqual(result.get("status"), "error")
            source.assert_not_called()
        with fixed_clock(), patch("cs2ml.map1_desk.fetch_event", return_value=wrong):
            result = self.desk.run("hltv_confirm", {
                "event_id": "123", "evidence_id": imported["evidence_id"], "reviewed": True})
        self.assertEqual(result.get("status"), "error")
        self.assertFalse(self.desk.store.contexts())
        self.assertEqual(set(self.desk.store.latest_hltv_evidence()), {"123"})

    def test_confirmation_persists_and_restart_is_paused(self):
        c = self.confirmed()
        self.assertEqual(c["confirmed_at"], T.isoformat())
        self.assertEqual(c["map_known_at"], T.isoformat())
        self.assertEqual(c["roster_known_at"], T.isoformat())
        self.desk.close()
        self.desk = PaperDesk(self.root)
        self.assertFalse(self.desk.running)
        self.assertEqual(self.desk.store.contexts()["123"]["context"], c)
        self.assertEqual(len(self.desk.store.hltv_identities()), 10)

    def test_unreviewed_match_cannot_confirm(self):
        imported = self.imported()
        with fixed_clock():
            self.desk.run("hltv_identities", identity_payload(imported["evidence_id"]))
        with fixed_clock(), patch("cs2ml.map1_desk.fetch_event", return_value=cs_event()):
            result = self.desk.run("hltv_confirm", {"event_id": "123", "evidence_id": imported["evidence_id"]})
        self.assertEqual(result["status"], "error")
        self.assertIn("match_review_required", result["reason"])
        self.assertFalse(self.desk.store.contexts())

    def test_manual_context_cannot_claim_hltv_verification(self):
        c = context(); c["hltv_match_id"] = "2398020"
        with fixed_clock(), patch("cs2ml.map1_desk.fetch_event") as network:
            result = self.desk.run("confirm", c)
        self.assertIn("manual_context_cannot_claim_hltv_verification", result["reason"])
        network.assert_not_called()

    def test_stale_evidence_blocks_before_market_capture(self):
        self.confirmed()
        at = T + pd.Timedelta(seconds=MAX_EVIDENCE_AGE + 1)
        with fixed_clock(at), patch("cs2ml.map1_desk.capture_once") as capture:
            result = self.desk.run("capture")
        capture.assert_not_called()
        self.assertEqual(result["ready"], 0)
        observed = self.desk.store.snapshots()[-1]
        self.assertEqual(observed["status"], "blocked")
        self.assertIn("hltv_evidence_stale_or_future", observed["reason"])
        self.assertNotIn("decision", observed)

    def test_new_source_failure_overrides_older_valid_evidence(self):
        self.confirmed()
        at = T + pd.Timedelta(minutes=5)
        self.imported(at, error=RuntimeError("HTTP 403"))
        with fixed_clock(at), patch("cs2ml.map1_desk.capture_once") as capture:
            result = self.desk.run("capture")
        capture.assert_not_called()
        self.assertEqual(result["ready"], 0)
        self.assertIn("hltv_source_unavailable", self.desk.store.snapshots()[-1]["reason"])

    def test_source_failure_cancels_existing_paper_pending_order_without_fill(self):
        c = self.confirmed()
        first = snapshot(); first["context"] = c
        with fixed_clock(), patch("cs2ml.map1_desk.capture_once", return_value=first):
            self.desk.run("capture")
            self.assertEqual(self.desk.report()["pending_count"], 1)
        at = T + pd.Timedelta(seconds=2)  # Past simulated order eligibility, before timeout.
        self.imported(at, error=RuntimeError("HTTP 403"))
        with fixed_clock(at), patch("cs2ml.map1_desk.capture_once") as capture:
            self.desk.run("capture")
            report = self.desk.report()
        capture.assert_not_called()
        self.assertEqual(report["pending_count"], 0)
        self.assertEqual(report["fill_count"], 0)
        self.assertEqual(report["cash"], 1000)
        self.assertFalse(report["open_positions"])
        self.assertEqual(report["ledger"][-1]["type"], "cancel")
        self.assertEqual(report["ledger"][-1]["reason"], "signal_no_longer_valid")

    def test_changed_map_or_roster_blocks_capture(self):
        self.confirmed()
        changed = parsed_match(); changed["map_name"] = "de_inferno"
        at = T + pd.Timedelta(minutes=5)
        self.imported(at, changed)
        with fixed_clock(at), patch("cs2ml.map1_desk.capture_once") as capture:
            result = self.desk.run("capture")
        capture.assert_not_called()
        self.assertEqual(result["ready"], 0)
        self.assertIn("changed_requires_new_review", self.desk.store.snapshots()[-1]["reason"])
        changed = parsed_match(); changed["teams"][0]["players"][0]["hltv_player_id"] = "99999"
        at += pd.Timedelta(minutes=5)
        self.imported(at, changed)
        with fixed_clock(at), patch("cs2ml.map1_desk.capture_once") as capture:
            self.desk.run("capture")
        capture.assert_not_called()
        self.assertIn("identities_unverified", self.desk.store.snapshots()[-1]["reason"])

    def test_identical_refresh_allows_capture_without_rewriting_locked_context(self):
        c = self.confirmed()
        first = snapshot(); first["context"] = c
        with fixed_clock(), patch("cs2ml.map1_desk.capture_once", return_value=first) as capture:
            self.assertEqual(self.desk.run("capture")["ready"], 1)
        capture.assert_called_once()
        at = T + pd.Timedelta(minutes=5)
        refreshed = self.imported(at)
        later = snapshot(at); later["context"] = c
        with fixed_clock(at), patch("cs2ml.map1_desk.capture_once", return_value=later):
            self.assertEqual(self.desk.run("capture")["ready"], 1)
        self.assertEqual(self.desk.store.contexts()["123"]["context"], c)
        self.assertNotEqual(refreshed["evidence_id"], c["hltv_evidence_id"])
        self.assertEqual(self.desk.store.snapshots()[-1]["hltv_source_evidence"]["evidence_id"], refreshed["evidence_id"])
        with fixed_clock(at), patch("cs2ml.map1_desk.fetch_event", return_value=cs_event()):
            result = self.desk.run("hltv_confirm", {"event_id": "123", "evidence_id": refreshed["evidence_id"], "reviewed": True})
        self.assertIn("context_locked", result["reason"])

    def test_capture_finishing_after_source_expiry_cannot_keep_buy_decision(self):
        c = self.confirmed()
        started = T + pd.Timedelta(seconds=MAX_EVIDENCE_AGE - 1)
        late = snapshot(T + pd.Timedelta(seconds=MAX_EVIDENCE_AGE + 1)); late["context"] = c
        late["decision"] = {"action": "buy", "reason": "fixture"}
        with fixed_clock(started), patch("cs2ml.map1_desk.capture_once", return_value=late):
            self.desk.run("capture")
        saved = self.desk.store.snapshots()[-1]
        self.assertEqual(saved["status"], "blocked")
        self.assertIn("hltv_evidence_stale_or_future", saved["reason"])
        self.assertNotIn("decision", saved)

    def test_refresh_only_confirmed_due_matches_and_stop_after_failure(self):
        self.imported()
        with fixed_clock(T + pd.Timedelta(minutes=6)), patch.object(self.desk, "import_hltv") as importer:
            self.desk.hltv.last_request = None
            self.assertEqual(self.desk.refresh_hltv_once()["refreshed"], 0)
        importer.assert_not_called()
        self.confirmed()
        at = T + pd.Timedelta(minutes=6)
        with fixed_clock(at), patch("cs2ml.map1_desk.fetch_event", return_value=cs_event()), \
             source_page(error=RuntimeError("HTTP 403")) as fetch:
            self.desk.hltv.last_request = None
            self.assertEqual(self.desk.refresh_hltv_once()["refreshed"], 1)
        fetch.assert_called_once()
        with fixed_clock(at + pd.Timedelta(minutes=6)), patch.object(self.desk, "import_hltv") as importer:
            self.desk.hltv.last_request = None
            self.assertEqual(self.desk.refresh_hltv_once()["refreshed"], 0)
        importer.assert_not_called()

    def test_refresh_is_one_source_request_per_cycle_even_if_multiple_are_due(self):
        c = self.confirmed()
        other = copy.deepcopy(c); other.update(event_id="456", hltv_match_id="2398021")
        self.desk.store.save_context(other, {}, T.isoformat())
        evidence = self.desk.store.hltv_evidence(c["hltv_evidence_id"])
        evidence["event_id"] = "456"; evidence.pop("evidence_id")
        evidence["parsed"]["match_id"] = "2398021"
        self.desk.store.add_hltv_evidence(evidence)
        with fixed_clock(T + pd.Timedelta(minutes=6)), patch.object(self.desk, "import_hltv", return_value={}) as importer:
            self.desk.hltv.last_request = None
            self.assertEqual(self.desk.refresh_hltv_once()["refreshed"], 1)
        importer.assert_called_once()


class HltvWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.desk = PaperDesk(Path(self.temp.name) / "map1")
        self.server = make_server(self.desk, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        with self.request("/api/state") as response:
            token = json.load(response)["csrf_token"]
        self.headers = {"Origin": self.base, "X-CSRF-Token": token, "Content-Type": "application/json"}

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.desk.close(); self.temp.cleanup()

    def request(self, path, body=None, headers=None):
        request = Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                          headers=headers or {})
        return urlopen(request, timeout=5)

    def action(self, path, payload):
        with self.request(path, payload, self.headers) as response:
            self.assertEqual(response.status, 202)
            self.assertTrue(json.load(response)["accepted"])
        return self.desk.future.result(timeout=5)

    def test_hltv_http_routes_review_and_raw_evidence_export(self):
        with fixed_clock(), patch("cs2ml.map1_desk.fetch_event", return_value=cs_event()), source_page() as source:
            imported = self.action("/api/hltv_import", {"event_id": "123", "url": MATCH_URL})
            self.assertFalse(self.desk.store.contexts())
            self.assertEqual(self.action("/api/hltv_identities", identity_payload(imported["evidence_id"]))[
                "identities_saved"], 10)
            confirmed = self.action("/api/hltv_confirm", {
                "event_id": "123", "evidence_id": imported["evidence_id"], "reviewed": True})
            self.assertEqual(confirmed["event_id"], "123")
            with self.request("/api/state") as response:
                state = json.load(response)
            self.assertTrue(state["events"][0]["confirmed"])
            self.assertTrue(state["events"][0]["hltv"]["confirmation_matches"])
            self.assertNotIn("html", state["events"][0]["hltv"])
            self.assertFalse(state["running"])
            self.assertFalse(state["live_trading_enabled"])
        source.assert_called_once()
        before = self.desk.store.hltv_evidence()
        with self.request("/api/export/hltv_evidence") as response:
            self.assertEqual(response.headers.get_content_type(), "application/x-ndjson")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            exported = [json.loads(line) for line in response.read().decode().splitlines()]
        self.assertEqual(exported, before)
        self.assertEqual(exported[0]["html"], "<html>offline fixture</html>")
        self.assertEqual(self.desk.store.hltv_evidence(), before)
        self.assertFalse(self.desk.store.snapshots())

    def test_hltv_mutation_routes_require_same_origin_token_and_json(self):
        with patch.object(self.desk, "submit") as submit:
            for path in ("/api/hltv_import", "/api/hltv_identities", "/api/hltv_confirm"):
                for override in ({"Origin": "https://evil.example"}, {"X-CSRF-Token": "wrong"},
                                 {"Content-Type": "text/plain"}):
                    with self.subTest(path=path, override=override):
                        with self.assertRaises(HTTPError) as error:
                            self.request(path, {}, {**self.headers, **override})
                        self.assertEqual(error.exception.code, 403)
                        error.exception.close()
            submit.assert_not_called()
        with self.assertRaises(HTTPError) as error:
            self.request("/api/export/hltv_evidence", headers={"Host": "evil.example"})
        self.assertEqual(error.exception.code, 403)
        error.exception.close()


if __name__ == "__main__":
    unittest.main()
