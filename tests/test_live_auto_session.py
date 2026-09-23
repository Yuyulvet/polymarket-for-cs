from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cs2ml import live_auto_discover as disc
from cs2ml.live_session_run import (
    build_session_spec, resolve_bout_map_names, session_slug, startable_pairs)
from cs2ml.live_session_audit import audit_session
from cs2ml.trend_protocol import validate_session_spec


def fivee_candidate(match_id="csgo_mc_1", plan_ts=1_800_000_000.0,
                    team1="G2 Esports", team2="MOUZ", status="0"):
    k1, k2 = disc.normalize_team(team1), disc.normalize_team(team2)
    return disc.FiveEMatchCandidate(match_id=match_id, plan_ts=plan_ts,
                                    team1=team1, team2=team2, team_key1=k1,
                                    team_key2=k2, format="3", stage="小组赛",
                                    status=status, raw={})


def poly_candidate(event_id="7001", start_ts=1_800_000_300.0,
                   team1="G2", team2="MOUZ", bouts=(1, 2)):
    k1, k2 = disc.normalize_team(team1), disc.normalize_team(team2)
    markets = [disc.PolyMapMarket(bout_num=b, market_name=f"Map {b} Winner",
                                  market_id=f"100{b}",
                                  outcomes=["Yes", "No"],
                                  tokens={f"t{b}a": f"tok{b}a", f"t{b}b": f"tok{b}b"})
               for b in bouts]
    return disc.PolyEventCandidate(event_id=event_id, start_ts=start_ts,
                                   title=f"Counter-Strike: {team1} vs {team2} (BO3) - Test League",
                                   team1=team1, team2=team2, team_key1=k1,
                                   team_key2=k2, bo="BO3", league="Test League",
                                   map_markets=markets, live=False, raw={})


def fivee_list_payload(rows):
    return {"success": True, "data": {"matches": rows, "live_matches": [],
                                      "state_ver": ""}}


def fivee_row(match_id, plan_ts, team1, team2, status="0"):
    return {"mc_info": {"id": match_id, "plan_ts": str(int(plan_ts)),
                        "format": "3", "tt_stage": "小组赛",
                        "t1_info": {"disp_name": team1},
                        "t2_info": {"disp_name": team2}},
            "state": {"status": status}}


def gamma_event_payload(event_id, title, start_iso, bouts=(1, 2)):
    markets = []
    for b in bouts:
        markets.append({
            "id": f"100{b}", "groupItemTitle": f"Map {b} Winner",
            "question": f"Will the team win map {b}?",
            "clobTokenIds": json.dumps([f"tok{b}a", f"tok{b}b"]),
            "outcomes": json.dumps(["Yes", "No"]),
        })
    return [{"id": event_id, "title": title, "startDate": start_iso,
             "closed": False, "liveAt": None, "markets": markets}]


class TeamNormalizationTests(unittest.TestCase):
    def test_case_punctuation_accents(self):
        self.assertEqual(disc.normalize_team("QUINTESSÊNCIA"), "quintessencia")
        self.assertEqual(disc.normalize_team("paiN Gaming"), "pain")
        self.assertEqual(disc.normalize_team("ex-RUSTEC"), "ex rustec")
        self.assertEqual(disc.normalize_team("G2 Esports"), "g2")
        self.assertEqual(disc.normalize_team("Natus Vincere"), "navi")

    def test_academy_and_junior_are_distinct_teams(self):
        self.assertNotEqual(disc.normalize_team("NAVI Junior"),
                            disc.normalize_team("NAVI"))
        self.assertNotEqual(disc.normalize_team("Spirit Academy"),
                            disc.normalize_team("Spirit"))

    def test_alias_file_overrides(self):
        aliases = {"MOUZ": "mousesports"}
        self.assertEqual(disc.normalize_team("MOUZ", aliases), "mousesports")

    def test_parse_poly_title(self):
        parsed = disc.parse_poly_title(
            "Counter-Strike: Black Phoenix vs Lavked (BO3) - CCT Europe Series #9")
        self.assertEqual(parsed["team1"], "Black Phoenix")
        self.assertEqual(parsed["team2"], "Lavked")
        self.assertEqual(parsed["bo"], "BO3")
        self.assertIn("CCT", parsed["league"])

    def test_parse_poly_title_rejects_non_match(self):
        self.assertIsNone(disc.parse_poly_title("Will FaZe win a Tier 1 event in 2026?"))


class FiveeListParsingTests(unittest.TestCase):
    def test_tbd_and_unknown_teams_are_dropped(self):
        payload = fivee_list_payload([
            fivee_row("csgo_mc_1", 1_800_000_000, "TBD", "MOUZ"),
            fivee_row("csgo_mc_2", 1_800_000_000, "G2", "待定"),
            fivee_row("csgo_mc_3", 1_800_000_000, "G2", "MOUZ"),
        ])
        candidates = disc.parse_fivee_list(payload)
        self.assertEqual([c.match_id for c in candidates], ["csgo_mc_3"])

    def test_missing_plan_ts_kept_but_unpairable(self):
        row = fivee_row("csgo_mc_9", 1_800_000_000, "G2", "MOUZ")
        del row["mc_info"]["plan_ts"]
        candidates = disc.parse_fivee_list(fivee_list_payload([row]))
        self.assertEqual(len(candidates), 1)
        self.assertIsNone(candidates[0].plan_ts)

    def test_non_success_payload_raises(self):
        with self.assertRaisesRegex(ValueError, "fivee_list_not_success"):
            disc.parse_fivee_list({"success": False, "message": "x"})


class FiveePaginationTests(unittest.TestCase):
    def test_pages_are_merged_and_deduped(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload
            def raise_for_status(self):
                pass
            def json(self):
                return self._payload

        def fake_get(url, params=None, **kwargs):
            page = (params or {}).get("page", 1)
            if page == 1:
                return FakeResponse(fivee_list_payload([
                    fivee_row("csgo_mc_1", 1_800_000_000, "G2", "MOUZ")]))
            if page == 2:
                # duplicate of page 1 plus a new match
                return FakeResponse(fivee_list_payload([
                    fivee_row("csgo_mc_1", 1_800_000_000, "G2", "MOUZ"),
                    fivee_row("csgo_mc_2", 1_800_000_100, "FaZe", "Nemiga")]))
            return FakeResponse(fivee_list_payload([]))

        candidates = disc.fetch_fivee_matches(fake_get, pages=3, limit=50)
        self.assertEqual([c.match_id for c in candidates],
                         ["csgo_mc_1", "csgo_mc_2"])


class PolyEventParsingTests(unittest.TestCase):
    def test_match_events_kept_and_map_markets_parsed(self):
        payload = gamma_event_payload(
            7001, "Counter-Strike: G2 vs MOUZ (BO3) - Test", "2026-09-19T10:00:00Z")
        events = disc.parse_poly_events(payload)
        self.assertEqual(len(events), 1)
        self.assertEqual([m.bout_num for m in events[0].map_markets], [1, 2])
        self.assertEqual(events[0].start_ts, 1_789_812_000.0)

    def test_longterm_markets_and_closed_events_rejected(self):
        payload = gamma_event_payload(
            7002, "Counter-Strike: G2 vs MOUZ (BO3) - Test", "2026-09-19T10:00:00Z")
        payload[0]["closed"] = True
        payload.append({"id": 7003, "title": "Will FaZe win a Tier 1 event in 2026?",
                        "startDate": "2026-09-19T10:00:00Z", "closed": False,
                        "markets": []})
        self.assertEqual(disc.parse_poly_events(payload), [])

    def test_non_map_markets_do_not_create_bouts(self):
        payload = gamma_event_payload(
            7004, "Counter-Strike: G2 vs MOUZ (BO3) - Test", "2026-09-19T10:00:00Z")
        payload[0]["markets"].append({
            "id": "9999", "groupItemTitle": "O/U 2.5 Games",
            "clobTokenIds": json.dumps(["x", "y"]),
            "outcomes": json.dumps(["Over", "Under"])})
        events = disc.parse_poly_events(payload)
        self.assertEqual(len(events[0].map_markets), 2)


class PairingGateTests(unittest.TestCase):
    def test_exact_pair_approved(self):
        report = disc.pair_candidates([fivee_candidate()], [poly_candidate()],
                                      tolerance_seconds=1800)
        self.assertEqual(len(report["paired"]), 1)
        self.assertEqual(report["ambiguous"], [])

    def test_time_outside_tolerance_not_paired(self):
        pm = poly_candidate(start_ts=1_800_090_000.0)  # +25 min
        report = disc.pair_candidates([fivee_candidate()], [pm],
                                      tolerance_seconds=1800)
        self.assertEqual(report["paired"], [])
        self.assertEqual(len(report["unmatched_fivee"]), 1)
        self.assertEqual(len(report["unmatched_poly"]), 1)

    def test_team_mismatch_not_paired(self):
        pm = poly_candidate(team1="G2", team2="FaZe")
        report = disc.pair_candidates([fivee_candidate()], [pm],
                                      tolerance_seconds=1800)
        self.assertEqual(report["paired"], [])

    def test_ambiguous_when_two_poly_events_match_one_fivee(self):
        pm1 = poly_candidate(event_id="7001")
        pm2 = poly_candidate(event_id="7002", start_ts=1_800_000_600.0)
        report = disc.pair_candidates([fivee_candidate()], [pm1, pm2],
                                      tolerance_seconds=1800)
        self.assertEqual(report["paired"], [])
        self.assertEqual(len(report["ambiguous"]), 2)
        self.assertEqual(len(report["unmatched_fivee"]), 0)

    def test_ambiguous_when_one_poly_event_matches_two_fivee_matches(self):
        fm1 = fivee_candidate(match_id="csgo_mc_1", plan_ts=1_800_000_000.0)
        fm2 = fivee_candidate(match_id="csgo_mc_2", plan_ts=1_800_000_500.0)
        report = disc.pair_candidates([fm1, fm2], [poly_candidate()],
                                      tolerance_seconds=1800)
        self.assertEqual(report["paired"], [])
        self.assertEqual(len(report["ambiguous"]), 2)

    def test_alias_bridges_platform_name_difference(self):
        fm = fivee_candidate(team1="MOUZ", team2="G2 Esports")
        pm = poly_candidate(team1="mousesports", team2="g2")
        aliases = {"MOUZ": "mousesports"}
        k1, k2 = disc.normalize_team("MOUZ", aliases), disc.normalize_team("G2 Esports", aliases)
        fm_alias = disc.FiveEMatchCandidate(match_id=fm.match_id, plan_ts=fm.plan_ts,
                                            team1=fm.team1, team2=fm.team2,
                                            team_key1=k1, team_key2=k2,
                                            format=fm.format, stage=fm.stage,
                                            status=fm.status, raw={})
        report = disc.pair_candidates([fm_alias], [pm], tolerance_seconds=1800)
        self.assertEqual(len(report["paired"]), 1)

    def test_poly_event_without_map_markets_never_pairs(self):
        pm = poly_candidate(bouts=())
        report = disc.pair_candidates([fivee_candidate()], [pm],
                                      tolerance_seconds=1800)
        self.assertEqual(report["paired"], [])


class StartableWindowTests(unittest.TestCase):
    def test_lead_and_lateness_window(self):
        fm = fivee_candidate(plan_ts=1_000_000.0)
        pm = poly_candidate(start_ts=1_000_000.0)
        pair = disc.pair_candidates([fm], [pm], tolerance_seconds=1800)["paired"][0]
        ser = {"paired": [disc.serialize_pairing({"paired": [pair], "ambiguous": [],
                                                 "unmatched_fivee": [],
                                                 "unmatched_poly": []})["paired"][0]
                          ], "ambiguous": []}
        # within lead window
        due = startable_pairs(ser, now=1_000_000.0 - 600, lead_seconds=1200,
                              max_lateness_seconds=7200)
        self.assertEqual(len(due), 1)
        # too early
        due = startable_pairs(ser, now=1_000_000.0 - 3600, lead_seconds=1200,
                              max_lateness_seconds=7200)
        self.assertEqual(due, [])
        # too late
        due = startable_pairs(ser, now=1_000_000.0 + 7201, lead_seconds=1200,
                              max_lateness_seconds=7200)
        self.assertEqual(due, [])

    def test_session_slug_is_safe(self):
        slug = session_slug("ex rustec", "honved", 1_800_000_000.0)
        self.assertRegex(slug, r"^[a-z0-9][a-z0-9-]*$")


class SessionSpecAndAuditTests(unittest.TestCase):
    def _write_session_files(self, root: Path) -> Path:
        session = root / "data" / "live_sessions" / "g2-vs-mouz-20260919"
        (session / "market").mkdir(parents=True)
        events_path = session / "fivee_events.jsonl"
        with events_path.open("w", encoding="utf-8") as fh:
            for bout, map_name in ((1, "de_ancient"), (1, "de_ancient"), (2, "de_mirage")):
                fh.write(json.dumps({"record_type": "event", "entry": {
                    "bout_num": bout, "map_name": map_name,
                    "update_version": 1, "log_info": "{}"}}) + "\n")
        (session / "fivee_states.jsonl").write_text(
            json.dumps({"record_type": "session_start", "match_id": "csgo_mc_1"})
            + "\n", encoding="utf-8")
        (session / "market" / "events.jsonl").write_text("", encoding="utf-8")
        metadata = {"binding": {"event_id": "7001", "market_id": "1001",
                                "outcome_tokens": {"Yes": "tok1a", "No": "tok1b"}},
                    "markets": [{"market_id": "1001", "tokens": {"Yes": "tok1a", "No": "tok1b"}},
                                {"market_id": "1002", "tokens": {"Yes": "tok2a", "No": "tok2b"}}]}
        (session / "market" / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8")
        (session / "market" / "summary.json").write_text(
            json.dumps({"exit_reason": "deadline", "messages": 0, "bytes": 0}),
            encoding="utf-8")
        return session

    def _pair(self):
        fm = fivee_candidate(plan_ts=1_800_000_000.0)
        pm = poly_candidate(start_ts=1_800_000_000.0)
        pair = disc.pair_candidates([fm], [pm], tolerance_seconds=1800)["paired"][0]
        return disc.serialize_pairing({"paired": [pair], "ambiguous": [],
                                       "unmatched_fivee": [],
                                       "unmatched_poly": []})["paired"][0]

    def test_spec_validates_and_bouts_get_map_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = self._write_session_files(root)
            event = gamma_event_payload(
                7001, "Counter-Strike: G2 vs MOUZ (BO3) - Test",
                "2026-09-19T10:00:00Z")[0]
            spec = build_session_spec(session.name, self._pair(), event, session, root)
            validate_session_spec(spec)
            names = {m["bout_num"]: m["map_name"] for m in spec["maps"]}
            self.assertEqual(names, {1: "de_ancient", 2: "de_mirage"})
            self.assertEqual(spec["match_id"], "csgo_mc_1")
            self.assertEqual(spec["event_id"], "7001")

    def test_bout_map_name_resolution_ignores_unknown_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for bout, map_name in ((1, "Ancient"), (1, ""), (1, "de_ancient")):
                    fh.write(json.dumps({"record_type": "event",
                                         "entry": {"bout_num": bout,
                                                   "map_name": map_name}}) + "\n")
            self.assertEqual(resolve_bout_map_names(path, [1]), {1: "de_ancient"})
            self.assertEqual(resolve_bout_map_names(path, [1, 2]), {1: "de_ancient"})

    def test_audit_blocks_incomplete_session_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = self._write_session_files(root)
            event = gamma_event_payload(
                7001, "Counter-Strike: G2 vs MOUZ (BO3) - Test",
                "2026-09-19T10:00:00Z")[0]
            spec = build_session_spec(session.name, self._pair(), event, session, root)
            report = audit_session(spec, root)
            self.assertFalse(report["eligible_for_trend_dataset"])
            self.assertIn("fivee_event_segment_missing_session_end",
                          report["blocking_reasons"])
            self.assertIn("no_strict_eligible_fivee_events",
                          report["blocking_reasons"])


class ManualModeTests(unittest.TestCase):
    def test_manual_dry_run_serializes_plain_dicts(self):
        from cs2ml.live_session_run import main
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--mode", "dry-run",
                         "--match-id", "csgo_mc_999", "--event-id", "7001",
                         "--out-root", str(Path(tmp) / "ls")])
            self.assertEqual(code, 0)
            snapshot = json.loads((Path(tmp) / "ls" / "discovery_latest.json")
                                  .read_text(encoding="utf-8"))
            self.assertEqual(snapshot["paired"][0]["fivee"]["match_id"],
                             "csgo_mc_999")

    def test_select_market_bindings_carries_event_id(self):
        from cs2ml.market_raw_capture import select_market_bindings
        event = gamma_event_payload(
            7001, "Counter-Strike: G2 vs MOUZ (BO3) - Test",
            "2026-09-19T10:00:00Z")[0]
        bindings = select_market_bindings(event, 7001, ["1001", "1002"])
        self.assertEqual([b["event_id"] for b in bindings], ["7001", "7001"])
        self.assertEqual(bindings[0]["tokens"],
                         {"Yes": "tok1a", "No": "tok1b"})


if __name__ == "__main__":
    unittest.main()
