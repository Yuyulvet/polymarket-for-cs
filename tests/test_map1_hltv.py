"""Synthetic match-only fixtures based on the public HLTV match page structure."""
from __future__ import annotations

import hashlib
import unittest
from unittest.mock import MagicMock, patch

import requests

from cs2ml.map1_hltv import (
    HLTVSourceError, MAX_HTML_BYTES, fetch_match_page, normalize_match_url,
    parse_match_page,
)


URL = "https://www.hltv.org/matches/2398020/drama-vs-mellren"


def lineup(team: int, ids: list[int] | None = None, *, legacy: bool = False,
           coach: bool = False) -> str:
    ids = ids if ids is not None else list(range(team * 10, team * 10 + 5))
    photos = "".join(f'<td class="player player-image"><div class="player-compare" '
                     f'data-player-id="{pid}"><img alt="Full name"></div></td>' for pid in ids)
    if legacy:
        names = "".join(f'<td class="player"><a href="/player/{pid}/p{pid}">'
                        f'<div class="text-ellipsis">p{pid}</div></a></td>' for pid in ids)
    else:
        names = "".join(f'<td class="player"><div class="player-compare flagAlign" '
                        f'data-player-id="{pid}" data-team-ordinal="{team}">'
                        f'<img class="flag" alt="Other"><div class="text-ellipsis">p{pid}</div>'
                        f'</div></td>' for pid in ids)
    coach_html = '<div class="coach"><a href="/player/999/coach">Coach</a></div>' if coach else ""
    return (f'<div class="lineup standard-box"><div class="box-headline flex-align-center">'
            f'<a href="/team/{team}/team{team}">Team {team}</a></div>'
            f'<div class="players"><table><tr>{photos}</tr><tr>{names}</tr></table></div>'
            f'{coach_html}</div>')


def fixture(*, map_names: tuple[str, ...] = ("Ancient", "Nuke", "Mirage"),
            countdown: str = "53m : 26s", time_ms: str = "1789554600000",
            lineups: str | None = None) -> str:
    holders = "".join(f'<div class="mapholder"><div class="played"><div class="mapname">'
                      f'{name}</div></div></div>' for name in map_names)
    if lineups is None:
        lineups = lineup(1, coach=True) + lineup(2)
    return f'''<!doctype html><html><head><link rel="canonical" href="{URL}"></head><body>
      <div class="standard-box teamsBox">
        <div class="team1-gradient"><a href="/team/1/team1"><div class="teamName">Team 1</div></a></div>
        <div class="timeAndEvent"><div class="time" data-unix="{time_ms}">00:30</div>
          <div class="date" data-unix="{time_ms}">localized date</div>
          <div class="event"><a href="/events/9000/test">Example Cup &amp; League</a></div>
          <div class="countdown" data-time-countdown="LIVE">{countdown}</div></div>
        <div class="team2-gradient"><a href="/team/2/team2"><div class="teamName">Team 2</div></a></div>
      </div><div class="maps">{holders}</div>
      <div class="sidebar"><a href="/player/888/outsider">Outsider</a>
        <div class="mapname">Dust2</div><div data-unix="123">Sidebar time</div></div>
      <div class="lineups" id="lineups">{lineups}</div>
    </body></html>'''


class MatchUrlTests(unittest.TestCase):
    def test_valid_urls_and_trailing_slash(self):
        self.assertEqual(normalize_match_url(URL + "/"), URL)
        self.assertEqual(normalize_match_url("https://www.hltv.org/matches/2398020"),
                         "https://www.hltv.org/matches/2398020")

    def test_ssrf_and_url_confusion_rejected(self):
        invalid = [
            "http://www.hltv.org/matches/1/x", "https://hltv.org/matches/1/x",
            "https://www.hltv.org.evil.test/matches/1/x", "https://127.0.0.1/matches/1/x",
            "https://www.hltv.org@evil.test/matches/1/x", "https://user@www.hltv.org/matches/1/x",
            "https://www.hltv.org:443/matches/1/x", "https://www.hltv.org/matches/1/x?foo=1",
            "https://www.hltv.org/matches/1/x?", "https://www.hltv.org/matches/1/x#",
            "https://www.hltv.org/matches/1/../2", "https://www.hltv.org/matches/1/%2e%2e",
            "https://www.hltv.org/matches/1/x%2Fy", "https://www.hltv.org\\@evil.test/matches/1/x",
            "https://www.hltv.org/matches/0/x", "https://www.hltv.org/matches/1/x/y",
            "https://www.hltv.org/matches/１/x", "https://www.hltv.org/\nmatches/1/x",
            URL + " ", " " + URL, "https://www.hltv.org/matches/1/\x00", None,
        ]
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(HLTVSourceError):
                normalize_match_url(url)


class MatchParserTests(unittest.TestCase):
    def test_explicit_map_order_and_exact_match_lineups(self):
        parsed = parse_match_page(fixture(), URL)
        self.assertEqual(parsed["issues"], [])
        self.assertEqual(parsed["match_id"], "2398020")
        self.assertEqual(parsed["map_name"], "de_ancient")
        self.assertEqual(parsed["map_names"], ["de_ancient", "de_nuke", "de_mirage"])
        self.assertEqual(parsed["status"], "prematch")
        self.assertEqual(parsed["event_name"], "Example Cup & League")
        self.assertEqual([t["hltv_team_id"] for t in parsed["teams"]], ["1", "2"])
        self.assertEqual(parsed["teams"][0]["players"],
                         [{"hltv_player_id": str(i), "name": f"p{i}"} for i in range(10, 15)])
        self.assertEqual(len(parsed["teams"][1]["players"]), 5)

    def test_timestamp_is_epoch_utc_not_displayed_time(self):
        parsed = parse_match_page(fixture(), URL)
        self.assertEqual(parsed["scheduled_start_at"], "2026-09-16T10:30:00Z")

    def test_first_map_tba_does_not_shift_known_second_map(self):
        parsed = parse_match_page(fixture(map_names=("TBA", "Nuke", "Mirage")), URL)
        self.assertIsNone(parsed["map_name"])
        self.assertEqual(parsed["map_names"], [None, "de_nuke", "de_mirage"])
        self.assertEqual(parsed["issues"], ["map1_not_announced"])

    def test_all_maps_tba(self):
        parsed = parse_match_page(fixture(map_names=("TBA", "TBA", "TBA")), URL)
        self.assertIsNone(parsed["map_name"])
        self.assertEqual(parsed["map_names"], [None, None, None])

    def test_unknown_map_fails_closed(self):
        parsed = parse_match_page(fixture(map_names=("New Unknown Map",)), URL)
        self.assertIsNone(parsed["map_name"])
        self.assertIn("map_name_unknown:1", parsed["issues"])

    def test_finished_and_live_are_not_prematch(self):
        for text, expected in (("LIVE", "live"), ("Match over", "finished"),
                               ("FINISHED", "finished"), ("Postponed", "unknown"),
                               ("", "unknown")):
            with self.subTest(text=text):
                parsed = parse_match_page(fixture(countdown=text), URL)
                self.assertEqual(parsed["status"], expected)
                if expected == "unknown":
                    self.assertIn("match_status_unknown", parsed["issues"])

    def test_invalid_or_missing_epoch_does_not_parse_localized_date(self):
        for stamp in ("", "1789554600", "September 16", "-1789554600000"):
            parsed = parse_match_page(fixture(time_ms=stamp), URL)
            self.assertIsNone(parsed["scheduled_start_at"])
            self.assertIn("scheduled_start_invalid", parsed["issues"])

    def test_five_duplicate_player_cells_are_not_deduplicated_as_valid(self):
        parsed = parse_match_page(fixture(lineups=lineup(1, [10, 10, 12, 13, 14]) + lineup(2)), URL)
        self.assertIn("duplicate_lineup_player:1", parsed["issues"])

    def test_missing_player_does_not_use_generic_profile_link(self):
        parsed = parse_match_page(fixture(lineups=lineup(1, [10, 11, 12, 13]) + lineup(2)), URL)
        self.assertIn("lineup_requires_five_players:1", parsed["issues"])
        self.assertEqual(len(parsed["teams"][0]["players"]), 4)

    def test_sixth_lineup_member_is_ambiguous(self):
        parsed = parse_match_page(fixture(lineups=lineup(1, list(range(10, 16))) + lineup(2)), URL)
        self.assertIn("lineup_requires_five_players:1", parsed["issues"])

    def test_coach_row_inside_players_table_does_not_count_as_a_starter(self):
        team = lineup(1).replace('</table>', '<tr class="coach"><td class="player">'
                                '<a href="/player/999/coach">Coach</a></td></tr></table>')
        parsed = parse_match_page(fixture(lineups=team + lineup(2)), URL)
        self.assertEqual(parsed["issues"], [])
        self.assertEqual(len(parsed["teams"][0]["players"]), 5)

    def test_player_cannot_appear_for_both_teams(self):
        parsed = parse_match_page(fixture(lineups=lineup(1) + lineup(2, [10, 21, 22, 23, 24])), URL)
        self.assertIn("players_not_distinct_across_teams", parsed["issues"])

    def test_lineup_team_ids_determine_association_not_dom_order(self):
        parsed = parse_match_page(fixture(lineups=lineup(2) + lineup(1)), URL)
        self.assertEqual(parsed["issues"], [])
        self.assertEqual(parsed["teams"][0]["players"][0]["hltv_player_id"], "10")

    def test_extra_or_duplicate_team_lineup_is_rejected(self):
        parsed = parse_match_page(fixture(lineups=lineup(1) + lineup(1) + lineup(2)), URL)
        self.assertIn("lineup_missing_or_ambiguous:1", parsed["issues"])
        parsed = parse_match_page(fixture(lineups=lineup(1) + lineup(3)), URL)
        self.assertIn("lineup_header_team_mismatch", parsed["issues"])

    def test_legacy_player_links_are_supported_only_in_match_lineup_cells(self):
        parsed = parse_match_page(fixture(lineups=lineup(1, legacy=True) + lineup(2, legacy=True)), URL)
        self.assertEqual(parsed["issues"], [])

    def test_changed_layout_and_challenge_are_ineligible(self):
        for html, issue in (("<html><h1>New layout</h1></html>", "match_header_missing_or_ambiguous"),
                            ("<html><title>Just a moment...</title></html>", "access_challenge")):
            parsed = parse_match_page(html, URL)
            self.assertEqual(parsed["status"], "unknown")
            self.assertIn(issue, parsed["issues"])

    def test_normal_site_security_script_is_not_itself_a_challenge_page(self):
        html = fixture() + "<script>var s='/cdn-cgi/challenge-platform/scripts/precursor/main.js';</script>"
        self.assertEqual(parse_match_page(html, URL)["issues"], [])

    def test_mismatched_canonical_match_id_blocks_page(self):
        html = fixture().replace(f'href="{URL}"', 'href="https://www.hltv.org/matches/999/other"')
        self.assertIn("canonical_match_id_mismatch", parse_match_page(html, URL)["issues"])

    def test_oversized_and_deep_html_are_rejected(self):
        self.assertEqual(parse_match_page("x" * (MAX_HTML_BYTES + 1), URL)["issues"],
                         ["html_missing_or_oversized"])
        self.assertEqual(parse_match_page("<div>" * 150, URL)["issues"], ["html_structure_invalid"])


class MatchFetchTests(unittest.TestCase):
    def fake_session(self, *, status=200, content_type="text/html; charset=utf-8",
                     body: bytes | None = None, content_length: str | None = None):
        session = MagicMock()
        session.__enter__.return_value = session
        response = session.get.return_value.__enter__.return_value
        response.status_code = status
        response.headers = {"Content-Type": content_type}
        if content_length is not None:
            response.headers["Content-Length"] = content_length
        response.iter_content.return_value = [fixture().encode() if body is None else body]
        return session

    def test_bounded_plain_get_retains_hash_and_observation_time(self):
        session = self.fake_session()
        with patch("cs2ml.map1_hltv.requests.Session", return_value=session):
            captured = fetch_match_page(URL)
        self.assertEqual(captured["html"], fixture())
        self.assertEqual(captured["sha256"], hashlib.sha256(fixture().encode()).hexdigest())
        self.assertTrue(captured["received_at"].endswith("Z"))
        self.assertFalse(session.trust_env)
        kwargs = session.get.call_args.kwargs
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (5, 8))
        session.get.assert_called_once()

    def test_redirect_forbidden_and_rate_limit_fail_without_retry(self):
        for status in (301, 302, 403, 404, 429, 500):
            session = self.fake_session(status=status)
            with patch("cs2ml.map1_hltv.requests.Session", return_value=session):
                with self.subTest(status=status), self.assertRaises(HLTVSourceError):
                    fetch_match_page(URL)
            session.get.assert_called_once()

    def test_size_limits_and_content_type_and_utf8(self):
        cases = [dict(content_length=str(MAX_HTML_BYTES + 1)),
                 dict(content_length="not a number"), dict(body=b"x" * (MAX_HTML_BYTES + 1)),
                 dict(content_type="application/json"), dict(body=b"\xff"), dict(body=b"")]
        for case in cases:
            with self.subTest(case=list(case)):
                session = self.fake_session(**case)
                with patch("cs2ml.map1_hltv.requests.Session", return_value=session), self.assertRaises(HLTVSourceError):
                    fetch_match_page(URL)

    def test_invalid_url_never_makes_request(self):
        with patch("cs2ml.map1_hltv.requests.Session") as factory, self.assertRaises(HLTVSourceError):
            fetch_match_page("https://localhost/")
        factory.assert_not_called()

    def test_timeout_is_reported_without_retry(self):
        session = self.fake_session()
        session.get.side_effect = requests.Timeout("private response details")
        with patch("cs2ml.map1_hltv.requests.Session", return_value=session):
            with self.assertRaisesRegex(HLTVSourceError, "Timeout"):
                fetch_match_page(URL)
        session.get.assert_called_once()

    def test_successful_challenge_is_preserved_but_parser_blocks_it(self):
        session = self.fake_session(body=b"<html><title>Just a moment...</title></html>")
        with patch("cs2ml.map1_hltv.requests.Session", return_value=session):
            captured = fetch_match_page(URL)
        self.assertIn("access_challenge", parse_match_page(captured["html"], URL)["issues"])


if __name__ == "__main__":
    unittest.main()
