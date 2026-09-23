"""Bounded public HLTV match reads and conservative, match-specific parsing.

This module makes no authenticated requests, follows no redirects, and does not
solve or work around access challenges. A successful HTTP read is evidence of
what was received, not a claim that the lineup or Map 1 has been announced.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Iterator
from urllib.parse import urlsplit

import requests

MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_READ_SECONDS = 25.0
_MATCH_PATH = re.compile(r"/matches/([1-9][0-9]{0,11})(?:/([A-Za-z0-9-]+))?/?\Z")
_TEAM_PATH = re.compile(r"/team/([1-9][0-9]*)/[^/?#]+\Z")
_PLAYER_PATH = re.compile(r"/player/([1-9][0-9]*)/[^/?#]+\Z")
_MAPS = {name: "de_" + name for name in (
    "ancient", "anubis", "cache", "cobblestone", "dust2", "inferno", "mirage",
    "nuke", "overpass", "train", "vertigo",
)}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
         "meta", "param", "source", "track", "wbr"}


class HLTVSourceError(ValueError):
    """Unavailable/untrusted source data; callers must not trade on it."""


def normalize_match_url(url: str) -> str:
    """Accept only a plain HTTPS HLTV match URL, not an arbitrary fetch target."""
    if not isinstance(url, str) or not url or url != url.strip():
        raise HLTVSourceError("HLTV match URL is required without surrounding whitespace")
    if any(ord(char) < 33 or ord(char) > 126 for char in url):
        raise HLTVSourceError("HLTV match URL must contain plain ASCII without whitespace")
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.netloc == "www.hltv.org"
                 and not parsed.query and not parsed.fragment
                 and "?" not in url and "#" not in url)
        match = _MATCH_PATH.fullmatch(parsed.path)
    except ValueError as exc:
        raise HLTVSourceError("Invalid HLTV match URL") from exc
    if not valid or not match:
        raise HLTVSourceError("Use https://www.hltv.org/matches/<match-id>/<slug>")
    # Retain the supplied slug: a canonical /x URL could require a redirect.
    return "https://www.hltv.org" + parsed.path.rstrip("/")


def fetch_match_page(url: str) -> dict[str, Any]:
    """Perform one bounded GET; return unchanged HTML for local evidence storage.

    SHA-256 covers the returned HTML encoded as UTF-8, so callers can verify the
    stored text without HTTP transfer encodings. Parsing is deliberately separate.
    """
    url = normalize_match_url(url)
    started = time.monotonic()
    try:
        with requests.Session() as session:
            # No ambient .netrc credentials, proxy credentials, or cookie jar.
            session.trust_env = False
            with session.get(
                url, headers={"User-Agent": "CS2Map1Research/1.0 (public match verification)",
                              "Accept": "text/html", "Accept-Encoding": "identity"},
                timeout=(5, 8), allow_redirects=False, stream=True,
            ) as response:
                if response.status_code != 200:
                    raise HLTVSourceError(f"HLTV HTTP {response.status_code}; no retry or bypass")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower().strip()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise HLTVSourceError("HLTV response is not HTML")
                size_header = response.headers.get("Content-Length")
                if size_header:
                    try:
                        if int(size_header) < 0 or int(size_header) > MAX_HTML_BYTES:
                            raise HLTVSourceError("HLTV page exceeds evidence size limit")
                    except ValueError as exc:
                        raise HLTVSourceError("HLTV invalid or oversized Content-Length") from exc
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(chunk_size=16 * 1024):
                    if time.monotonic() - started > MAX_READ_SECONDS:
                        raise HLTVSourceError("HLTV page exceeded bounded read deadline")
                    total += len(chunk)
                    if total > MAX_HTML_BYTES:
                        raise HLTVSourceError("HLTV page exceeds evidence size limit")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                try:
                    html = raw.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise HLTVSourceError("HLTV page is not valid UTF-8") from exc
    except requests.RequestException as exc:
        # Avoid echoing potential third-party response bodies into the UI.
        raise HLTVSourceError(f"HLTV read unavailable: {type(exc).__name__}") from exc
    received_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not html.strip():
        raise HLTVSourceError("HLTV returned an empty page")
    # Successful challenge pages are also returned as evidence; parse marks them
    # ineligible. HTTP 403/429 and redirects never return usable observations.
    return {"url": url, "received_at": received_at, "html": html,
            "sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()}


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str]
    children: list[Any] = field(default_factory=list)

    def has_class(self, name: str) -> bool:
        return name in self.attrs.get("class", "").split()

    def walk(self) -> Iterator[_Node]:
        pending = [self]
        while pending:
            node = pending.pop()
            yield node
            pending.extend(child for child in reversed(node.children) if isinstance(child, _Node))

    def find(self, class_name: str) -> list[_Node]:
        return [node for node in self.walk() if node.has_class(class_name)]

    def text(self) -> str:
        pending: list[Any] = [self]
        parts = []
        while pending:
            node = pending.pop()
            if isinstance(node, str):
                parts.append(node)
            elif node.tag not in {"script", "style"}:
                pending.extend(reversed(node.children))
        return " ".join(" ".join(parts).split())


class _Tree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", {})
        self.stack = [self.root]
        self.node_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.node_count += 1
        if self.node_count > 100_000 or len(self.stack) > 128:
            raise HLTVSourceError("HLTV HTML structure exceeds parser limits")
        node = _Node(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def _link_id(node: _Node, pattern: re.Pattern[str]) -> str | None:
    match = pattern.fullmatch(node.attrs.get("href", ""))
    return match.group(1) if match and node.tag == "a" else None


def _lineup_players(lineup: _Node, issues: list[str], team_id: str) -> list[dict[str, str]]:
    scopes = lineup.find("players")
    if len(scopes) != 1:
        issues.append(f"lineup_players_section_missing_or_ambiguous:{team_id}")
        return []
    players: list[dict[str, str]] = []
    # HLTV has one photo row and one nickname row. Only nickname cells count;
    # the photo row repeats every player ID and must not become a second roster.
    cells = []
    pending = [scopes[0]]
    while pending:
        node = pending.pop()
        if any("coach" in value.lower() for value in node.attrs.get("class", "").split()):
            continue
        if node.tag == "td" and node.has_class("player") and not node.has_class("player-image"):
            cells.append(node)
        pending.extend(child for child in reversed(node.children) if isinstance(child, _Node))
    for cell in cells:
        if cell.find("coach"):
            continue
        named = [node for node in cell.walk()
                 if node.has_class("player-compare") and node.has_class("flagAlign")]
        if named:
            if len(named) != 1:
                issues.append(f"ambiguous_player_cell:{team_id}")
                continue
            player_id = named[0].attrs.get("data-player-id", "")
            nicknames = named[0].find("text-ellipsis")
            nickname = nicknames[0].text() if len(nicknames) == 1 else ""
        else:
            # Earlier match layouts use explicit profile anchors in nickname
            # cells. Never search generic page/player/team-roster links.
            anchors = [(node, _link_id(node, _PLAYER_PATH)) for node in cell.walk()]
            anchors = [(node, player_id) for node, player_id in anchors if player_id]
            if len(anchors) != 1:
                issues.append(f"missing_or_ambiguous_player_identity:{team_id}")
                continue
            anchor, player_id = anchors[0]
            names = anchor.find("text-ellipsis")
            nickname = names[0].text() if len(names) == 1 else anchor.text()
        if not re.fullmatch(r"[1-9][0-9]*", player_id or "") or not nickname or len(nickname) > 100:
            issues.append(f"invalid_player_identity:{team_id}")
            continue
        players.append({"hltv_player_id": str(player_id), "name": nickname})
    if len(players) != 5:
        issues.append(f"lineup_requires_five_players:{team_id}")
    ids = [player["hltv_player_id"] for player in players]
    if len(set(ids)) != len(ids):
        issues.append(f"duplicate_lineup_player:{team_id}")
    return players


def parse_match_page(html: str, url: str) -> dict[str, Any]:
    """Extract only explicit match-page evidence; unknown/incomplete means issues.

    All external IDs are strings. ``map_names`` preserves positions, using None
    for TBA/unknown maps. Thus a known Map 2 can never be mistaken for Map 1.
    An empty ``issues`` list does not itself mean eligible: consumers also must
    require prematch status, source freshness, and their market/identity checks.
    """
    url = normalize_match_url(url)
    match_id = _MATCH_PATH.fullmatch(urlsplit(url).path).group(1)
    result: dict[str, Any] = {"match_id": match_id, "url": url,
        "scheduled_start_at": None, "event_name": None, "teams": [],
        "map_name": None, "map_names": [], "status": "unknown", "issues": []}
    issues: list[str] = result["issues"]
    if not isinstance(html, str) or len(html.encode("utf-8")) > MAX_HTML_BYTES:
        issues.append("html_missing_or_oversized")
        return result
    lower = html.lower()
    if any(marker in lower for marker in (
        "cf-chl-", "cf-turnstile", "verify you are human",
        "checking your browser", "<title>just a moment", "access denied",
    )):
        issues.append("access_challenge")
        return result
    tree = _Tree()
    try:
        tree.feed(html)
        tree.close()
    except (HLTVSourceError, ValueError, RecursionError):
        issues.append("html_structure_invalid")
        return result
    headers = tree.root.find("teamsBox")
    if len(headers) != 1:
        issues.append("match_header_missing_or_ambiguous")
        return result
    header = headers[0]
    # If provided by HLTV, canonical identity must agree with the requested ID.
    canonical = [node.attrs.get("href", "") for node in tree.root.walk()
                 if node.tag == "link" and node.attrs.get("rel") == "canonical"]
    for candidate in canonical:
        try:
            canonical_id = _MATCH_PATH.fullmatch(urlsplit(normalize_match_url(candidate)).path).group(1)
            if canonical_id != match_id:
                issues.append("canonical_match_id_mismatch")
        except (HLTVSourceError, AttributeError):
            issues.append("canonical_match_url_invalid")
    time_boxes = header.find("timeAndEvent")
    if len(time_boxes) == 1:
        time_box = time_boxes[0]
        time_nodes = time_box.find("time")
        if len(time_nodes) == 1:
            unix_ms = time_nodes[0].attrs.get("data-unix", "")
            try:
                if not re.fullmatch(r"[0-9]{13}", unix_ms):
                    raise ValueError("not epoch milliseconds")
                start = datetime.fromtimestamp(int(unix_ms) / 1000, timezone.utc)
                result["scheduled_start_at"] = start.isoformat().replace("+00:00", "Z")
            except (ValueError, OverflowError, OSError):
                issues.append("scheduled_start_invalid")
        else:
            issues.append("scheduled_start_missing_or_ambiguous")
        events = time_box.find("event")
        if len(events) == 1 and events[0].text():
            result["event_name"] = events[0].text()
        else:
            issues.append("event_name_missing_or_ambiguous")
        countdowns = time_box.find("countdown")
        if len(countdowns) == 1:
            countdown = countdowns[0]
            text = countdown.text().strip().lower()
            if text in {"match over", "finished", "ended"}:
                result["status"] = "finished"
            elif text == "live":
                result["status"] = "live"
            elif (countdown.attrs.get("data-time-countdown", "").upper() == "LIVE"
                  and re.fullmatch(r"[0-9]+[wdhms](?:\s*:\s*[0-9]+[wdhms])*", text)):
                result["status"] = "prematch"
    else:
        issues.append("time_and_event_missing_or_ambiguous")
    if result["status"] == "unknown":
        issues.append("match_status_unknown")
    for side in (1, 2):
        groups = header.find(f"team{side}-gradient")
        if len(groups) != 1:
            issues.append(f"team_header_missing_or_ambiguous:{side}")
            continue
        links = [(node, _link_id(node, _TEAM_PATH)) for node in groups[0].walk()]
        links = [(node, team_id) for node, team_id in links if team_id]
        names = groups[0].find("teamName")
        if len(links) != 1 or len(names) != 1 or not names[0].text():
            issues.append(f"team_identity_missing_or_ambiguous:{side}")
            continue
        result["teams"].append({"hltv_team_id": links[0][1], "name": names[0].text(), "players": []})
    team_ids = [team["hltv_team_id"] for team in result["teams"]]
    if len(team_ids) != 2 or len(set(team_ids)) != 2:
        issues.append("requires_two_distinct_teams")
    maps = tree.root.find("maps")
    if len(maps) == 1:
        holders = maps[0].find("mapholder")
        for index, holder in enumerate(holders):
            names = holder.find("mapname")
            name = names[0].text().lower().replace(" ", "") if len(names) == 1 else ""
            map_name = _MAPS.get(name)
            result["map_names"].append(map_name)
            if not map_name and name not in {"tba", "tbd", "default", "-"}:
                issues.append(f"map_name_unknown:{index + 1}")
        if holders:
            result["map_name"] = result["map_names"][0]
        else:
            issues.append("map_holders_missing")
    else:
        issues.append("maps_section_missing_or_ambiguous")
    if result["map_name"] is None:
        issues.append("map1_not_announced")
    lineups = [node for node in tree.root.find("lineups") if node.attrs.get("id") == "lineups"]
    if len(lineups) != 1:
        issues.append("lineups_section_missing_or_ambiguous")
    else:
        lineup_by_team: dict[str, list[_Node]] = {}
        for lineup in lineups[0].find("lineup"):
            heads = lineup.find("box-headline")
            links = [(_link_id(node, _TEAM_PATH)) for head in heads for node in head.walk()]
            ids = [team_id for team_id in links if team_id]
            if len(ids) != 1:
                issues.append("lineup_team_identity_missing_or_ambiguous")
                continue
            lineup_by_team.setdefault(ids[0], []).append(lineup)
        if set(lineup_by_team) != set(team_ids):
            issues.append("lineup_header_team_mismatch")
        for team in result["teams"]:
            entries = lineup_by_team.get(team["hltv_team_id"], [])
            if len(entries) != 1:
                issues.append(f"lineup_missing_or_ambiguous:{team['hltv_team_id']}")
            else:
                team["players"] = _lineup_players(entries[0], issues, team["hltv_team_id"])
        all_ids = [player["hltv_player_id"] for team in result["teams"] for player in team["players"]]
        if len(set(all_ids)) != len(all_ids):
            issues.append("players_not_distinct_across_teams")
    return result
