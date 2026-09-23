"""Auto-discovery and fail-closed pairing of 5EPlay matches with Polymarket events.

Verified public sources (2026-09-19, evidence in data/fivee/discovery_probe/):
  5E:         GET https://esports-data.5eplaycdn.com/v1/api/csgo/matches?page=&limit=
  Polymarket: GET https://gamma-api.polymarket.com/events?active=true&closed=false
                    &tag_slug=counter-strike-2&order=startDate

Pairing gates (all must hold, else the candidate is recorded but NOT started):
  1. both team names known on both sides (5E "TBD" is rejected);
  2. normalized team sets are equal (aliases applied, accents/punctuation stripped);
  3. |5E plan_ts - Polymarket startDate| <= tolerance seconds;
  4. uniqueness in BOTH directions: the pair is the only candidate for its 5E
     match and the only candidate for its Polymarket event.

Unknown enums/fields are recorded raw, never guessed. All IO goes through
injectable request_get so tests never touch the network.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

FIVEE_LIST_URL = "https://esports-data.5eplaycdn.com/v1/api/csgo/matches"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
POLY_TAG_SLUG = "counter-strike-2"
TITLE_RE = re.compile(
    r"^Counter-Strike:\s*(?P<t1>.+?)\s+vs\.?\s+(?P<t2>.+?)\s*"
    r"\((?P<bo>BO\d+)\s*\)\s*-\s*(?P<league>.+?)\s*$",
    re.IGNORECASE,
)
MAP_WINNER_RE = re.compile(r"^Map\s+(\d+)\s+Winner$", re.IGNORECASE)
UNKNOWN_TEAM_NAMES = {"", "tbd", "tbd ", "待定", "unknown", "n/a"}

# Minimal built-in aliases; extend via alias_file {alias: canonical}.
BUILTIN_TEAM_ALIASES = {
    "natus vincere": "navi",
    "natus vincere junior": "navi junior",
}
GENERIC_SUFFIXES = (" esports", " e sports", " gaming", " club", " gg", " team")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_team(name: str, aliases: dict[str, str] | None = None) -> str:
    """Case/punctuation/accent-insensitive team key; keeps academy/junior/fe."""
    table = dict(BUILTIN_TEAM_ALIASES)
    if aliases:
        table.update({str(k).strip().casefold(): str(v).strip().casefold()
                      for k, v in aliases.items()})
    raw = str(name or "")
    folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    folded = folded.casefold().replace("&", " and ")
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    for suffix in GENERIC_SUFFIXES:
        if folded.endswith(suffix) and len(folded) > len(suffix) + 1:
            folded = folded[: -len(suffix)].strip()
    return table.get(folded, folded)


def parse_poly_title(title: str) -> dict | None:
    match = TITLE_RE.match(str(title or "").strip())
    if not match:
        return None
    return {
        "team1": match.group("t1").strip(),
        "team2": match.group("t2").strip(),
        "bo": match.group("bo").upper(),
        "league": match.group("league").strip(),
    }


def _iso_ts(value) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _json_array(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


# ---------------------------------------------------------------- 5E side
@dataclass
class FiveEMatchCandidate:
    match_id: str
    plan_ts: float | None
    team1: str
    team2: str
    team_key1: str
    team_key2: str
    format: str
    stage: str
    status: str
    raw: dict = field(repr=False, default_factory=dict)


def parse_fivee_list(payload: dict, aliases: dict[str, str] | None = None) -> list[FiveEMatchCandidate]:
    if not isinstance(payload, dict) or not payload.get("success"):
        raise ValueError(f"fivee_list_not_success:{payload.get('message')!r}")
    data = payload.get("data") or {}
    rows = data.get("matches") or []
    if not isinstance(rows, list):
        raise ValueError("fivee_matches_not_array")
    out: list[FiveEMatchCandidate] = []
    for row in rows:
        mc = row.get("mc_info") if isinstance(row, dict) else None
        if not isinstance(mc, dict):
            continue
        team1 = str((mc.get("t1_info") or {}).get("disp_name") or "").strip()
        team2 = str((mc.get("t2_info") or {}).get("disp_name") or "").strip()
        key1, key2 = normalize_team(team1, aliases), normalize_team(team2, aliases)
        if (key1 in UNKNOWN_TEAM_NAMES or key2 in UNKNOWN_TEAM_NAMES
                or not key1 or not key2 or key1 == key2):
            continue  # TBD/placeholder teams can never pair safely
        try:
            plan_ts = float(mc.get("plan_ts"))
        except (TypeError, ValueError):
            plan_ts = None
        state = row.get("state") if isinstance(row.get("state"), dict) else {}
        out.append(FiveEMatchCandidate(
            match_id=str(mc.get("id") or ""), plan_ts=plan_ts,
            team1=team1, team2=team2, team_key1=key1, team_key2=key2,
            format=str(mc.get("format") or ""), stage=str(mc.get("tt_stage") or ""),
            status=str(state.get("status") or ""), raw=row))
    return [c for c in out if c.match_id]


def fetch_fivee_matches(request_get: Callable, *, pages: int = 4, limit: int = 50,
                        timeout: float = 20.0,
                        aliases: dict[str, str] | None = None) -> list[FiveEMatchCandidate]:
    """Fetch several list pages and dedup by match_id.

    The list is tournament-blocked rather than time-ordered (verified
    2026-09-19: page 1 = Oct 1-11, page 2 = Sep 26-Oct 1), so one page can
    miss near-term matches; paging is required for reliable coverage.
    """
    seen_ids: set[str] = set()
    out: list[FiveEMatchCandidate] = []
    for page in range(1, max(1, int(pages)) + 1):
        response = request_get(FIVEE_LIST_URL, params={"page": page, "limit": limit},
                               impersonate="chrome", timeout=timeout)
        response.raise_for_status()
        candidates = parse_fivee_list(response.json(), aliases)
        for cand in candidates:
            if cand.match_id not in seen_ids:
                seen_ids.add(cand.match_id)
                out.append(cand)
    return out


# ---------------------------------------------------------------- Polymarket side
@dataclass
class PolyMapMarket:
    bout_num: int
    market_name: str
    market_id: str
    outcomes: list
    tokens: dict


@dataclass
class PolyEventCandidate:
    event_id: str
    start_ts: float | None
    title: str
    team1: str
    team2: str
    team_key1: str
    team_key2: str
    bo: str
    league: str
    map_markets: list
    live: bool
    raw: dict = field(repr=False, default_factory=dict)


def map_winner_markets(event: dict) -> list[PolyMapMarket]:
    markets = []
    for market in event.get("markets") or []:
        name = str(market.get("groupItemTitle") or market.get("question") or "")
        match = MAP_WINNER_RE.match(name.strip())
        if not match:
            continue
        outcomes = [str(o) for o in _json_array(market.get("outcomes"))]
        tokens = _json_array(market.get("clobTokenIds"))
        if len(outcomes) != 2 or len(tokens) != 2:
            continue
        markets.append(PolyMapMarket(
            bout_num=int(match.group(1)), market_name=name.strip(),
            market_id=str(market.get("id") or ""),
            outcomes=outcomes, tokens=dict(zip(outcomes, [str(t) for t in tokens]))))
    return sorted(markets, key=lambda m: m.bout_num)


def parse_poly_events(payload: list, aliases: dict[str, str] | None = None) -> list[PolyEventCandidate]:
    if not isinstance(payload, list):
        raise ValueError("poly_events_not_array")
    out: list[PolyEventCandidate] = []
    for event in payload:
        if not isinstance(event, dict) or event.get("closed"):
            continue
        parsed_title = parse_poly_title(event.get("title") or "")
        if parsed_title is None:
            continue
        key1 = normalize_team(parsed_title["team1"], aliases)
        key2 = normalize_team(parsed_title["team2"], aliases)
        if not key1 or not key2 or key1 == key2:
            continue
        out.append(PolyEventCandidate(
            event_id=str(event.get("id") or ""), start_ts=_iso_ts(event.get("startDate")),
            title=str(event.get("title") or ""),
            team1=parsed_title["team1"], team2=parsed_title["team2"],
            team_key1=key1, team_key2=key2, bo=parsed_title["bo"],
            league=parsed_title["league"], map_markets=map_winner_markets(event),
            live=bool(event.get("liveAt")), raw=event))
    return [c for c in out if c.event_id]


def fetch_poly_events(request_get: Callable, *, limit: int = 50, timeout: float = 30.0,
                      aliases: dict[str, str] | None = None) -> list[PolyEventCandidate]:
    response = request_get(GAMMA_EVENTS_URL, params={
        "active": "true", "closed": "false", "limit": limit,
        "tag_slug": POLY_TAG_SLUG, "order": "startDate", "ascending": "false"},
        impersonate="chrome", timeout=timeout)
    response.raise_for_status()
    return parse_poly_events(response.json(), aliases)


# ---------------------------------------------------------------- pairing
def _teams_match(a1: str, a2: str, b1: str, b2: str) -> bool:
    return {a1, a2} == {b1, b2}


def pair_candidates(fivee: list[FiveEMatchCandidate], poly: list[PolyEventCandidate],
                    *, tolerance_seconds: float) -> dict:
    """Fail-closed bipartite pairing. Ambiguity is recorded, never resolved."""
    candidates: list[dict] = []
    for fm in fivee:
        if fm.plan_ts is None:
            continue
        for pm in poly:
            if pm.start_ts is None or not pm.map_markets:
                continue
            if abs(fm.plan_ts - pm.start_ts) > tolerance_seconds:
                continue
            if not _teams_match(fm.team_key1, fm.team_key2, pm.team_key1, pm.team_key2):
                continue
            candidates.append({"fivee": fm, "poly": pm,
                               "start_skew_seconds": fm.plan_ts - pm.start_ts})
    by_fivee: dict[str, list[dict]] = {}
    by_poly: dict[str, list[dict]] = {}
    for cand in candidates:
        by_fivee.setdefault(cand["fivee"].match_id, []).append(cand)
        by_poly.setdefault(cand["poly"].event_id, []).append(cand)
    paired, ambiguous = [], []
    for cand in candidates:
        if (len(by_fivee[cand["fivee"].match_id]) == 1
                and len(by_poly[cand["poly"].event_id]) == 1):
            paired.append(cand)
        else:
            ambiguous.append({
                "reason": "multiple_pairing_candidates",
                "fivee_match_id": cand["fivee"].match_id,
                "poly_event_id": cand["poly"].event_id,
                "fivee_candidates": [c["fivee"].match_id for c in by_poly[cand["poly"].event_id]],
                "poly_candidates": [c["poly"].event_id for c in by_fivee[cand["fivee"].match_id]],
            })
    paired_fivee = {c["fivee"].match_id for c in paired} | set(by_fivee)
    paired_poly = {c["poly"].event_id for c in paired} | set(by_poly)
    return {
        "paired": paired,
        "ambiguous": ambiguous,
        "unmatched_fivee": [
            {"match_id": fm.match_id, "team1": fm.team1, "team2": fm.team2,
             "plan_ts": fm.plan_ts, "status": fm.status}
            for fm in fivee if fm.match_id not in paired_fivee],
        "unmatched_poly": [
            {"event_id": pm.event_id, "title": pm.title, "start_ts": pm.start_ts,
             "map_markets": len(pm.map_markets)}
            for pm in poly if pm.event_id not in paired_poly],
    }


def load_aliases(path) -> dict[str, str]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("alias_file_must_be_object")
    return {str(k): str(v) for k, v in data.items()}


def serialize_pairing(report: dict) -> dict:
    def cand_json(c):
        fm, pm = c["fivee"], c["poly"]
        return {
            "fivee": {"match_id": fm.match_id, "team1": fm.team1, "team2": fm.team2,
                      "plan_ts": fm.plan_ts, "plan_utc": _utc(fm.plan_ts),
                      "format": fm.format, "stage": fm.stage, "status": fm.status},
            "poly": {"event_id": pm.event_id, "title": pm.title,
                     "start_ts": pm.start_ts, "start_utc": _utc(pm.start_ts),
                     "bo": pm.bo, "league": pm.league, "live": pm.live,
                     "map_markets": [{"bout_num": m.bout_num, "market_id": m.market_id,
                                      "market_name": m.market_name} for m in pm.map_markets]},
            "start_skew_seconds": c["start_skew_seconds"],
        }
    return {
        "created_at": utc_now(),
        "paired": [cand_json(c) for c in report["paired"]],
        "ambiguous": report["ambiguous"],
        "unmatched_fivee": report["unmatched_fivee"],
        "unmatched_poly": report["unmatched_poly"],
    }


def _utc(ts) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat()
