"""Read-only discovery and conservative, explicit binary resolution evidence."""
from __future__ import annotations

import re

import pandas as pd

from .map1_data import utc
from .map1_market import as_list, select_map1_market, team_key

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def public_get(url, params=None):
    # Only fixed public market-data origins; never fetch a user-entered source URL.
    if not url.startswith((GAMMA + "/", CLOB + "/")):
        raise ValueError("unsupported_public_origin")
    from curl_cffi import requests
    response = requests.get(url, params=params, impersonate="chrome", timeout=20)
    response.raise_for_status()
    return response.json()


def describe_event(event: dict, at, horizon_hours=72) -> dict:
    eid = str(event.get("id", ""))
    if not eid.isdigit():
        raise ValueError("invalid_event_id")
    summary = {"event_id": eid, "title": str(event.get("title", eid)),
               "slug": str(event.get("slug", "")), "status": "excluded", "reason": ""}
    try:
        if not (summary["slug"].startswith("cs2-") or summary["title"].startswith("Counter-Strike:")):
            raise ValueError("not_a_cs2_match")
        market = select_map1_market(event)
        start = utc(market["gameStartTime"])
        outcomes = as_list(market["outcomes"])
        tokens = [str(t) for t in as_list(market["clobTokenIds"])]
        condition = str(market.get("conditionId", ""))
        if (len(outcomes) != 2 or len(set(map(team_key, outcomes))) != 2
                or len(tokens) != 2 or len(set(tokens)) != 2 or not all(t.isdigit() for t in tokens)
                or not re.fullmatch(r"0x[0-9a-fA-F]{64}", condition)):
            raise ValueError("invalid_market_identity")
        summary.update(market_id=str(market["id"]), condition_id=condition,
                       scheduled_start_at=start.isoformat(), outcomes=outcomes, tokens=tokens)
        if event.get("closed") is not False or market.get("closed") is not False:
            raise ValueError("market_closed")
        if event.get("live") is True or event.get("ended") is True or start <= utc(at):
            raise ValueError("prematch_window_closed")
        if start > utc(at) + pd.Timedelta(hours=horizon_hours):
            raise ValueError("outside_discovery_horizon")
        if (market.get("active") is not True or market.get("acceptingOrders") is not True
                or market.get("enableOrderBook") is not True or market.get("negRisk") is not False):
            raise ValueError("market_not_supported_or_accepting")
        summary.update(status="awaiting_confirmation", reason="map_and_roster_unconfirmed")
    except (KeyError, TypeError, ValueError) as exc:
        summary["reason"] = str(exc)
    return summary


def discover(at, get=public_get, max_pages=10, page_size=100) -> dict:
    if not 1 <= max_pages <= 20 or not 1 <= page_size <= 100:
        raise ValueError("invalid_discovery_bounds")
    sports = get(GAMMA + "/sports")
    matches = [s for s in sports if s.get("sport") == "cs2"]
    if len(matches) != 1 or not str(matches[0].get("primaryTagId", "")).isdigit():
        raise ValueError("missing_cs2_primary_tag")
    tag = str(matches[0]["primaryTagId"])
    events, seen, cursors = [], set(), set()
    cursor = None
    for _ in range(max_pages):
        params = {"tag_id": tag, "closed": "false", "limit": page_size,
                  "order": "id", "ascending": "false"}
        if cursor:
            params["after_cursor"] = cursor
        page = get(GAMMA + "/events/keyset", params)
        if not isinstance(page, dict) or not isinstance(page.get("events"), list):
            raise ValueError("invalid_event_page")
        for event in page["events"]:
            if str(event.get("id")) not in seen:
                events.append(event)
                seen.add(str(event.get("id")))
        cursor = page.get("next_cursor")
        if not cursor:
            break
        if cursor in cursors:
            raise ValueError("repeated_discovery_cursor")
        cursors.add(cursor)
    return {"events": events, "truncated": bool(cursor), "received_at": str(at),
            "sport_metadata": matches[0], "pages": len(cursors) + (not bool(cursor))}


def binary_resolution(event: dict, clob: dict, expected: dict, received_at) -> dict | None:
    """Only resolved Gamma + closed CLOB + exactly one explicit winner qualify.

    Prices (including 0/1 and 0.5/0.5) are deliberately never used as payouts.
    Ambiguous, split and disputed outcomes remain unsettled for human review.
    """
    if str(event.get("id")) != str(expected["event_id"]):
        raise ValueError("resolution_event_mismatch")
    market = select_map1_market(event)
    if str(market.get("id")) != str(expected["market_id"]) or market.get("conditionId") != expected["condition_id"]:
        raise ValueError("resolution_market_mismatch")
    wanted = {expected["token_a"], expected["token_b"]}
    if set(map(str, as_list(market["clobTokenIds"]))) != wanted:
        raise ValueError("resolution_gamma_tokens_mismatch")
    if market.get("closed") is not True or market.get("umaResolutionStatus") != "resolved":
        return None
    if clob.get("condition_id") != expected["condition_id"]:
        raise ValueError("resolution_clob_condition_mismatch")
    tokens = clob.get("tokens", [])
    if len(tokens) != 2 or {str(t.get("token_id")) for t in tokens} != wanted:
        raise ValueError("resolution_clob_tokens_mismatch")
    if clob.get("closed") is not True:
        return None
    if not all(type(t.get("winner")) is bool for t in tokens) or sum(t["winner"] for t in tokens) != 1:
        raise ValueError("ambiguous_or_fractional_resolution_requires_review")
    return {"event_id": str(expected["event_id"]), "condition_id": expected["condition_id"],
            "resolved_at": utc(received_at).isoformat(),
            "payouts": {str(t["token_id"]): float(t["winner"]) for t in tokens},
            "source": CLOB + "/markets/" + expected["condition_id"],
            "evidence": {"gamma_event": event, "clob_market": clob},
            "time_basis": "first_local_observation_of_confirmed_resolution"}
