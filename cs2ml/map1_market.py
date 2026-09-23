"""Strict public-data adapter and executable-price checks for Map 1 only.

No wallet, authenticated client or order placement is implemented here.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata

import pandas as pd

from .map1_data import map_name, roster_key, utc


def team_key(name: str) -> str:
    # Keep Academy/Youth/region markers; no substring or fuzzy matching.
    return "".join(c for c in unicodedata.normalize("NFKC", str(name)).casefold() if c.isalnum())


def as_list(value) -> list:
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list")
    return value


def number(value, label: str, low=0.0, high=float("inf")) -> float:
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f"Invalid {label}")
    return v


def validate_context(context: dict, at) -> dict:
    now = utc(at)
    if context.get("map_no") != 1:
        raise ValueError("map1_only")
    arena = map_name(context["map_name"])
    for key in ("map", "roster"):
        known = utc(context[f"{key}_known_at"])
        if known > now:
            raise ValueError(f"{key}_not_known_at_decision")
        if not str(context.get(f"{key}_source", "")).strip():
            raise ValueError(f"missing_{key}_source")
    ra = roster_key(context["team_a"]["roster"])
    rb = roster_key(context["team_b"]["roster"])
    if set(ra.split(",")) & set(rb.split(",")):
        raise ValueError("overlapping_rosters")
    if not str(context.get("event_id", "")).isdigit():
        raise ValueError("missing_event_id")
    if utc(context["scheduled_start_at"]) <= now:
        raise ValueError("prematch_window_closed")
    return {**context, "map_name": arena, "roster_a": ra, "roster_b": rb}


def select_map1_market(event: dict) -> dict:
    found = []
    for market in event.get("markets", []):
        if market.get("sportsMarketType") != "child_moneyline":
            continue
        question = str(market.get("question", ""))
        title = str(market.get("groupItemTitle", ""))
        if not re.search(r"\bMap\s+1\s+Winner$", question, re.I):
            continue
        if title and not re.fullmatch(r"Map\s+1\s+Winner", title, re.I):
            continue
        found.append(market)
    if len(found) != 1:
        raise ValueError("missing_or_ambiguous_map1_market")
    return found[0]


def bind_market(event: dict, context: dict, at) -> dict:
    c = validate_context(context, at)
    if str(event.get("id")) != str(c["event_id"]):
        raise ValueError("event_id_mismatch")
    if event.get("closed") is not False or event.get("live") is True or event.get("ended") is True:
        raise ValueError("event_not_prematch_open")
    m = select_map1_market(event)
    if (m.get("closed") is not False or m.get("active") is not True
            or m.get("acceptingOrders") is not True or m.get("enableOrderBook") is not True):
        raise ValueError("market_not_accepting_orders")
    if m.get("negRisk") is not False:
        raise ValueError("unsupported_or_unknown_neg_risk")
    start = utc(m["gameStartTime"])
    if start != utc(c["scheduled_start_at"]):
        raise ValueError("schedule_changed_or_wrong_match")
    if start <= utc(at):
        raise ValueError("prematch_window_closed")
    outcomes, tokens = as_list(m["outcomes"]), as_list(m["clobTokenIds"])
    if len(outcomes) != 2 or len(tokens) != 2 or len(set(map(str, tokens))) != 2:
        raise ValueError("nonbinary_or_invalid_tokens")
    keys = [team_key(o) for o in outcomes]
    a, b = team_key(c["team_a"]["outcome"]), team_key(c["team_b"]["outcome"])
    if not a or a == b or len(set(keys)) != 2 or set(keys) != {a, b}:
        raise ValueError("outcome_identity_mismatch")
    if not all(str(t).isdigit() for t in tokens):
        raise ValueError("invalid_token_id")
    condition = str(m.get("conditionId", ""))
    if not re.fullmatch(r"0x[0-9a-fA-F]{64}", condition):
        raise ValueError("invalid_condition_id")
    rules = str(m.get("description", "")).strip()
    if not rules:
        raise ValueError("missing_resolution_rules")
    fees_enabled = m.get("feesEnabled")
    if fees_enabled is False:
        rate = 0.0
    elif fees_enabled is True:
        schedule = m.get("feeSchedule") or {}
        if schedule.get("exponent") != 1 or schedule.get("takerOnly") is not True:
            raise ValueError("unsupported_fee_schedule")
        rate = number(schedule["rate"], "fee rate", 0, 1)
    else:
        raise ValueError("unknown_fee_schedule")
    return {
        "event_id": str(event["id"]), "market_id": str(m["id"]), "condition_id": condition,
        "question": m["question"], "map_no": 1, "map_name": c["map_name"],
        "roster_a": c["roster_a"], "roster_b": c["roster_b"],
        "token_a": str(tokens[keys.index(a)]), "token_b": str(tokens[keys.index(b)]),
        "outcome_a": outcomes[keys.index(a)], "outcome_b": outcomes[keys.index(b)],
        "scheduled_start_at": start.isoformat(), "fee_rate": rate,
        "seconds_delay": number(m["secondsDelay"], "secondsDelay", 0, 60),
        "rules": rules, "resolution_source": m.get("resolutionSource", event.get("resolutionSource")),
        "metadata_received_at": utc(at).isoformat(),
    }


def book_time(value):
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        return pd.to_datetime(float(value), unit="ms", utc=True)
    return utc(value)


def levels(book: dict, side: str) -> list[tuple[float, float]]:
    raw = book.get(side)
    if not isinstance(raw, list):
        raise ValueError("missing_book_depth")
    out = []
    for lv in raw:
        p = number(lv["price"], "level price", 0, 1)
        size = number(lv["size"], "level size")
        if not 0 < p < 1:
            raise ValueError("nontradeable_price")
        if size:
            out.append((p, size))
    return sorted(out, reverse=side == "bids")


def validate_book(book: dict, token: str, condition: str, at, max_age: float = 10) -> dict:
    if str(book.get("asset_id")) != token or book.get("market") != condition:
        raise ValueError("book_identity_mismatch")
    age = (utc(at) - book_time(book["timestamp"])).total_seconds()
    if age < -1 or age > max_age:
        raise ValueError("stale_or_future_book")
    bids, asks = levels(book, "bids"), levels(book, "asks")
    if not bids or not asks:
        raise ValueError("empty_book")
    if bids[0][0] >= asks[0][0]:
        raise ValueError("crossed_book")
    return {"bid": bids[0][0], "ask": asks[0][0], "mid": (bids[0][0] + asks[0][0]) / 2,
            "age_seconds": age, "min_order_size": number(book["min_order_size"], "minimum size"),
            "tick_size": number(book["tick_size"], "tick size", .000001, 1)}


def walk_book(book: dict, shares: float, fee_rate: float, side: str = "asks", limit: float | None = None) -> dict:
    quantity = number(shares, "shares", .000001)
    rate = number(fee_rate, "fee rate", 0, 1)
    if side not in {"asks", "bids"}:
        raise ValueError("Unknown book side")
    remaining, notional, fees, worst = quantity, 0.0, 0.0, None
    for p, size in levels(book, side):
        if limit is not None and ((side == "asks" and p > limit + 1e-10)
                                  or (side == "bids" and p < limit - 1e-10)):
            break
        take = min(size, remaining)
        notional += take * p
        fees += take * rate * p * (1 - p)
        remaining -= take
        worst = p
        if remaining < 1e-8:
            break
    if remaining >= 1e-8:
        raise ValueError("insufficient_depth_at_limit")
    return {"shares": quantity, "notional": notional, "fee": fees,
            "vwap": notional / quantity, "worst_price": worst,
            "cash": notional + fees if side == "asks" else notional - fees}


def proposal(snapshot: dict, shares: float = 10, min_edge: float = .05, max_age: float = 10) -> dict:
    """Evaluate both independently quoted tokens, never 1 - the other token's ask."""
    if snapshot.get("status") != "ready":
        return {"action": "skip", "reason": snapshot.get("reason", "snapshot_not_ready")}
    try:
        if snapshot.get("model_target") != "map1_winner":
            raise ValueError("model_target_mismatch")
        p = number(snapshot["p_model"], "model probability", 0, 1)
        min_edge = number(min_edge, "minimum edge", 0, 1)
        shares = number(shares, "shares", .000001)
        at, binding = utc(snapshot["received_at"]), snapshot["binding"]
        context = validate_context(snapshot["context"], at)
        if str(snapshot.get("event_id")) != str(binding["event_id"]):
            raise ValueError("snapshot_event_mismatch")
        if any(str(context[k]) != str(binding[k]) for k in ("event_id", "map_no", "map_name", "roster_a", "roster_b")):
            raise ValueError("context_binding_mismatch")
        if utc(snapshot["train_max_available_at"]) >= utc(snapshot["model_fitted_at"]) or utc(snapshot["model_fitted_at"]) > at:
            raise ValueError("model_time_leakage")
        if utc(binding["scheduled_start_at"]) <= at:
            raise ValueError("prematch_window_closed")
        quotes = {s: validate_book(snapshot["books"][s], binding[f"token_{s}"],
                                  binding["condition_id"], at, max_age) for s in ("a", "b")}
        p_market = quotes["a"]["mid"] / (quotes["a"]["mid"] + quotes["b"]["mid"])
        options, exclusions = [], []
        for s, prob in (("a", p), ("b", 1 - p)):
            if shares < quotes[s]["min_order_size"]:
                exclusions.append("below_minimum_size")
                continue
            try:
                fill = walk_book(snapshot["books"][s], shares, binding["fee_rate"])
            except ValueError as exc:
                exclusions.append(str(exc))
                continue
            edge = prob - fill["cash"] / shares
            if edge >= min_edge:
                options.append({"action": "signal", "side": s, "token": binding[f"token_{s}"],
                                "payout_probability": prob, "p_model": p, "p_market": p_market,
                                "net_edge": edge, "limit": fill["worst_price"], "estimated_fill": fill})
        if not options:
            return {"action": "skip", "reason": ",".join(sorted(set(exclusions))) or "edge_below_threshold",
                    "p_model": p, "p_market": p_market}
        return max(options, key=lambda x: x["net_edge"])
    except (ValueError, KeyError, TypeError) as exc:
        return {"action": "skip", "reason": str(exc)}


def fetch_event(event_id: str) -> dict:
    if not str(event_id).isdigit():
        raise ValueError("Invalid event ID")
    from curl_cffi import requests
    response = requests.get(f"https://gamma-api.polymarket.com/events/{event_id}",
                            impersonate="chrome", timeout=25)
    response.raise_for_status()
    return response.json()


def fetch_book(token: str) -> dict:
    if not str(token).isdigit():
        raise ValueError("Invalid token ID")
    from curl_cffi import requests
    response = requests.get("https://clob.polymarket.com/book", params={"token_id": token},
                            impersonate="chrome", timeout=25)
    response.raise_for_status()
    return response.json()
