"""Deterministic, delayed, limit-price paper fills and explicit settlement.

One entry per event; hold to settlement. No invented end-of-file liquidation,
no fills at midpoint, no inferred payouts from last trade prices.
"""
from __future__ import annotations

import pandas as pd

from .map1_data import utc
from .map1_market import book_time, number, proposal, validate_book, walk_book
from .map1_model import forward_comparison


def replay(snapshots: list[dict], resolutions: list[dict] = (), initial_cash: float = 1000,
           shares: float = 10, min_edge: float = .05, latency_seconds: float = .5,
           max_age: float = 10, fill_window_seconds: float = 10, as_of=None) -> dict:
    cash = number(initial_cash, "initial cash", 0)
    shares = number(shares, "shares", .000001)
    number(min_edge, "minimum edge", 0, 1)
    number(latency_seconds, "latency", 0, 60)
    number(max_age, "maximum book age", 0, 300)
    number(fill_window_seconds, "fill window", .000001, 300)
    pending, positions, visited = {}, {}, set()
    ledger, observations, candidates = [], [], {}
    events = [(utc(s["received_at"]), 1, "snapshot", s) for s in snapshots]
    events += [(utc(r["resolved_at"]), 0, "resolution", r) for r in resolutions]
    if as_of is not None:
        events = [e for e in events if e[0] <= utc(as_of)]
    # Resolutions are processed when known, not retroactively before a fill.
    realized = 0.0
    last_snapshot = {}
    for at, _, kind, obj in sorted(events, key=lambda x: (x[0], x[1])):
        if kind == "resolution":
            event = str(obj["event_id"])
            if not str(obj.get("source", "")).strip():
                raise ValueError("Settlement must have an explicit source")
            payouts = obj.get("payouts", {})
            if len(payouts) != 2:
                raise ValueError("Settlement needs both token payouts")
            amounts = [number(v, "payout", 0, 1) for v in payouts.values()]
            if abs(sum(amounts) - 1) > 1e-8:
                raise ValueError("Token payouts must sum to one")
            candidate = candidates.pop(event, None)
            if candidate:
                binding = candidate.pop("binding")
                if set(payouts) != {binding["token_a"], binding["token_b"]} or obj.get("condition_id") != binding["condition_id"]:
                    raise ValueError("Settlement identity mismatch")
                observations.append({**candidate, "resolved_at": at.isoformat(),
                                     "payout_a": payouts[binding["token_a"]]})
            if event in pending:
                ledger.append({"type": "cancel", "event_id": event, "reason": "resolved_while_pending"})
                pending.pop(event)
            pos = positions.get(event)
            if pos:
                expected_tokens = {pos["binding"]["token_a"], pos["binding"]["token_b"]}
                if set(payouts) != expected_tokens or obj.get("condition_id") != pos["binding"]["condition_id"]:
                    raise ValueError("Settlement identity mismatch")
                proceeds = pos["shares"] * float(payouts[pos["token"]])
                pnl = proceeds - pos["cash"]
                cash += proceeds
                realized += pnl
                ledger.append({"type": "settle", "event_id": event, "at": at.isoformat(),
                               "cash_received": proceeds, "pnl": pnl, "source": obj["source"]})
                del positions[event]
            visited.add(event)
            continue

        event = str(obj.get("event_id", ""))
        if not event:
            raise ValueError("Snapshot is missing event_id")
        if event in visited or event in positions:
            continue
        if last_snapshot.get(event) == at:
            continue
        last_snapshot[event] = at
        signal = proposal(obj, shares, min_edge, max_age)
        if "p_market" in signal and event not in candidates:
            candidates[event] = {"event_id": event, "decision_at": at.isoformat(),
                                 "p_model": signal["p_model"], "p_market": signal["p_market"],
                                 "binding": obj["binding"]}
        if event in pending:
            order = pending[event]
            if signal["action"] != "signal" or signal.get("token") != order["token"]:
                ledger.append({"type": "cancel", "event_id": event, "reason": "signal_no_longer_valid"})
                pending.pop(event)
                continue
            if at < order["eligible_at"]:
                continue
            if (at - order["eligible_at"]).total_seconds() > fill_window_seconds:
                ledger.append({"type": "cancel", "event_id": event, "reason": "missing_timely_post_delay_book"})
                pending.pop(event)
                visited.add(event)
                continue
            try:
                binding = obj["binding"]
                if any(binding[k] != order["binding"][k] for k in
                       ("condition_id", "token_a", "token_b", "fee_rate", "rules", "seconds_delay",
                        "map_name", "roster_a", "roster_b", "scheduled_start_at")):
                    raise ValueError("market_parameters_changed")
                book = obj["books"][order["side"]]
                quote = validate_book(book, order["token"], binding["condition_id"], at, max_age)
                if book_time(book["timestamp"]) < order["eligible_at"]:
                    raise ValueError("book_predates_eligibility")
                if shares < quote["min_order_size"]:
                    raise ValueError("below_minimum_size")
                fill = walk_book(book, shares, binding["fee_rate"], limit=order["limit"])
                if fill["cash"] > cash:
                    raise ValueError("insufficient_paper_cash")
                cash -= fill["cash"]
                positions[event] = {**order, **fill, "eligible_at": order["eligible_at"].isoformat(),
                                    "filled_at": at.isoformat()}
                ledger.append({"type": "fill", "event_id": event, "at": at.isoformat(),
                               "token": order["token"], **fill})
                visited.add(event)
            except (KeyError, ValueError, TypeError) as exc:
                ledger.append({"type": "cancel", "event_id": event, "reason": str(exc)})
                visited.add(event)
            pending.pop(event)
        elif signal["action"] == "signal":
            binding = obj["binding"]
            delay = binding["seconds_delay"] + latency_seconds
            eligible = at + pd.Timedelta(seconds=delay)
            if eligible >= utc(binding["scheduled_start_at"]):
                ledger.append({"type": "skip", "event_id": event, "reason": "delay_crosses_match_start"})
                continue
            pending[event] = {**signal, "event_id": event, "binding": binding, "decision_at": at.isoformat(),
                              "eligible_at": eligible}
            ledger.append({"type": "pending", "event_id": event, "at": at.isoformat(),
                           "token": signal["token"], "limit": signal["limit"],
                           "eligible_at": eligible.isoformat()})
        else:
            ledger.append({"type": "skip", "event_id": event, "at": at.isoformat(),
                           "reason": signal["reason"]})
    # A live desk has a clock even when its feed is disconnected. Never keep a
    # paper order pending indefinitely just because no next snapshot arrived.
    if as_of is not None:
        for event, order in list(pending.items()):
            if utc(as_of) > order["eligible_at"] + pd.Timedelta(seconds=fill_window_seconds):
                ledger.append({"type": "cancel", "event_id": event, "at": utc(as_of).isoformat(),
                               "reason": "missing_timely_post_delay_book"})
                del pending[event]
    return {"mode": "paper_only", "initial_cash": initial_cash, "cash": cash,
            "realized_pnl": realized, "open_cost_basis": sum(p["cash"] for p in positions.values()),
            "open_positions": list(positions.values()), "pending_count": len(pending),
            "unrealized_pnl": None, "ledger": ledger,
            "probability_comparison_first_valid_snapshot_per_event": forward_comparison(observations),
            "limitations": ["Displayed book replay cannot prove historical queue access or an actual fill.",
                            "Fees are accounted in USD equivalent; quantities are simulated shares.",
                            "Entry EV assumes ordinary binary settlement; void payouts are accounted when provided.",
                            "No end-of-file liquidation or profit is assumed for unresolved positions."]}
