"""Explicitly opt-in public API smoke check; never creates forecasts or orders.

Run from project root: python -B -m scripts.smoke_map1_desk --live
"""
from __future__ import annotations

import argparse
import json

from cs2ml.map1 import now
from cs2ml.map1_market import as_list, select_map1_market
from cs2ml.map1_sources import CLOB, GAMMA, binary_resolution, describe_event, discover, public_get


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Allow read-only public internet requests")
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to explicitly allow public market-data GET requests")
    discovered = discover(now(), max_pages=3)
    summaries = [describe_event(e, now()) for e in discovered["events"]]
    eligible = [s for s in summaries if s["status"] == "awaiting_confirmation"]
    print(json.dumps({"check": "public_discovery", "seen": len(summaries), "eligible": len(eligible),
                      "truncated": discovered["truncated"], "examples": eligible[:2]}, ensure_ascii=False), flush=True)
    page = public_get(GAMMA + "/events/keyset", {"tag_id": discovered["sport_metadata"]["primaryTagId"],
                      "closed": "true", "limit": 20, "order": "id", "ascending": "false"})
    for event in page["events"]:
        try:
            market = select_map1_market(event)
        except ValueError:
            continue
        if market.get("umaResolutionStatus") != "resolved":
            continue
        tokens = list(map(str, as_list(market["clobTokenIds"])))
        if len(tokens) != 2:
            continue
        expected = {"event_id": str(event["id"]), "market_id": str(market["id"]),
                    "condition_id": market["conditionId"], "token_a": tokens[0], "token_b": tokens[1]}
        clob = public_get(CLOB + "/markets/" + expected["condition_id"])
        result = binary_resolution(event, clob, expected, now())
        if result is not None:
            print(json.dumps({"check": "explicit_binary_resolution", "event_id": result["event_id"],
                              "payouts": result["payouts"], "status": "passed",
                              "note": "Diagnostic only; nothing added to forward-validation samples."}), flush=True)
            return
    raise RuntimeError("No usable recent binary resolution found; smoke check incomplete")


if __name__ == "__main__":
    main()
