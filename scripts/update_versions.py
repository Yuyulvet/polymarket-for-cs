"""Fetch the latest CS2 updates from the official Steam news feed and flag
gameplay-relevant patches newer than the current version timeline.

Usage (run weekly, or via Task Scheduler):
    python scripts/update_versions.py [--since YYYY-MM-DD]

Source: Steam Web API GetNewsForApp (appid 730 = CS2), no auth needed.
This is a REVIEW tool: it lists candidate patches with a gameplay-relevance
guess, but the final "does this change the meta" call is a human/domain
judgment — confirmed major patches get appended to cs2ml/versions.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import requests

from cs2ml.versions import CS2_MAJOR_PATCHES

APPID = 730
NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"

# keywords that suggest a patch materially changes gameplay (map pool, guns,
# economy, hit-reg, animation, engine) rather than cosmetics.
GAMEPLAY_KEYWORDS = [
    "map", "active duty", "weapon", "economy", "price", "damage", "hit",
    "recoil", "animation", "movement", "jump", "grenade", "smoke", "flash",
    "molotov", "engine", "tick", "plant", "defuse", "buy menu", "armor",
    "pistol", "rifle", "awp", "round", "bomb", "penetration", "tagging",
    "accuracy", "bhop", "bunny", "spread", "reload", "stamina",
]


def fetch_news(count: int = 300) -> list[dict]:
    r = requests.get(NEWS_URL, params={
        "appid": APPID, "count": count, "maxlength": 600, "format": "json",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["appnews"]["newsitems"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="only show updates after this date (default: newest known in versions.py)")
    args = ap.parse_args()

    latest_known = max(p[0] for p in CS2_MAJOR_PATCHES)
    since = args.since or latest_known
    print(f"latest known major patch: {latest_known}; scanning Steam news for newer updates\n")

    items = fetch_news()
    seen = 0
    for it in items:
        d = datetime.fromtimestamp(it["date"], tz=timezone.utc).strftime("%Y-%m-%d")
        if d <= since:
            continue
        title = (it.get("title") or "").strip()
        text = (it.get("contents") or "").lower()
        flagged = any(k in text for k in GAMEPLAY_KEYWORDS)
        seen += 1
        flag = "  [gameplay?]" if flagged else ""
        print(f"{d}{flag}  {title}")
    if not seen:
        print(f"no updates newer than {since}")


if __name__ == "__main__":
    main()
