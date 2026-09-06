"""Historical backfill from bo3.gg.

Usage:
    python -m cs2ml.backfill --start 2024-01-01 --end 2026-08-23
    python -m cs2ml.backfill --retry-failed     # re-fetch days recorded as error
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

from . import config, store
from .bo3gg import Bo3Client


def _daterange(start: str, end: str):
    d = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while d <= e:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def backfill(start: str | None = None, end: str | None = None, retry_failed: bool = False) -> None:
    client = Bo3Client()
    conn = store.connect()

    if retry_failed:
        days = [r["date"] for r in conn.execute(
            "SELECT date FROM backfill_days WHERE status='error' ORDER BY date")]
        print(f"retrying {len(days)} failed days")
    else:
        assert start and end, "--start/--end required"
        days = list(_daterange(start, end))
        # resume: skip days already fetched successfully
        done = {r["date"] for r in conn.execute(
            "SELECT date FROM backfill_days WHERE status IN ('ok','empty')")}
        days = [d for d in days if d not in done]
        print(f"{len(days)} days to fetch ({start} .. {end})")

    t0 = time.time()
    n_ok = n_match = 0
    for i, day in enumerate(days, 1):
        try:
            payload = client.matches_for_date(day, "finished")
            matches, teams, tournaments = Bo3Client.parse_day(payload)
            fetched_at = store._now()
            with conn:
                store.upsert_teams(conn, teams, fetched_at)
                store.upsert_tournaments(conn, tournaments)
                for m in matches:
                    store.upsert_match(conn, m, fetched_at)
                store.set_meta(conn, "last_backfill_day", day)
                conn.execute(
                    "INSERT OR REPLACE INTO backfill_days (date, status, match_count, fetched_at) VALUES (?,?,?,?)",
                    (day, "ok" if matches else "empty", len(matches), fetched_at))
            n_ok += 1
            n_match += len(matches)
        except Exception as e:  # noqa: BLE001 - record and continue
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO backfill_days (date, status, match_count, error, fetched_at) VALUES (?,?,?,?,?)",
                    (day, "error", 0, str(e)[:500], store._now()))
            print(f"  {day}: ERROR {e}")
        if i % 25 == 0:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  progress {i}/{len(days)} days, {n_match} matches, {rate:.1f} days/s, eta {int((len(days)-i)/max(rate,1e-9))}s")
    conn.close()
    print(f"done: {n_ok}/{len(days)} days ok, {n_match} matches stored")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    backfill(args.start, args.end, args.retry_failed)
