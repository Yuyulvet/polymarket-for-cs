"""Unified auto capture session: discover -> pair (fail-closed) -> capture x3 -> audit.

Phase-1 stage-2 entry point. Reuses the three verified collectors; never creates
a fourth capture format:
  - fivee_live.FiveERecorder        (5E event log, HTTP poll)
  - fivee_mqtt.FiveEMqttRecorder    (5E live state, MQTT push)
  - market_raw_capture.capture_event_markets (Polymarket full-depth raw WS)

Modes:
  --mode dry-run   discovery + pairing report only, never starts capture
  --mode once      one discovery pass; start the earliest startable approved pair
  --mode watch     poll discovery every --poll-minutes; start approved pairs as
                   they become startable (one per pass), loop until --hours budget

A pair is approved only via live_auto_discover.pair_candidates gates
(team-name normalization, time tolerance, uniqueness in BOTH directions). Any
ambiguity is recorded in the discovery report and never started. Manual
override: --match-id + --event-id skips discovery but keeps all start gates
(fresh event re-validation, Map 1 Winner presence, map-market binding).

Read-only and paper-only: no orders, no wallet, no private keys.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from curl_cffi import requests as creq

from . import live_auto_discover as disc
from .fivee_live import FiveERecorder, SNAPSHOT_LIMIT
from .fivee_mqtt import BROKER, BROKER_PORT, FiveEMqttRecorder, topic_for
from .fivee_state import BASE as FIVEE_BASE, DETAIL_PATH
from .live_session_audit import audit_session
from .market_raw_capture import capture_event_markets
from .trend_protocol import PROTOCOL_ID, validate_session_spec

GAMMA = "https://gamma-api.polymarket.com"
FIVEE_EVENT_POLL_SECONDS = 2.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_slug(team_key1: str, team_key2: str, plan_ts: float) -> str:
    day = datetime.fromtimestamp(plan_ts, timezone.utc).strftime("%Y%m%d")
    slug = re.sub(r"[^a-z0-9]+", "-", f"{team_key1}-vs-{team_key2}-{day}".casefold()).strip("-")
    return slug[:80]


# ---------------------------------------------------------------- start gates
def refresh_event(request_get, event_id: str, timeout: float = 30.0) -> dict:
    response = request_get(f"{GAMMA}/events/{int(event_id)}",
                           impersonate="chrome", timeout=timeout)
    response.raise_for_status()
    event = response.json()
    if not isinstance(event, dict) or event.get("closed"):
        raise ValueError(f"poly_event_closed_or_invalid:{event_id}")
    return event


def validate_pair_at_start(pair: dict, request_get) -> dict:
    """Re-validate the pairing against a FRESH event payload. Fail closed."""
    event = refresh_event(request_get, pair["poly"]["event_id"])
    title = disc.parse_poly_title(event.get("title") or "")
    if title is None:
        raise ValueError("poly_title_unparseable_at_start")
    key1 = disc.normalize_team(title["team1"])
    key2 = disc.normalize_team(title["team2"])
    manual = not pair["fivee"]["team_key1"] and not pair["fivee"]["team_key2"]
    if not manual and {key1, key2} != {pair["fivee"]["team_key1"],
                                       pair["fivee"]["team_key2"]}:
        raise ValueError("team_identity_changed_since_discovery")
    if manual:  # explicit ids were given: adopt the freshly validated identity
        pair["fivee"]["team_key1"], pair["fivee"]["team_key2"] = key1, key2
        pair["fivee"]["team1"], pair["fivee"]["team2"] = title["team1"], title["team2"]
    map_markets = disc.map_winner_markets(event)
    if not any(m.bout_num == 1 for m in map_markets):
        raise ValueError("map1_winner_market_missing_at_start")
    return event


# ---------------------------------------------------------------- spec build
def _canonical_map(value) -> str:
    value = str(value or "").strip().casefold()
    return value if value.startswith("de_") else ("de_" + value if value else "")


def resolve_bout_map_names(events_path: Path, bout_nums: list[int]) -> dict[int, str]:
    """Most frequent canonical map_name per bout from the captured 5E event log."""
    from collections import Counter
    votes: dict[int, Counter] = {int(b): Counter() for b in bout_nums}
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") != "event":
                continue
            entry = row.get("entry") or {}
            try:
                bout = int(entry.get("bout_num"))
            except (TypeError, ValueError):
                continue
            name = _canonical_map(entry.get("map_name"))
            if bout in votes and re.fullmatch(r"de_[a-z0-9_]+", name):
                votes[bout][name] += 1
    return {bout: counter.most_common(1)[0][0] for bout, counter in votes.items()
            if counter}


def build_session_spec(session_id: str, pair: dict, event: dict,
                       session_dir: Path, root: Path) -> dict:
    rel = lambda p: str(Path(p).resolve().relative_to(root.resolve()))
    map_markets = disc.map_winner_markets(event)
    bout_nums = [m.bout_num for m in map_markets]
    map_names = resolve_bout_map_names(session_dir / "fivee_events.jsonl", bout_nums)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "session_id": session_id,
        "match_id": pair["fivee"]["match_id"],
        "event_id": str(pair["poly"]["event_id"]),
        "teams": [pair["fivee"]["team1"], pair["fivee"]["team2"]],
        "maps": [{
            "bout_num": m.bout_num,
            "map_name": map_names.get(m.bout_num, ""),
            "market_id": m.market_id,
            "market_name": m.market_name,
        } for m in map_markets],
        "sources": {
            "fivee_events": [rel(session_dir / "fivee_events.jsonl")],
            "fivee_states": [rel(session_dir / "fivee_states.jsonl")],
            "market": {
                "format": "raw_depth",
                "events": rel(session_dir / "market" / "events.jsonl"),
                "metadata": rel(session_dir / "market" / "metadata.json"),
                "summary": rel(session_dir / "market" / "summary.json"),
            },
        },
        "pairing": {
            "fivee_plan_utc": disc._utc(pair["fivee"]["plan_ts"]),
            "poly_start_utc": disc._utc(pair["poly"]["start_ts"]),
            "start_skew_seconds": pair["start_skew_seconds"],
            "bo": pair["poly"]["bo"],
            "league": pair["poly"]["league"],
        },
    }


# ---------------------------------------------------------------- capture
def _thread_record(results: dict, key: str, fn) -> None:
    try:
        results[key] = {"exit_code": int(fn() or 0)}
    except Exception as exc:  # never let a collector kill the session silently
        results[key] = {"exit_code": 2, "error_type": type(exc).__name__,
                        "error": str(exc)[:500]}


def run_capture_session(pair: dict, event: dict, session_dir: Path, *,
                        duration_seconds: float, max_bytes: float,
                        request_get=None) -> dict:
    request_get = request_get or creq.get
    session_dir.mkdir(parents=True, exist_ok=True)  # 重启续采:已落盘数据保留
    # market/ 由 capture_event_markets 自己创建/续采（exist_ok 防重）
    match_id = pair["fivee"]["match_id"]
    market_ids = [m.market_id for m in disc.map_winner_markets(event)]
    started_at = utc_now()

    fivee_meta = {
        "schema_version": 2, "match_id": match_id,
        "source": "esports-data.5eplaycdn.com",
        "timing_basis": "local_receive_time; source_update_version_preserved_separately",
        "strict_backtest_policy": "exclude_initial_and_recovery_snapshots",
        "poll_seconds": FIVEE_EVENT_POLL_SECONDS, "snapshot_limit": SNAPSHOT_LIMIT,
        "started_utc": started_at, "session_auto": True,
    }
    (session_dir / "fivee_events.meta.json").write_text(
        json.dumps(fivee_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    mqtt_meta = {
        "schema_version": 1, "match_id": match_id,
        "topic": topic_for(match_id), "broker": BROKER, "broker_port": BROKER_PORT,
        "initial_source": FIVEE_BASE + DETAIL_PATH.format(match_id=match_id),
        "timing_basis": "local_receive_time",
        "strict_backtest_policy": "exclude initial HTTP and first post-reconnect MQTT snapshots",
        "started_utc": started_at, "session_auto": True,
    }
    (session_dir / "fivee_states.meta.json").write_text(
        json.dumps(mqtt_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    results: dict[str, dict] = {}
    fivee_recorder = FiveERecorder(
        match_id, session_dir / "fivee_events.jsonl",
        duration_seconds=duration_seconds, poll_seconds=FIVEE_EVENT_POLL_SECONDS)
    mqtt_recorder = FiveEMqttRecorder(
        match_id, session_dir / "fivee_states.jsonl",
        duration_seconds=duration_seconds)

    threads = [
        threading.Thread(target=_thread_record,
                         args=(results, "fivee_events", fivee_recorder.run), daemon=True),
        threading.Thread(target=_thread_record,
                         args=(results, "fivee_states", mqtt_recorder.run), daemon=True),
        threading.Thread(
            target=_thread_record, args=(results, "market",
                                         lambda: capture_event_markets(
                                             event, pair["poly"]["event_id"], market_ids,
                                             session_dir / "market",
                                             seconds=duration_seconds,
                                             max_bytes=max_bytes)), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    manifest = {
        "session_id": session_dir.name, "started_at": started_at,
        "ended_at": utc_now(), "duration_seconds": duration_seconds,
        "pairing": pair, "market_ids": market_ids, "collectors": results,
        "paper_only": True, "eligible_for_trading": False,
    }
    (session_dir / "session_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def finalize_session(session_dir: Path, pair: dict, event: dict, root: Path,
                     report_dir: Path) -> dict:
    """Build spec (fail-closed) and run the frozen audit."""
    session_id = session_dir.name
    try:
        spec = build_session_spec(session_id, pair, event, session_dir, root)
        validate_session_spec(spec)
    except Exception as exc:
        report = {
            "schema_version": 1, "created_at": utc_now(),
            "protocol_id": PROTOCOL_ID, "paper_only": True,
            "session_id": session_id, "eligible_for_trend_dataset": False,
            "blocking_reasons": [f"session_spec_build_failed:{type(exc).__name__}"],
            "warnings": [], "three_source_overlap_seconds": 0.0,
            "spec_error": str(exc)[:500],
        }
        (session_dir / "session_spec.json").write_text(
            json.dumps(locals().get("spec", {}), ensure_ascii=False, indent=2),
            encoding="utf-8")
    else:
        (session_dir / "session_spec.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        report = audit_session(spec, root)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


# ---------------------------------------------------------------- discovery loop
def startable_pairs(report: dict, *, now: float, lead_seconds: float,
                    max_lateness_seconds: float) -> list[dict]:
    out = []
    for cand in report["paired"]:
        plan_ts = cand["fivee"]["plan_ts"]
        if plan_ts is None:
            continue
        if plan_ts - lead_seconds <= now <= plan_ts + max_lateness_seconds:
            out.append(cand)
    out.sort(key=lambda c: c["fivee"]["plan_ts"])
    return out


def discovery_pass(*, aliases, tolerance_seconds: float, limit: int,
                   pages: int = 4, request_get=None) -> dict:
    request_get = request_get or creq.get
    fivee = disc.fetch_fivee_matches(request_get, pages=pages, limit=limit,
                                     aliases=aliases)
    poly = disc.fetch_poly_events(request_get, limit=limit, aliases=aliases)
    return disc.pair_candidates(fivee, poly, tolerance_seconds=tolerance_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dry-run", "once", "watch"],
                        default="dry-run")
    parser.add_argument("--hours", type=float, default=4.0, help="单次采集时长预算")
    parser.add_argument("--watch-hours", type=float, default=6.0,
                        help="watch 模式总守候时长")
    parser.add_argument("--lead-minutes", type=float, default=20.0,
                        help="开赛前几多分钟启动采集")
    parser.add_argument("--max-lateness-minutes", type=float, default=120.0,
                        help="开赛后最多晚多久仍启动（已开赛场次）")
    parser.add_argument("--tolerance-minutes", type=float, default=45.0,
                        help="5E 与 Polymarket 开赛时间配对容差")
    parser.add_argument("--poll-minutes", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=100,
                        help="每个来源单次拉取条数（Polymarket）")
    parser.add_argument("--pages", type=int, default=4,
                        help="5E 列表翻页数（每页 --page-size 条；列表按赛事分块非时间序）")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--alias-file", type=Path, default=None)
    parser.add_argument("--max-bytes", type=float, default=100_000_000)
    parser.add_argument("--out-root", type=Path, default=Path("data/live_sessions"))
    parser.add_argument("--report-root", type=Path, default=Path("reports"))
    parser.add_argument("--match-id", default=None, help="手动模式：5E match_id")
    parser.add_argument("--event-id", default=None, help="手动模式：Polymarket event_id")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    try:
        aliases = disc.load_aliases(args.alias_file)
    except Exception as exc:
        sys.exit(f"alias_file_invalid:{exc}")

    started_sessions: list[str] = []
    started_match_ids: set[str] = set()
    deadline = time.monotonic() + args.watch_hours * 3600

    def write_discovery(snapshot: dict) -> Path:
        args.out_root.mkdir(parents=True, exist_ok=True)
        path = args.out_root / "discovery_latest.json"
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    while True:
        now = time.time()
        if args.match_id and args.event_id:
            pair = {"fivee": {"match_id": args.match_id, "team1": "", "team2": "",
                              "team_key1": "", "team_key2": "", "plan_ts": now,
                              "format": "", "stage": "", "status": "manual"},
                    "poly": {"event_id": str(args.event_id), "title": "", "start_ts": now,
                             "bo": "", "league": "manual", "live": True, "map_markets": []},
                    "start_skew_seconds": 0.0}
            report = {"paired": [pair], "ambiguous": [],
                      "unmatched_fivee": [], "unmatched_poly": []}
            snapshot = dict(report)  # 手动模式的 pair 已是纯 dict，可直接序列化
        else:
            report = discovery_pass(aliases=aliases,
                                    tolerance_seconds=args.tolerance_minutes * 60,
                                    limit=args.limit, pages=args.pages)
            snapshot = disc.serialize_pairing(report)
        snapshot["mode"] = args.mode
        snapshot["now_utc"] = utc_now()
        path = write_discovery(snapshot)
        print(f"[{utc_now()}] paired={len(report['paired'])} "
              f"ambiguous={len(report['ambiguous'])} "
              f"unmatched_fivee={len(report['unmatched_fivee'])} "
              f"unmatched_poly={len(report['unmatched_poly'])} -> {path}",
              flush=True)

        if args.mode == "dry-run":
            print(json.dumps(snapshot, ensure_ascii=False, indent=2)[:4000])
            return 0

        due = startable_pairs(report, now=now,
                              lead_seconds=args.lead_minutes * 60,
                              max_lateness_seconds=args.max_lateness_minutes * 60)
        due = [c for c in due if c["fivee"]["match_id"] not in started_match_ids]
        if due:
            pair = due[0]
            if pair["fivee"]["team_key1"]:
                session_id = session_slug(pair["fivee"]["team_key1"],
                                          pair["fivee"]["team_key2"],
                                          pair["fivee"]["plan_ts"] or now)
            else:  # manual override: ids are the only stable identity
                tail = re.sub(r"[^a-z0-9]+", "", str(pair["fivee"]["match_id"]).casefold())[-10:]
                session_id = f"manual-{pair['poly']['event_id']}-{tail}"[:80]
            session_dir = args.out_root / session_id
            try:
                event = validate_pair_at_start(pair, creq.get)
            except Exception as exc:
                print(f"[{utc_now()}] start_gate_rejected {session_id}: {exc}",
                      flush=True)
                rejected = {"rejected_at": utc_now(), "session_id": session_id,
                            "pair": pair,
                            "reason": f"{type(exc).__name__}:{exc}"}
                (args.out_root / "start_rejections.jsonl").open("a", encoding="utf-8").write(
                    json.dumps(rejected, ensure_ascii=False) + "\n")
            else:
                print(f"[{utc_now()}] starting {session_id} "
                      f"(5E {pair['fivee']['match_id']} <-> PM {pair['poly']['event_id']})",
                      flush=True)
                run_capture_session(pair, event, session_dir,
                                    duration_seconds=args.hours * 3600,
                                    max_bytes=args.max_bytes)
                audit = finalize_session(session_dir, pair, event, Path.cwd(),
                                         args.report_root / f"live_session_{session_id}")
                started_sessions.append(session_id)
                started_match_ids.add(pair["fivee"]["match_id"])
                print(f"[{utc_now()}] session {session_id} done: "
                      f"eligible={audit['eligible_for_trend_dataset']} "
                      f"blockers={audit['blocking_reasons']}", flush=True)
                if args.mode == "once":
                    return 0 if audit["eligible_for_trend_dataset"] else 2

        if args.mode == "once":
            print(f"[{utc_now()}] no startable approved pair; report at {path}",
                  flush=True)
            return 3
        if time.monotonic() >= deadline:
            print(f"[{utc_now()}] watch budget exhausted; "
                  f"started={started_sessions}", flush=True)
            return 0 if started_sessions else 3
        time.sleep(max(30.0, args.poll_minutes * 60))


if __name__ == "__main__":
    raise SystemExit(main())
