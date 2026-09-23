"""Audit whether a capture session can enter the frozen trend dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .trend_protocol import (MINIMUM_SOURCE_OVERLAP_SECONDS, PROTOCOL_ID,
                             protocol_hash, protocol_payload, validate_session_spec)


NORMAL_MARKET_ENDS = {"deadline", "stop_file"}
# --- v2 amendments (docs/live-audit-v2-amendments.md, effective 2026-09-21) ---
AUDIT_VERSION = 2
# R1: an unclosed segment is acceptable iff a successor session_start in the
# SAME file begins within this gap (designed resume path for crash recovery).
MAX_RESTART_GAP_SECONDS = 300.0
# R2: a post-match clean socket close (connection_closed) counts as a normal
# end iff the market tape covers up to the end of the 5E event stream.
MARKET_COVERAGE_GRACE_SECONDS = 60.0


def _timestamp(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _canonical_map(value) -> str:
    value = str(value or "").strip().casefold()
    if value and not value.startswith("de_"):
        value = "de_" + value
    return value


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_object_required:{path}:{line_number}")
            yield row


def _coverage(times: list[float]) -> dict:
    if not times:
        return {"start": None, "end": None, "seconds": 0.0}
    return {"start": min(times), "end": max(times),
            "seconds": max(0.0, max(times)-min(times))}


def audit_fivee_events(paths: list[Path]) -> dict:
    counts, maps, bouts, times = Counter(), set(), set(), []
    match_ids, round_ones, terminal_maps = set(), set(), set()
    complete_segments = 0
    segments: list[dict] = []
    current: dict | None = None
    rows_outside_segments = 0
    last_match_event: float | None = None
    for path in paths:
        for row in _rows(path):
            counts[row.get("record_type", "unknown")] += 1
            when = _timestamp(row.get("recv_utc"))
            if when is not None:
                times.append(when)
            kind = row.get("record_type")
            if kind == "event" and when is not None:
                last_match_event = when if last_match_event is None \
                    else max(last_match_event, when)
            if kind == "session_start":
                if current is not None:  # prior segment never closed: keep as-is
                    segments.append(current)
                current = {"path": str(path), "start": when, "last": when,
                           "end": False}
                if row.get("match_id"):
                    match_ids.add(str(row["match_id"]))
            elif kind == "session_end":
                if current is not None:
                    current["end"] = True
                    segments.append(current)
                    current = None
                else:
                    rows_outside_segments += 1  # end without an open segment
            elif current is not None and when is not None:
                current["last"] = when
            elif current is None:
                rows_outside_segments += 1  # data before any session_start
            if kind == "poll_ok" and row.get("snapshot_may_be_truncated"):
                counts["truncated_polls"] += 1
            elif kind == "event":
                entry = row.get("entry") or {}
                map_name = _canonical_map(entry.get("map_name"))
                try:
                    bout = int(entry.get("bout_num"))
                except (TypeError, ValueError):
                    bout = None
                if map_name:
                    maps.add(map_name)
                if bout is not None:
                    bouts.add(bout)
                if not row.get("decision_eligible"):
                    counts["ineligible_events"] += 1
                    continue
                counts["eligible_events"] += 1
                try:
                    log = json.loads(entry.get("log_info") or "{}")
                except json.JSONDecodeError:
                    counts["invalid_log_info"] += 1
                    continue
                event_type = str(log.get("type", ""))
                counts[f"eligible_type_{event_type}"] += 1
                if event_type == "1":
                    try:
                        number = int((log.get("round_start") or {}).get("round_num"))
                    except (TypeError, ValueError):
                        number = None
                    if number == 1 and bout is not None:
                        round_ones.add(bout)
                elif event_type == "2":
                    end = log.get("round_end") or {}
                    try:
                        scores = (int(end.get("ct_score")), int(end.get("t_score")))
                    except (TypeError, ValueError):
                        scores = None
                    if scores and bout is not None:
                        high, low = max(scores), min(scores)
                        if (high == 13 and low <= 11) or (high >= 16 and high-low >= 2):
                            terminal_maps.add(bout)
        # a segment must not span files; an unclosed one at EOF stays open
        if current is not None:
            segments.append(current)
            current = None
        complete_segments += sum(1 for s in segments
                                 if s["path"] == str(path) and s["end"])
    restarts_accepted, unrecovered = _classify_segments(segments)
    return {
        "paths": [str(path) for path in paths], "segments": len(paths),
        "complete_segments": complete_segments, "match_ids": sorted(match_ids),
        "maps": sorted(maps), "bouts": sorted(bouts),
        "round_one_bouts": sorted(round_ones), "terminal_bouts": sorted(terminal_maps),
        "counts": dict(counts), "coverage": _coverage(times),
        "last_match_event": last_match_event,
        "segments_v2": [{"path": s["path"], "end": s["end"],
                         "start_utc": _iso(s["start"]), "last_utc": _iso(s["last"])}
                        for s in segments],
        "restarts_accepted": restarts_accepted,
        "unrecovered_incomplete_segments": unrecovered,
        "rows_outside_segments": rows_outside_segments,
    }


def _iso(epoch) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _classify_segments(segments: list[dict]) -> tuple[int, list[dict]]:
    """R1: an unclosed segment is fine iff a same-file successor starts within
    MAX_RESTART_GAP_SECONDS (designed operator-restart resume path)."""
    restarts, unrecovered = 0, []
    for index, segment in enumerate(segments):
        if segment["end"]:
            continue
        successor = segments[index + 1] if index + 1 < len(segments) else None
        gap = None
        accepted = False
        if (successor is not None and successor["path"] == segment["path"]
                and segment["last"] is not None and successor["start"] is not None):
            gap = successor["start"] - segment["last"]
            accepted = gap <= MAX_RESTART_GAP_SECONDS
        if accepted:
            restarts += 1
        else:
            unrecovered.append({"path": segment["path"],
                                "last_utc": _iso(segment["last"]),
                                "restart_gap_seconds": gap})
    return restarts, unrecovered


def audit_fivee_states(paths: list[Path]) -> dict:
    counts, times, match_ids = Counter(), [], set()
    complete_segments = 0
    segments: list[dict] = []
    current: dict | None = None
    rows_outside_segments = 0
    for path in paths:
        for row in _rows(path):
            kind = row.get("record_type", "unknown")
            counts[kind] += 1
            when = _timestamp(row.get("recv_utc"))
            if when is not None:
                times.append(when)
            if kind == "session_start":
                if current is not None:
                    segments.append(current)
                current = {"path": str(path), "start": when, "last": when,
                           "end": False}
                if row.get("match_id"):
                    match_ids.add(str(row["match_id"]))
            elif kind == "session_end":
                if current is not None:
                    current["end"] = True
                    segments.append(current)
                    current = None
                else:
                    rows_outside_segments += 1
            elif current is not None and when is not None:
                current["last"] = when
            elif current is None and kind != "session_start":
                rows_outside_segments += 1
            if kind == "state_snapshot":
                strict = (row.get("decision_eligible") is True
                          and row.get("source_transport") == "mqtt"
                          and row.get("timing_quality") == "mqtt_push_receive_time")
                counts["strict_eligible_snapshots" if strict else "ineligible_snapshots"] += 1
        if current is not None:
            segments.append(current)
            current = None
        complete_segments += sum(1 for s in segments
                                 if s["path"] == str(path) and s["end"])
    restarts_accepted, unrecovered = _classify_segments(segments)
    return {
        "paths": [str(path) for path in paths], "segments": len(paths),
        "complete_segments": complete_segments, "match_ids": sorted(match_ids),
        "counts": dict(counts), "coverage": _coverage(times),
        "segments_v2": [{"path": s["path"], "end": s["end"],
                         "start_utc": _iso(s["start"]), "last_utc": _iso(s["last"])}
                        for s in segments],
        "restarts_accepted": restarts_accepted,
        "unrecovered_incomplete_segments": unrecovered,
        "rows_outside_segments": rows_outside_segments,
    }


def _market_binding(meta: dict, required_market_ids: set[str]) -> tuple[str | None, set[str], set[str]]:
    event_id = meta.get("event_id")
    market_ids, tokens = set(), set()
    binding = meta.get("binding")
    if isinstance(binding, dict):
        event_id = binding.get("event_id", event_id)
        binding_market_id = str(binding.get("market_id", ""))
        if binding_market_id in required_market_ids:
            market_ids.add(binding_market_id)
            tokens.update(str(value) for value in
                          (binding.get("outcome_tokens") or binding.get("tokens") or {}).values())
    for market in meta.get("markets") or []:
        market_id = str(market.get("market_id", ""))
        if market_id in required_market_ids:
            market_ids.add(market_id)
            tokens.update(str(value) for value in (market.get("tokens") or {}).values())
    return None if event_id is None else str(event_id), market_ids, tokens


def audit_market(source: dict, root: Path, required_market_ids: set[str]) -> dict:
    events = _resolve(root, source["events"])
    metadata = _read_json(_resolve(root, source["metadata"]))
    event_id, market_ids, expected_tokens = _market_binding(metadata, required_market_ids)
    times, counts, depth_tokens = [], Counter(), set()
    for row in _rows(events):
        counts[row.get("type", "unknown")] += 1
        when = _timestamp(row.get("received_at") or row.get("recv_utc") or row.get("local_ts"))
        if when is not None:
            times.append(when)
        if source["format"] != "raw_depth" or row.get("type") != "market_raw":
            continue
        try:
            payload = json.loads(row.get("raw") or "null")
        except json.JSONDecodeError:
            counts["invalid_raw_messages"] += 1
            continue
        messages = payload if isinstance(payload, list) else [payload]
        for message in messages:
            if not isinstance(message, dict):
                continue
            if isinstance(message.get("bids"), list) and isinstance(message.get("asks"), list):
                token = message.get("asset_id")
                if token is not None:
                    depth_tokens.add(str(token))
    exit_reason = None
    if source["format"] == "raw_depth":
        summary = _read_json(_resolve(root, source["summary"]))
        exit_reason = summary.get("exit_reason")
    return {
        "format": source["format"], "events": str(events),
        "event_id": event_id, "market_ids": sorted(market_ids),
        "expected_tokens": sorted(expected_tokens), "depth_tokens": sorted(depth_tokens),
        "has_full_depth_for_expected_tokens": bool(expected_tokens) and expected_tokens <= depth_tokens,
        "exit_reason": exit_reason, "normal_end": exit_reason in NORMAL_MARKET_ENDS,
        "counts": dict(counts), "coverage": _coverage(times),
    }


def _resolve(root: Path, value: str) -> Path:
    root = root.resolve()
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source_outside_root:{path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _market_normal_end_v2(market: dict, event: dict) -> bool:
    """R2: deadline/stop_file as v1, plus a post-match clean socket close whose
    tape covers the last observed match event (verifiable completeness).

    Compared against last_match_event (last record_type=event row), NOT the
    raw coverage end: recorders keep polling and write session_end at the
    hours-deadline, long after the final round, and the market socket is
    expected to close once the match goes quiet."""
    if market["normal_end"]:
        return True
    if market.get("exit_reason") != "connection_closed":
        return False
    market_end = market["coverage"].get("end")
    event_end = event.get("last_match_event") or event["coverage"].get("end")
    return (market_end is not None and event_end is not None
            and market_end >= event_end - MARKET_COVERAGE_GRACE_SECONDS)


def _bout_status(expected_bouts: set[int], event: dict) -> dict[int, str]:
    """R3: per-bout observation status (see docs/live-audit-v2-amendments.md)."""
    observed = set(event["bouts"])
    status = {}
    for bout in sorted(expected_bouts):
        if bout not in observed:
            status[bout] = "unobserved"
        elif bout not in set(event["round_one_bouts"]):
            status[bout] = "incomplete_no_round_one"
        elif bout not in set(event["terminal_bouts"]):
            status[bout] = "incomplete_no_terminal"
        else:
            status[bout] = "complete"
    return status


def audit_session(spec: dict, root: Path,
                  audit_version: int = AUDIT_VERSION) -> dict:
    validate_session_spec(spec)
    sources = spec["sources"]
    event_paths = [_resolve(root, value) for value in sources["fivee_events"]]
    state_paths = [_resolve(root, value) for value in sources["fivee_states"]]
    event = audit_fivee_events(event_paths)
    state = audit_fivee_states(state_paths)
    expected_market_ids = {str(item["market_id"]) for item in spec["maps"]}
    market = audit_market(sources["market"], root, expected_market_ids)
    market["normal_end_v2"] = _market_normal_end_v2(market, event)

    # ---- v1 verdict (frozen for comparison, reported as v1_blocking_reasons)
    v1_blockers: list[str] = []
    if event["complete_segments"] != event["segments"]:
        v1_blockers.append("fivee_event_segment_missing_session_end")
    if event["counts"].get("eligible_events", 0) == 0:
        v1_blockers.append("no_strict_eligible_fivee_events")
    if not state_paths:
        v1_blockers.append("fivee_mqtt_state_source_missing")
    elif state["complete_segments"] != state["segments"]:
        v1_blockers.append("fivee_state_segment_missing_session_end")
    if state["counts"].get("strict_eligible_snapshots", 0) == 0:
        v1_blockers.append("no_strict_eligible_mqtt_snapshots")
    if not market["normal_end"]:
        v1_blockers.append("market_capture_missing_normal_end")
    if not market["has_full_depth_for_expected_tokens"]:
        v1_blockers.append("market_full_depth_missing")
    if event["match_ids"] != [spec["match_id"]]:
        v1_blockers.append("fivee_event_match_identity_mismatch")
    if state_paths and state["match_ids"] != [spec["match_id"]]:
        v1_blockers.append("fivee_state_match_identity_mismatch")
    if market["event_id"] != str(spec["event_id"]):
        v1_blockers.append("market_event_identity_mismatch")
    if not expected_market_ids <= set(market["market_ids"]):
        v1_blockers.append("expected_map_market_missing")
    expected_bouts = {int(item["bout_num"]) for item in spec["maps"]}
    if not expected_bouts <= set(event["round_one_bouts"]):
        v1_blockers.append("capture_did_not_observe_round_one_live")
    if not expected_bouts <= set(event["terminal_bouts"]):
        v1_blockers.append("capture_did_not_observe_terminal_round_live")

    # ---- v2 verdict (docs/live-audit-v2-amendments.md)
    bout_status = _bout_status(expected_bouts, event)
    incomplete_bouts = sorted(b for b, s in bout_status.items()
                              if s.startswith("incomplete"))
    blockers = list(v1_blockers)
    if audit_version >= 2:
        blockers = [b for b in blockers
                    if b not in ("fivee_event_segment_missing_session_end",
                                 "fivee_state_segment_missing_session_end",
                                 "market_capture_missing_normal_end",
                                 "capture_did_not_observe_round_one_live",
                                 "capture_did_not_observe_terminal_round_live")]
        # R1: operator-restart resume (only unrecovered incomplete segments block;
        # stray rows outside any session segment block too — data we cannot
        # attribute to a started session is not evidence of coverage)
        if event["unrecovered_incomplete_segments"] or event["rows_outside_segments"]:
            blockers.append("fivee_event_segment_missing_session_end")
        if (state_paths and (state["unrecovered_incomplete_segments"]
                             or state["rows_outside_segments"])):
            blockers.append("fivee_state_segment_missing_session_end")
        # R2: post-match clean close with verifiable tape completeness
        if not market["normal_end_v2"]:
            blockers.append("market_capture_missing_normal_end")
        # R3: session needs at least one fully-observed bout; incomplete bouts
        # are excluded from the trend dataset at the row level.
        if not any(s == "complete" for s in bout_status.values()):
            blockers.append("no_complete_bout")

    coverages = [event["coverage"], state["coverage"], market["coverage"]]
    if all(item["start"] is not None and item["end"] is not None for item in coverages):
        overlap = max(0.0, min(item["end"] for item in coverages)
                      - max(item["start"] for item in coverages))
    else:
        overlap = 0.0
    if overlap < MINIMUM_SOURCE_OVERLAP_SECONDS:
        blockers.append("insufficient_three_source_overlap")
    warnings = []
    if event["counts"].get("truncated_polls", 0):
        warnings.append("fivee_event_snapshot_limit_reached_review_for_gaps")
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_hash(),
        "audit_version": audit_version,
        "paper_only": True, "session_id": spec["session_id"],
        "eligible_for_trend_dataset": not blockers,
        "blocking_reasons": sorted(set(blockers)),
        "v1_blocking_reasons": sorted(set(v1_blockers)),
        "bout_status": {str(k): v for k, v in bout_status.items()},
        "incomplete_bouts": incomplete_bouts,
        "restarts_accepted": event["restarts_accepted"] + state["restarts_accepted"],
        "warnings": warnings,
        "three_source_overlap_seconds": overlap,
        "sources": {"fivee_events": event, "fivee_states": state, "market": market},
        "protocol": protocol_payload(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = _read_json(args.spec)
    report = audit_session(spec, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output),
                      "eligible": report["eligible_for_trend_dataset"],
                      "blocking_reasons": report["blocking_reasons"]}, ensure_ascii=False))
    return 0 if report["eligible_for_trend_dataset"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
