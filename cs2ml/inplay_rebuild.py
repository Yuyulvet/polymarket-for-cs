"""Versioned, restartable offline demo rebuild with per-map audit evidence.

No network, model promotion or trading. Raw demos and legacy caches are read
only. Source file size/mtime are change guards, NOT content hashes of demos.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import pandas as pd

from . import config
from .duel import discover_demos
from .inround import compute_inround
from .inround_model import prepare_events
from .map1_data import first_ct_is_ct, path_key, roster_key, terminal_score, utc
from .midround import compute_midround, _validate_cache


VERSION = "inplay_rebuild_audit_v1"
CODE_FILES = ("inplay_rebuild.py", "inround.py", "inround_model.py", "midround.py", "map1_data.py")
ARTIFACTS = ("inround_events", "round_states", "midround_v2", "map_labels")


def _json(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    p = Path(path).resolve()
    stat = p.stat()
    return {"path": str(p), "map_id": path_key(p), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def validate_map_label(rounds, completion):
    """Recompute the complete ledger; never infer its label from sparse events."""
    if not completion.get("complete"):
        raise ValueError("extractor_map_incomplete:" + str(completion.get("reason_code")))
    required = {"demo_path", "map_name", "round_num", "ct_roster", "t_roster", "winner_side"}
    if required - set(rounds) or rounds.empty:
        raise ValueError("missing_complete_round_ledger")
    r = rounds.sort_values("round_num")
    if r.round_num.tolist() != list(range(1, len(r) + 1)):
        raise ValueError("noncontiguous_round_ledger")
    if len(r) != completion.get("n_rounds"):
        raise ValueError("raw_and_retained_round_count_disagree")
    if r.demo_path.map(path_key).nunique() != 1 or r.map_name.nunique() != 1:
        raise ValueError("mixed_map_identity")
    first = r.iloc[0]
    a, b = sorted([roster_key(first.ct_roster), roster_key(first.t_roster)])
    if set(a.split(",")) & set(b.split(",")):
        raise ValueError("overlapping_rosters")
    scores = {a: 0, b: 0}
    first_ct, first_t = roster_key(first.ct_roster), roster_key(first.t_roster)
    for row in r.itertuples(index=False):
        ct, tt = roster_key(row.ct_roster), roster_key(row.t_roster)
        if {ct, tt} != {a, b}:
            raise ValueError("roster_changed")
        if ct != (first_ct if first_ct_is_ct(row.round_num) else first_t):
            raise ValueError("unsupported_side_schedule")
        side = str(row.winner_side).upper().replace("TERRORIST", "T")
        if side not in {"CT", "T"}:
            raise ValueError("unknown_round_winner")
        # These are past-only features, independently verified against prior rows.
        for name, expected in (("score_a_before", scores[a]), ("score_b_before", scores[b]),
                               ("ct_is_a", int(ct == a))):
            if not hasattr(row, name) or getattr(row, name) != expected:
                raise ValueError("invalid_past_only_score_or_side")
        scores[ct if side == "CT" else tt] += 1
        if terminal_score(scores[a], scores[b]) and row.round_num != len(r):
            raise ValueError("rounds_after_terminal_score")
    if not terminal_score(scores[a], scores[b]):
        raise ValueError("incomplete_or_nonstandard_map")
    winner = a if scores[a] > scores[b] else b
    for key, value in (("roster_a", a), ("roster_b", b), ("score_a", scores[a]),
                       ("score_b", scores[b]), ("winner_roster", winner)):
        if completion.get(key) != value:
            raise ValueError("extractor_completion_disagrees:" + key)
    return {"demo_path": first.demo_path, "map_id": path_key(first.demo_path),
            "match_id": Path(first.demo_path).parent.name, "map_name": first.map_name,
            "roster_a": a, "roster_b": b, "winner_roster": winner,
            "score_a": scores[a], "score_b": scores[b], "n_rounds": len(r),
            "complete": True, "overtime": len(r) > 24, "n_overtime_rounds": max(0, len(r)-24)}


def compare_history(label, history):
    """History supplies chronology only; disagreement is an explicit quarantine."""
    if history is None:
        return "missing_chronology"
    try:
        if utc(history.get("available_at")) <= utc(history.get("start_at")):
            return "invalid_chronology"
    except (ValueError, TypeError):
        return "invalid_chronology"
    expected = {"roster_a": label["roster_a"], "roster_b": label["roster_b"],
                "round_wins_a": label["score_a"], "round_wins_b": label["score_b"],
                "rounds": label["n_rounds"], "map_name": label["map_name"],
                "match_id": label["match_id"]}
    if any(history.get(k) != v for k, v in expected.items()):
        return "new_ledger_disagrees_with_history"
    return "matched"


def _write_frame(folder, name, frame, result):
    frame = frame.copy()
    frame.attrs = {}  # Per-demo audits belong in audit.json, not pandas attrs.
    target = folder / (name + ".parquet")
    with target.open("xb") as handle:
        frame.to_parquet(handle, index=False)
    result["artifacts"][name] = {"rows": len(frame), "sha256": _hash(target)}


def rebuild_one(source, folder, history=None, *, extract=compute_inround, mid_extract=compute_midround):
    """One durable shard; errors are visible and do not silently fill from caches."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    result = {"source": source, "status": "excluded", "artifacts": {}, "errors": {},
              "extraction": {}, "map_eligible": False, "chronology": "not_checked"}
    started = time.monotonic()
    if fingerprint(source["path"]) != source:
        result["errors"]["source"] = "source_changed_before_parse"
        _json(folder / "audit.json", result)
        return result
    try:
        parsed = extract(source["path"], audit=result["extraction"])
        if parsed is not None:
            raw, rounds = parsed
            events, audit = prepare_events(raw, rounds)
            result["events"] = audit
            if not events.empty:
                _write_frame(folder, "inround_events", events, result)
            else:
                result["errors"]["events"] = "no_nonterminal_valid_events"
            if not rounds.empty:
                _write_frame(folder, "round_states", rounds, result)
                result["status"] = "round_data_only"
                try:
                    label = validate_map_label(rounds, result["extraction"].get("map_completion", {}))
                    result["chronology"] = compare_history(label, history)
                    label["history_check"] = result["chronology"]
                    # Complete maps without chronology remain useful raw evidence.
                    # Evaluator can only use a map with a matched history check.
                    _write_frame(folder, "map_labels", pd.DataFrame([label]), result)
                    result["map_eligible"] = result["chronology"] == "matched"
                    result["status"] = "map_complete"
                except ValueError as exc:
                    result["errors"]["map_label"] = str(exc)
        else:
            result["errors"]["events"] = result["extraction"].get("reason_code", "no_event_data")
    except Exception as exc:
        result["errors"]["events"] = f"{type(exc).__name__}: {exc}"
    try:
        mid = mid_extract(Path(source["path"]))
        if mid is not None and not mid.empty:
            _validate_cache(mid)
            _write_frame(folder, "midround_v2", mid, result)
        else:
            result["errors"]["midround"] = "no_valid_30second_states"
    except Exception as exc:
        result["errors"]["midround"] = f"{type(exc).__name__}: {exc}"
    if fingerprint(source["path"]) != source:
        result["status"] = "source_changed"
        result["map_eligible"] = False
        result["errors"]["source"] = "source_changed_during_parse"
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _json(folder / "audit.json", result)
    return result


def _verify_shard(folder, source):
    audit_path = folder / "audit.json"
    if not audit_path.is_file():
        raise ValueError(f"Incomplete shard retained for inspection: {folder}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["source"] != source or fingerprint(source["path"]) != source:
        raise ValueError("Resume input fingerprint differs")
    for name, details in audit["artifacts"].items():
        if name not in ARTIFACTS or _hash(folder / (name + ".parquet")) != details["sha256"]:
            raise ValueError("Resume artifact integrity failure")
    return audit


def aggregate(directory, entries, audits):
    counts = Counter(audit["status"] for audit in audits)
    reasons = Counter()
    for audit in audits:
        reasons.update(f"{scope}:{reason}" for scope, reason in audit["errors"].items())
    report = {"version": VERSION, "status": "completed", "input_maps": len(entries),
              "input_gib": sum(x["size"] for x in entries) / 1024**3,
              "map_statuses": dict(counts), "exclusion_reasons": dict(reasons),
              "map_eligible_with_chronology": sum(x["map_eligible"] for x in audits),
              "chronology_checks": dict(Counter(x["chronology"] for x in audits)),
              "artifact_rows": {}, "artifacts": {},
              "timing_basis": "demo_tick_not_live_receipt; history_series_result_availability_proxy",
              "live_trading_enabled": False, "execution_pnl": None}
    for name in ARTIFACTS:
        frames = []
        for index, audit in enumerate(audits):
            if name in audit["artifacts"] and audit["status"] != "source_changed":
                frames.append(pd.read_parquet(directory / "shards" / f"{index:05d}" / (name + ".parquet")))
        if frames:
            frame = pd.concat(frames, ignore_index=True)
            _write_frame(directory, name, frame, report)
            report["artifact_rows"][name] = len(frame)
    _json(directory / "audit.json", report)
    return report


def run(paths, directory, history_path, *, workers=1, resume=False, selection=None):
    directory, history_path = Path(directory), Path(history_path).resolve()
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2 to limit parser memory pressure")
    if not paths:
        raise ValueError("No input demos")
    entries = [fingerprint(p) for p in paths]
    if len({x["map_id"] for x in entries}) != len(entries):
        raise ValueError("Duplicate input demo paths")
    h = pd.read_parquet(history_path)
    h["map_id"] = h.map_id.map(path_key)
    if h.map_id.duplicated().any():
        raise ValueError("Duplicate map chronology")
    history = h.set_index("map_id").to_dict("index")
    manifest = {"version": VERSION, "inputs": entries, "history_path": str(history_path),
                "selection": selection,
                "history_sha256": _hash(history_path),
                "code_sha256": {name: _hash(Path(__file__).parent / name) for name in CODE_FILES},
                "runtime": {"python": sys.version, "packages": {name: version(name) for name in
                            ("demoparser2", "pandas", "numpy", "pyarrow")}},
                "demo_fingerprint_basis": "absolute_path_size_mtime_ns_not_content_hash"}
    if resume:
        existing = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Resume manifest/code/history differs; use a new output directory")
        if (directory / "audit.json").exists():
            raise FileExistsError("Completed rebuild; existing artifacts remain unchanged")
        if any((directory / (name + ".parquet")).exists() for name in ARTIFACTS):
            raise FileExistsError("Incomplete aggregation retained; use a new directory")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory / "manifest.json", manifest)
    audits = [None] * len(entries)
    pending = []
    for index, entry in enumerate(entries):
        folder = directory / "shards" / f"{index:05d}"
        if folder.exists():
            audits[index] = _verify_shard(folder, entry)
        else:
            pending.append((index, entry, folder, history.get(entry["map_id"])))
    print(f"Rebuild {len(entries)} maps; {len(pending)} pending; {workers} worker(s)", flush=True)
    def record(index, audit):
        audits[index] = audit
        line = {"index": index, "map_id": entries[index]["map_id"], "status": audit["status"],
                "map_eligible": audit["map_eligible"], "elapsed_seconds": audit.get("elapsed_seconds"),
                "errors": audit["errors"]}
        with (directory / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            handle.flush()
        print(f"[{sum(x is not None for x in audits)}/{len(entries)}] {Path(entries[index]['path']).name}: "
              f"{audit['status']} {audit.get('elapsed_seconds')}s", flush=True)
    if workers == 1:
        for index, entry, folder, old in pending:
            record(index, rebuild_one(entry, folder, old))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(rebuild_one, entry, folder, old): index
                       for index, entry, folder, old in pending}
            for future in as_completed(futures):
                record(futures[future], future.result())
    return aggregate(directory, entries, audits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--history", type=Path, default=config.DATA_DIR / "map1/history.parquet")
    parser.add_argument("--demo", type=Path, action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--history-only", action="store_true", help="Only maps with existing chronology; selection is recorded")
    args = parser.parse_args(argv)
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    paths = args.demo or discover_demos()
    candidates = list(paths)
    excluded = []
    if args.history_only:
        keys = set(pd.read_parquet(args.history).map_id.map(path_key))
        excluded = [{"map_id": path_key(p), "reason": "outside_existing_clean_chronology"}
                    for p in paths if path_key(p) not in keys]
        paths = [p for p in paths if path_key(p) in keys]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        excluded += [{"map_id": path_key(p), "reason": "explicit_limit"} for p in paths[args.limit:]]
        paths = paths[:args.limit]
    directory = args.output_dir or config.DATA_DIR / "map1/rebuild" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    selection = {"candidate_maps": len(candidates), "selected_maps": len(paths),
                 "history_only": args.history_only, "excluded": excluded}
    report = run(paths, directory, args.history, workers=args.workers, resume=args.resume, selection=selection)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
