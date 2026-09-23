"""CLI for the Map 1 research -> public recording -> paper settlement pilot.

python -m cs2ml.map1 prepare
python -m cs2ml.map1 inspect --event EVENT_ID
python -m cs2ml.map1 record --context context.json --count 2 --interval 3
python -m cs2ml.map1 replay --snapshots snapshots.jsonl --resolutions resolutions.jsonl
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import time

import pandas as pd

from . import config
from .map1_data import FEATURES, build_features, features_at, load_history, utc
from .map1_eligibility import MIN_ROSTER_HISTORY, roster_history_eligibility
from .map1_market import bind_market, fetch_book, fetch_event, proposal, validate_context
from .map1_model import evaluation_report, fit_model, training_rows, walk_forward
from .map1_paper import replay

DEFAULT_DIR = config.DATA_DIR / "map1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare(output: Path, data_dir: Path, min_train: int, lead_seconds: float, publication_lag: float) -> dict:
    history, audit = load_history(data_dir, publication_lag)
    print(f"Usable maps: {len(history)}; Map 1: {audit['usable_map1']}; overtime: {audit['overtime_maps']}", flush=True)
    features = build_features(history, lead_seconds)
    print(f"Built {len(features)} time-aware Map 1 feature rows; evaluating ...", flush=True)
    predictions = walk_forward(features, min_train=min_train)
    report = evaluation_report(predictions, audit)
    report.update(schema_version=1, generated_at=now(), lead_seconds=lead_seconds,
                  min_train=min_train, feature_columns=FEATURES)
    output.mkdir(parents=True, exist_ok=True)
    history.to_parquet(output / "history.parquet", index=False)
    features.to_parquet(output / "features.parquet", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    write_json(output / "validation.json", report)
    return report


def capture_once(context: dict, history: pd.DataFrame, features: pd.DataFrame,
                 min_train: int = 60, min_history: int = MIN_ROSTER_HISTORY) -> dict:
    started = now()
    result = {"schema_version": 1, "event_id": str(context.get("event_id", "")),
              "context": context, "request_started_at": started, "received_at": started,
              "status": "blocked", "model_target": "map1_winner"}
    try:
        c = validate_context(context, started)
        event = fetch_event(c["event_id"])
        result["event_metadata"] = event
        at = now()
        binding = bind_market(event, c, at)
        result["binding"] = binding
        # Preserve market observations even when the model's roster coverage fails.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {s: pool.submit(fetch_book, binding[f"token_{s}"]) for s in ("a", "b")}
            books = {s: future.result() for s, future in futures.items()}
        result.update(books=books, received_at=now())
        train = training_rows(features, at)
        if len(train) < min_train or train.y.nunique() < 2:
            raise ValueError("insufficient_training_history")
        f, coverage = features_at(history, c["roster_a"], c["roster_b"], c["map_name"], at)
        result["coverage"] = coverage
        eligibility = roster_history_eligibility(coverage, min_history)
        if not eligibility["eligible"]:
            raise ValueError(eligibility["reason"])
        model = fit_model(train)
        p = float(model.predict_proba(pd.DataFrame([f])[FEATURES].to_numpy())[0, 1])
        result.update(p_model=p, feature_values=f, model_fitted_at=at,
                      train_max_available_at=train.available_at.max().isoformat(), n_train=len(train))
        result.update(books=books, received_at=now(), status="ready")
        # Revalidate after network/model latency; starting mid-request closes the window.
        validate_context(context, result["received_at"])
        decision = proposal(result)
        result["decision"] = decision
    except Exception as exc:
        result.update(received_at=now(), status="blocked", reason=f"{type(exc).__name__}: {exc}")
    return result


def read_jsonl(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Build clean history and conditional offline validation")
    prep.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    prep.add_argument("--output", type=Path, default=DEFAULT_DIR)
    prep.add_argument("--min-train", type=int, default=60)
    prep.add_argument("--lead-seconds", type=float, default=60)
    prep.add_argument("--publication-lag", type=float, default=300)
    inspect = sub.add_parser("inspect", help="Read a public event's Map 1 metadata; no orders")
    inspect.add_argument("--event", required=True)
    record = sub.add_parser("record", help="Record bounded public snapshots using confirmed context")
    record.add_argument("--context", type=Path, required=True)
    record.add_argument("--data", type=Path, default=DEFAULT_DIR)
    record.add_argument("--out", type=Path, default=DEFAULT_DIR / "snapshots.jsonl")
    record.add_argument("--count", type=int, default=1)
    record.add_argument("--interval", type=float, default=3)
    paper = sub.add_parser("replay", help="Replay delayed paper fills and supplied final payouts")
    paper.add_argument("--snapshots", type=Path, required=True)
    paper.add_argument("--resolutions", type=Path)
    paper.add_argument("--out", type=Path, default=DEFAULT_DIR / "paper_result.json")
    paper.add_argument("--shares", type=float, default=10)
    paper.add_argument("--cash", type=float, default=1000)
    paper.add_argument("--min-edge", type=float, default=.05)
    paper.add_argument("--latency", type=float, default=.5)
    paper.add_argument("--max-book-age", type=float, default=10)
    args = ap.parse_args()
    if args.command == "prepare":
        report = prepare(args.output, args.data_dir, args.min_train, args.lead_seconds, args.publication_lag)
        print(json.dumps({"report": str(args.output / "validation.json"),
                          "metrics": {k: {n: {q: x for q, x in v.items() if q != "calibration"}
                                           for n, v in m.items()} for k, m in report["metrics"].items()},
                          "execution_status": report["execution_status"]}, ensure_ascii=False, indent=2))
    elif args.command == "inspect":
        event = fetch_event(args.event)
        markets = [m for m in event.get("markets", []) if m.get("groupItemTitle") == "Map 1 Winner"]
        fields = ("id", "question", "conditionId", "outcomes", "clobTokenIds", "description",
                  "gameStartTime", "active", "closed", "acceptingOrders", "feeSchedule", "secondsDelay")
        print(json.dumps({"event_id": event.get("id"), "title": event.get("title"),
                          "markets": [{k: m.get(k) for k in fields} for m in markets]}, ensure_ascii=False, indent=2))
    elif args.command == "record":
        if not 1 <= args.count <= 1000 or not 0 < args.interval <= 60:
            ap.error("count must be 1..1000 and interval must be in (0, 60] seconds")
        context = json.loads(args.context.read_text(encoding="utf-8"))
        history = pd.read_parquet(args.data / "history.parquet")
        features = pd.read_parquet(args.data / "features.parquet")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for i in range(args.count):
            snapshot = capture_once(context, history, features)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps({"status": snapshot["status"], "reason": snapshot.get("reason"),
                              "decision": snapshot.get("decision"), "out": str(args.out)}, ensure_ascii=False), flush=True)
            if i + 1 < args.count:
                time.sleep(args.interval)
    else:
        result = replay(read_jsonl(args.snapshots), read_jsonl(args.resolutions), args.cash,
                        args.shares, args.min_edge, args.latency, args.max_book_age)
        write_json(args.out, result)
        print(json.dumps({k: result[k] for k in ("mode", "cash", "realized_pnl", "open_cost_basis", "pending_count")}, indent=2))


if __name__ == "__main__":
    main()
