"""Phase-7 forward acceptance runner.

Runs the FROZEN scheme (and nothing else) on post-freeze sessions,
appends results to an append-only acceptance ledger, and evaluates the
frozen acceptance criteria:

  PASS  iff  n_map_units >= min_maps
         AND n_triggers_evaluated >= min_triggers
         AND pooled 95% CI lower bound of per-trade net > 0

Hard rules:
  * The ledger is append-only and idempotent: a session_id already
    present is skipped, so restarts and double-runs never double-count.
  * Every entry records the frozen scheme's sha256; if the frozen file
    changes (new freeze round), processing refuses unless --new-round is
    given, which ARCHIVES the old ledger (never deletes) and starts over.
  * wf_* fair sources need --train-rows (pre-freeze era data): the exit
    forecaster is fit once on frozen-era rows and applied forward, never
    refit on acceptance data.
  * Verdicts are recomputed from the full ledger on every run; there is
    no stored verdict that could drift from the entries.

Paper-only: replays labels, sends no orders, touches no wallet.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import trend_dataset as td
from .trend_market_baseline import load_rows
from .trend_scheme_freeze import (CandidateScheme, fit_exit_forecaster,
                                  load_frozen, run_frozen)

LEDGER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AcceptanceCriteria:
    min_maps: int = 30
    min_triggers: int = 100


# ----------------------------------------------------------------- ledger
def load_ledger(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def write_ledger(path: Path, entries: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def start_new_round(ledger_path: Path) -> None:
    """Archive the old ledger (suffix .archived-<ts>); never delete."""
    ledger_path = Path(ledger_path)
    if not ledger_path.is_file():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = ledger_path.with_name(ledger_path.name + f".archived-{stamp}")
    shutil.copy2(ledger_path, archive)


# ------------------------------------------------------------- processing
def load_audit_exclusions(session_dir: Path,
                          reports_root: Path | None = None) -> set[int]:
    """audit v2 R3: bouts the audit marked incomplete must not produce rows.

    Prefers reports/live_session_<session_id>/audit_v2.json (audit v2,
    docs/live-audit-v2-amendments.md); falls back to audit.json for
    pre-v2 reports. Missing report means no exclusions (caller is
    responsible for session gating).
    """
    session_dir = Path(session_dir)
    root = Path(reports_root) if reports_root else Path("reports")
    report_dir = root / f"live_session_{session_dir.name}"
    report_path = report_dir / "audit_v2.json"
    if not report_path.is_file():
        report_path = report_dir / "audit.json"
    if not report_path.is_file():
        return set()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {int(b) for b in report.get("incomplete_bouts") or []}


def process_session(session_dir: Path, frozen: dict, *,
                    train_rows: pd.DataFrame | None = None,
                    outcome_is_team1: object | None = None,
                    spec_path: Path | None = None,
                    reports_root: Path | None = None) -> dict:
    """Build rows for one session and replay the frozen scheme on them.

    outcome_is_team1: callable outcome_label -> bool（game_heuristic 冻结
    方案需要；在执行器内部按构建出的行对齐）。
    """
    session_dir = Path(session_dir)
    spec_file = Path(spec_path) if spec_path else session_dir / "session_spec.json"
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    excluded = load_audit_exclusions(session_dir, reports_root)
    frame, build_report = td.build_rows(session_dir, spec,
                                        root=session_dir,
                                        excluded_bouts=excluded)
    scheme = CandidateScheme(**frozen["selection"]["winner"]["scheme"])
    team1 = None
    if outcome_is_team1 is not None:
        team1 = pd.Series([bool(outcome_is_team1(o)) for o in frame["outcome"]],
                          index=frame.index)
    if scheme.fair_source in ("wf_ridge_exit", "wf_xgboost_exit"):
        if train_rows is None:
            raise ValueError("wf_scheme_requires_train_rows")
        predict = fit_exit_forecaster(scheme, train_rows)
        result = run_frozen(frame, frozen, team1,
                            p_fair_override=predict(frame))
    else:
        result = run_frozen(frame, frozen, team1)
    report = result["report"]
    mechanisms = set(scheme.mechanisms)
    candidates = frame[frame["decision_kind"].isin(mechanisms)]
    n_triggers = int(len(candidates))
    n_map_units = int(candidates[["session_id", "bout_num"]].drop_duplicates()
                      .shape[0]) if n_triggers else 0
    nets = result["ledger"]["net"].tolist() if len(result["ledger"]) else []
    return {"schema_version": LEDGER_SCHEMA_VERSION,
            "session_id": spec.get("session_id", session_dir.name),
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_sha256": frozen["sha256"],
            "scheme_hash": scheme.scheme_hash(),
            "built_rows": int(len(frame)),
            "build_drop_reasons": build_report.get("drop_reasons", {}),
            "n_map_units": n_map_units, "n_triggers": n_triggers,
            "n_trades": report["n_trades"],
            "skip_reasons": report["skip_reasons"],
            "nets": nets,
            "net_total": report["net_total"],
            "inputs_sha256": build_report.get("inputs_sha256", {})}


# ---------------------------------------------------------------- verdict
def evaluate_acceptance(entries: list[dict],
                        criteria: AcceptanceCriteria) -> dict:
    nets = np.array([n for e in entries for n in e.get("nets", [])])
    n_maps = int(sum(e.get("n_map_units", 0) for e in entries))
    n_triggers = int(sum(e.get("n_triggers", 0) for e in entries))
    n_trades = int(len(nets))
    ci_lower = None
    if n_trades >= 2:
        ci_lower = float(np.mean(nets) - 1.96 * np.std(nets, ddof=1)
                         / np.sqrt(n_trades))
    thresholds_met = (n_maps >= criteria.min_maps
                      and n_triggers >= criteria.min_triggers)
    if not thresholds_met:
        status = "in_progress"
    elif ci_lower is not None and ci_lower > 0:
        status = "passed"
    else:
        status = "failed"
    return {"status": status,
            "criteria": {"min_maps": criteria.min_maps,
                         "min_triggers": criteria.min_triggers},
            "n_sessions": len(entries), "n_map_units": n_maps,
            "n_triggers_evaluated": n_triggers, "n_trades": n_trades,
            "net_total": float(np.sum(nets)) if n_trades else 0.0,
            "net_mean": float(np.mean(nets)) if n_trades else None,
            "net_median": float(np.median(nets)) if n_trades else None,
            "ci95_lower_bound": ci_lower,
            "ci_rule": "mean - 1.96 * std / sqrt(n) (normal approx, diagnostic)"}


def run_acceptance(frozen_path: Path, ledger_path: Path,
                   session_dirs: list[Path], *,
                   criteria: AcceptanceCriteria | None = None,
                   train_rows_paths: list[Path] | None = None,
                   new_round: bool = False) -> dict:
    criteria = criteria or AcceptanceCriteria()
    frozen = load_frozen(frozen_path)
    ledger_path = Path(ledger_path)
    entries = load_ledger(ledger_path)
    if new_round and entries:
        start_new_round(ledger_path)
        entries = []
    if entries and entries[0].get("frozen_sha256") != frozen["sha256"]:
        raise ValueError("frozen_scheme_changed:use_new_round_to_archive_"
                         "old_ledger_and_restart_acceptance")
    train_rows = (load_rows(list(train_rows_paths))
                  if train_rows_paths else None)
    known = {e["session_id"] for e in entries}
    added, skipped_dup = [], []
    for session_dir in session_dirs:
        spec = json.loads((Path(session_dir) / "session_spec.json")
                          .read_text(encoding="utf-8"))
        session_id = spec.get("session_id", Path(session_dir).name)
        if session_id in known:
            skipped_dup.append(session_id)
            continue
        # 注意：CLI 路径不支持 game_heuristic 的 outcome->team 映射；
        # 需要映射的冻结方案请走 Python API (process_session)。
        entry = process_session(session_dir, frozen, train_rows=train_rows)
        entries.append(entry)
        added.append(session_id)
        known.add(session_id)
    write_ledger(ledger_path, entries)
    verdict = evaluate_acceptance(entries, criteria)
    return {"verdict": verdict, "added_sessions": added,
            "skipped_duplicates": skipped_dup,
            "ledger_path": str(ledger_path),
            "frozen_sha256": frozen["sha256"]}


# ------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--train-rows", type=Path, nargs="+", default=None,
                        help="冻结前时期 rows.parquet（wf_* 方案必填）")
    parser.add_argument("--min-maps", type=int, default=30)
    parser.add_argument("--min-triggers", type=int, default=100)
    parser.add_argument("--new-round", action="store_true",
                        help="冻结方案已换：归档旧账本并开启新验收轮")
    args = parser.parse_args(argv)
    result = run_acceptance(
        args.frozen, args.ledger, args.session_dir,
        criteria=AcceptanceCriteria(min_maps=args.min_maps,
                                    min_triggers=args.min_triggers),
        train_rows_paths=args.train_rows, new_round=args.new_round)
    print(json.dumps(result["verdict"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
