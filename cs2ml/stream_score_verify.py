"""Manual evidence gate for stream score transitions.

The stream watcher intentionally does not perform OCR.  This module turns a
small, human-reviewed subset of its candidate frames into strict score events.
Only ``verified_score_events.jsonl`` is accepted by the lag analysis.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import unicodedata
from pathlib import Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalize_team(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(ch for ch in value if ch.isalnum() and not unicodedata.combining(ch))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_metadata(session_dir: str | Path) -> tuple[Path, dict]:
    folder = Path(session_dir)
    path = folder / "metadata.json"
    if not path.exists():
        raise ValueError("missing_session_metadata")
    meta = json.loads(path.read_text(encoding="utf-8"))
    required = {"event_id", "team_a", "team_b", "map_number", "market"}
    if int(meta.get("schema_version", 0)) < 2 or not required.issubset(meta.get("binding", {})):
        raise ValueError("legacy_or_incomplete_session_binding")
    return folder, meta


def _save_metadata(folder: Path, meta: dict) -> None:
    (folder / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _quarantine(folder: Path, meta: dict, reason: str, submitted: dict) -> None:
    record = {"quarantined_at": utc_now(), "reason": reason, "submitted": submitted,
              "binding": meta.get("binding")}
    (folder / "quarantine.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["binding_status"] = "quarantined_identity_mismatch"
    meta["eligible_for_inference"] = False
    _save_metadata(folder, meta)


def _known_frame(events: list[dict], frame: int) -> bool:
    for row in events:
        if row.get("type") in {"calibration_frame", "identity_frame"} \
                and int(row.get("frame", -1)) == frame:
            return True
        if row.get("type") == "candidate_visual_change" and frame in {
                int(row.get("before_frame", -1)), int(row.get("after_frame", -1))}:
            return True
    return False


def confirm_binding(session_dir: str | Path, observed_left: str, observed_right: str,
                    observed_map_number: int, evidence_frame: int) -> dict:
    """Confirm that one saved frame shows exactly the bound teams and map."""
    folder, meta = load_metadata(session_dir)
    binding = meta["binding"]
    submitted = {"observed_left": observed_left, "observed_right": observed_right,
                 "observed_map_number": int(observed_map_number),
                 "evidence_frame": int(evidence_frame)}
    expected = {normalize_team(binding["team_a"]), normalize_team(binding["team_b"])}
    observed = {normalize_team(observed_left), normalize_team(observed_right)}
    events = read_jsonl(folder / "events.jsonl")
    if not _known_frame(events, int(evidence_frame)):
        raise ValueError("unknown_evidence_frame")
    if (len(observed) != 2 or observed != expected
            or int(observed_map_number) != int(binding["map_number"])):
        _quarantine(folder, meta, "observed_identity_does_not_match_binding", submitted)
        raise ValueError("observed_identity_does_not_match_binding")

    left_is = ("team_a" if normalize_team(observed_left) == normalize_team(binding["team_a"])
               else "team_b")
    confirmation = {**submitted, "display_left_is": left_is,
                    "confirmed_at": utc_now(), "binding": binding}
    existing_path = folder / "binding_confirmation.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        same_observation = all(existing.get(key) == confirmation.get(key) for key in
                               ("observed_left", "observed_right", "observed_map_number",
                                "evidence_frame", "display_left_is", "binding"))
        if not same_observation:
            raise ValueError("binding_already_confirmed_with_different_evidence")
        return existing
    existing_path.write_text(
        json.dumps(confirmation, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["binding_status"] = "confirmed"
    meta["binding_confirmed_at"] = confirmation["confirmed_at"]
    # This only makes frames eligible for lag measurement, never model/trading input.
    meta["eligible_for_inference"] = False
    _save_metadata(folder, meta)
    return confirmation


def _canonical_scores(left: int, right: int, left_is: str) -> tuple[int, int]:
    return (left, right) if left_is == "team_a" else (right, left)


def _candidate_for_frame(folder: Path, after_frame: int) -> dict:
    candidates = [row for row in read_jsonl(folder / "events.jsonl")
                  if row.get("type") == "candidate_visual_change"
                  and int(row.get("after_frame", -1)) == int(after_frame)]
    if len(candidates) != 1:
        raise ValueError("candidate_after_frame_must_be_unique")
    return candidates[0]


def session_status(session_dir: str | Path) -> dict:
    """Return the small review queue without treating candidates as scores."""
    folder, meta = load_metadata(session_dir)
    verified = {int(row["after_frame"])
                for row in read_jsonl(folder / "verified_score_events.jsonl")}
    candidates = []
    for row in read_jsonl(folder / "events.jsonl"):
        if row.get("type") != "candidate_visual_change":
            continue
        candidates.append({key: row.get(key) for key in
                           ("before_frame", "after_frame", "before_received_at",
                            "received_at", "changed_fraction", "before_path", "after_path")})
        candidates[-1]["verified"] = int(row["after_frame"]) in verified
    return {"binding": meta["binding"], "binding_status": meta.get("binding_status"),
            "candidate_count": len(candidates), "verified_count": len(verified),
            "candidates": candidates}


def verify_score_transition(session_dir: str | Path, after_frame: int,
                            before_left: int, before_right: int,
                            after_left: int, after_right: int) -> dict:
    """Append one verified, continuous, single-round score transition."""
    folder, meta = load_metadata(session_dir)
    if meta.get("binding_status") != "confirmed":
        raise ValueError("session_binding_not_confirmed")
    confirmation_path = folder / "binding_confirmation.json"
    if not confirmation_path.exists():
        raise ValueError("missing_binding_confirmation")
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    candidate = _candidate_for_frame(folder, after_frame)
    binding = meta["binding"]
    for key in ("event_id", "map_number", "market", "team_a", "team_b"):
        if str(candidate.get(key)) != str(binding.get(key)):
            _quarantine(folder, meta, "candidate_binding_mismatch", candidate)
            raise ValueError("candidate_binding_mismatch")

    values = [before_left, before_right, after_left, after_right]
    if any(isinstance(value, bool) or int(value) != value or int(value) < 0
           for value in values):
        raise ValueError("scores_must_be_nonnegative_integers")
    delta_left = int(after_left) - int(before_left)
    delta_right = int(after_right) - int(before_right)
    if sorted((delta_left, delta_right)) != [0, 1]:
        raise ValueError("exactly_one_display_score_must_increase_by_one")

    left_is = confirmation["display_left_is"]
    before_a, before_b = _canonical_scores(int(before_left), int(before_right), left_is)
    after_a, after_b = _canonical_scores(int(after_left), int(after_right), left_is)
    scoring_team = binding["team_a"] if after_a > before_a else binding["team_b"]
    output = folder / "verified_score_events.jsonl"
    prior = read_jsonl(output)
    if any(int(row["after_frame"]) == int(after_frame) for row in prior):
        raise ValueError("score_transition_already_verified")
    if prior:
        latest = prior[-1]
        if (before_a, before_b) != (int(latest["after_score_a"]),
                                    int(latest["after_score_b"])):
            raise ValueError("score_transition_is_not_continuous")
        if str(candidate["received_at"]) <= str(latest["received_at"]):
            raise ValueError("score_transition_time_not_increasing")

    record = {"schema_version": 1, "type": "verified_score_transition",
              **binding, "before_frame": int(candidate["before_frame"]),
              "after_frame": int(candidate["after_frame"]),
              "before_received_at": candidate.get("before_received_at"),
              "received_at": candidate["received_at"],
              "before_score_a": before_a, "before_score_b": before_b,
              "after_score_a": after_a, "after_score_b": after_b,
              "scoring_team": scoring_team,
              "before_path": candidate.get("before_path"),
              "after_path": candidate.get("after_path"),
              "verified_at": utc_now(), "eligible_for_lag_analysis": True,
              "eligible_for_inference": False}
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("confirm-binding")
    bind.add_argument("--session-dir", required=True)
    bind.add_argument("--observed-left", required=True)
    bind.add_argument("--observed-right", required=True)
    bind.add_argument("--observed-map-number", type=int, required=True)
    bind.add_argument("--evidence-frame", type=int, required=True)
    score = sub.add_parser("add-score")
    score.add_argument("--session-dir", required=True)
    score.add_argument("--after-frame", type=int, required=True)
    score.add_argument("--before-left", type=int, required=True)
    score.add_argument("--before-right", type=int, required=True)
    score.add_argument("--after-left", type=int, required=True)
    score.add_argument("--after-right", type=int, required=True)
    status = sub.add_parser("status")
    status.add_argument("--session-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "confirm-binding":
        result = confirm_binding(args.session_dir, args.observed_left, args.observed_right,
                                 args.observed_map_number, args.evidence_frame)
    elif args.command == "add-score":
        result = verify_score_transition(args.session_dir, args.after_frame,
                                         args.before_left, args.before_right,
                                         args.after_left, args.after_right)
    else:
        result = session_status(args.session_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
