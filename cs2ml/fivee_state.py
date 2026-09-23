"""Record 5EPlay live CS2 state snapshots with local receive timing.

The match-detail snapshot contains the same player rows rendered by the 5E live
page: cash, HP, one displayed weapon, armour, defuse kit, C4 and K/D/A. It is
not a complete inventory feed and must not be labelled as total equipment
value. Raw match payloads are retained so field interpretation can be revised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from curl_cffi import requests


BASE = "https://esports-data.5eplaycdn.com"
DETAIL_PATH = "/v1/api/csgo/matches/{match_id}/data"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _integer(value) -> int | None:
    try:
        if value in (None, "") or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _flag(value) -> bool | None:
    """Decode the 5E 1/2 enum while preserving blank/unknown values."""
    parsed = _integer(value)
    if parsed == 1:
        return True
    if parsed in (0, 2):
        return False
    return None


def _opposite(side: str | None) -> str | None:
    return {"CT": "T", "T": "CT"}.get(side)


def _current_side(first_half_side: str | None, round_number: int | None) -> str | None:
    if first_half_side not in ("CT", "T") or round_number is None:
        return None
    if 1 <= round_number <= 12:
        return first_half_side
    if 13 <= round_number <= 24:
        return _opposite(first_half_side)
    # Do not guess overtime side order from a first-half field.
    return None


def _player_summary(player: dict) -> dict:
    return {
        "id": player.get("id") or None,
        "name": player.get("name") or None,
        "hp": _integer(player.get("hp")),
        "money": _integer(player.get("money")),
        # 5E renders this as one weapon icon; it is not the full inventory.
        "display_weapon": player.get("weapon") or None,
        "display_weapon_logo": player.get("weapon_logo") or None,
        "helmet": _flag(player.get("helmet")),
        "kevlar": _flag(player.get("kevlar")),
        "defuse_kit": _flag(player.get("has_defusekit")),
        "c4": _flag(player.get("c4")),
        "kills": _integer(player.get("kill")),
        "deaths": _integer(player.get("death")),
        "assists": _integer(player.get("assist")),
    }


def _team_summary(
    label: str, info: dict, stats: dict, players: list[dict], round_number: int | None,
) -> dict:
    normalized = sorted(
        (_player_summary(player) for player in players if isinstance(player, dict)),
        key=lambda player: (player["id"] or "", player["name"] or ""),
    )
    money = [player["money"] for player in normalized if player["money"] is not None]
    hp = [player["hp"] for player in normalized if player["hp"] is not None]
    weapons = Counter(
        player["display_weapon"] for player in normalized if player["display_weapon"]
    )
    first_half_side = stats.get("fh_role") or None
    return {
        "label": label,
        "id": info.get("id") or None,
        "name": info.get("disp_name") or None,
        "first_half_side": first_half_side,
        "current_side": _current_side(first_half_side, round_number),
        "score": _integer(stats.get("all_score")),
        "first_half_score": _integer(stats.get("fh_score")),
        "second_half_score": _integer(stats.get("sh_score")),
        "overtime_score": _integer(stats.get("ot_score")),
        "players": normalized,
        "economy": {
            "player_count": len(normalized),
            "money_known_players": len(money),
            "money_sum": sum(money) if len(money) == len(normalized) and normalized else None,
            "hp_known_players": len(hp),
            "hp_sum": sum(hp) if len(hp) == len(normalized) and normalized else None,
            "alive_count": (
                sum(value > 0 for value in hp)
                if len(hp) == len(normalized) and normalized else None
            ),
            "display_weapon_known_players": sum(bool(p["display_weapon"]) for p in normalized),
            "display_weapon_counts": dict(sorted(weapons.items())),
            "helmet_count": sum(p["helmet"] is True for p in normalized),
            "kevlar_count": sum(p["kevlar"] is True for p in normalized),
            "defuse_kit_count": sum(p["defuse_kit"] is True for p in normalized),
        },
    }


def summarize_payload(payload: dict, expected_match_id: str | None = None) -> tuple[dict, dict]:
    """Validate a 5E response and return (stable summary, raw match object)."""
    if not isinstance(payload, dict) or not payload.get("success"):
        message = payload.get("message") if isinstance(payload, dict) else None
        raise ValueError(f"api_not_success:{message!r}")
    match = (payload.get("data") or {}).get("match")
    if not isinstance(match, dict):
        raise ValueError("api_match_not_object")
    mc_info = match.get("mc_info") or {}
    match_id = mc_info.get("id")
    if expected_match_id and match_id != expected_match_id:
        raise ValueError(f"match_id_mismatch:{match_id!r}")
    global_state = match.get("global_state") or {}
    bouts = match.get("bouts_state") or []
    if not isinstance(bouts, list):
        raise ValueError("bouts_state_not_array")

    live_bouts = []
    for bout in bouts:
        if not isinstance(bout, dict) or str(bout.get("status")) != "1":
            continue
        team1_stats, team2_stats = bout.get("t1_stats") or {}, bout.get("t2_stats") or {}
        team1_score, team2_score = (
            _integer(team1_stats.get("all_score")),
            _integer(team2_stats.get("all_score")),
        )
        completed_rounds = (
            team1_score + team2_score
            if team1_score is not None and team2_score is not None else None
        )
        # curr_round_num was observed one or more rounds ahead on a live 5E
        # match. The active round implied by the scoreboard is stable and is
        # what the historical model uses; retain the source field for audit.
        round_number = None if completed_rounds is None else completed_rounds + 1
        source_round_number = _integer(bout.get("curr_round_num"))
        live_bouts.append({
            "bout_id": f"{match_id}_{bout.get('bout_num')}",
            "bout_num": _integer(bout.get("bout_num")),
            "map_name": bout.get("map_name") or None,
            "status": _integer(bout.get("status")),
            "round_number": round_number,
            "completed_rounds": completed_rounds,
            "source_round_number": source_round_number,
            "source_round_number_matches_score": (
                None if round_number is None or source_round_number is None
                else source_round_number == round_number),
            "round_number_basis": "score_sum_plus_one_for_live_bout",
            "round_start_time": _integer(bout.get("round_start_time")),
            "game_time": _integer(bout.get("game_time")),
            "stage": bout.get("curr_bout_stage") or None,
            "bomb_state": _integer(bout.get("bomb_planted")),
            "team1": _team_summary(
                "t1", mc_info.get("t1_info") or {}, bout.get("t1_stats") or {},
                bout.get("t1_pr_stats") or [], round_number,
            ),
            "team2": _team_summary(
                "t2", mc_info.get("t2_info") or {}, bout.get("t2_stats") or {},
                bout.get("t2_pr_stats") or [], round_number,
            ),
        })
    live_bouts.sort(key=lambda bout: (bout["bout_num"] is None, bout["bout_num"] or 0))
    summary = {
        "schema_version": 1,
        "match_id": match_id,
        "series": {
            "status": _integer(global_state.get("status")),
            "team1_score": _integer(global_state.get("t1_score")),
            "team2_score": _integer(global_state.get("t2_score")),
            "team1_id": (mc_info.get("t1_info") or {}).get("id") or None,
            "team1_name": (mc_info.get("t1_info") or {}).get("disp_name") or None,
            "team2_id": (mc_info.get("t2_info") or {}).get("id") or None,
            "team2_name": (mc_info.get("t2_info") or {}).get("disp_name") or None,
        },
        "live_bouts": live_bouts,
    }
    return summary, match


def state_hash(summary: dict) -> str:
    canonical = json.dumps(
        summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class FiveEStateRecorder:
    """Poll the public match snapshot and persist only meaningful state changes."""

    def __init__(
        self, match_id: str, out_path: Path, *, duration_seconds: float,
        poll_seconds: float = 1.0, request_timeout: float = 20.0,
        max_backoff_seconds: float = 30.0, max_consecutive_errors: int = 0,
        request_get: Callable | None = None, wall_clock: Callable[[], str] = utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not str(match_id).strip():
            raise ValueError("match_id_required")
        for name, value in (
            ("duration_seconds", duration_seconds), ("poll_seconds", poll_seconds),
            ("request_timeout", request_timeout),
            ("max_backoff_seconds", max_backoff_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name}_must_be_positive")
        if isinstance(max_consecutive_errors, bool) or max_consecutive_errors < 0:
            raise ValueError("max_consecutive_errors_must_be_nonnegative")
        self.match_id = str(match_id)
        self.out_path = Path(out_path)
        self.duration_seconds = float(duration_seconds)
        self.poll_seconds = float(poll_seconds)
        self.request_timeout = float(request_timeout)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.request_get = request_get or requests.get
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self.sleeper = sleeper
        self.url = BASE + DETAIL_PATH.format(match_id=self.match_id)
        self.records_written = self.snapshots_written = self.poll_index = 0

    def _write(self, handle, record_type: str, **fields) -> None:
        row = {
            "schema_version": 1, "record_type": record_type,
            "recv_utc": self.wall_clock(), "recv_mono": self.monotonic_clock(), **fields,
        }
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        self.records_written += 1

    def _sleep(self, seconds: float, deadline: float) -> None:
        remaining = deadline - self.monotonic_clock()
        if remaining > 0:
            self.sleeper(min(seconds, remaining))

    def run(self) -> int:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = self.monotonic_clock() + self.duration_seconds
        errors, outage_start, backoff = 0, None, 1.0
        first_success, previous_hash = True, None
        with self.out_path.open("a", encoding="utf-8") as handle:
            self._write(handle, "session_start", match_id=self.match_id, url=self.url)
            while self.monotonic_clock() < deadline:
                self.poll_index += 1
                try:
                    response = self.request_get(
                        self.url, impersonate="chrome", timeout=self.request_timeout,
                    )
                    response.raise_for_status()
                    summary, raw_match = summarize_payload(response.json(), self.match_id)
                    digest = state_hash(summary)
                    recovering = errors > 0
                    changed = digest != previous_hash
                    if recovering:
                        self._write(
                            handle, "source_recovered", poll_index=self.poll_index,
                            outage_seconds=max(0.0, self.monotonic_clock() - outage_start),
                            consecutive_errors=errors, state_changed=changed,
                        )
                    if changed:
                        timing_quality = (
                            "initial_snapshot" if first_success else
                            "recovery_snapshot" if recovering else "normal_poll_receive_time"
                        )
                        self._write(
                            handle, "state_snapshot", poll_index=self.poll_index,
                            state_hash=digest, initial_snapshot=first_success,
                            recovery_snapshot=recovering,
                            decision_eligible=not first_success and not recovering,
                            timing_quality=timing_quality, summary=summary, raw_match=raw_match,
                        )
                        self.snapshots_written += 1
                        previous_hash = digest
                    self._write(
                        handle, "poll_ok", poll_index=self.poll_index, state_hash=digest,
                        state_changed=changed, live_bouts=len(summary["live_bouts"]),
                    )
                    first_success, errors, outage_start, backoff = False, 0, None, 1.0
                    self._sleep(self.poll_seconds, deadline)
                except Exception as exc:
                    errors += 1
                    outage_start = (
                        self.monotonic_clock() if outage_start is None else outage_start
                    )
                    delay = min(backoff, self.max_backoff_seconds)
                    self._write(
                        handle, "source_error", poll_index=self.poll_index,
                        error_type=type(exc).__name__, error=str(exc)[:500],
                        consecutive_errors=errors, retry_in_seconds=delay,
                    )
                    if self.max_consecutive_errors and errors >= self.max_consecutive_errors:
                        self._write(
                            handle, "session_end", exit_reason="max_consecutive_errors",
                            snapshots_written=self.snapshots_written, polls=self.poll_index,
                        )
                        return 2
                    self._sleep(delay, deadline)
                    backoff = min(backoff * 2.0, self.max_backoff_seconds)
            self._write(
                handle, "session_end", exit_reason="deadline",
                snapshots_written=self.snapshots_written, polls=self.poll_index,
                ended_during_outage=errors > 0,
            )
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=0)
    parser.add_argument("--out-dir", default="data/fivee")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_path = out_dir / f"state_{args.match_id}_{stamp}.jsonl"
    meta_path = out_dir / f"state_{args.match_id}_{stamp}.meta.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 1, "match_id": args.match_id,
        "source": BASE + DETAIL_PATH.format(match_id=args.match_id),
        "timing_basis": "local_receive_time; upstream event time not guaranteed",
        "strict_backtest_policy": "exclude initial and recovery snapshots",
        "inventory_semantics": (
            "cash, HP, one displayed weapon, armour, kit and C4; not total equipment value"
        ),
        "poll_seconds": args.poll_seconds, "started_utc": utc_now(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder = FiveEStateRecorder(
        args.match_id, out_path, duration_seconds=args.hours * 3600.0,
        poll_seconds=args.poll_seconds, request_timeout=args.request_timeout,
        max_backoff_seconds=args.max_backoff_seconds,
        max_consecutive_errors=args.max_consecutive_errors,
    )
    code = recorder.run()
    print(f"{recorder.snapshots_written} state changes -> {out_path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
