"""Frozen v1 contract for paper-only CS2 swing-trend research."""
from __future__ import annotations

import hashlib
import json
import math
import re


PROTOCOL_ID = "cs2_swing_trend_v1"
HORIZONS_SECONDS = (15, 30, 60, 120)
DECISION_KINDS = (
    "post_pistol_freeze",
    "round_end_economy_update",
    "prematch_prior_dislocation",
)
MINIMUM_SOURCE_OVERLAP_SECONDS = 300


def protocol_payload() -> dict:
    """Canonical research choices. Changing this requires a new protocol id."""
    return {
        "protocol_id": PROTOCOL_ID,
        "mode": "paper_only",
        "decision_unit": "one row per mechanism transition, never per quote or video frame",
        "decision_kinds": list(DECISION_KINDS),
        "horizons_seconds": list(HORIZONS_SECONDS),
        "entry": (
            "first executable ask at or after state receipt + measured model latency "
            "+ market secondsDelay"),
        "exit": "first executable bid at or after the fixed horizon",
        "net_label": "exit proceeds - entry cost - observed market fees",
        "price_prohibitions": [
            "no midpoint fills", "no window maximum as an exit",
            "no quote received before the executable decision time",
        ],
        "model_comparison": ["market_only", "game_only", "market_plus_game"],
        "cohort_rule": "same decision rows and whole-series forward splits for every model",
        "timing_rule": (
            "initial snapshots and recovery backfills are ineligible; unknown event time stays unknown"),
        "execution_rule": (
            "full depth is required for size-aware PnL; otherwise the session is diagnostic only"),
        "minimum_source_overlap_seconds": MINIMUM_SOURCE_OVERLAP_SECONDS,
        "promotion": "none; a separate frozen future acceptance window is required",
    }


def protocol_hash() -> str:
    raw = json.dumps(protocol_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _nonempty(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}_required")
    return value.strip()


def validate_session_spec(spec: dict) -> dict:
    """Fail closed on ambiguous match, market or map bindings."""
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("session_spec_schema_version")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("session_spec_protocol_mismatch")
    session_id = _nonempty(spec.get("session_id"), "session_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,100}", session_id):
        raise ValueError("invalid_session_id")
    match_id = _nonempty(spec.get("match_id"), "match_id")
    event_id = _nonempty(str(spec.get("event_id", "")), "event_id")
    if not event_id.isdigit():
        raise ValueError("event_id_must_be_numeric")
    teams = spec.get("teams")
    if (not isinstance(teams, list) or len(teams) != 2
            or len({_nonempty(team, "team") for team in teams}) != 2):
        raise ValueError("exactly_two_distinct_teams_required")
    maps = spec.get("maps")
    if not isinstance(maps, list) or not maps:
        raise ValueError("expected_maps_required")
    bouts = set()
    for item in maps:
        if not isinstance(item, dict):
            raise ValueError("invalid_map_binding")
        bout = item.get("bout_num")
        if isinstance(bout, bool) or not isinstance(bout, int) or bout <= 0 or bout in bouts:
            raise ValueError("invalid_or_duplicate_bout_num")
        bouts.add(bout)
        name = _nonempty(item.get("map_name"), "map_name").casefold()
        if not re.fullmatch(r"de_[a-z0-9_]+", name):
            raise ValueError("map_name_must_be_canonical")
        _nonempty(str(item.get("market_id", "")), "market_id")
        _nonempty(item.get("market_name"), "market_name")
    sources = spec.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("sources_required")
    event_paths = sources.get("fivee_events")
    state_paths = sources.get("fivee_states")
    market = sources.get("market")
    if not isinstance(event_paths, list) or not all(isinstance(x, str) and x for x in event_paths):
        raise ValueError("fivee_event_paths_required")
    if not isinstance(state_paths, list) or not all(isinstance(x, str) and x for x in state_paths):
        raise ValueError("fivee_state_paths_must_be_list")
    if not isinstance(market, dict) or market.get("format") not in {"raw_depth", "normalized_top"}:
        raise ValueError("invalid_market_source")
    for key in ("events", "metadata"):
        _nonempty(market.get(key), f"market_{key}")
    if market["format"] == "raw_depth":
        _nonempty(market.get("summary"), "market_summary")
    return spec


def net_swing_pnl(entry_cost: float, exit_proceeds: float,
                  entry_fee: float, exit_fee: float) -> float:
    values = [entry_cost, exit_proceeds, entry_fee, exit_fee]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("nonnegative_finite_cashflows_required")
    return float(exit_proceeds - entry_cost - entry_fee - exit_fee)

