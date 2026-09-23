"""Offline-only, past-only SteamID profiles for a small Map 1 ablation.

This deliberately does not change the production model or consume live/demo
events. A player's history follows their SteamID through roster substitutions;
each current five-player roster is an equal-weight mean of five profiles.
Unknown players retain fixed priors, not a prior estimated from the full pool.
Raw damage is excluded: the legacy cache includes armor damage, not normal ADR.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .map1_data import map_name, path_key, roster_key, utc


HALF_LIFE_DAYS = 90.0
PROFILE_NAMES = ("win", "kd", "opening", "opening_death", "trade", "headshot")
PLAYER_FEATURES = [f"player_{scope}_{name}_gap" for scope in ("all", "map")
                   for name in PROFILE_NAMES]
PLAYER_STATS = ("kills", "deaths", "rounds_played", "opening_kills",
                "opening_deaths", "trade_kills", "headshots")
# Prior strengths have denominator units: maps, deaths, rounds, or kills.
PRIORS = {"win": (.5, 5.), "kd": (1., 50.), "opening": (.1, 100.),
          "opening_death": (.1, 100.), "trade": (.15, 50.),
          "headshot": (.5, 50.)}


def _require(frame: pd.DataFrame, columns, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _clean_players(history: pd.DataFrame, player_maps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Use clean map metadata as sole source of identity, labels and availability."""
    _require(history, ("map_id", "match_id", "map_name", "available_at", "roster_a",
                       "roster_b", "y", "rounds"), "history")
    _require(player_maps, ("demo_path", "steamid", "roster_key", *PLAYER_STATS), "player_maps")
    clean = history.copy()
    clean["map_id"] = clean.map_id.map(path_key)
    if clean.map_id.duplicated().any():
        raise ValueError("Duplicate clean-history map_id; resolve before player research")
    clean["match_id"] = clean.match_id.map(str)
    clean["available_at"] = pd.to_datetime(clean.available_at.map(utc), utc=True)
    clean["map_name"] = clean.map_name.map(map_name)
    for side in ("a", "b"):
        clean[f"roster_{side}"] = clean[f"roster_{side}"].map(roster_key)
    if not clean.y.isin((0, 1)).all():
        raise ValueError("Clean-history labels must be binary")
    for row in clean.itertuples(index=False):
        if set(row.roster_a.split(",")) & set(row.roster_b.split(",")):
            raise ValueError("Overlapping clean-history rosters")
        if not np.isfinite(float(row.rounds)) or row.rounds <= 0 or int(row.rounds) != row.rounds:
            raise ValueError("Clean-history rounds must be positive integers")
    metadata = clean.set_index("map_id").to_dict("index")
    raw = player_maps.copy()
    raw["map_id"] = raw.demo_path.map(path_key)
    raw["steamid"] = raw.steamid.map(lambda value: str(value).strip())
    # A duplicate may be an accidental repeated file or conflicting extraction.
    # Either way, discard the entire affected map rather than arbitrarily keep one.
    duplicates = set(raw.loc[raw.duplicated(["map_id", "steamid"], keep=False), "map_id"])
    excluded = Counter()
    rows = []
    for row in raw.to_dict("records"):
        mid = row["map_id"]
        m = metadata.get(mid)
        if m is None:
            excluded["map_not_in_clean_history"] += 1
            continue
        if mid in duplicates:
            excluded["duplicate_player_map"] += 1
            continue
        try:
            roster = roster_key(row["roster_key"])
        except (ValueError, TypeError):
            excluded["invalid_roster"] += 1
            continue
        if roster not in (m["roster_a"], m["roster_b"]) or row["steamid"] not in roster.split(","):
            excluded["roster_or_player_identity_mismatch"] += 1
            continue
        if "match_id" in row and str(row["match_id"]) != m["match_id"]:
            excluded["series_identity_mismatch"] += 1
            continue
        if "map_name" in row:
            try:
                same_map = map_name(row["map_name"]) == m["map_name"]
            except ValueError:
                same_map = False
            if not same_map:
                excluded["map_name_mismatch"] += 1
                continue
        try:
            values = {stat: float(row[stat]) for stat in PLAYER_STATS}
        except (TypeError, ValueError):
            excluded["invalid_player_counts"] += 1
            continue
        if not all(np.isfinite(v) and v >= 0 and v.is_integer() for v in values.values()):
            excluded["invalid_player_counts"] += 1
            continue
        if values["rounds_played"] != m["rounds"]:
            excluded["round_count_mismatch"] += 1
            continue
        if (values["deaths"] > values["rounds_played"] or
                values["opening_kills"] > min(values["kills"], values["rounds_played"]) or
                values["opening_deaths"] > values["deaths"] or
                values["headshots"] > values["kills"] or
                values["trade_kills"] > values["kills"]):
            excluded["inconsistent_player_counts"] += 1
            continue
        rows.append({"map_id": mid, "match_id": m["match_id"], "map_name": m["map_name"],
                     "available_at": m["available_at"], "steamid": row["steamid"],
                     "won": int(m["y"] if roster == m["roster_a"] else 1 - m["y"]), **values})
    columns = ["map_id", "match_id", "map_name", "available_at", "steamid", "won", *PLAYER_STATS]
    joined = pd.DataFrame(rows, columns=columns)
    joined["available_at"] = pd.to_datetime(joined.available_at, utc=True)
    return joined, {
        "source_player_rows": len(raw), "accepted_player_rows": len(joined),
        "accepted_clean_maps": int(joined.map_id.nunique()),
        "duplicate_player_maps": len(duplicates & set(metadata)),
        "excluded_player_rows": dict(sorted(excluded.items())),
    }


def _profile(rows: pd.DataFrame, at: pd.Timestamp) -> dict[str, float]:
    weight = np.exp2(-(at - rows.available_at).dt.total_seconds().to_numpy() /
                     (HALF_LIFE_DAYS * 86400.))
    totals = {name: float(np.dot(rows[name].to_numpy(dtype=float), weight))
              for name in ("won", *PLAYER_STATS)}

    def ratio(name: str, numerator: float, denominator: float) -> float:
        prior, strength = PRIORS[name]
        return (numerator + prior * strength) / (denominator + strength)

    return {"win": ratio("win", totals["won"], float(weight.sum())),
            "kd": float(np.log(ratio("kd", totals["kills"], totals["deaths"]))),
            "opening": ratio("opening", totals["opening_kills"], totals["rounds_played"]),
            "opening_death": ratio("opening_death", totals["opening_deaths"], totals["rounds_played"]),
            "trade": ratio("trade", totals["trade_kills"], totals["kills"]),
            "headshot": ratio("headshot", totals["headshots"], totals["kills"])}


def build_player_features(history: pd.DataFrame, player_maps: pd.DataFrame,
                          targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return *new columns only*, with exactly the targets' index and row order.

    Targets require match_id, roster_a, roster_b, map_name and decision_at. If
    map_id is present it is excluded too, protecting against a mislabelled series.
    Every profile uses available_at strictly before decision_at and never the
    target series. Availability inherits the clean history's match-end-plus-lag
    proxy; this does not establish historical public veto or lineup availability.

    Sparse profiles shrink to constants and remain finite; there is no hidden
    minimum-sample filter. Coverage exposes per-side known players and minimum /
    mean map counts so the caller can report fixed-cohort and sparse-cohort results.
    """
    _require(targets, ("match_id", "roster_a", "roster_b", "map_name", "decision_at"), "targets")
    players, audit = _clean_players(history, player_maps)
    results = []
    for target in targets.to_dict("records"):
        at = utc(target["decision_at"])
        rosters = {side: roster_key(target[f"roster_{side}"]).split(",") for side in ("a", "b")}
        if set(rosters["a"]) & set(rosters["b"]):
            raise ValueError("Overlapping target rosters")
        arena = map_name(target["map_name"])
        past = players[players.available_at.lt(at) & players.match_id.ne(str(target["match_id"]))]
        if "map_id" in target:
            past = past[past.map_id.ne(path_key(target["map_id"]))]
        current_players = rosters["a"] + rosters["b"]
        past = past[past.steamid.isin(current_players)]
        latest = past.available_at.max()
        result = {"player_max_history_available_at": None if pd.isna(latest) else latest.isoformat()}
        for scope, frame in (("all", past), ("map", past[past.map_name.eq(arena)])):
            profiles = {}
            for side, ids in rosters.items():
                parts = [frame[frame.steamid.eq(sid)] for sid in ids]
                counts = [len(part) for part in parts]
                individual = [_profile(part, at) for part in parts]
                profiles[side] = {key: float(np.mean([p[key] for p in individual])) for key in PROFILE_NAMES}
                result[f"player_{scope}_known_{side}"] = sum(n > 0 for n in counts)
                result[f"player_{scope}_min_maps_{side}"] = min(counts)
                result[f"player_{scope}_mean_maps_{side}"] = float(np.mean(counts))
                result[f"player_{scope}_unknown_{side}"] = [sid for sid, n in zip(ids, counts) if n == 0]
            result.update({f"player_{scope}_{key}_gap": profiles["a"][key] - profiles["b"][key]
                           for key in PROFILE_NAMES})
        results.append(result)
    coverage = [f"player_{scope}_{name}_{side}" for scope in ("all", "map")
                for side in ("a", "b") for name in ("known", "min_maps", "mean_maps", "unknown")]
    frame = pd.DataFrame(results, index=targets.index,
                         columns=[*PLAYER_FEATURES, *coverage, "player_max_history_available_at"])
    audit.update({
        "target_rows": len(targets), "feature_columns": list(PLAYER_FEATURES),
        "half_life_days": HALF_LIFE_DAYS,
        "priors": {key: {"mean": value[0], "denominator_strength": value[1]}
                   for key, value in PRIORS.items()},
        "minimum_profile_maps": 0,
        "aggregation": "equal_weight_mean_of_five_current_SteamID_profiles_including_unknown_priors",
        "availability_basis": "inherited_clean_history_available_at_not_observed_live_receipt",
        "same_series_excluded": True, "strictly_past_only": True,
        "limitations": [
            "Offline exploratory candidate, not a validated or deployed trading model.",
            "Raw counts inherit the legacy parser's event semantics; no fresh demo-level event audit here.",
            "Damage and ADR excluded because legacy damage includes armor.",
            "Player map wins reflect former teammates/opponents; not an opponent-adjusted causal skill estimate.",
            "No target result, global fitted prior, nickname matching, or current-map event features.",
        ],
    })
    return frame, audit
