"""Strict-time historical CS2 pistol-round models for paper research.

Round 1 is predicted from map, starting side, roster history and player pistol
history. Round 13 additionally uses only the completed first half. Labels from
the target series never enter its features; historical results must have an
``available_at`` strictly before the target start.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .map1_data import FEATURES as MAP_FEATURES
from .map1_data import map_name, path_key, roster_key, stats_names, utc
from .map1_model import probability_metrics


CONTEXT = ["first_half_score_gap", "first_pistol_gap", "first_kill_gap",
           "last3_gap", "stomp_gap"]
PLAYER = ["player_all_pistol_gap", "player_map_pistol_gap", "player_side_pistol_gap"]
ROSTER = [name for name in MAP_FEATURES if name not in {"all_adr_gap", "map_adr_gap"}]
LOGISTIC_C = .002


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_event_manifest(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"event", "starts_at", "recent_window_starts_at", "teams", "sources"}
    if not required.issubset(data) or len(data["teams"]) != 8:
        raise ValueError("invalid_target_event_manifest")
    names, slugs = set(), set()
    for team in data["teams"]:
        if not team.get("name") or len(team.get("players") or []) != 5:
            raise ValueError("invalid_target_team_or_lineup")
        names.add(team["name"])
        slugs.update(str(value).casefold() for value in team.get("path_slugs") or [])
    if len(names) != 8 or not slugs:
        raise ValueError("duplicate_or_missing_target_teams")
    data["_slugs"] = sorted(slugs)
    data["_starts_at"] = utc(data["starts_at"])
    data["_recent_at"] = utc(data["recent_window_starts_at"])
    return data


def matchup_slugs(map_id: str) -> tuple[str, str] | None:
    stem = PurePosixPath(path_key(map_id)).stem
    match = re.fullmatch(r"(.+)-vs-(.+)-m[1-5]-[a-z0-9_]+", stem)
    return None if match is None else (match.group(1), match.group(2))


def _winner_from_side(row) -> str:
    side = str(row.winner_side).upper()
    if side == "CT":
        return roster_key(row.ct_roster)
    if side in {"T", "TERRORIST"}:
        return roster_key(row.t_roster)
    raise ValueError("unknown_round_winner")


def _first_kill_roster(row) -> str | None:
    side = str(row.first_kill_side).upper()
    if side == "CT":
        return roster_key(row.ct_roster)
    if side in {"T", "TERRORIST"}:
        return roster_key(row.t_roster)
    return None


def extract_pistol_events(rounds: pd.DataFrame, history: pd.DataFrame,
                          labels: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, dict]:
    """Build two oriented targets per eligible map and quarantine bad maps."""
    h = history.copy()
    h["map_id"] = h.map_id.map(path_key)
    h["start_at"] = h.start_at.map(utc)
    h["available_at"] = h.available_at.map(utc)
    if h.map_id.duplicated().any() or (h.available_at <= h.start_at).any():
        raise ValueError("invalid_history_identity_or_time")
    eligible = labels.copy()
    eligible["map_id"] = eligible.map_id.map(path_key)
    eligible = eligible[eligible.complete.eq(True) & eligible.history_check.eq("matched")]
    if eligible.map_id.duplicated().any():
        raise ValueError("duplicate_map_labels")
    allowed = set(eligible.map_id)
    meta = h.set_index("map_id").to_dict("index")
    r = rounds.copy()
    r["map_id"] = r.demo_path.map(path_key)
    rejected, records = Counter(), []
    target_slugs = set(manifest["_slugs"])
    for map_id, group in r[r.map_id.isin(allowed)].groupby("map_id", sort=True):
        m = meta.get(map_id)
        try:
            if m is None:
                raise ValueError("missing_history")
            d = group.sort_values("round_num")
            if d.round_num.tolist() != list(range(1, len(d) + 1)) or len(d) < 13:
                raise ValueError("incomplete_round_ledger")
            a, b = roster_key(m["roster_a"]), roster_key(m["roster_b"])
            if set(d.roster_a.map(roster_key)) != {a} or set(d.roster_b.map(roster_key)) != {b}:
                raise ValueError("roster_identity_mismatch")
            matchup = matchup_slugs(map_id)
            priority = bool(matchup and set(matchup) & target_slugs)
            recent = manifest["_recent_at"] <= m["start_at"] < manifest["_starts_at"]
            first_half = d[d.round_num.le(12)]
            winner_a = first_half.apply(_winner_from_side, axis=1).eq(a)
            first_kills = first_half.apply(_first_kill_roster, axis=1)
            opening_gap = (int(first_kills.eq(a).sum()) - int(first_kills.eq(b).sum())) / 12
            last3 = int(winner_a.tail(3).sum()) - int((~winner_a.tail(3)).sum())
            stomps = first_half.outcome_type.astype(str).str.casefold().eq("stomp")
            stomp_gap = int((stomps & winner_a).sum()) - int((stomps & ~winner_a).sum())
            for pistol_round in (1, 13):
                row = d[d.round_num.eq(pistol_round)].iloc[0]
                winner = _winner_from_side(row)
                if winner not in {a, b}:
                    raise ValueError("pistol_winner_outside_rosters")
                record = {
                    "map_id": map_id, "match_id": m["match_id"], "map_name": m["map_name"],
                    "pistol_round": pistol_round, "roster_a": a, "roster_b": b,
                    "start_at": m["start_at"], "available_at": m["available_at"],
                    "y": int(winner == a), "a_is_ct": 1.0 if bool(row.ct_is_a) else -1.0,
                    "target_event_team": priority, "recent_three_months": bool(recent),
                    "matchup_slugs": "" if matchup is None else "|".join(matchup),
                    "first_half_score_gap": 0.0, "first_pistol_gap": 0.0,
                    "first_kill_gap": 0.0, "last3_gap": 0.0, "stomp_gap": 0.0,
                }
                if pistol_round == 13:
                    score_gap = int(row.score_a_before) - int(row.score_b_before)
                    if score_gap != int(winner_a.sum()) - int((~winner_a).sum()):
                        raise ValueError("first_half_score_mismatch")
                    first_winner = _winner_from_side(d[d.round_num.eq(1)].iloc[0])
                    record.update(first_half_score_gap=float(score_gap),
                                  first_pistol_gap=1.0 if first_winner == a else -1.0,
                                  first_kill_gap=float(opening_gap), last3_gap=float(last3),
                                  stomp_gap=float(stomp_gap))
                records.append(record)
        except (ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
    if not records:
        raise ValueError("empty_or_duplicate_pistol_targets")
    result = pd.DataFrame(records).sort_values(
        ["start_at", "match_id", "map_id", "pistol_round"]).reset_index(drop=True)
    if result.duplicated(["map_id", "pistol_round"]).any():
        raise ValueError("empty_or_duplicate_pistol_targets")
    return result, {"source_round_maps": int(r.map_id.nunique()),
                    "eligible_label_maps": len(allowed), "pistol_rows": len(result),
                    "maps": int(result.map_id.nunique()), "rejected_maps": dict(rejected)}


def _player_event_table(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in events.to_dict("records"):
        for label, roster, won, is_ct in (
            ("a", event["roster_a"], event["y"], event["a_is_ct"] > 0),
            ("b", event["roster_b"], 1-event["y"], event["a_is_ct"] < 0),
        ):
            for player in roster_key(roster).split(","):
                rows.append({"player": player, "won": int(won), "is_ct": bool(is_ct),
                             "map_name": event["map_name"], "match_id": event["match_id"],
                             "available_at": event["available_at"]})
    return pd.DataFrame(rows)


def _roster_history_index(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Orient history once per roster instead of rescanning all maps per target."""
    identity = ["match_id", "map_name", "available_at", "rounds"]
    measures = ["round_wins", "pistol_wins", "ct_wins", "ct_rounds", *stats_names()]
    parts = []
    for side in ("a", "b"):
        part = history[["roster_" + side, "y", *identity,
                        *[name + "_" + side for name in measures]]].copy()
        part = part.rename(columns={"roster_" + side: "roster", **{
            name + "_" + side: name for name in measures}})
        part["roster"] = part.roster.map(roster_key)
        part["won"] = part.y if side == "a" else 1 - part.y
        parts.append(part.drop(columns="y"))
    oriented = pd.concat(parts, ignore_index=True).sort_values("available_at")
    return {roster: group.reset_index(drop=True)
            for roster, group in oriented.groupby("roster", sort=False)}


def _indexed_roster_features(by_roster: dict[str, pd.DataFrame], roster_a: str,
                             roster_b: str, arena: str, at: pd.Timestamp,
                             exclude_match: str | None) -> tuple[dict, dict]:
    a, b = roster_key(roster_a), roster_key(roster_b)
    if set(a.split(",")) & set(b.split(",")):
        raise ValueError("overlapping_rosters")

    def prior(roster: str) -> pd.DataFrame:
        frame = by_roster.get(roster)
        if frame is None:
            return pd.DataFrame()
        frame = frame[frame.available_at.lt(at)]
        if exclude_match is not None:
            frame = frame[frame.match_id.ne(exclude_match)]
        return frame

    histories = {a: prior(a), b: prior(b)}
    max_values = [frame.available_at.max() for frame in histories.values() if not frame.empty]
    max_at = max(max_values) if max_values else pd.NaT
    coverage = {"max_history_available_at": None if pd.isna(max_at) else max_at.isoformat()}

    def profile(roster: str, arena_filter: str | None):
        frame = histories[roster]
        if arena_filter and not frame.empty:
            frame = frame[frame.map_name.eq(arena_filter)]
        if frame.empty:
            # The formulas below use fixed priors, so an empty oriented table is valid.
            return {"win": .5, "round": .5, "pistol": .5, "ct": .5,
                    "kd": 0.0, "adr": 70.0, "opening": .1, "trade": .15}, 0
        weights = np.exp2(-(at - frame.available_at).dt.total_seconds().to_numpy() /
                          (90 * 86400))

        def weighted(name: str) -> float:
            return float(np.dot(frame[name].to_numpy(dtype=float), weights))

        def ratio(num: str, den: str, prior_value: float, strength: float) -> float:
            valid = frame[[num, den]].notna().all(axis=1).to_numpy()
            valid_weights = weights[valid]
            numerator = np.dot(frame.loc[valid, num].to_numpy(dtype=float), valid_weights)
            denominator = np.dot(frame.loc[valid, den].to_numpy(dtype=float), valid_weights)
            return float((numerator + prior_value * strength) / (denominator + strength))

        n = float(weights.sum())
        pistols = np.where(frame["rounds"].to_numpy(dtype=float) >= 13, 2, 1)
        return {
            "win": float((np.dot(frame.won.to_numpy(dtype=float), weights) + 2.5) / (n + 5)),
            "round": ratio("round_wins", "rounds", .5, 60),
            "pistol": float((weighted("pistol_wins") + 2) /
                            (np.dot(pistols, weights) + 4)),
            "ct": ratio("ct_wins", "ct_rounds", .5, 30),
            "kd": float(np.log(ratio("kills", "deaths", 1, 100))),
            "adr": ratio("damage", "rounds_played", 70, 150),
            "opening": ratio("opening_kills", "rounds_played", .1, 150),
            "trade": ratio("trade_kills", "kills", .15, 100),
        }, len(frame)

    result = {}
    for scope, arena_filter in (("all", None), ("map", map_name(arena))):
        pa, na = profile(a, arena_filter)
        pb, nb = profile(b, arena_filter)
        coverage[f"history_{scope}_a"] = na
        coverage[f"history_{scope}_b"] = nb
        result.update({f"{scope}_{name}_gap": pa[name] - pb[name] for name in pa})
    return result, coverage


def _player_history_index(events: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for player, group in events.groupby("player", sort=False):
        group = group.sort_values("available_at")
        result[player] = {
            # Parquet commonly loads timestamps at microsecond precision while
            # Timestamp.value is nanoseconds; normalize before binary search.
            "available_ns": group.available_at.dt.as_unit("ns").array.asi8.copy(),
            "arena": group.map_name.to_numpy(dtype=object),
            "is_ct": group.is_ct.to_numpy(dtype=bool),
            "won": group.won.to_numpy(dtype=float),
        }
    return result


def _player_rate(by_player: dict[str, dict[str, np.ndarray]], players: list[str], at: pd.Timestamp,
                 *, arena: str | None = None, is_ct: bool | None = None) -> tuple[float, float]:
    rates, counts = [], []
    cutoff = at.value
    for player in players:
        history = by_player.get(player)
        if history is None:
            rates.append(.5)
            counts.append(0.0)
            continue
        end = int(np.searchsorted(history["available_ns"], cutoff, side="left"))
        mask = np.ones(end, dtype=bool)
        if arena is not None:
            mask &= history["arena"][:end] == arena
        if is_ct is not None:
            mask &= history["is_ct"][:end] == is_ct
        available = history["available_ns"][:end][mask]
        if not len(available):
            rates.append(.5)
            counts.append(0.0)
            continue
        age = (cutoff - available) / (86400 * 1_000_000_000)
        weights = np.exp2(-np.maximum(age, 0) / 90.0)
        rates.append(float((np.dot(history["won"][:end][mask], weights) + 1) /
                           (weights.sum() + 2)))
        counts.append(float(weights.sum()))
    return float(np.mean(rates)), float(np.mean(counts))


def build_features(events: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add strictly prior exact-roster and transferable player pistol profiles."""
    h = history.copy()
    h["map_id"] = h.map_id.map(path_key)
    h["start_at"] = h.start_at.map(utc)
    h["available_at"] = h.available_at.map(utc)
    player_events = _player_event_table(events)
    player_events["available_at"] = player_events.available_at.map(utc)
    by_player = _player_history_index(player_events)
    by_roster = _roster_history_index(h)
    map_side_columns = ["side_" + name for name in sorted(events.map_name.unique())]
    records = []
    prematch_cache = {}
    for row in events.to_dict("records"):
        at = utc(row["start_at"])
        cache_key = row["map_id"]
        if cache_key not in prematch_cache:
            prematch_cache[cache_key] = _indexed_roster_features(
                by_roster, row["roster_a"], row["roster_b"], row["map_name"], at,
                exclude_match=row["match_id"])
        prematch, coverage = prematch_cache[cache_key]
        a_players, b_players = row["roster_a"].split(","), row["roster_b"].split(",")
        a_ct = row["a_is_ct"] > 0
        pa_all, na = _player_rate(by_player, a_players, at)
        pb_all, nb = _player_rate(by_player, b_players, at)
        pa_map, _ = _player_rate(by_player, a_players, at, arena=row["map_name"])
        pb_map, _ = _player_rate(by_player, b_players, at, arena=row["map_name"])
        pa_side, _ = _player_rate(by_player, a_players, at, is_ct=a_ct)
        pb_side, _ = _player_rate(by_player, b_players, at, is_ct=not a_ct)
        record = {**row, **prematch, **coverage,
                  "player_all_pistol_gap": pa_all-pb_all,
                  "player_map_pistol_gap": pa_map-pb_map,
                  "player_side_pistol_gap": pa_side-pb_side,
                  "player_history_weight_a": na, "player_history_weight_b": nb}
        for column in map_side_columns:
            record[column] = row["a_is_ct"] if column == "side_" + row["map_name"] else 0.0
        records.append(record)
    result = pd.DataFrame(records).sort_values(
        ["start_at", "match_id", "map_id", "pistol_round"]).reset_index(drop=True)
    required = ["a_is_ct", *map_side_columns, *ROSTER, *PLAYER, *CONTEXT]
    if not np.isfinite(result[required].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite_pistol_features")
    for row in result.to_dict("records"):
        latest = row.get("max_history_available_at")
        if pd.notna(latest) and utc(latest) >= utc(row["start_at"]):
            raise ValueError("future_roster_history")
    return result, {"map_side_columns": map_side_columns,
                    "zero_player_history_rows": int(((result.player_history_weight_a == 0) |
                                                     (result.player_history_weight_b == 0)).sum()),
                    "player_history_basis": "past_series_available_at_only_half_life_90_days"}


def model_definitions(map_side_columns: list[str], pistol_round: int) -> dict[str, list[str]]:
    # Global side only. The v1 development run included this plus mutually
    # exclusive map-side columns, a collinear design that overfit rare maps.
    side = ["a_is_ct"]
    definitions = {"side_only": side,
                   "roster_history": side + ROSTER,
                   "roster_plus_players": side + ROSTER + PLAYER}
    if pistol_round == 13:
        definitions["roster_players_first_half"] = side + ROSTER + PLAYER + CONTEXT
    return definitions


def forward_splits(data: pd.DataFrame, n_blocks: int = 6):
    starts = data.start_at.map(utc)
    available = data.available_at.map(utc)
    times = sorted(starts.unique())
    for block in np.array_split(np.asarray(times, dtype=object), min(n_blocks, len(times)))[1:]:
        test = starts.isin(block)
        test_start = utc(block[0])
        test_matches = set(data.loc[test, "match_id"])
        train = available.lt(test_start) & starts.lt(test_start) & ~data.match_id.isin(test_matches)
        if train.any() and test.any():
            yield np.flatnonzero(train), np.flatnonzero(test), test_start


def training_weights(train: pd.DataFrame, reference_at: pd.Timestamp) -> np.ndarray:
    """Equalize series, then emphasize recent target-event teams without filtering."""
    age = (reference_at - train.start_at.map(utc)).dt.total_seconds().to_numpy() / 86400
    recency = np.exp2(-np.maximum(age, 0) / 90.0)
    recent_target = train.target_event_team.to_numpy(dtype=bool) & (age <= 92)
    priority = np.where(recent_target, 3.0, np.where(train.target_event_team, 1.5, 1.0))
    series = 1 / train.groupby("match_id").match_id.transform("size").to_numpy(dtype=float)
    weights = recency * priority * series
    return weights * len(weights) / weights.sum()


def fit_model(train: pd.DataFrame, features: list[str], reference_at: pd.Timestamp):
    if len(train) < 2 or train.y.nunique() < 2:
        raise ValueError("training_requires_both_pistol_outcomes")
    x, y = train[features].to_numpy(dtype=float), train.y.to_numpy(dtype=int)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_training_features")
    weights = training_weights(train, reference_at)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=LOGISTIC_C, fit_intercept=False, max_iter=2000))
    model.fit(np.concatenate([x, -x]), np.concatenate([y, 1-y]),
              logisticregression__sample_weight=np.tile(weights, 2))
    return model


def _metrics_for(frame: pd.DataFrame, probability_columns: dict[str, str]) -> dict:
    result = {}
    for name, column in probability_columns.items():
        result[name] = probability_metrics(frame.y, frame[column])
    return result


def evaluate_round(data: pd.DataFrame, pistol_round: int, map_side_columns: list[str],
                   min_train: int = 80) -> tuple[pd.DataFrame, dict]:
    d = data[data.pistol_round.eq(pistol_round)].sort_values(
        ["start_at", "match_id", "map_id"]).reset_index(drop=True).copy()
    if d.map_id.duplicated().any():
        raise ValueError("duplicate_map_pistol_target")
    definitions = model_definitions(map_side_columns, pistol_round)
    d["p_constant_0_5"] = .5
    for name in definitions:
        d["p_" + name] = np.nan
    d["train_rows"] = 0
    d["train_max_available_at"] = ""
    folds = []
    for train_index, test_index, test_start in forward_splits(d):
        train, test = d.iloc[train_index], d.iloc[test_index]
        if len(train) < min_train or train.y.nunique() < 2:
            continue
        if train.available_at.map(utc).max() >= test_start:
            raise ValueError("future_training_label")
        if set(train.match_id) & set(test.match_id):
            raise ValueError("series_overlap")
        for name, features in definitions.items():
            model = fit_model(train, features, test_start)
            d.loc[test_index, "p_" + name] = model.predict_proba(
                test[features].to_numpy(dtype=float))[:, 1]
        d.loc[test_index, "train_rows"] = len(train)
        d.loc[test_index, "train_max_available_at"] = train.available_at.max().isoformat()
        folds.append({"test_start": test_start.isoformat(), "train_rows": len(train),
                      "train_series": int(train.match_id.nunique()), "test_rows": len(test),
                      "test_series": int(test.match_id.nunique()),
                      "recent_target_train_rows": int((train.target_event_team &
                                                       train.recent_three_months).sum())})
    columns = {"constant_0_5": "p_constant_0_5",
               **{name: "p_" + name for name in definitions}}
    common = d.dropna(subset=list(columns.values()))
    recent_target = common[common.target_event_team & common.recent_three_months]
    paired = {}
    if len(common):
        baseline = (common.p_side_only-common.y) ** 2
        for name in definitions:
            paired[name] = float((((common["p_"+name]-common.y) ** 2)-baseline).mean())
    report = {"pistol_round": pistol_round, "target_rows": len(d),
              "scored_rows": len(common), "scored_series": int(common.match_id.nunique()),
              "recent_target_scored_rows": len(recent_target),
              "models": {name: {"features": features} for name, features in definitions.items()},
              "all_scored_metrics": _metrics_for(common, columns),
              "recent_target_metrics": _metrics_for(recent_target, columns),
              "paired_brier_minus_side_only": paired, "folds": folds}
    return d, report


def _target_coverage(events: pd.DataFrame, manifest: dict) -> dict:
    maps = events.drop_duplicates("map_id")
    result = {}
    for team in manifest["teams"]:
        slugs = set(value.casefold() for value in team["path_slugs"])
        involved = maps.matchup_slugs.map(
            lambda value: bool(set(str(value).split("|")) & slugs))
        recent = involved & maps.recent_three_months
        result[team["name"]] = {"all_maps": int(involved.sum()),
                                "recent_three_month_maps": int(recent.sum()),
                                "recent_series": int(maps.loc[recent, "match_id"].nunique()),
                                "lineup": team["players"]}
    return result


def run(*, history_path: Path, rounds_path: Path, labels_path: Path,
        manifest_path: Path, output_dir: Path, min_train: int = 80) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing_overwrite:{output_dir}")
    inputs = {"history": Path(history_path), "rounds": Path(rounds_path),
              "labels": Path(labels_path), "manifest": Path(manifest_path)}
    hashes = {name: file_hash(path) for name, path in inputs.items()}
    manifest = load_event_manifest(manifest_path)
    history, rounds, labels = (pd.read_parquet(inputs[name]) for name in ("history", "rounds", "labels"))
    raw, extraction_audit = extract_pistol_events(rounds, history, labels, manifest)
    data, feature_audit = build_features(raw, history)
    map_side_columns = feature_audit["map_side_columns"]
    predictions, evaluations, models = [], {}, {}
    for pistol_round in (1, 13):
        pred, evaluation = evaluate_round(data, pistol_round, map_side_columns, min_train)
        predictions.append(pred)
        evaluations[str(pistol_round)] = evaluation
        final_train = data[data.pistol_round.eq(pistol_round) &
                           data.available_at.map(utc).lt(manifest["_starts_at"])]
        models[str(pistol_round)] = {
            name: {"features": features,
                   "model": fit_model(final_train, features, manifest["_starts_at"])}
            for name, features in model_definitions(map_side_columns, pistol_round).items()
        }
    combined = pd.concat(predictions, ignore_index=True).sort_values(
        ["start_at", "match_id", "map_id", "pistol_round"]).reset_index(drop=True)
    report = {
        "version": 3, "mode": "offline_pistol_probability_research",
        "event": manifest["event"], "event_starts_at": manifest["starts_at"],
        "live_trading_enabled": False, "promoted_model": None, "execution_pnl": None,
        "audit": {**extraction_audit, **feature_audit},
        "target_team_coverage": _target_coverage(raw, manifest),
        "weighting": {"series": "equal aggregate weight within each pistol model",
                      "recency_half_life_days": 90,
                      "target_team_multiplier_recent": 3.0,
                      "target_team_multiplier_older": 1.5,
                      "recent_definition_for_manifest": manifest["recent_window_starts_at"]},
        "model_stability": {"logistic_C": LOGISTIC_C,
                            "side_design": "single_global_ct_indicator_no_collinear_map_side_terms",
                            "feature_query": "strict roster index plus nanosecond-normalized player time index",
                            "status": "v2 chosen after v1 development diagnostic; still requires new forward acceptance data"},
        "evaluation": evaluations,
        "validation": "whole-series forward blocks; labels require available_at < block start; mirrored only after split",
        "limitations": [
            "Historical data ends before the event and has already been inspected; this is development evidence, not a fresh acceptance set.",
            "Map and starting side are assumed known at decision time.",
            "No synchronized historical Polymarket order books exist for these maps, so profitability is not evaluated.",
            "magic, MIBR and NRG have sparse or zero recent local demo coverage; targeted collection remains required.",
            "The round-13 model uses completed first-half state but no round-13 equipment because the current live feed does not reliably expose it."
        ],
        "inputs": {name: {"path": str(path.resolve()), "sha256": hashes[name]}
                   for name, path in inputs.items()},
        "sources": manifest["sources"],
    }
    if hashes != {name: file_hash(path) for name, path in inputs.items()}:
        raise ValueError("input_changed_while_running")
    output_dir.mkdir(parents=True, exist_ok=False)
    data.to_parquet(output_dir / "dataset.parquet", index=False)
    combined.to_parquet(output_dir / "predictions.parquet", index=False)
    joblib.dump(models, output_dir / "models.joblib")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = config.DATA_DIR / "map1"
    parser.add_argument("--history", type=Path, default=root / "history.parquet")
    parser.add_argument("--rounds", type=Path,
                        default=root / "rebuild" / "full-20260916" / "round_states.parquet")
    parser.add_argument("--labels", type=Path,
                        default=root / "rebuild" / "full-20260916" / "map_labels.parquet")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("starladder_fall_2026.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=80)
    args = parser.parse_args(argv)
    report = run(history_path=args.history, rounds_path=args.rounds, labels_path=args.labels,
                 manifest_path=args.manifest, output_dir=args.output, min_train=args.min_train)
    compact = {"output": str(args.output.resolve()), "audit": report["audit"],
               "target_team_coverage": report["target_team_coverage"],
               "evaluation": {key: value["all_scored_metrics"]
                              for key, value in report["evaluation"].items()},
               "live_trading_enabled": False, "promoted_model": None}
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
