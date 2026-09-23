"""Fixed offline event-time Map Winner ablations on newly verified demo caches.

All candidates predict canonical roster A winning the complete map at identical
nonterminal event-state ticks. No market prior, market prices, live feed, trading,
automatic model selection or untouched-holdout claim is provided here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .inround_model import attach_chronology, terminal_reason
from .map1_data import first_ct_is_ct, map_name, path_key, roster_key, terminal_score, utc


VERSION = "event_time_map_winner_fixed_ablation_v3"
CATEGORICAL_BASE = ["map_side"]
BASE = ["score_gap", "score_gap_x_round", "a_is_ct", "round_num", "elapsed_seconds"]
FIVEE_BASE = ["score_gap", "score_gap_x_round", "a_is_ct", "round_num"]
FIVEE_STATE = FIVEE_BASE + ["alive_gap", "hp_gap"]
ECONOMY = ["starting_equip_gap"]
STATE = ["alive_gap", "hp_gap", "bomb_advantage"]
# Layer 4: geometry/attack-defense structure proxies + current-kill weapon class.
# contact/spread/aiming are geometric proxies, not true line-of-sight; NaN (~1.7%)
# is imputed with paired-train means (exactly 0 for these odd features) inside
# _fit_predict, so the common cohort stays identical to earlier layers.
STRUCTURE = ["contact_gap", "spread_gap", "aiming_gap"]
VARIANTS = {"score_side": BASE,
            "fivee_score": FIVEE_BASE,
            "fivee_state": FIVEE_STATE,
            "starting_economy": BASE + ECONOMY,
            "alive_hp_bomb": BASE + ECONOMY + STATE,
            "geometry_weapon": BASE + ECONOMY + STATE + STRUCTURE}
CATEGORICAL = {name: list(CATEGORICAL_BASE) for name in VARIANTS}
CATEGORICAL["geometry_weapon"] = CATEGORICAL_BASE + ["weapon_class"]
ODD_FEATURES = ["score_gap", "score_gap_x_round", "a_is_ct", *ECONOMY, *STATE, *STRUCTURE]
ALL_NUMERIC = BASE + ECONOMY + STATE
APPROVED_NUMERIC = ALL_NUMERIC + STRUCTURE
WEAPON_CLASS = {"awp": "awp", "ssg08": "scout",
                "ak47": "rifle", "m4a1": "rifle", "m4a1_silencer": "rifle",
                "galilar": "rifle", "famas": "rifle", "aug": "rifle", "sg556": "rifle",
                "mp9": "smg", "mac10": "smg", "mp7": "smg", "ump45": "smg", "mp5sd": "smg",
                "usp_silencer": "pistol", "usp_silencer_off": "pistol", "glock": "pistol",
                "deagle": "pistol", "tec9": "pistol", "p250": "pistol",
                "fiveseven": "pistol", "hkp2000": "pistol", "cz75a": "pistol",
                "revolver": "pistol", "elite": "pistol",
                "nova": "shotgun", "xm1014": "shotgun", "mag7": "shotgun"}
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_DRAWS = 2000
FORBIDDEN_FEATURES = {"y", "label_ct_win", "winner_side", "winner_roster", "score_a", "score_b",
                      "n_rounds", "round_end_tick", "summary_state_tick", "event_count", "event_rank",
                      "round_len_ticks", "round_len_sec", "outcome_type", "final_margin",
                      "max_gap_winner", "min_gap_winner", "available_at"}


def _require(frame, columns, name):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{name}_missing_columns:{','.join(sorted(missing))}")


def _count(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("invalid_integer_count")
    number = float(value)
    if not np.isfinite(number) or number < 0 or number != int(number):
        raise ValueError("invalid_integer_count")
    return int(number)


def _verified_rounds(group, label):
    """Recompute full-map score identities; outcomes remain target metadata only."""
    a, b = roster_key(label["roster_a"]), roster_key(label["roster_b"])
    if (a != label["roster_a"] or b != label["roster_b"] or a >= b
            or set(a.split(",")) & set(b.split(","))):
        raise ValueError("invalid_canonical_map_rosters")
    if not isinstance(label["complete"], (bool, np.bool_)) or not label["complete"]:
        raise ValueError("map_not_verified_complete")
    if label["history_check"] != "matched":
        raise ValueError("map_history_check_not_matched")
    end_a, end_b, count = (_count(label[key]) for key in ("score_a", "score_b", "n_rounds"))
    if not terminal_score(end_a, end_b) or end_a + end_b != count:
        raise ValueError("invalid_complete_map_score")
    winner = a if end_a > end_b else b
    if label["winner_roster"] != winner:
        raise ValueError("map_winner_label_conflicts_with_score")
    d = group.sort_values("round_num").copy()
    if [_count(value) for value in d.round_num] != list(range(1, count + 1)):
        raise ValueError("incomplete_or_duplicate_map_round_history")
    if d.map_name.map(map_name).nunique() != 1:
        raise ValueError("mixed_map_names_in_round_history")
    first_ct = roster_key(d.iloc[0].ct_roster)
    first_t = b if first_ct == a else a
    score_a = score_b = 0
    last_end = -1
    for row in d.to_dict("records"):
        ct, tt = roster_key(row["ct_roster"]), roster_key(row["t_roster"])
        if {ct, tt} != {a, b} or row["roster_a"] != a or row["roster_b"] != b:
            raise ValueError("round_roster_identity_mismatch")
        if ct != (first_ct if first_ct_is_ct(int(row["round_num"])) else first_t):
            raise ValueError("round_side_schedule_mismatch")
        if row["ct_is_a"] not in (0, 1, False, True) or bool(row["ct_is_a"]) != (ct == a):
            raise ValueError("round_side_orientation_mismatch")
        start, end = _count(row["round_start_tick"]), _count(row["round_end_tick"])
        if not last_end < start < end:
            raise ValueError("invalid_complete_round_boundaries")
        last_end = end
        if (_count(row["score_a_before"]), _count(row["score_b_before"])) != (score_a, score_b):
            raise ValueError("round_prior_score_mismatch")
        winning_side = str(row["winner_side"]).upper().replace("TERRORIST", "T")
        if winning_side not in {"CT", "T"}:
            raise ValueError("invalid_round_winner_side")
        winning_roster = ct if winning_side == "CT" else tt
        if row["winner_roster"] != winning_roster:
            raise ValueError("round_winner_roster_mismatch")
        score_a += int(winning_roster == a)
        score_b += int(winning_roster == b)
        if row["round_num"] < count and terminal_score(score_a, score_b):
            raise ValueError("rounds_after_terminal_map_score")
    if (score_a, score_b) != (end_a, end_b):
        raise ValueError("map_label_disagrees_with_complete_round_history")
    return d, a, b, int(winner == a)


def _oriented_gap(row, sign, ct_col, t_col, *, t_minus_ct):
    """A-oriented geometry gap from raw CT/T columns; NaN when a side is missing."""
    ct, tt = float(row[ct_col]), float(row[t_col])
    if not (np.isfinite(ct) and np.isfinite(tt)):
        return float("nan")
    gap = tt - ct if t_minus_ct else ct - tt
    return sign * gap


def _weapon_class(event_type, weapon):
    if event_type != "kill" or pd.isna(weapon):
        return "none"
    return WEAPON_CLASS.get(str(weapon).lower(), "other")


def prepare_map_events(events, rounds, map_labels, history):
    """Build one common event cohort, using complete-map labels and past scores.

    Partial/ambiguous maps are excluded wholesale. Event rows provide only their
    current snapshots; current-round results, final scores and terminal ticks are
    never model inputs. History supplies chronology, not the new map target.
    """
    event_columns = {"demo_path", "round_num", "tick", "event_tick", "state_tick", "event_type",
                     "schema_version", "observation_basis", "map_name", "ct_roster", "t_roster",
                     "ct_alive", "t_alive", "ct_hp", "t_hp", "bomb_state",
                     "weapon", "ct_contact", "t_contact", "ct_spread", "t_spread",
                     "ct_aiming", "t_aiming"}
    round_columns = {"demo_path", "round_num", "ct_roster", "t_roster", "roster_a", "roster_b",
                     "ct_is_a", "score_a_before", "score_b_before", "winner_roster", "winner_side",
                     "round_start_tick", "round_end_tick", "ct_equip0", "t_equip0", "map_name"}
    label_columns = {"demo_path", "map_id", "roster_a", "roster_b", "winner_roster",
                     "score_a", "score_b", "n_rounds", "complete", "history_check"}
    _require(events, event_columns, "events")
    _require(rounds, round_columns, "rounds")
    _require(map_labels, label_columns, "map_labels")
    e, r, labels = events.copy(), rounds.copy(), map_labels.copy()
    for frame in (e, r, labels):
        frame["demo_path"] = frame.demo_path.map(path_key)
    labels["map_id"] = labels.map_id.map(path_key)
    if labels.map_id.duplicated().any() or not labels.map_id.eq(labels.demo_path).all():
        raise ValueError("duplicate_or_conflicting_map_label_identity")
    if e.duplicated(["demo_path", "round_num", "state_tick"]).any():
        raise ValueError("duplicate_event_state_identity")
    label_by_map = labels.set_index("map_id").to_dict("index")
    round_groups = {key: group for key, group in r.groupby("demo_path", sort=False)}
    rejected_maps, rejected_events = Counter(), Counter()
    records = []
    verified_maps = 0
    for mid, group in e.groupby("demo_path", sort=True):
        label = label_by_map.get(mid)
        if label is None:
            rejected_maps["missing_verified_map_label"] += 1
            rejected_events["map_ineligible"] += len(group)
            continue
        if mid not in round_groups:
            rejected_maps["missing_complete_map_rounds"] += 1
            rejected_events["map_ineligible"] += len(group)
            continue
        try:
            completed, a, b, y = _verified_rounds(round_groups[mid], label)
        except (TypeError, ValueError) as exc:
            rejected_maps[str(exc)] += 1
            rejected_events["map_ineligible"] += len(group)
            continue
        verified_maps += 1
        round_lookup = completed.set_index("round_num").to_dict("index")
        for row in group.to_dict("records"):
            reason = None
            try:
                number = _count(row["round_num"])
                current = round_lookup.get(number)
                if current is None:
                    raise ValueError("event_round_missing")
                event_tick, state_tick = _count(row["event_tick"]), _count(row["state_tick"])
                start, end = _count(current["round_start_tick"]), _count(current["round_end_tick"])
                if (row["schema_version"] != 2 or row["observation_basis"] != "first_full_tick_after_event_batch"
                        or event_tick != _count(row["tick"]) or state_tick != event_tick + 1):
                    raise ValueError("unverified_event_observation_timing")
                if not start <= event_tick < state_tick < end:
                    raise ValueError("event_outside_active_round")
                if (roster_key(row["ct_roster"]) != current["ct_roster"]
                        or roster_key(row["t_roster"]) != current["t_roster"]):
                    raise ValueError("event_round_roster_mismatch")
                arena = map_name(row["map_name"])
                if arena != map_name(current["map_name"]):
                    raise ValueError("event_round_map_name_mismatch")
                ct_alive, t_alive, bomb = (_count(row[key]) for key in ("ct_alive", "t_alive", "bomb_state"))
                if ct_alive > 5 or t_alive > 5 or bomb not in (0, 1, 2, 3):
                    raise ValueError("invalid_event_state")
                ct_hp, t_hp = float(row["ct_hp"]), float(row["t_hp"])
                ct_equip, t_equip = float(current["ct_equip0"]), float(current["t_equip0"])
                if (not np.isfinite([ct_hp, t_hp, ct_equip, t_equip]).all()
                        or not (0 <= ct_hp <= 100 * ct_alive and 0 <= t_hp <= 100 * t_alive)
                        or min(ct_equip, t_equip) < 0):
                    raise ValueError("invalid_event_hp_or_starting_equipment")
                reason = terminal_reason(ct_alive, t_alive, bomb, row["event_type"])
                if reason:
                    raise ValueError(reason)
                sign = 1 if current["ct_is_a"] else -1
                score_a, score_b = _count(current["score_a_before"]), _count(current["score_b_before"])
                record = {"map_id": mid, "demo_path": mid, "round_num": number,
                          "event_tick": event_tick, "state_tick": state_tick, "event_type": row["event_type"],
                          "map_name": arena, "map_side": arena + ("|A_CT" if sign == 1 else "|A_T"),
                          "roster_a": a, "roster_b": b, "y": y,
                          "score_a_before": score_a, "score_b_before": score_b,
                          "score_gap": score_a - score_b, "score_gap_x_round": (score_a - score_b) * number,
                          "a_is_ct": sign, "elapsed_seconds": (state_tick - start) / 64.,
                          "starting_equip_gap": sign * (ct_equip - t_equip),
                          "alive_gap": sign * (ct_alive - t_alive), "hp_gap": sign * (ct_hp - t_hp),
                          "bomb_advantage": -sign * int(bomb == 1),
                          "contact_gap": _oriented_gap(row, sign, "ct_contact", "t_contact", t_minus_ct=True),
                          "spread_gap": _oriented_gap(row, sign, "ct_spread", "t_spread", t_minus_ct=True),
                          "aiming_gap": _oriented_gap(row, sign, "ct_aiming", "t_aiming", t_minus_ct=False),
                          "weapon_class": _weapon_class(row["event_type"], row.get("weapon"))}
                if "match_id" in row:
                    record["match_id"] = str(row["match_id"])
                records.append(record)
            except (TypeError, ValueError) as exc:
                rejected_events[str(exc)] += 1
    columns = ["map_id", "demo_path", "round_num", "event_tick", "state_tick", "event_type", "map_name",
               "map_side", "roster_a", "roster_b", "y", "score_a_before", "score_b_before",
               *ALL_NUMERIC, *STRUCTURE, "weapon_class"]
    columns = list(dict.fromkeys(columns))
    data = pd.DataFrame(records, columns=columns + (["match_id"] if "match_id" in e else []))
    # Validate chronology even when a map's new label disagrees with old history.
    # The old y/final score columns are not passed through attach_chronology.
    _require(history, {"map_id", "match_id", "start_at", "available_at"}, "history")
    for row in history[["start_at", "available_at"]].to_dict("records"):
        if utc(row["available_at"]) <= utc(row["start_at"]):
            raise ValueError("invalid_history_result_availability")
    data, chronology = attach_chronology(data, history)
    data = data.sort_values(["start_at", "match_id", "map_id", "round_num", "state_tick"]).reset_index(drop=True)
    return data, {"source_event_rows": len(events), "source_event_maps": int(e.demo_path.nunique()),
                  "label_maps": len(labels), "verified_complete_maps_with_events": verified_maps,
                  "excluded_maps": dict(sorted(rejected_maps.items())),
                  "excluded_event_rows": dict(sorted(rejected_events.items())),
                  "retained_event_rows": len(data), "retained_maps": int(data.map_id.nunique()),
                  "retained_series": int(data.match_id.nunique()), "chronology": chronology,
                  "labels": "new_verified_complete_round_history_not_legacy_history_y"}


def hierarchical_weights(data):
    """Each series has mass one, then equal maps, rounds within map, and events."""
    if data.empty:
        return np.array([], dtype=float)
    keys = ["match_id", "map_id", "round_num"]
    events_per_round = data.groupby(keys).map_id.transform("size").to_numpy(dtype=float)
    maps = data[["match_id", "map_id"]].drop_duplicates().groupby("match_id").size()
    rounds = data[keys].drop_duplicates().groupby(["match_id", "map_id"]).size()
    index = pd.MultiIndex.from_frame(data[["match_id", "map_id"]])
    return 1. / (events_per_round * data.match_id.map(maps).to_numpy(dtype=float)
                 * rounds.reindex(index).to_numpy(dtype=float))


def forward_splits(data, *, n_folds=4, initial_fraction=.4, min_train_series=20):
    if (isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 1
            or isinstance(min_train_series, bool) or not isinstance(min_train_series, int) or min_train_series < 1
            or not np.isfinite(initial_fraction) or not 0 < initial_fraction < 1):
        raise ValueError("invalid_forward_split_parameters")
    starts, available = data.start_at.map(utc), data.available_at.map(utc)
    if not available.gt(starts).all():
        raise ValueError("invalid_result_availability")
    for _, group in data.assign(_start=starts, _available=available).groupby("match_id"):
        if group._start.nunique() != 1 or group._available.nunique() != 1:
            raise ValueError("inconsistent_series_chronology")
    times = sorted(starts.unique())
    if len(times) < 2:
        return
    initial = min(len(times) - 1, max(1, int(np.ceil(len(times) * initial_fraction))))
    remaining = np.asarray(times[initial:], dtype=object)
    for block in np.array_split(remaining, min(n_folds, len(remaining))):
        test = starts.isin(block)
        train = starts.lt(block[0]) & available.lt(block[0]) & ~data.match_id.isin(data.loc[test, "match_id"])
        if data.loc[train, "match_id"].nunique() >= min_train_series:
            yield np.flatnonzero(train), np.flatnonzero(test)


def mirror_features(data):
    result = data.copy()
    present = [column for column in ODD_FEATURES if column in result]
    result[present] = -result[present]
    result["map_side"] = result.map_name + np.where(result.a_is_ct.eq(1), "|A_CT", "|A_T")
    return result


def fit_variant_bundle(train, numeric, categorical):
    if set(numeric) - set(APPROVED_NUMERIC) or set(numeric) & FORBIDDEN_FEATURES:
        raise ValueError("unapproved_model_features")
    structure = [column for column in numeric if column in STRUCTURE]
    dense = [column for column in numeric if column not in STRUCTURE]
    # Structure features may carry NaN before train-only imputation; all others
    # must be finite up front.
    if not np.isfinite(train[dense].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite_model_features")
    weights = hierarchical_weights(train)
    weights /= weights.mean()
    paired = pd.concat([train, mirror_features(train)], ignore_index=True)
    paired_weights = np.tile(weights, 2)
    # Impute geometry NaN with paired-train means (train-only information; for
    # these odd features the mirrored mean is exactly 0). Keeps the common
    # cohort identical across layers instead of dropping geometry-missing rows.
    impute = paired[structure].mean() if structure else None

    def _prepared(frame):
        filled = frame.copy()
        if structure:
            filled[structure] = filled[structure].fillna(impute)
        if not np.isfinite(filled[numeric].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite_after_train_imputation")
        return filled

    train_p, paired_p = _prepared(train), _prepared(paired)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(paired_p[categorical])
    scaler = StandardScaler().fit(paired_p[numeric].to_numpy(dtype=float), sample_weight=paired_weights)

    def matrix(frame):
        prepared = _prepared(frame)
        return np.column_stack([encoder.transform(prepared[categorical]),
                                scaler.transform(prepared[numeric].to_numpy(dtype=float))])

    model = LogisticRegression(C=.25, fit_intercept=False, max_iter=2000)
    model.fit(matrix(paired), np.concatenate([train.y.to_numpy(), 1 - train.y.to_numpy()]), sample_weight=paired_weights)
    return {"schema_version": 1, "numeric": list(numeric),
            "categorical": list(categorical), "structure_impute": impute,
            "encoder": encoder, "scaler": scaler, "model": model}


def predict_variant_bundle(bundle, frame):
    numeric, categorical = bundle["numeric"], bundle["categorical"]
    structure = [column for column in numeric if column in STRUCTURE]

    def matrix(source):
        prepared = source.copy()
        if structure:
            prepared[structure] = prepared[structure].fillna(bundle["structure_impute"])
        if not np.isfinite(prepared[numeric].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite_live_model_features")
        return np.column_stack([
            bundle["encoder"].transform(prepared[categorical]),
            bundle["scaler"].transform(prepared[numeric].to_numpy(dtype=float)),
        ])

    # Explicit averaging enforces A/B complements even under optimizer tolerance.
    return .5 * (bundle["model"].predict_proba(matrix(frame))[:, 1]
                 + 1 - bundle["model"].predict_proba(matrix(mirror_features(frame)))[:, 1])


def _fit_predict(train, test, numeric, categorical):
    return predict_variant_bundle(fit_variant_bundle(train, numeric, categorical), test)


def _metrics(frame, probabilities):
    if frame.empty:
        return {"event_rows": 0, "series": 0, "maps": 0, "rounds": 0}
    p = np.asarray(probabilities, dtype=float)
    y = frame.y.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid_prediction_probability")
    weights = hierarchical_weights(frame)
    clipped = np.clip(p, 1e-8, 1 - 1e-8)
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = []
    for index in range(10):
        mask = bins == index
        if mask.any():
            calibration.append({"low": index / 10, "event_rows": int(mask.sum()),
                                "weight": float(weights[mask].sum()),
                                "predicted": float(np.average(p[mask], weights=weights[mask])),
                                "actual": float(np.average(y[mask], weights=weights[mask]))})
    return {"event_rows": len(frame), "series": int(frame.match_id.nunique()), "maps": int(frame.map_id.nunique()),
            "rounds": int(frame[["map_id", "round_num"]].drop_duplicates().shape[0]),
            "brier": float(np.average((p - y) ** 2, weights=weights)),
            "log_loss": float(np.average(-y * np.log(clipped) - (1 - y) * np.log(1 - clipped), weights=weights)),
            "calibration": calibration, "weighting": "equal_series_map_round_event"}


def paired_series_increment(frame, candidate, reference, *, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS):
    """Whole-series resampling, not event-IID or a multiple-comparison test.

    Series can themselves share teams, leagues and calendar regimes. These
    percentile intervals are diagnostic under a resampled-series approximation,
    not proof of significance or an untouched future-acceptance decision.
    """
    weights, y = hierarchical_weights(frame), frame.y.to_numpy(dtype=float)
    candidate_p = frame[candidate].to_numpy(dtype=float)
    reference_p = frame[reference].to_numpy(dtype=float)
    pc, pr = np.clip(candidate_p, 1e-8, 1-1e-8), np.clip(reference_p, 1e-8, 1-1e-8)
    losses = pd.DataFrame({"match_id": frame.match_id.to_numpy(),
                           "brier": weights * ((candidate_p-y)**2 - (reference_p-y)**2),
                           "log_loss": weights * (-y*np.log(pc)-(1-y)*np.log(1-pc)+y*np.log(pr)+(1-y)*np.log(1-pr))})
    series = losses.groupby("match_id", sort=True)[["brier", "log_loss"]].sum()
    bootstrap = {"resampling_unit": "whole_series", "series_n": len(series), "draws": draws,
                 "seed": seed, "nominal_level": .95,
                 "brier_interval": None, "log_loss_interval": None,
                 "interpretation": "diagnostic_no_temporal_dependence_or_multiple_candidate_correction"}
    if len(series) >= 2:
        sample = np.random.default_rng(seed).integers(0, len(series), size=(draws, len(series)))
        simulated = series.to_numpy()[sample].mean(axis=1)
        bounds = np.quantile(simulated, [.025, .975], axis=0)
        for index, name in enumerate(("brier", "log_loss")):
            bootstrap[name + "_interval"] = [float(value) for value in bounds[:, index]]
    return {**{name: float(series[name].mean()) for name in ("brier", "log_loss")},
            "series_resampling": bootstrap}


def evaluate_frame(data, *, n_folds=4, initial_fraction=.4, min_train_series=20, progress=None):
    emit = progress or (lambda message: None)
    d = data.sort_values(["start_at", "match_id", "map_id", "round_num", "state_tick"]).reset_index(drop=True).copy()
    common = np.isfinite(d[ALL_NUMERIC].to_numpy(dtype=float)).all(axis=1) & d.y.isin([0, 1])
    excluded = int((~common).sum())
    d = d.loc[common].reset_index(drop=True)
    if d.duplicated(["map_id", "round_num", "state_tick"]).any():
        raise ValueError("duplicate_evaluation_event_identity")
    for name in ["constant_50", *VARIANTS]:
        d["p_" + name] = np.nan
    d["fold"] = -1
    d["train_max_available_at"] = ""
    folds = []
    for number, (tr, te) in enumerate(forward_splits(d, n_folds=n_folds, initial_fraction=initial_fraction,
                                                   min_train_series=min_train_series), 1):
        train, test = d.iloc[tr], d.iloc[te]
        if train.y.nunique() < 2:
            folds.append({"fold": number, "status": "skipped_single_training_class"})
            continue
        emit(f"Map Winner fold {number}: {train.match_id.nunique()} training / {test.match_id.nunique()} test series")
        for name, numeric in VARIANTS.items():
            d.loc[te, "p_" + name] = _fit_predict(train, test, numeric, CATEGORICAL[name])
        d.loc[te, "p_constant_50"] = .5
        d.loc[te, "fold"] = number
        d.loc[te, "train_max_available_at"] = train.available_at.max().isoformat()
        folds.append({"fold": number, "status": "scored", "train_series": int(train.match_id.nunique()),
                      "test_series": int(test.match_id.nunique()), "train_maps": int(train.map_id.nunique()),
                      "test_maps": int(test.map_id.nunique()), "train_event_rows": len(train), "test_event_rows": len(test),
                      "train_max_available_at": train.available_at.max().isoformat(),
                      "test_min_start_at": test.start_at.min().isoformat(), "test_max_start_at": test.start_at.max().isoformat(),
                      "train_series_ids": sorted(train.match_id.unique().tolist()),
                      "test_series_ids": sorted(test.match_id.unique().tolist()),
                      "series_overlap": len(set(train.match_id) & set(test.match_id))})
    prediction_columns = ["p_" + name for name in ["constant_50", *VARIANTS]]
    scored = d[np.isfinite(d[prediction_columns].to_numpy(dtype=float)).all(axis=1)]
    metrics = {name: _metrics(scored, scored["p_" + name]) for name in ["constant_50", *VARIANTS]}
    paired = {}
    if len(scored):
        for candidate, reference in (("score_side", "constant_50"),
                                     ("fivee_score", "constant_50"),
                                     ("fivee_state", "fivee_score"),
                                     ("fivee_state", "score_side"),
                                     ("starting_economy", "score_side"),
                                     ("alive_hp_bomb", "starting_economy"), ("alive_hp_bomb", "score_side"),
                                     ("geometry_weapon", "alive_hp_bomb"), ("geometry_weapon", "score_side")):
            paired[candidate + "_minus_" + reference] = paired_series_increment(
                scored, "p_" + candidate, "p_" + reference)
    report = {"version": VERSION, "status": "diagnostic_only" if len(scored) else "insufficient_past_series",
              "target": "canonical_roster_a_map_winner", "source_event_rows": len(data),
              "common_event_rows": len(d), "excluded_nonfinite_or_invalid_target": excluded,
              "scored_event_rows": len(scored), "scored_maps": int(scored.map_id.nunique()),
              "scored_series": int(scored.match_id.nunique()), "folds": folds, "models": metrics,
              "paired_loss_increment_candidate_minus_reference": paired,
              "parameters": {"n_folds": n_folds, "initial_time_fraction": initial_fraction,
                             "min_train_series": min_train_series, "logistic_C": .25,
                             "series_bootstrap_draws": BOOTSTRAP_DRAWS, "series_bootstrap_seed": BOOTSTRAP_SEED,
                             "features": {name: CATEGORICAL[name] + columns for name, columns in VARIANTS.items()},
                             "mirrored_after_split": True, "weighted_scaler_train_only": True},
              "acceptance_status": "existing_history_diagnostic_not_untouched_acceptance",
              "promoted_model": None, "live_trading_enabled": False, "execution_pnl": None,
              "limitations": ["Score baseline has no prematch team-strength prior and is not a market-price baseline.",
                              "fivee_state matches the verified live field subset but historical event-triggered sampling "
                              "does not match 5E's irregular delayed push cadence.",
                              "Starting equipment is freeze-time equipment, not current equipment or cash.",
                              "Bomb state lacks a verified remaining-bomb-time feature; positions are intentionally excluded.",
                              "geometry_weapon contact/spread/aiming are geometric proxies without walls, smokes or vertical occlusion; "
                              "weapon_class describes the current event only and is 'none' for non-kill events.",
                              "Same-event comparison estimates information gain conditional on this event-sampling policy.",
                              "Event, round and map rows are dependent; no event-IID confidence or profitability claim.",
                              "Current history was already available for research, not a new unseen validation window.",
                              "Demo clock and series availability proxies do not establish live game/market synchronization."]}
    return d, report


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(data_dir, output_dir, history_path=None, *, n_folds=4, initial_fraction=.4, min_train_series=20):
    folder, output = Path(data_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError("research_output_exists_refusing_overwrite")
    sources = {"events": folder / "inround_events.parquet", "rounds": folder / "round_states.parquet",
               "labels": folder / "map_labels.parquet",
               "history": Path(history_path) if history_path is not None else config.DATA_DIR / "map1" / "history.parquet"}
    hashes = {key: _hash(path) for key, path in sources.items()}
    code_paths = {name: Path(__file__).with_name(name) for name in ("inplay_map_research.py", "inround_model.py", "map1_data.py")}
    code_hashes = {name: _hash(path) for name, path in code_paths.items()}
    events, rounds, labels, history = (pd.read_parquet(sources[key]) for key in ("events", "rounds", "labels", "history"))
    data, audit = prepare_map_events(events, rounds, labels, history)
    predictions, report = evaluate_frame(data, n_folds=n_folds, initial_fraction=initial_fraction,
                                         min_train_series=min_train_series, progress=lambda text: print(text, flush=True))
    generated_at = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(generated_at)
    training = data[data.available_at.map(utc).lt(cutoff)].copy()
    common = np.isfinite(training[ALL_NUMERIC].to_numpy(dtype=float)).all(axis=1) & training.y.isin([0, 1])
    training = training.loc[common].reset_index(drop=True)
    paper_models = {
        name: fit_variant_bundle(training, VARIANTS[name], CATEGORICAL[name])
        for name in ("fivee_score", "fivee_state")
    }
    if hashes != {key: _hash(path) for key, path in sources.items()}:
        raise ValueError("research_inputs_changed_while_running")
    if code_hashes != {name: _hash(path) for name, path in code_paths.items()}:
        raise ValueError("research_code_changed_while_running")
    report.update(generated_at=generated_at.isoformat(), quality_audit=audit,
                  inputs={key: {"path": str(path.resolve()), "sha256": hashes[key]} for key, path in sources.items()},
                  code_hashes=code_hashes,
                  paper_model_artifact={"path": "models.joblib", "promoted": False,
                                        "training_cutoff": generated_at.isoformat(),
                                        "training_event_rows": len(training),
                                        "training_maps": int(training.map_id.nunique()),
                                        "training_series": int(training.match_id.nunique()),
                                        "variants": ["fivee_score", "fivee_state"]})
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    joblib.dump({"schema_version": 1, "research_version": VERSION,
                 "paper_only": True, "promoted": False,
                 "trained_before": generated_at.isoformat(),
                 "variants": paper_models}, output / "models.joblib")
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-series", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=4)
    args = parser.parse_args(argv)
    report = run(args.data_dir, args.output_dir, args.metadata_path,
                 n_folds=args.n_folds, min_train_series=args.min_train_series)
    print(json.dumps({"status": report["status"], "scored_series": report["scored_series"],
                      "report": str((args.output_dir / "report.json").resolve()), "promoted_model": None}, indent=2))


if __name__ == "__main__":
    main()
