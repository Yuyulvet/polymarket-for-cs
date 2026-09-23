"""Map-isolated, availability-aware MR12/MR3 economic simulation.

The legacy cache labels overtime sides as if there were one side switch. Repair
those labels by stable roster identity; never treat cached winner_roster as truth.
Incomplete maps and missing probabilities/transitions block predictions. Historical
match-end availability is a proxy, not evidence of live timing or executable profit.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config
from .map1_data import first_ct_is_ct, path_key, roster_key, terminal_score, utc
from .round_model import _round_class, build_round_dataset

CAT = ["map_name", "round_class", "ct_tier", "t_tier"]
TIERS = ("eco", "force", "full")


def _phase(round_no: int) -> str:
    return "regulation" if round_no <= 24 else "overtime"


def _model_round_class(round_no: int) -> str:
    return _round_class(round_no) if round_no <= 24 else "overtime"


def _is_reset(round_no: int) -> bool:
    return round_no == 13 or (round_no >= 25 and (round_no - 25) % 3 == 0)


def _map_groups(frame: pd.DataFrame):
    key = "demo_path" if "demo_path" in frame else "map_id"
    if key not in frame or frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
        raise ValueError("missing_map_identity")
    return frame.groupby(frame[key].map(path_key), sort=True)


def normalize_map_rounds(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a complete map and reorient legacy roster-based CT/T fields.

    All paired ct_*/t_* fields follow their stored roster. winner_side is a raw
    engine-side outcome and must NOT be swapped. Legacy first_kill_side and gaps
    are derived from those roster fields and therefore need reconstruction too.
    This function never changes cached files.
    """
    required = {"match_id", "map_name", "round_num", "ct_roster", "t_roster",
                "ct_tier", "t_tier", "winner_side"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("missing_round_fields")
    if len(list(_map_groups(frame))) != 1:
        raise ValueError("mixed_map_identity")
    d = frame.sort_values("round_num").copy().reset_index(drop=True)
    if d.round_num.tolist() != list(range(1, len(d) + 1)):
        raise ValueError("non_contiguous_rounds")
    for field in ("match_id", "map_name"):
        if d[field].isna().any() or d[field].nunique() != 1 or not str(d.iloc[0][field]).strip():
            raise ValueError("mixed_or_missing_map_metadata")
    if "map_id" in d and (d.map_id.isna().any() or d.map_id.nunique() != 1):
        raise ValueError("mixed_map_identity")
    first_ct, first_t = (roster_key(d.iloc[0][side + "_roster"]) for side in ("ct", "t"))
    if set(first_ct.split(",")) & set(first_t.split(",")):
        raise ValueError("overlapping_rosters")
    scores = {first_ct: 0, first_t: 0}
    rows = []
    for i, row in enumerate(d.to_dict("records"), 1):
        if terminal_score(scores[first_ct], scores[first_t]):
            raise ValueError("rounds_after_terminal_score")
        old_ct, old_t = (roster_key(row[side + "_roster"]) for side in ("ct", "t"))
        if {old_ct, old_t} != {first_ct, first_t}:
            raise ValueError("roster_changed_within_map")
        ct = first_ct if first_ct_is_ct(i) else first_t
        t = first_t if ct == first_ct else first_ct
        if i <= 24 and old_ct != ct:
            raise ValueError("unexpected_regulation_side_assignment")
        winner_side = str(row["winner_side"]).upper().replace("TERRORIST", "T")
        if winner_side not in {"CT", "T"}:
            raise ValueError("unknown_round_winner")
        if row["ct_tier"] not in TIERS or row["t_tier"] not in TIERS:
            raise ValueError("missing_or_unknown_equipment_tier")
        if old_ct != ct:
            for key in list(row):
                other = "t_" + key[3:]
                if key.startswith("ct_") and other in row:
                    row[key], row[other] = row[other], row[key]
            if row.get("first_kill_side") in {"CT", "T"}:
                row["first_kill_side"] = "T" if row["first_kill_side"] == "CT" else "CT"
        for key in list(row):
            if key.endswith("_gap"):
                base = key[:-4]
                if "ct_" + base in row and "t_" + base in row:
                    row[key] = row["ct_" + base] - row["t_" + base]
        if "rwaff" in row:
            first = row.get("first_kill_side")
            row["rwaff"] = int(first == winner_side) if first in {"CT", "T"} else np.nan
        winner = ct if winner_side == "CT" else t
        scores[winner] += 1
        row.update(ct_roster=ct, t_roster=t, winner_side=winner_side,
                   winner_roster=winner, label_ct_win=int(winner_side == "CT"),
                   round_class=_model_round_class(i),
                   score0=scores[first_ct], score1=scores[first_t],
                   legacy_side_repaired=old_ct != ct)
        rows.append(row)
    if not terminal_score(scores[first_ct], scores[first_t]):
        raise ValueError("incomplete_or_nonstandard_map")
    return pd.DataFrame(rows)


def _prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    valid, rejected = [], Counter()
    for _, group in _map_groups(frame):
        try:
            valid.append(normalize_map_rounds(group))
        except (ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
    data = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
    return data, {"usable_maps": len(valid), "excluded_maps": dict(rejected),
                  "reoriented_overtime_rows": int(data.legacy_side_repaired.sum()) if len(data) else 0}


def _fit_round_lookup(train: pd.DataFrame) -> dict:
    if train.empty or train.label_ct_win.nunique() != 2:
        raise ValueError("training_requires_both_round_outcomes")
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="error"), CAT)])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    model.fit(train[CAT], train.label_ct_win)
    # No synthetic unseen-map/OT-phase or unseen equipment category lookup.
    pairs = train[["map_name", "round_class"]].drop_duplicates().itertuples(index=False, name=None)
    combos = pd.DataFrame([(m, rc, ct, t) for m, rc in pairs for ct in TIERS for t in TIERS], columns=CAT)
    for col in ("ct_tier", "t_tier"):
        combos = combos[combos[col].isin(train[col].unique())]
    probabilities = model.predict_proba(combos[CAT])[:, 1]
    return dict(zip(combos.itertuples(index=False, name=None), probabilities))


def _fit_transition(train: pd.DataFrame) -> dict:
    """Fit adjacent-round, roster-oriented transitions and joint reset buys."""
    normal, resets = defaultdict(Counter), defaultdict(Counter)
    for _, group in _map_groups(train):
        d = normalize_map_rounds(group)
        for row in d.itertuples(index=False):
            if _is_reset(row.round_num):
                resets[_phase(row.round_num)][(row.ct_tier, row.t_tier)] += 1
        for i in range(len(d) - 1):
            current, following = d.iloc[i], d.iloc[i + 1]
            if _is_reset(int(following.round_num)):
                continue
            next_tiers = {following.ct_roster: following.ct_tier, following.t_roster: following.t_tier}
            for side in ("ct", "t"):
                roster = current[side + "_roster"]
                key = (_phase(int(current.round_num)), current[side + "_tier"],
                       int(current.winner_roster == roster))
                normal[key][next_tiers[roster]] += 1
    def distributions(counts):
        return {key: {outcome: n / sum(values.values()) for outcome, n in values.items()}
                for key, values in counts.items()}
    return {"normal": distributions(normal), "resets": distributions(resets)}


def _draw(distribution: dict | None, rng: np.random.Generator):
    if not distribution:
        raise ValueError("missing_equipment_transition")
    values, weights = list(distribution), np.asarray(list(distribution.values()), dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("invalid_transition_distribution")
    return values[int(rng.choice(len(values), p=weights))]


def _validate_start(start_round: int, s0: int, s1: int, n_sims: int, max_rounds: int):
    vals = (start_round, s0, s1, n_sims, max_rounds)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in vals):
        raise ValueError("integer_simulation_state_required")
    if s0 < 0 or s1 < 0 or start_round != s0 + s1 + 1 or n_sims < 1 or max_rounds < start_round:
        raise ValueError("invalid_simulation_state")
    if terminal_score(s0, s1):
        raise ValueError("map_already_finished")
    base = 12 + 3 * ((s0 + s1 - 24) // 6)
    if (start_round <= 25 and max(s0, s1) > 12) or (start_round > 25 and not (
            min(s0, s1) >= base and max(s0, s1) <= base + 3)):
        raise ValueError("unreachable_score")


def _simulate(lookup: dict, trans: dict, map_name: str, start_round: int,
              s0: int, s1: int, t0: str, t1: str, n_sims: int,
              rng: np.random.Generator, max_rounds: int = 600) -> float:
    """P(first-half CT roster wins), conditional on observed start-round buys.

    Future reset purchases use training data, never an assumed full/eco buy.
    If any sampled path is unsupported or reaches the cap, fail the whole
    prediction instead of assigning a winner or discarding that path.
    """
    _validate_start(start_round, s0, s1, n_sims, max_rounds)
    if t0 not in TIERS or t1 not in TIERS:
        raise ValueError("missing_or_unknown_equipment_tier")
    wins = 0
    for _ in range(n_sims):
        a, b, ta, tb, r = s0, s1, t0, t1, start_round
        while True:
            if r > max_rounds:
                raise ValueError("simulation_round_cap_reached")
            first_ct = first_ct_is_ct(r)
            ct_tier, t_tier = (ta, tb) if first_ct else (tb, ta)
            key = (map_name, _model_round_class(r), ct_tier, t_tier)
            if key not in lookup:
                raise ValueError("missing_round_probability")
            p = float(lookup[key])
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("invalid_round_probability")
            won0 = (rng.random() < p) == first_ct
            a, b = a + int(won0), b + int(not won0)
            if terminal_score(a, b):
                wins += int(a > b)
                break
            r += 1
            if _is_reset(r):
                ct_buy, t_buy = _draw(trans.get("resets", {}).get(_phase(r)), rng)
                ta, tb = (ct_buy, t_buy) if first_ct_is_ct(r) else (t_buy, ct_buy)
            else:
                distributions = trans.get("normal", {})
                ta = _draw(distributions.get((_phase(r), ta, int(won0))), rng)
                tb = _draw(distributions.get((_phase(r), tb, int(not won0))), rng)
            if ta not in TIERS or tb not in TIERS:
                raise ValueError("invalid_transition_equipment_tier")
    return wins / n_sims


def _actual_winner(sub: pd.DataFrame) -> str | None:
    try:
        d = normalize_map_rounds(sub)
    except (ValueError, TypeError):
        return None
    return str(d.iloc[0].ct_roster if d.iloc[-1].score0 > d.iloc[-1].score1 else d.iloc[0].t_roster)


def attach_history_times(data: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Strict many-to-one attachment; no schedule-time/result-time fallbacks."""
    required = {"map_id", "match_id", "start_at", "available_at"}
    if not required.issubset(history.columns):
        raise ValueError("missing_availability_metadata")
    meta = history[list(required)].copy()
    meta["map_id"] = meta.map_id.map(path_key)
    if meta.map_id.duplicated().any():
        raise ValueError("ambiguous_map_metadata")
    meta["start_at"], meta["available_at"] = meta.start_at.map(utc), meta.available_at.map(utc)
    if not meta.available_at.gt(meta.start_at).all():
        raise ValueError("invalid_availability_metadata")
    if meta.groupby("match_id")[["start_at", "available_at"]].nunique().gt(1).any().any():
        raise ValueError("inconsistent_series_metadata")
    d = data.copy()
    d["map_id"] = d["demo_path" if "demo_path" in d else "map_id"].map(path_key)
    d = d.drop(columns=["start_at", "available_at"], errors="ignore")
    # Both identity keys must match. Unlinked maps stay excluded from evaluation.
    return d.merge(meta, on=["map_id", "match_id"], how="inner", validate="many_to_one")


def evaluate_forward(n_sims: int = 500, history: pd.DataFrame | None = None,
                     min_train_series: int = 2) -> dict:
    df, audit = _prepare_dataset(build_round_dataset())
    if df.empty:
        return {"error": "no_valid_maps", "audit": audit}
    if min_train_series < 1:
        raise ValueError("min_train_series_must_be_positive")
    try:
        if history is None:
            history = pd.read_parquet(config.DATA_DIR / "map1" / "history.parquet")
        before = len(list(_map_groups(df)))
        df = attach_history_times(df, history)
        audit["missing_metadata_maps"] = before - (len(list(_map_groups(df))) if len(df) else 0)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"error": "availability_metadata_required:" + str(exc), "audit": audit}
    checkpoints = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 28, 31, 34]
    acc, base_acc = {c: [] for c in checkpoints}, {c: [] for c in checkpoints}
    skipped, folds = Counter(), []
    for at, test in df.groupby("start_at", sort=True):
        train = df[df.available_at.lt(at) & df.start_at.lt(at) & ~df.match_id.isin(test.match_id)]
        if train.match_id.nunique() < min_train_series:
            skipped["insufficient_training_series"] += int(test.match_id.nunique())
            continue
        try:
            lookup, trans = _fit_round_lookup(train), _fit_transition(train)
        except ValueError as exc:
            skipped["fold:" + str(exc)] += 1
            continue
        folds.append({"cutoff": at.isoformat(), "train_max_available_at": train.available_at.max().isoformat(),
                      "train_series": int(train.match_id.nunique()), "test_series": int(test.match_id.nunique())})
        rng = np.random.default_rng(0)
        for _, sub in _map_groups(test):
            sub = sub.sort_values("round_num")
            roster0, winner = sub.iloc[0].ct_roster, _actual_winner(sub)
            for c in checkpoints:
                row = sub[sub.round_num == c]
                if row.empty:
                    continue
                prior, current = sub[sub.round_num < c], row.iloc[0]
                s0 = int(prior.winner_roster.eq(roster0).sum())
                s1 = len(prior) - s0
                t0, t1 = ((current.ct_tier, current.t_tier) if current.ct_roster == roster0
                          else (current.t_tier, current.ct_tier))
                try:
                    p = _simulate(lookup, trans, current.map_name, c, s0, s1, t0, t1, n_sims, rng)
                except ValueError as exc:
                    skipped[str(exc)] += 1
                    continue
                truth = winner == roster0
                acc[c].append(int((p > .5) == truth))
                base_acc[c].append(int((s0 >= s1) == truth))
    result = {"acc": {c: float(np.mean(v)) for c, v in acc.items() if v},
              "baseline": {c: float(np.mean(v)) for c, v in base_acc.items() if v},
              "n": {c: len(v) for c, v in acc.items()}, "skipped": dict(skipped), "audit": audit,
              "folds": folds, "validation": "expanding_series_isolated_result_availability_proxy",
              "limitations": ["No observed live feed receipt times or executable market prices.",
                              "Tier transitions pool teams/maps within regulation or overtime.",
                              "Standard MR12/MR3 with sides retained between overtime blocks only."]}
    print(result)
    return result


if __name__ == "__main__":
    evaluate_forward()
