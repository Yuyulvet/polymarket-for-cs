"""Past-only 30-second round-state research, not a live or market-edge model.

Legacy midround.parquet is unsafe and is never read or overwritten here.
Explicitly rebuild a new cache with ``python -m cs2ml.midround --rebuild``;
evaluate it separately with ``python -m cs2ml.midround --evaluate``.

Snapshots describe the end of freeze_end + 30 * 64 ticks, inclusive. A round
must still be open at that cutoff. Actual snapshot sides handle overtime;
future kills cannot change the first-kill feature. Tick time is demo time,
not evidence of when an external live feed or Polymarket received information.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .duel import discover_demos
from .map1_data import path_key, utc

SECONDS = 30
TICKS = 64
CACHE_VERSION = 2
CACHE_NAME = "midround_v2.parquet"
CAT = ["map_name", "round_class", "ct_tier", "t_tier"]
NUM_BASE = ["ct_equip", "t_equip", "round_num"]
NUM_30 = ["ct_alive", "t_alive"]
CAT_30 = ["first_kill_side"]
STATE_COLUMNS = ["schema_version", "demo_path", "match_id", "round_num",
                 "freeze_tick", "cutoff_tick", "round_end_tick", "seconds", "tick_rate",
                 "ct_deaths30", "t_deaths30", "ct_alive", "t_alive", "first_kill_side"]


def _integers(values, name):
    converted = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(converted).all() or (converted < 0).any() or (converted != np.floor(converted)).any():
        raise ValueError(f"invalid_{name}")
    return converted.astype(np.int64)


def round_windows(freeze, round_end, *, seconds=SECONDS, tick_rate=TICKS, terminal_ticks=()):
    """Pure cutoff policy: no inferred ends, restarted rounds or closed snapshots."""
    if not np.isfinite([seconds, tick_rate]).all() or seconds <= 0 or tick_rate <= 0:
        raise ValueError("invalid_snapshot_timing")
    duration = seconds * tick_rate
    if duration != int(duration):
        raise ValueError("snapshot_duration_requires_integer_ticks")
    ticks = sorted(_integers(freeze["tick"], "freeze_ticks"))
    terminal = _integers(terminal_ticks, "terminal_ticks")
    if len(set(ticks)) != len(ticks):
        raise ValueError("duplicate_freeze_ticks")
    ends = round_end.copy()
    for column in ("tick", "round"):
        ends[column] = _integers(ends[column], f"round_end_{column}")
    ends = ends[ends["round"].gt(0)]
    if ends["round"].duplicated().any():
        raise ValueError("ambiguous_round_end_identity")
    by_round = ends.set_index("round").to_dict("index")
    rows = []
    for index, freeze_tick in enumerate(ticks):
        number = index + 1
        end = by_round.get(number)
        if end is None or str(end.get("winner", "")).upper() not in ("CT", "T", "TERRORIST"):
            continue
        cutoff = int(freeze_tick + duration)
        end_tick = int(end["tick"])
        next_freeze = ticks[index + 1] if index + 1 < len(ticks) else np.inf
        if not freeze_tick < cutoff < end_tick < next_freeze:
            continue
        if ((terminal >= freeze_tick) & (terminal <= cutoff)).any():
            continue
        rows.append({"round_num": number, "freeze_tick": int(freeze_tick),
                     "cutoff_tick": cutoff, "round_end_tick": end_tick})
    return pd.DataFrame(rows, columns=["round_num", "freeze_tick", "cutoff_tick", "round_end_tick"])


def states_from_events(dp, windows, kills, snapshots, *, seconds=SECONDS, tick_rate=TICKS):
    """Pure extraction; no winner or after-cutoff event enters a feature.

    Require ten distinct players at the exact tick. Alive state handles world
    damage/suicides. First kill means enemy kill; opposite same-tick openings are
    simultaneous because within-tick order is unavailable.
    """
    path = Path(dp)
    if windows.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)
    snap = snapshots.copy()
    snap["tick"] = _integers(snap["tick"], "snapshot_ticks")
    snap["steamid"] = snap["steamid"].astype(str)
    snap["side"] = snap["team_name"].astype(str).str.upper().replace({"TERRORIST": "T"})
    snap = snap[snap.side.isin(["CT", "T"])]
    deaths = kills.copy()
    if deaths.empty:
        deaths = pd.DataFrame(columns=["tick", "attacker_steamid", "user_steamid"])
    else:
        deaths["tick"] = _integers(deaths["tick"], "death_ticks")
    rows = []
    for window in windows.to_dict("records"):
        cutoff = window["cutoff_tick"]
        if not window["freeze_tick"] < cutoff < window["round_end_tick"]:
            raise ValueError("snapshot_not_in_active_round")
        current = snap[snap.tick.eq(cutoff)]
        if (len(current) != 10 or current.steamid.duplicated().any()
                or not current.steamid.str.fullmatch(r"[1-9][0-9]*").all()
                or current.groupby("side").size().to_dict() != {"CT": 5, "T": 5}
                or not current.is_alive.isin([True, False, 0, 1]).all()):
            continue
        sides = current.set_index("steamid").side.to_dict()
        alive = current[current.is_alive.astype(bool)].groupby("side").size().to_dict()
        before = deaths[deaths.tick.ge(window["freeze_tick"]) & deaths.tick.le(cutoff)].copy()
        before["attacker_side"] = before.attacker_steamid.astype(str).map(sides)
        before["victim_side"] = before.user_steamid.astype(str).map(sides)
        enemy = before[before.attacker_side.notna() & before.victim_side.notna()
                       & before.attacker_side.ne(before.victim_side)]
        first = "none"
        if not enemy.empty:
            opening_sides = set(enemy.loc[enemy.tick.eq(enemy.tick.min()), "attacker_side"])
            first = next(iter(opening_sides)) if len(opening_sides) == 1 else "simultaneous"
        ct_alive, t_alive = int(alive.get("CT", 0)), int(alive.get("T", 0))
        # round_end can lag elimination. No bomb-state field exists here, so
        # conservatively exclude even T=0 with a planted bomb still active.
        if min(ct_alive, t_alive) == 0:
            continue
        rows.append({"schema_version": CACHE_VERSION, "demo_path": str(path), "match_id": path.parent.name,
                     **window, "seconds": seconds, "tick_rate": tick_rate,
                     "ct_deaths30": 5 - ct_alive, "t_deaths30": 5 - t_alive,
                     "ct_alive": ct_alive, "t_alive": t_alive, "first_kill_side": first})
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


def compute_midround(dp: Path) -> pd.DataFrame | None:
    parser = DemoParser(str(dp))
    freeze, ends = parser.parse_event("round_freeze_end"), parser.parse_event("round_end")
    if freeze is None or not len(freeze) or ends is None or not len(ends):
        return None
    terminal_ticks = []
    for event_name in ("bomb_defused", "bomb_exploded"):
        terminal = parser.parse_event(event_name)
        if terminal is not None and len(terminal):
            terminal_ticks.extend(terminal.tick.tolist())
    windows = round_windows(freeze, ends, terminal_ticks=terminal_ticks)
    if windows.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)
    snapshots = parser.parse_ticks(["team_name", "is_alive"], ticks=windows.cutoff_tick.tolist())
    if snapshots is None or not len(snapshots):
        return None
    kills = parser.parse_event("player_death")
    if kills is None or not len(kills):
        kills = pd.DataFrame()
    return states_from_events(dp, windows, kills, snapshots)


def _validate_cache(frame):
    if frame.empty or not set(STATE_COLUMNS).issubset(frame.columns):
        raise ValueError("missing_safe_midround_cache_schema_rebuild_required")
    for column in ("round_num", "freeze_tick", "cutoff_tick", "round_end_tick",
                   "ct_alive", "t_alive", "ct_deaths30", "t_deaths30"):
        _integers(frame[column], f"cache_{column}")
    if (not frame.schema_version.eq(CACHE_VERSION).all()
            or not frame.seconds.eq(SECONDS).all() or not frame.tick_rate.eq(TICKS).all()
            or not frame.round_num.ge(1).all()
            or not frame.cutoff_tick.lt(frame.round_end_tick).all()
            or not frame.cutoff_tick.eq(frame.freeze_tick + SECONDS * TICKS).all()
            or not frame.ct_alive.between(1, 5).all() or not frame.t_alive.between(1, 5).all()
            or not frame.ct_deaths30.eq(5 - frame.ct_alive).all()
            or not frame.t_deaths30.eq(5 - frame.t_alive).all()
            or not frame.first_kill_side.isin(["none", "CT", "T", "simultaneous"]).all()
            or frame.assign(demo_path=frame.demo_path.map(path_key)).duplicated(["demo_path", "round_num"]).any()):
        raise ValueError("unsafe_midround_cache_rebuild_required")
    return frame


def build_midround(dem_paths: list[Path] | None = None, *, rebuild=False, cache_path=None) -> pd.DataFrame:
    """Explicit paths parse in memory. Cache writes require rebuild=True.

    Never use the old cache, implicitly rebuild, or overwrite any artifact.
    """
    cache = Path(cache_path) if cache_path is not None else config.DATA_DIR / CACHE_NAME
    if dem_paths is not None and (rebuild or cache_path is not None):
        raise ValueError("explicit_demo_paths_are_in_memory_only")
    if dem_paths is None:
        if rebuild and cache.exists():
            raise FileExistsError(f"Refusing to overwrite {cache}; choose a new --cache path")
        if not rebuild:
            if not cache.exists():
                raise FileNotFoundError(f"Safe cache missing: {cache}; explicitly rebuild with --rebuild")
            return _validate_cache(pd.read_parquet(cache))
    demos = discover_demos() if dem_paths is None else list(dem_paths)
    frames = []
    for dp in demos:
        try:
            frame = compute_midround(Path(dp))
            if frame is not None and not frame.empty:
                frames.append(frame)
        except Exception as error:
            print(f"Skipped {Path(dp).name}: {error}")
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STATE_COLUMNS)
    if dem_paths is None and not result.empty:
        _validate_cache(result)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("xb") as handle:
            result.to_parquet(handle, index=False)
    return result


def temporal_series_splits(frame, min_train_series=2):
    """Expanding result-aware splits; same-series maps and tied starts stay out."""
    if isinstance(min_train_series, bool) or not isinstance(min_train_series, int) or min_train_series < 1:
        raise ValueError("invalid_minimum_training_series")
    required = {"match_id", "start_at", "available_at"}
    if not required.issubset(frame.columns):
        raise ValueError("series_timing_metadata_required_no_group_kfold_fallback")
    data = frame.reset_index(drop=True).copy()
    data["start_at"], data["available_at"] = data.start_at.map(utc), data.available_at.map(utc)
    if data.match_id.isna().any() or data.groupby("match_id")[["start_at", "available_at"]].nunique().gt(1).any().any():
        raise ValueError("inconsistent_series_timing")
    if not data.available_at.gt(data.start_at).all():
        raise ValueError("series_result_not_after_start")
    for at, targets in data.groupby("start_at", sort=True):
        train = data[data.available_at.lt(at) & data.start_at.lt(at)
                     & ~data.match_id.isin(targets.match_id)]
        if train.match_id.nunique() >= min_train_series:
            yield train.index.to_numpy(), targets.index.to_numpy()


def _auc(df: pd.DataFrame, features: list[str]) -> float:
    if not set(features).issubset(CAT + NUM_BASE + NUM_30 + CAT_30):
        raise ValueError("unapproved_midround_model_feature")
    cat = [column for column in features if column in CAT + CAT_30]
    num = [column for column in features if column not in CAT + CAT_30]
    proba = np.full(len(df), np.nan)
    for train, test in temporal_series_splits(df):
        if df.iloc[train].label_ct_win.nunique() < 2:
            continue
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                                 ("num", StandardScaler(), num)])
        model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
        model.fit(df.iloc[train][features], df.iloc[train].label_ct_win)
        proba[test] = model.predict_proba(df.iloc[test][features])[:, 1]
    valid = np.isfinite(proba)
    if not valid.any() or df.loc[valid, "label_ct_win"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(df.loc[valid, "label_ct_win"], proba[valid]))


def evaluate(*, cache_path=None, history_path=None) -> dict:
    # Read only; do not trigger a legacy round cache rebuild as a side effect.
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    mr = build_midround(cache_path=cache_path)
    metadata = pd.read_parquet(history_path or config.DATA_DIR / "map1" / "history.parquet")
    metadata = metadata[["map_id", "match_id", "start_at", "available_at"]].copy()
    metadata["map_id"] = metadata.map_id.map(path_key)
    df = rd.merge(mr, on=["demo_path", "round_num"], how="inner", suffixes=("", "_m"), validate="one_to_one")
    if not df.match_id.eq(df.match_id_m).all():
        raise ValueError("midround_series_identity_mismatch")
    df["map_id"] = df.demo_path.map(path_key)
    merged_rows = len(df)
    df = df.merge(metadata, on=["map_id", "match_id"], how="inner", validate="many_to_one")
    timed_rows = len(df)
    # Legacy base economics only flip sides once: OT is not a safe comparator.
    df = df[df.round_num.le(24)].dropna(subset=["ct_equip", "t_equip"]).copy()
    if df.empty:
        raise ValueError("no_safe_midround_rows_with_clean_series_timing")
    result = {"mode": "offline_temporal_round_winner_diagnostic", "n_rounds": len(df),
              "n_series": int(df.match_id.nunique()),
              "excluded_without_clean_timing": merged_rows - timed_rows,
              "excluded_overtime_or_missing_base": timed_rows - len(df),
              "base": _auc(df, CAT + NUM_BASE),
              "mid": _auc(df, CAT + NUM_BASE + NUM_30 + CAT_30),
              "only_adv": _auc(df, CAT + NUM_BASE + NUM_30),
              "timing_basis": "series_end_plus_lag_proxy_not_observed_live_receipt",
              "baseline_scope": "regulation_rounds_only_legacy_economic_side_mapping",
              "snapshot_scope": "both_sides_have_survivors_no_bomb_state_so_all_zero_survivor_sides_excluded",
              "execution_pnl": None, "live_trading_enabled": False}
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rebuild", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--history", type=Path)
    args = parser.parse_args()
    if args.rebuild:
        result = build_midround(rebuild=True, cache_path=args.cache)
        print(f"New safe 30-second cache: {len(result)} rows; old artifacts unchanged")
    else:
        evaluate(cache_path=args.cache, history_path=args.history)


if __name__ == "__main__":
    main()
