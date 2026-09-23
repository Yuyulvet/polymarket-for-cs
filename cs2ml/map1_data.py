"""Map-level, roster-oriented MR12/MR3 history for the Map 1 pilot.

Never infer a team's identity from a market's final price. Legacy round caches
only switch sides once; rebuild winners from winner_side, including overtime.
All exclusions and the assumed result-publication lag are reported explicitly.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError("A non-null timezone-aware timestamp is required")
    return t.tz_convert("UTC")


def roster_key(value) -> str:
    players = value.split(",") if isinstance(value, str) else list(value)
    players = [str(p).strip() for p in players]
    if len(players) != 5 or len(set(players)) != 5 or not all(p.isdigit() for p in players):
        raise ValueError("A roster must contain five distinct Steam IDs")
    return ",".join(sorted(players))


def map_name(value: str) -> str:
    name = str(value).strip().lower().removeprefix("de_").replace(" ", "")
    if not re.fullmatch(r"[a-z0-9_]+", name) or name in {
            "", "unknown", "tbd", "tba", "none", "default", "tobeannounced", "tobedetermined"}:
        raise ValueError("A confirmed map name is required")
    return "de_" + name


def path_key(value: str) -> str:
    return re.sub(r"/+", "/", str(value).replace("\\", "/")).lower()


def first_ct_is_ct(round_no: int) -> bool:
    """Standard MR12 + repeated MR3; sides carry over between overtime blocks."""
    if round_no < 1:
        raise ValueError("round_no must be positive")
    if round_no <= 24:
        return round_no <= 12
    swaps = 1 + (round_no - 25 + 3) // 6
    return swaps % 2 == 0


def terminal_score(a: int, b: int) -> bool:
    hi, lo = max(a, b), min(a, b)
    if lo < 12:
        return hi == 13
    block = (lo - 12) // 3
    return hi == 16 + 3 * block and lo <= 14 + 3 * block


def summarize_map(rounds: pd.DataFrame) -> dict:
    d = rounds.sort_values("round_num")
    if d["round_num"].tolist() != list(range(1, len(d) + 1)):
        raise ValueError("non_contiguous_rounds")
    if d["match_id"].nunique() != 1 or d["map_name"].nunique() != 1:
        raise ValueError("mixed_map_identity")
    first_ct = roster_key(d.iloc[0]["ct_roster"])
    first_t = roster_key(d.iloc[0]["t_roster"])
    if set(first_ct.split(",")) & set(first_t.split(",")):
        raise ValueError("overlapping_rosters")
    a, b = sorted((first_ct, first_t))
    score = {a: 0, b: 0}
    pistol = {a: 0, b: 0}
    ct_wins = {a: 0, b: 0}
    ct_rounds = {a: 0, b: 0}
    for i, row in enumerate(d.itertuples(index=False), 1):
        if {roster_key(row.ct_roster), roster_key(row.t_roster)} != {a, b}:
            raise ValueError("roster_changed_within_map")
        side = str(row.winner_side).upper()
        if side not in {"CT", "T", "TERRORIST"}:
            raise ValueError("unknown_round_winner")
        ct = first_ct if first_ct_is_ct(i) else first_t
        other = b if ct == a else a
        winner = ct if side == "CT" else other
        score[winner] += 1
        ct_rounds[ct] += 1
        ct_wins[ct] += int(winner == ct)
        if i in (1, 13):
            pistol[winner] += 1
        if terminal_score(score[a], score[b]) and i != len(d):
            raise ValueError("rounds_after_terminal_score")
    if not terminal_score(score[a], score[b]):
        raise ValueError("incomplete_or_nonstandard_map")
    # Anchor to the final filename suffix: the team name M80 is not map 80.
    match = re.search(r"(?:-m|map)([1-5])-[a-z0-9_]+\.dem$", path_key(d.iloc[0]["demo_path"]))
    if not match:
        raise ValueError("unknown_map_number")
    return {
        "match_id": str(d.iloc[0]["match_id"]),
        "map_id": path_key(d.iloc[0]["demo_path"]),
        "map_no": int(match.group(1)), "map_name": map_name(d.iloc[0]["map_name"]),
        "roster_a": a, "roster_b": b, "y": int(score[a] > score[b]),
        "rounds": len(d), "overtime": len(d) > 24,
        **{f"{k}_{s}": values[r] for k, values in
           (("round_wins", score), ("pistol_wins", pistol),
            ("ct_wins", ct_wins), ("ct_rounds", ct_rounds))
           for s, r in (("a", a), ("b", b))},
    }


def load_history(data_dir: Path = config.DATA_DIR, publication_lag_seconds: float = 300) -> tuple[pd.DataFrame, dict]:
    """Use linked match END + lag as an explicit historical availability proxy."""
    if not np.isfinite(publication_lag_seconds) or publication_lag_seconds < 0:
        raise ValueError("publication lag must be finite and nonnegative")
    data_dir = Path(data_dir)
    rd = pd.read_parquet(data_dir / "round_dataset.parquet")
    with closing(sqlite3.connect((data_dir / "cs2.db").resolve().as_uri() + "?mode=ro", uri=True)) as con:
        dm = pd.read_sql_query(
            "SELECT d.hltv_match_id, d.ambiguous, m.start_date, m.end_date "
            "FROM demos d JOIN matches m ON m.id=d.bo3gg_match_id", con)
    duplicate_ids = set(dm.loc[dm.duplicated("hltv_match_id", keep=False), "hltv_match_id"])
    meta = dm.drop_duplicates("hltv_match_id").set_index("hltv_match_id").to_dict("index")
    rejected = Counter()
    rows = []
    for path, group in rd.groupby("demo_path", sort=True):
        hid = re.search(r"hltv-(\d+)", path_key(path))
        ident = int(hid.group(1)) if hid else None
        m = meta.get(ident)
        if m is None or ident in duplicate_ids or m["ambiguous"]:
            rejected["missing_or_ambiguous_match_metadata"] += 1
            continue
        try:
            row = summarize_map(group)
            start, end = utc(m["start_date"]), utc(m["end_date"])
            if end <= start:
                raise ValueError("invalid_match_end_time")
            row.update(start_at=start, available_at=end + pd.Timedelta(seconds=publication_lag_seconds))
            rows.append(row)
        except (ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
    maps = pd.DataFrame(rows)
    if maps.empty:
        raise ValueError(f"No usable maps: {dict(rejected)}")
    if maps.duplicated(["match_id", "map_no"]).any():
        raise ValueError("Multiple demo files for the same series/map number; resolve duplicates first")

    # Add observed historical performance, never target-map performance features.
    pf = pd.read_parquet(data_dir / "player_features.parquet")
    pf["map_id"] = pf["demo_path"].map(path_key)
    pf["roster_key"] = pf["roster_key"].map(lambda r: ",".join(sorted(str(r).split(","))))
    stats = ["kills", "deaths", "damage", "rounds_played", "opening_kills", "trade_kills"]
    agg = pf.groupby(["map_id", "roster_key"])[stats].sum(min_count=1)
    for side in ("a", "b"):
        index = pd.MultiIndex.from_arrays([maps.map_id, maps[f"roster_{side}"]])
        values = agg.reindex(index)
        for stat in stats:
            maps[f"{stat}_{side}"] = values[stat].to_numpy()
    maps = maps.sort_values(["start_at", "match_id", "map_no"]).reset_index(drop=True)
    audit = {
        "source_maps": int(rd.demo_path.nunique()), "usable_maps": len(maps),
        "usable_map1": int(maps.map_no.eq(1).sum()), "overtime_maps": int(maps.overtime.sum()),
        "excluded_maps": dict(rejected), "publication_lag_seconds": publication_lag_seconds,
        "availability_basis": "match_end_plus_lag_proxy_not_observed_publication",
        "ruleset": "MR12 / MR3 / sides retained between overtime blocks",
        "known_limitations": [
            "Historical veto/lineup announcement times are missing; offline evaluation is conditional research only.",
            "Match-end plus lag approximates stat availability; it is not a recorded feed receipt time.",
            "Exact five-player rosters do not merge aliases or substitute lineups.",
        ],
    }
    return maps, audit


FEATURES = [f"{scope}_{name}_gap" for scope in ("all", "map") for name in
            ("win", "round", "pistol", "kd", "adr", "opening", "trade", "ct")]


def features_at(history: pd.DataFrame, roster_a: str, roster_b: str, arena: str,
                decision_at, exclude_match: str | None = None) -> tuple[dict, dict]:
    """Symmetric gaps; fixed priors, no global fitted statistics or future rows."""
    at = utc(decision_at)
    a, b = roster_key(roster_a), roster_key(roster_b)
    if set(a.split(",")) & set(b.split(",")):
        raise ValueError("overlapping_rosters")
    hist = history[history.available_at.lt(at)]
    if exclude_match is not None:
        hist = hist[hist.match_id.ne(exclude_match)]
    selected = hist[hist.roster_a.isin([a, b]) | hist.roster_b.isin([a, b])]
    max_at = selected.available_at.max()
    coverage = {"max_history_available_at": None if pd.isna(max_at) else max_at.isoformat()}

    def profile(roster: str, arena_filter: str | None):
        parts = []
        for side in ("a", "b"):
            h = hist[hist[f"roster_{side}"].eq(roster)].copy()
            if arena_filter:
                h = h[h.map_name.eq(arena_filter)]
            vals = {k: h[f"{k}_{side}"].to_numpy(dtype=float) for k in
                    ("round_wins", "pistol_wins", "ct_wins", "ct_rounds", *stats_names())}
            vals["won"] = h.y.to_numpy() if side == "a" else 1 - h.y.to_numpy()
            vals["rounds"] = h["rounds"].to_numpy()
            vals["w"] = np.exp2(-(at - h.available_at).dt.total_seconds().to_numpy() / (90 * 86400))
            parts.append(pd.DataFrame(vals))
        q = pd.concat(parts, ignore_index=True)
        n = float(q.w.sum())

        def ratio(num, den, prior, strength):
            valid = q[[num, den, "w"]].dropna()
            return float(((valid[num] * valid.w).sum() + prior * strength) /
                         ((valid[den] * valid.w).sum() + strength))

        return {
            "win": float(((q.won * q.w).sum() + 2.5) / (n + 5)),
            "round": ratio("round_wins", "rounds", .5, 60),
            "pistol": float(((q.pistol_wins * q.w).sum() + 2) /
                            ((np.where(q["rounds"] >= 13, 2, 1) * q.w).sum() + 4)),
            "ct": ratio("ct_wins", "ct_rounds", .5, 30),
            "kd": np.log(ratio("kills", "deaths", 1, 100)),
            "adr": ratio("damage", "rounds_played", 70, 150),
            "opening": ratio("opening_kills", "rounds_played", .1, 150),
            "trade": ratio("trade_kills", "kills", .15, 100),
        }, len(q)

    result = {}
    for scope, filt in (("all", None), ("map", map_name(arena))):
        pa, na = profile(a, filt)
        pb, nb = profile(b, filt)
        coverage[f"history_{scope}_a"] = na
        coverage[f"history_{scope}_b"] = nb
        result.update({f"{scope}_{k}_gap": pa[k] - pb[k] for k in pa})
    return result, coverage


def stats_names():
    return ("kills", "deaths", "damage", "rounds_played", "opening_kills", "trade_kills")


def build_features(history: pd.DataFrame, lead_seconds: float = 60) -> pd.DataFrame:
    if not np.isfinite(lead_seconds) or lead_seconds < 0:
        raise ValueError("lead_seconds must be finite and nonnegative")
    rows = []
    for row in history[history.map_no.eq(1)].to_dict("records"):
        at = row["start_at"] - pd.Timedelta(seconds=lead_seconds)
        f, coverage = features_at(history, row["roster_a"], row["roster_b"], row["map_name"], at, row["match_id"])
        rows.append({**row, **f, **coverage, "decision_at": at})
    return pd.DataFrame(rows)
