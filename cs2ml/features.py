"""As-of feature construction - the anti-leakage core.

Walks matches in chronological order; every feature for match M uses ONLY
matches with start_date strictly before M.start_date. Ratings are updated
strictly after the feature row is emitted.

Reusable pieces (shared by training and live prediction):
    build_states()          walk all finished matches, return team states
    compute_features()      one match's feature row from pre-match states
    apply_result()          fold a finished match into the states
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from . import config, store
from .ratings import Glicko2Rating, update_pair
from .versions import version_era

FEATURE_COLUMNS = [
    "rating_diff", "rd_mean", "rd_diff",
    "form5_diff", "form10_diff", "form20_diff",
    "h2h_winrate", "h2h_games",
    "exp_diff", "rest_days_diff", "team_age_diff",
    "map_overlap", "map_entropy_diff",
    "bo_type", "tier_num", "log_prize", "stars",
]

_TIER_NUM = {"s": 4, "a": 3, "b": 2, "c": 1}
_FORM_PRIOR = 2.0  # pull winrate toward 0.5 when few games
_MAP_WINDOW_DAYS = 120.0  # recent-map window for map-pool features
_MAP_MAX = 30             # cap on recent maps considered per team


def _elite_weight(r1: float, r2: float) -> float:
    """Higher learning weight for matches between higher-rated (elite) teams.

    Maps the pair's average pre-match rating to [0.5, 1.5], centered on 1500,
    so a 2000-rated pair carries up to 3x the weight of a 1000-rated pair.
    """
    avg = (r1 + r2) / 2.0
    return max(0.5, min(1.5, 1.0 + (avg - 1500.0) / 500.0))


def _day(ts: str) -> float:
    """ISO timestamp -> fractional-day float (UTC)."""
    if not ts:
        return float("nan")
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() / 86400.0


@dataclass
class TeamState:
    rating: Glicko2Rating = field(default_factory=Glicko2Rating)
    first_day: float | None = None
    last_day: float | None = None
    games: int = 0
    history: list[tuple[float, int]] = field(default_factory=list)  # (day, win 0/1)
    h2h: dict[int, list[tuple[float, int]]] = field(default_factory=dict)
    recent_maps: list[tuple[float, str]] = field(default_factory=list)  # (day, map_name)


def _form(state: TeamState, day: float, window: int) -> float:
    """Win rate over last `window` games within 90 days, with 0.5 prior."""
    wins = 0
    n = 0
    for d, w in reversed(state.history):
        if day - d > 90.0:
            break
        wins += w
        n += 1
        if n >= window:
            break
    return (wins + _FORM_PRIOR * 0.5) / (n + _FORM_PRIOR)


def _h2h(state: TeamState, opp_id: int, day: float) -> tuple[float, int]:
    games = [g for g in state.h2h.get(opp_id, []) if day - g[0] <= config.H2H_WINDOW_DAYS]
    if not games:
        return 0.5, 0
    wins = sum(w for _, w in games)
    return (wins + _FORM_PRIOR * 0.5) / (len(games) + _FORM_PRIOR), len(games)


def _rest_days(state: TeamState, day: float) -> float:
    if state.last_day is None:
        return 30.0
    return min(day - state.last_day, 60.0)


def _recent_maps(state: TeamState, day: float) -> list[str]:
    """Map names from matches within _MAP_WINDOW_DAYS (newest first), capped."""
    maps: list[str] = []
    for d, m in reversed(state.recent_maps):
        if day - d > _MAP_WINDOW_DAYS:
            break
        maps.append(m)
        if len(maps) >= _MAP_MAX:
            break
    return maps


def _map_overlap(m1: list[str], m2: list[str]) -> float:
    """Jaccard similarity of the two teams' recent map pools (0..1)."""
    s1, s2 = set(m1), set(m2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _map_entropy(maps: list[str]) -> float:
    """Shannon entropy of a team's recent map distribution (0 = predictable)."""
    if not maps:
        return 0.0
    c = Counter(maps)
    n = len(maps)
    return -sum((v / n) * math.log(v / n) for v in c.values())


def load_matches(conn) -> list[dict]:
    """Finished, decided matches (defwin excluded - no competitive signal)."""
    rows = conn.execute(
        """
        SELECT id, slug, start_date, end_date, bo_type, winner_team_id, tier,
               team1_id, team2_id, team1_score, team2_score, stars, tournament_id,
               ai_pred_winner, maps
        FROM matches
        WHERE status = 'finished'
          AND winner_team_id IN (team1_id, team2_id)
          AND team1_id IS NOT NULL AND team2_id IS NOT NULL AND team1_id != team2_id
        ORDER BY start_date, id
        """
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        d["maps"] = json.loads(d["maps"]) if d.get("maps") else []
        out.append(d)
    return out


def _load_tournament_prizes(conn) -> dict[int, float]:
    prizes: dict[int, float] = {}
    for r in conn.execute("SELECT id, prize FROM tournaments"):
        prizes[r["id"]] = float(r["prize"] or 0.0)
    return prizes


def compute_features(
    s1: TeamState,
    s2: TeamState,
    day: float,
    m: dict,
    prizes: dict[int, float] | None = None,
) -> dict | None:
    """Feature row for a match from PRE-match states (states not modified here,
    except rating idle decay which is idempotent and information-safe)."""
    t1, t2 = int(m["team1_id"]), int(m["team2_id"])
    s1.rating.apply_idle(day)
    s2.rating.apply_idle(day)

    if s1.games < config.MIN_TEAM_HISTORY or s2.games < config.MIN_TEAM_HISTORY:
        return None

    tier = (m.get("tier") or "c").lower()
    prize = (prizes or {}).get(m.get("tournament_id"), 0.0)

    f1_5, f1_10, f1_20 = (_form(s1, day, w) for w in config.FORM_WINDOWS)
    f2_5, f2_10, f2_20 = (_form(s2, day, w) for w in config.FORM_WINDOWS)
    h2h_wr, h2h_n = _h2h(s1, t2, day)

    age1 = min(day - s1.first_day, 1000.0) if s1.first_day is not None else 0.0
    age2 = min(day - s2.first_day, 1000.0) if s2.first_day is not None else 0.0

    rm1 = _recent_maps(s1, day)
    rm2 = _recent_maps(s2, day)

    return {
        "match_id": m["id"],
        "start_date": m["start_date"],
        "team1_id": t1, "team2_id": t2,
        "p_glicko": s1.rating.win_prob(s2.rating),
        "tier": tier,
        "bo_type": m.get("bo_type") or 1,
        # features (team1 - team2 orientation)
        "rating_diff": s1.rating.r - s2.rating.r,
        "rd_mean": (s1.rating.rd + s2.rating.rd) / 2.0,
        "rd_diff": s1.rating.rd - s2.rating.rd,
        "form5_diff": f1_5 - f2_5,
        "form10_diff": f1_10 - f2_10,
        "form20_diff": f1_20 - f2_20,
        "h2h_winrate": h2h_wr,
        "h2h_games": float(h2h_n),
        "exp_diff": (s1.games + 1) ** 0.5 - (s2.games + 1) ** 0.5,
        "rest_days_diff": _rest_days(s1, day) - _rest_days(s2, day),
        "team_age_diff": age1 - age2,
        "tier_num": float(_TIER_NUM.get(tier, 1.0)),
        "log_prize": math.log1p(prize) if prize else 0.0,
        "stars": float(m.get("stars") or 0),
        "map_overlap": _map_overlap(rm1, rm2),
        "map_entropy_diff": _map_entropy(rm1) - _map_entropy(rm2),
        # metadata (not model features): annotation + learning weight
        "weight": _elite_weight(s1.rating.r, s2.rating.r),
        "version_era": version_era(m["start_date"]),
    }


def apply_result(states: dict[int, TeamState], m: dict, day: float) -> None:
    """Fold a finished match into team states."""
    t1, t2 = int(m["team1_id"]), int(m["team2_id"])
    s1 = states.setdefault(t1, TeamState())
    s2 = states.setdefault(t2, TeamState())
    y = 1 if int(m["winner_team_id"]) == t1 else 0

    # idle decay before the update (idempotent; no-op in build_features where
    # compute_features already decayed to this day)
    s1.rating.apply_idle(day)
    s2.rating.apply_idle(day)

    winner, loser = (s1, s2) if y == 1 else (s2, s1)
    update_pair(winner.rating, loser.rating)
    for s, w, opp_id in ((s1, y, t2), (s2, 1 - y, t1)):
        if s.first_day is None:
            s.first_day = day
        s.last_day = day
        s.games += 1
        s.history.append((day, w))
        s.h2h.setdefault(opp_id, []).append((day, w))
    for mn in (m.get("maps") or []):
        s1.recent_maps.append((day, mn))
        s2.recent_maps.append((day, mn))


def build_states(matches: list[dict] | None = None, conn=None) -> dict[int, TeamState]:
    """Walk all finished matches chronologically, building final team states."""
    close_conn = False
    if matches is None:
        if conn is None:
            conn = store.connect()
            close_conn = True
        matches = load_matches(conn)
    states: dict[int, TeamState] = {}
    for m in matches:
        day = _day(m["start_date"])
        if day != day:  # NaN
            continue
        apply_result(states, m, day)
    if close_conn:
        conn.close()
    return states


def build_features(matches: list[dict] | None = None, conn=None) -> pd.DataFrame:
    """Emit one feature row per eligible finished match, as-of its start_date."""
    close_conn = False
    if conn is None:
        conn = store.connect()
        close_conn = True
    if matches is None:
        matches = load_matches(conn)

    prizes = _load_tournament_prizes(conn) if conn is not None else {}
    states: dict[int, TeamState] = {}
    out: list[dict] = []

    for m in matches:
        day = _day(m["start_date"])
        if day != day:  # NaN
            continue
        t1, t2 = int(m["team1_id"]), int(m["team2_id"])
        s1 = states.setdefault(t1, TeamState())
        s2 = states.setdefault(t2, TeamState())

        y = 1 if int(m["winner_team_id"]) == t1 else 0
        row = compute_features(s1, s2, day, m, prizes)
        if row is not None:
            row["y"] = y
            row["p_ai"] = 0.5 if m.get("ai_pred_winner") is None else (1.0 if int(m["ai_pred_winner"]) == t1 else 0.0)
            row["ai_available"] = 0 if m.get("ai_pred_winner") is None else 1
            out.append(row)

        apply_result(states, m, day)

    if close_conn:
        conn.close()

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("start_date").reset_index(drop=True)
    return df
