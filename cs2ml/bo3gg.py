"""bo3.gg public API client.

Response shape (verified 2026-08-23):
  GET /api/v2/matches/finished?date=YYYY-MM-DD&filter[discipline_id][eq]=1
  {
    "data": {"tiers": {"low_tier": {"codes": ["c"], "tournaments": [...], "matches": [...]}, ...}},
    "included": [{"teams": {"<id>": {"slug","name","image_url"?}},
                  "tournaments": {"<id>": {"id","slug","name","tier","prize"?}}}],
    "meta": {"date","prev_date","next_date"}
  }
Match fields: id, slug, status (finished|defwin|...), parsed_status, bo_type (1/3/5),
winner_team_id, tier (s|a|b|c), start_date/end_date (ISO+00:00), team1_id, team2_id,
team1_score, team2_score (maps won), stars, ai_predictions, team1/team2/tournament (string id refs).
"""
from __future__ import annotations

import time
from typing import Any

import requests

from . import config


class Bo3Client:
    def __init__(self, base_url: str = config.BO3_BASE, timeout: int = config.BO3_TIMEOUT_SEC):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.BO3_UA, "Accept": "application/json"})
        self._last_request = 0.0

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        params.setdefault("filter[discipline_id][eq]", config.DISCIPLINE_CS2)
        url = f"{self.base}{path}"
        last_err: Exception | None = None
        for attempt in range(config.BO3_MAX_RETRIES):
            # rate limit
            wait = config.BO3_REQUEST_INTERVAL_SEC - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"HTTP {resp.status_code}")
                    time.sleep(2.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as e:
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"bo3.gg request failed after {config.BO3_MAX_RETRIES} retries: {url} ({last_err})")

    # --- public endpoints ---

    def matches_for_date(self, date: str, kind: str = "finished") -> dict[str, Any]:
        """date: YYYY-MM-DD, kind: 'finished' | 'upcoming'."""
        return self._get(f"/api/v2/matches/{kind}", {"date": date})

    @staticmethod
    def parse_day(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict], dict[str, dict]]:
        """Extract (matches, teams_by_id, tournaments_by_id) from a day payload.

        `included` may be a dict {"teams": {...}, "tournaments": {...}} (observed live)
        or a list of such objects (defensive); both are handled.
        """
        teams: dict[str, dict] = {}
        tournaments: dict[str, dict] = {}
        included = payload.get("included") or {}
        inc_items = included if isinstance(included, list) else [included]
        for inc in inc_items:
            if not isinstance(inc, dict):
                continue
            for tid, t in (inc.get("teams") or {}).items():
                teams.setdefault(str(tid), t)
            for tid, t in (inc.get("tournaments") or {}).items():
                tournaments.setdefault(str(tid), t)

        matches: list[dict[str, Any]] = []
        tiers_root = ((payload.get("data") or {}).get("tiers") or {})
        for group in tiers_root.values():
            for m in group.get("matches") or []:
                matches.append(m)
        return matches, teams, tournaments

    def team_rankings(self, page: int = 1, per_page: int = 100) -> dict[str, Any]:
        return self._get(
            "/api/v2/team_rankings",
            {"page": page, "per_page": per_page},
        )

    def team_profile(self, slug: str) -> dict[str, Any]:
        return self._get(f"/api/v1/teams/{slug}", {})
