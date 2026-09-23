"""Evidence-backed HLTV review, never a nickname-based trading shortcut."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from .map1 import now
from .map1_data import utc
from .map1_hltv import fetch_match_page, normalize_match_url, parse_match_page
from .map1_market import as_list, select_map1_market, team_key
from .map1_sources import describe_event

MAX_EVIDENCE_AGE = 900
REFRESH_SECONDS = 300
MIN_REQUEST_INTERVAL = 15
SCHEDULE_TOLERANCE_SECONDS = 300


def nickname(value):
    # Exact, case-insensitive candidate lookup only; NOT an identity verifier.
    return str(value).strip().casefold()


def steam_id(value):
    value = str(value).strip()
    if not value.isascii() or not value.isdigit() or not 76561197960265728 <= int(value) <= 76561202255233023:
        raise ValueError("invalid_individual_steamid64")
    return value


def evidence_signature(preview):
    body = {k: preview[k] for k in ("match_id", "map_name", "scheduled_start_at")}
    body["teams"] = sorted([
        {"id": t["hltv_team_id"], "outcome": team_key(t["outcome"]),
         "players": sorted((p["hltv_player_id"], p["steamid"]) for p in t["players"])}
        for t in preview["teams"]], key=lambda t: t["id"])
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class HltvReview:
    def __init__(self, store, data_dir):
        self.store = store
        self.last_request = None
        self.parser_sha256 = hashlib.sha256(Path(__file__).with_name("map1_hltv.py").read_bytes()).hexdigest()
        self.candidates = {}
        self.candidate_warning = None
        # Both are demo-derived, including the misleading legacy CSV name.
        # Retain historical nickname aliases, but never infer an HLTV identity
        # or use post-settlement team maps to assign a current player.
        for filename in ("player_features.parquet", "player_hlvt_all.csv"):
            path = Path(data_dir).parent / filename
            if not path.exists():
                continue
            try:
                if path.suffix == ".parquet":
                    rows = pd.read_parquet(path, columns=["name", "steamid", "demo_path"])
                else:
                    rows = pd.read_csv(path, dtype=str, usecols=["name", "steamid"])
                for row in rows.dropna(subset=["name", "steamid"]).to_dict("records"):
                    sid = steam_id(row["steamid"])
                    candidate = {"steamid": sid, "name": row["name"],
                                 "source": (f"{path.resolve()} :: {row['demo_path']}"
                                            if row.get("demo_path") else str(path.resolve())),
                                 "verified": False}
                    pool = self.candidates.setdefault(nickname(row["name"]), [])
                    if not any(c["steamid"] == sid for c in pool):
                        pool.append(candidate)
            except (ValueError, OSError, KeyError) as exc:
                self.candidate_warning = f"local_identity_hints_unavailable: {exc}"

    def import_match(self, event, url):
        url = normalize_match_url(url)
        eid = str(event["id"])
        if describe_event(event, now())["status"] != "awaiting_confirmation":
            raise ValueError("hltv_import_requires_open_prematch_event")
        if self.last_request is not None and time.monotonic() - self.last_request < MIN_REQUEST_INTERVAL:
            raise ValueError("hltv_rate_limit_wait_15_seconds")
        self.last_request = time.monotonic()
        evidence = {"event_id": eid, "url": url, "received_at": now(), "html": None,
                    "sha256": None, "parser_sha256": self.parser_sha256, "parsed": None, "error": None}
        try:
            page = fetch_match_page(url)
            # Stamp receipt here too, after the whole bounded response arrived.
            # User input and a page's displayed clock cannot set this timestamp.
            evidence.update(html=page["html"], received_at=now())
            evidence["sha256"] = hashlib.sha256(page["html"].encode("utf-8")).hexdigest()
            evidence["parsed"] = parse_match_page(page["html"], url)
        except Exception as exc:
            evidence["received_at"] = now()
            evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["evidence_id"] = self.store.add_hltv_evidence(evidence)
        return self.preview(evidence, event, now())

    def preview(self, evidence, event, at, identities=None):
        identities = self.store.hltv_identities() if identities is None else identities
        parsed = copy.deepcopy(evidence.get("parsed") or {})
        result = {k: evidence.get(k) for k in ("evidence_id", "url", "received_at", "sha256", "error")}
        result.update({k: parsed.get(k) for k in
                       ("match_id", "event_name", "scheduled_start_at", "map_name", "status")})
        result["teams"] = parsed.get("teams", [])
        issues = list(parsed.get("issues") or [])
        if str(evidence["event_id"]) != str(event["id"]):
            issues.append("hltv_evidence_event_mismatch")
        if evidence.get("error"):
            issues.append("hltv_source_unavailable_manual_retry_required")
        age = (utc(at) - utc(evidence["received_at"])).total_seconds()
        if age < 0 or age > MAX_EVIDENCE_AGE:
            issues.append("hltv_evidence_stale_or_future")
        if parsed.get("status") != "prematch":
            issues.append("hltv_match_not_prematch")
        if not parsed.get("map_name"):
            issues.append("hltv_map1_not_announced")
        try:
            summary = describe_event(event, at)
            if summary["status"] != "awaiting_confirmation":
                issues.append("hltv_market_not_prematch")
            market = select_map1_market(event)
            outcomes = as_list(market["outcomes"])
            keys = {team_key(o): o for o in outcomes}
            teams = result["teams"]
            if len(teams) != 2 or len(keys) != 2 or {team_key(t["name"]) for t in teams} != set(keys):
                issues.append("hltv_team_identity_mismatch")
            for team in teams:
                team["outcome"] = keys.get(team_key(team["name"]))
            start = utc(parsed["scheduled_start_at"])
            if start <= utc(at):
                issues.append("hltv_match_not_prematch")
            if abs((start - utc(market["gameStartTime"])).total_seconds()) > SCHEDULE_TOLERANCE_SECONDS:
                issues.append("hltv_schedule_mismatch")
        except (KeyError, ValueError, TypeError):
            issues.append("hltv_missing_match_identity_or_schedule")
        players = []
        for team in result["teams"]:
            lineup = team.get("players", [])
            if len(lineup) != 5:
                issues.append("hltv_incomplete_lineup")
            for player in lineup:
                pid = str(player["hltv_player_id"])
                player["hltv_player_id"] = pid
                mapping = identities.get(pid)
                player.update(steamid=mapping["steamid"] if mapping else None, verified=bool(mapping),
                              candidates=self.candidates.get(nickname(player["name"]), []))
                players.append(player)
        if len(players) != 10 or len({p["hltv_player_id"] for p in players}) != 10:
            issues.append("hltv_incomplete_or_duplicate_players")
        if any(not p["verified"] for p in players):
            issues.append("hltv_player_identities_unverified")
        steamids = [p["steamid"] for p in players if p["steamid"]]
        if len(steamids) != len(set(steamids)):
            issues.append("hltv_overlapping_steam_ids")
        contexts = self.store.contexts()
        for eid, saved in contexts.items():
            if eid != str(event["id"]) and saved["context"].get("hltv_match_id") == parsed.get("match_id"):
                issues.append("hltv_match_already_bound_to_another_event")
        result["issues"] = sorted(set(issues))
        result["ready"] = not issues
        result["candidate_warning"] = self.candidate_warning
        context = contexts.get(str(event["id"]), {}).get("context", {})
        result["confirmation_matches"] = (
            bool(result["ready"] and evidence_signature(result) == context.get("hltv_signature"))
            if context.get("hltv_match_id") else None)
        return result

    def current(self, event_id, evidence_id):
        evidence = self.store.hltv_evidence(evidence_id)
        if evidence["event_id"] != str(event_id):
            raise ValueError("hltv_evidence_event_mismatch")
        latest = self.store.latest_hltv_evidence().get(str(event_id))
        if not latest or latest["evidence_id"] != evidence["evidence_id"]:
            raise ValueError("hltv_evidence_superseded_reload_review")
        return evidence

    def save_identities(self, submitted):
        if submitted.get("verified") is not True:
            raise ValueError("hltv_identity_attestation_required")
        evidence = self.current(submitted["event_id"], submitted["evidence_id"])
        if not evidence.get("parsed") or evidence.get("error"):
            raise ValueError("hltv_valid_source_required")
        players = {str(p["hltv_player_id"]): p for t in evidence["parsed"].get("teams", []) for p in t.get("players", [])}
        submitted_rows = submitted.get("mappings")
        if not isinstance(submitted_rows, list) or not 1 <= len(submitted_rows) <= 10:
            raise ValueError("hltv_invalid_identity_batch")
        rows, seen = [], set()
        for row in submitted_rows:
            pid = str(row["hltv_player_id"])
            source = str(row.get("source", "")).strip()
            if pid not in players or pid in seen or not source or len(source) > 2000:
                raise ValueError("hltv_invalid_identity_or_missing_evidence")
            seen.add(pid)
            rows.append({"hltv_player_id": pid, "steamid": steam_id(row["steamid"]),
                         "name_at_confirmation": players[pid]["name"], "source": source,
                         "confirmed_at": now(), "evidence_id": evidence["evidence_id"],
                         "verification_method": "manual_identity_attestation_not_nickname_inference"})
        self.store.save_hltv_identities(rows)
        return {"identities_saved": len(rows), "event_id": str(submitted["event_id"])}

    def confirmed_context(self, submitted, event, at):
        if submitted.get("reviewed") is not True:
            raise ValueError("hltv_match_review_required")
        if str(submitted["event_id"]) != str(event["id"]):
            raise ValueError("event_id_mismatch")
        evidence = self.current(submitted["event_id"], submitted["evidence_id"])
        view = self.preview(evidence, event, at)
        if not view["ready"]:
            raise ValueError(",".join(view["issues"]))
        teams = [{"outcome": t["outcome"], "roster": [p["steamid"] for p in t["players"]]} for t in view["teams"]]
        market = select_map1_market(event)
        return {"event_id": str(event["id"]), "map_no": 1, "map_name": view["map_name"],
                "scheduled_start_at": utc(market["gameStartTime"]).isoformat(),
                "team_a": teams[0], "team_b": teams[1],
                "map_source": evidence["url"], "roster_source": evidence["url"],
                "map_known_at": at, "roster_known_at": at, "confirmed_at": at,
                "source_received_at": evidence["received_at"], "hltv_evidence_id": evidence["evidence_id"],
                "hltv_match_id": view["match_id"], "hltv_signature": evidence_signature(view),
                "confirmation_method": "hltv_page_with_manual_match_and_identity_review"}

    def capture_guard(self, context, at):
        latest = self.store.latest_hltv_evidence().get(str(context["event_id"]))
        if not latest:
            raise ValueError("hltv_source_evidence_missing")
        # Use saved market only for source comparison; capture_once independently
        # refetches the live market and validates exact schedule and token IDs.
        event = next((r["event"] for r in self.store.events() if str(r["event"]["id"]) == str(context["event_id"])), None)
        if event is None:
            raise ValueError("hltv_market_evidence_missing")
        view = self.preview(latest, event, at)
        if not view["ready"]:
            raise ValueError(",".join(view["issues"]))
        if evidence_signature(view) != context["hltv_signature"]:
            raise ValueError("hltv_map_roster_or_match_changed_requires_new_review")
        return {"evidence_id": latest["evidence_id"], "received_at": latest["received_at"],
                "sha256": latest["sha256"]}
