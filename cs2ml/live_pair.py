"""Pair eligible 5E live states with token-level Polymarket CLOB quotes.

All joins use local receive time.  Quotes at or before a decision timestamp are
inputs; later horizons are labelled evaluation-only.  The output is paper-only
and contains one mirrored feature row per market outcome.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Iterable


WEAPON_CLASSES = {
    "glock": "pistol", "hkp2000": "pistol", "usp_silencer": "pistol",
    "elite": "pistol", "p250": "pistol", "tec9": "pistol",
    "fiveseven": "pistol", "cz75a": "pistol", "deagle": "pistol",
    "revolver": "pistol",
    "mp9": "smg", "mac10": "smg", "mp7": "smg", "mp5sd": "smg",
    "ump45": "smg", "p90": "smg", "bizon": "smg",
    "ak47": "rifle", "m4a1": "rifle", "m4a1_silencer": "rifle",
    "famas": "rifle", "galilar": "rifle", "aug": "rifle", "sg556": "rifle",
    "awp": "sniper", "ssg08": "sniper", "scar20": "sniper", "g3sg1": "sniper",
    "nova": "shotgun", "xm1014": "shotgun", "mag7": "shotgun",
    "sawedoff": "shotgun", "m249": "lmg", "negev": "lmg",
}
WEAPON_CLASS_NAMES = ("pistol", "smg", "rifle", "sniper", "shotgun", "lmg", "other")


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _version_timestamp(value) -> float | None:
    try:
        result = int(str(value)) / 10.0
    except (TypeError, ValueError):
        return None
    return result if 1_000_000_000 <= result <= 4_000_000_000 else None


def _price(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0 <= result <= 1 else None


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _difference(left, right) -> float | None:
    a, b = _number(left), _number(right)
    return None if a is None or b is None else a - b


def canonical_team(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def load_binding(path: Path, *, event_id: str | None = None,
                 market: str | None = None) -> tuple[dict, dict]:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if meta.get("timing_basis") != "local_receive_time":
        raise ValueError("market_timing_basis_not_local_receive_time")
    if event_id is not None and str(meta.get("event_id")) != str(event_id):
        raise ValueError("market_meta_event_id_mismatch")
    markets = meta.get("markets")
    if not isinstance(markets, list):
        raise ValueError("market_meta_markets_missing")
    selected = [item for item in markets if isinstance(item, dict)
                and (market is None or item.get("market") == market)]
    if len(selected) != 1:
        raise ValueError(f"market_binding_count:{len(selected)}")
    binding = selected[0]
    tokens = binding.get("tokens")
    outcomes = binding.get("outcomes")
    if not isinstance(tokens, dict) or not isinstance(outcomes, list) or len(outcomes) != 2:
        raise ValueError("market_binding_not_binary")
    if set(tokens) != set(outcomes) or any(not str(tokens[name]) for name in outcomes):
        raise ValueError("market_binding_token_mismatch")
    if not binding.get("condition_id") or not binding.get("market_id"):
        raise ValueError("market_binding_identity_incomplete")
    return meta, binding


def _weapon_class_counts(team: dict) -> dict[str, int]:
    counts = {name: 0 for name in WEAPON_CLASS_NAMES}
    for player in team.get("players") or []:
        weapon = str(player.get("display_weapon") or "").casefold()
        if weapon:
            counts[WEAPON_CLASSES.get(weapon, "other")] += 1
    return counts


def load_states(paths: Iterable[Path], *, bout_num: int,
                include_ineligible: bool = False,
                allowed_timing_qualities: tuple[str, ...] | None = (
                    "mqtt_push_receive_time",),
                max_source_lag_seconds: float | None = 5.0) -> tuple[list[dict], dict]:
    states, seen = [], set()
    audit = {"rows_scanned": 0, "snapshots": 0, "duplicates": 0,
             "ineligible": 0, "wrong_bout_or_not_live": 0,
             "disallowed_timing_quality": 0, "source_lag_missing": 0,
             "source_lag_rejected": 0, "non_monotonic": 0, "invalid": 0}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                audit["rows_scanned"] += 1
                try:
                    row = json.loads(line)
                    if row.get("record_type") != "state_snapshot":
                        continue
                    audit["snapshots"] += 1
                    digest = str(row["state_hash"])
                    if digest in seen:
                        audit["duplicates"] += 1
                        continue
                    seen.add(digest)
                    if not row.get("decision_eligible") and not include_ineligible:
                        audit["ineligible"] += 1
                        continue
                    if (allowed_timing_qualities is not None
                            and row.get("timing_quality") not in allowed_timing_qualities):
                        audit["disallowed_timing_quality"] += 1
                        continue
                    state_ts = _timestamp(row["recv_utc"])
                    source_lag = _number(row.get("source_lag_seconds"))
                    if source_lag is None:
                        source_ts = _version_timestamp(row.get("source_this_ver"))
                        source_lag = None if source_ts is None else state_ts - source_ts
                    if max_source_lag_seconds is not None:
                        if source_lag is None:
                            audit["source_lag_missing"] += 1
                            continue
                        if source_lag < -2.0 or source_lag > max_source_lag_seconds:
                            audit["source_lag_rejected"] += 1
                            continue
                    summary = row["summary"]
                    bouts = [bout for bout in summary.get("live_bouts", [])
                             if int(bout.get("bout_num", -1)) == bout_num]
                    if len(bouts) != 1:
                        audit["wrong_bout_or_not_live"] += 1
                        continue
                    states.append({
                        "state_hash": digest, "state_ts": state_ts,
                        "state_recv_utc": row["recv_utc"],
                        "source_lag_seconds": source_lag,
                        "decision_eligible": bool(row.get("decision_eligible")),
                        "timing_quality": row.get("timing_quality"),
                        "match_id": summary.get("match_id"), "bout": bouts[0],
                    })
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    audit["invalid"] += 1
    states.sort(key=lambda item: item["state_ts"])
    monotonic, last_by_bout = [], {}
    for state in states:
        bout = state["bout"]
        key = (state["match_id"], bout.get("bout_id"), bout.get("bout_num"))
        current = (
            _number(bout.get("round_number")),
            _number((bout.get("team1") or {}).get("score")),
            _number((bout.get("team2") or {}).get("score")),
        )
        previous = last_by_bout.get(key)
        comparable = previous is not None and all(value is not None for value in current + previous)
        if comparable and any(now < before for now, before in zip(current, previous)):
            audit["non_monotonic"] += 1
            continue
        monotonic.append(state)
        if all(value is not None for value in current):
            last_by_bout[key] = current
    return monotonic, audit


@dataclass(frozen=True)
class Quote:
    ts: float
    bid: float | None
    ask: float | None
    source_ts: float | None

    @property
    def mid(self) -> float | None:
        return None if self.bid is None or self.ask is None else (self.bid + self.ask) / 2


class QuoteSeries:
    def __init__(self, quotes: list[Quote]):
        self.quotes = sorted(quotes, key=lambda quote: quote.ts)
        self.times = [quote.ts for quote in self.quotes]

    def before(self, timestamp: float, max_age: float) -> Quote | None:
        index = bisect_right(self.times, timestamp) - 1
        if index < 0:
            return None
        quote = self.quotes[index]
        return quote if timestamp - quote.ts <= max_age else None

    def after(self, timestamp: float, max_wait: float) -> Quote | None:
        index = bisect_left(self.times, timestamp)
        if index >= len(self.quotes):
            return None
        quote = self.quotes[index]
        return quote if quote.ts - timestamp <= max_wait else None


def load_quotes(path: Path, meta: dict, binding: dict) -> tuple[dict[str, QuoteSeries], dict]:
    outcomes = list(binding["outcomes"])
    values = {outcome: [] for outcome in outcomes}
    last = {outcome: None for outcome in outcomes}
    audit = {"rows_scanned": 0, "accepted": 0, "duplicates": 0,
             "coalesced_unchanged_top": 0, "invalid": 0,
             "cross_event_or_market": 0, "identity_mismatch": 0}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            audit["rows_scanned"] += 1
            try:
                row = json.loads(line)
                if (str(row.get("event_id")) != str(meta["event_id"])
                        or row.get("market") != binding["market"]):
                    audit["cross_event_or_market"] += 1
                    continue
                outcome = row.get("outcome")
                token = str(row.get("token"))
                if outcome not in values or token != str(binding["tokens"][outcome]):
                    audit["identity_mismatch"] += 1
                    continue
                ts = float(row["local_ts"])
                bid, ask = _price(row.get("best_bid")), _price(row.get("best_ask"))
                if bid is None and ask is None:
                    continue
                source_ts = _number(row.get("source_ts"))
                signature = (bid, ask)
                if signature == last[outcome]:
                    audit["duplicates"] += 1
                    audit["coalesced_unchanged_top"] += 1
                    # For top-of-book pairing the newest receipt is the useful
                    # freshness boundary, even when the price did not move.
                    values[outcome][-1] = Quote(ts, bid, ask, source_ts)
                    continue
                last[outcome] = signature
                values[outcome].append(Quote(ts, bid, ask, source_ts))
                audit["accepted"] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                audit["invalid"] += 1
    return {outcome: QuoteSeries(items) for outcome, items in values.items()}, audit


def _quote_fields(quote: Quote | None, target_ts: float, *, after: bool = False) -> dict:
    if quote is None:
        return {"quote_ts": None, "quote_age_seconds": None,
                "quote_wait_seconds": None, "quote_offset_seconds": None,
                "bid": None, "ask": None, "mid": None, "source_ts": None}
    offset = quote.ts - target_ts
    return {"quote_ts": quote.ts,
            "quote_age_seconds": None if after else -offset,
            "quote_wait_seconds": offset if after else None,
            "quote_offset_seconds": offset,
            "bid": quote.bid, "ask": quote.ask, "mid": quote.mid,
            "source_ts": quote.source_ts}


def _perspective_features(bout: dict, focal_key: str, opponent_key: str) -> dict:
    focal, opponent = bout[focal_key], bout[opponent_key]
    focal_e, opponent_e = focal.get("economy") or {}, opponent.get("economy") or {}
    focal_weapons, opponent_weapons = _weapon_class_counts(focal), _weapon_class_counts(opponent)
    features = {
        "round_number": bout.get("round_number"), "map_name": bout.get("map_name"),
        "stage": bout.get("stage"), "bomb_state": bout.get("bomb_state"),
        "focal_side": focal.get("current_side"),
        "focal_is_ct": None if focal.get("current_side") is None
        else focal.get("current_side") == "CT",
        "focal_score": focal.get("score"), "opponent_score": opponent.get("score"),
        "score_diff": _difference(focal.get("score"), opponent.get("score")),
        "focal_money": focal_e.get("money_sum"),
        "opponent_money": opponent_e.get("money_sum"),
        "money_diff": _difference(focal_e.get("money_sum"), opponent_e.get("money_sum")),
        "focal_hp": focal_e.get("hp_sum"), "opponent_hp": opponent_e.get("hp_sum"),
        "hp_diff": _difference(focal_e.get("hp_sum"), opponent_e.get("hp_sum")),
        "focal_alive": focal_e.get("alive_count"),
        "opponent_alive": opponent_e.get("alive_count"),
        "alive_diff": _difference(focal_e.get("alive_count"), opponent_e.get("alive_count")),
        "helmet_diff": _difference(focal_e.get("helmet_count"), opponent_e.get("helmet_count")),
        "kevlar_diff": _difference(focal_e.get("kevlar_count"), opponent_e.get("kevlar_count")),
        "defuse_kit_diff": _difference(
            focal_e.get("defuse_kit_count"), opponent_e.get("defuse_kit_count")),
        "focal_display_weapon_known": focal_e.get("display_weapon_known_players"),
        "opponent_display_weapon_known": opponent_e.get("display_weapon_known_players"),
    }
    for name in WEAPON_CLASS_NAMES:
        features[f"focal_{name}_count"] = focal_weapons[name]
        features[f"opponent_{name}_count"] = opponent_weapons[name]
        features[f"{name}_count_diff"] = focal_weapons[name] - opponent_weapons[name]
    return features


def pair_states(
    states: list[dict], quotes: dict[str, QuoteSeries], meta: dict, binding: dict, *,
    quote_max_age: float = 5.0, local_model_latency: float = 0.0,
    next_quote_max_wait: float = 5.0, horizons: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
) -> tuple[list[dict], dict]:
    outcomes = list(binding["outcomes"])
    outcome_by_canonical = {canonical_team(outcome): outcome for outcome in outcomes}
    if len(outcome_by_canonical) != 2:
        raise ValueError("ambiguous_market_outcomes")
    market_delay = float(binding.get("seconds_delay") or 0)
    rows = []
    audit = {"states": len(states), "paired_states": 0, "rows": 0,
             "team_identity_mismatch": 0, "missing_effective_quote": 0}
    for state in states:
        bout = state["bout"]
        teams = [bout.get("team1") or {}, bout.get("team2") or {}]
        mapped = []
        for team in teams:
            outcome = outcome_by_canonical.get(canonical_team(team.get("name") or ""))
            if outcome is None:
                mapped = []
                break
            mapped.append(outcome)
        if len(mapped) != 2 or len(set(mapped)) != 2:
            audit["team_identity_mismatch"] += 1
            continue
        effective_ts = state["state_ts"] + local_model_latency + market_delay
        effective = {outcome: quotes[outcome].before(effective_ts, quote_max_age)
                     for outcome in outcomes}
        if any(value is None for value in effective.values()):
            audit["missing_effective_quote"] += 1
            continue
        for focal_index, focal_outcome in enumerate(mapped):
            opponent_index = 1 - focal_index
            opponent_outcome = mapped[opponent_index]
            focal_key = "team1" if focal_index == 0 else "team2"
            opponent_key = "team2" if focal_index == 0 else "team1"
            row = {
                "schema_version": 1, "event_id": str(meta["event_id"]),
                "market_id": str(binding["market_id"]),
                "condition_id": binding["condition_id"], "market": binding["market"],
                "match_id": state["match_id"], "bout_num": bout.get("bout_num"),
                "state_hash": state["state_hash"], "state_ts": state["state_ts"],
                "state_recv_utc": state["state_recv_utc"],
                "state_source_lag_seconds": state["source_lag_seconds"],
                "decision_eligible": state["decision_eligible"],
                "state_timing_quality": state["timing_quality"],
                "local_model_latency_seconds": local_model_latency,
                "market_delay_seconds": market_delay, "effective_decision_ts": effective_ts,
                "focal_team": teams[focal_index].get("name"),
                "focal_team_id": teams[focal_index].get("id"),
                "focal_outcome": focal_outcome,
                "focal_token": binding["tokens"][focal_outcome],
                "opponent_team": teams[opponent_index].get("name"),
                "opponent_team_id": teams[opponent_index].get("id"),
                "opponent_outcome": opponent_outcome,
                "opponent_token": binding["tokens"][opponent_outcome],
                "features": _perspective_features(bout, focal_key, opponent_key),
                "quote_at_state": _quote_fields(
                    quotes[focal_outcome].before(state["state_ts"], quote_max_age),
                    state["state_ts"],
                ),
                "quote_at_effective_decision": _quote_fields(
                    effective[focal_outcome], effective_ts),
                "opponent_quote_at_effective_decision": _quote_fields(
                    effective[opponent_outcome], effective_ts),
                "first_quote_after_effective_decision": _quote_fields(
                    quotes[focal_outcome].after(effective_ts, next_quote_max_wait),
                    effective_ts, after=True,
                ),
                "evaluation_only": {},
            }
            for horizon in horizons:
                target = effective_ts + horizon
                row["evaluation_only"][str(horizon)] = _quote_fields(
                    quotes[focal_outcome].before(target, quote_max_age), target)
            rows.append(row)
        audit["paired_states"] += 1
    audit["rows"] = len(rows)
    return rows, audit


def run(
    *, state_paths: list[Path], quote_path: Path, market_meta_path: Path,
    output_dir: Path, bout_num: int, market: str, include_ineligible: bool = False,
    allow_http_polling: bool = False,
    max_source_lag_seconds: float = 5.0,
    quote_max_age: float = 5.0, local_model_latency: float = 0.0,
    horizons: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
) -> dict:
    meta, binding = load_binding(market_meta_path, market=market)
    states, state_audit = load_states(
        state_paths, bout_num=bout_num, include_ineligible=include_ineligible,
        allowed_timing_qualities=None if allow_http_polling else (
            "mqtt_push_receive_time",),
        max_source_lag_seconds=None if allow_http_polling else max_source_lag_seconds)
    quotes, quote_audit = load_quotes(quote_path, meta, binding)
    rows, pair_audit = pair_states(
        states, quotes, meta, binding, quote_max_age=quote_max_age,
        local_model_latency=local_model_latency, horizons=horizons)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "paired_states.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_only": True, "timing_basis": "strict_local_receive_time",
        "event_id": str(meta["event_id"]), "market": market,
        "market_id": str(binding["market_id"]), "condition_id": binding["condition_id"],
        "bout_num": bout_num, "include_ineligible": include_ineligible,
        "allow_http_polling": allow_http_polling,
        "max_source_lag_seconds": (
            None if allow_http_polling else max_source_lag_seconds),
        "accepted_state_timing": (
            "all timing qualities (diagnostic only)" if allow_http_polling
            else "mqtt_push_receive_time only"),
        "quote_max_age_seconds": quote_max_age,
        "local_model_latency_seconds": local_model_latency,
        "market_delay_seconds": float(binding.get("seconds_delay") or 0),
        "horizons_seconds": list(horizons), "state_audit": state_audit,
        "quote_audit": quote_audit, "pair_audit": pair_audit,
        "artifacts": {"paired_states": str(rows_path)},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fivee", action="append", required=True, type=Path)
    parser.add_argument("--quotes", required=True, type=Path)
    parser.add_argument("--market-meta", required=True, type=Path)
    parser.add_argument("--bout-num", required=True, type=int)
    parser.add_argument("--market", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-ineligible", action="store_true")
    parser.add_argument(
        "--allow-http-polling", action="store_true",
        help="Diagnostic only: allow cached polling snapshots into the join")
    parser.add_argument("--quote-max-age", type=float, default=5.0)
    parser.add_argument("--max-source-lag", type=float, default=5.0)
    parser.add_argument("--local-model-latency", type=float, default=0.0)
    parser.add_argument("--horizons", default="1,2,5,10")
    args = parser.parse_args(argv)
    horizons = tuple(float(value) for value in args.horizons.split(",") if value.strip())
    report = run(
        state_paths=args.fivee, quote_path=args.quotes,
        market_meta_path=args.market_meta, output_dir=args.output_dir,
        bout_num=args.bout_num, market=args.market,
        include_ineligible=args.include_ineligible,
        allow_http_polling=args.allow_http_polling,
        max_source_lag_seconds=args.max_source_lag,
        quote_max_age=args.quote_max_age,
        local_model_latency=args.local_model_latency, horizons=horizons,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
