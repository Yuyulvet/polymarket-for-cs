"""Paper-only event study for fading CS2 consecutive-kill price reactions.

Two timing modes are intentionally separate. ``source_update_version`` studies
historical market response but is not executable evidence. ``strict_receive``
uses only normally polled events at their local receive time and therefore can
support a paper execution backtest.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from .fivee_live import entry_key, source_version


@dataclass(frozen=True)
class Kill:
    source_ts: float
    recv_ts: float
    round_num: int
    team: str
    player: str
    decision_eligible: bool


@dataclass(frozen=True)
class Quote:
    ts: float
    bid: float | None
    ask: float | None

    @property
    def mid(self) -> float | None:
        return None if self.bid is None or self.ask is None else (self.bid + self.ask) / 2


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _price(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0 <= result <= 1 else None


def load_kills(paths: Iterable[Path], bout_num: int,
               player_to_team: dict[str, str]) -> tuple[list[Kill], dict]:
    """Parse, deduplicate and attach round numbers without using later rounds."""
    raw, seen = [], set()
    audit = {"rows": 0, "schema_v1_rows": 0, "schema_v2_rows": 0,
             "events": 0, "duplicates": 0, "invalid": 0,
             "initial_or_recovery": 0, "unknown_killer_team": 0}
    for path in paths:
        file_rows, first_recv = [], None
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                audit["rows"] += 1
                try:
                    row = json.loads(line)
                    if row.get("schema_version") == 2:
                        audit["schema_v2_rows"] += 1
                    else:
                        audit["schema_v1_rows"] += 1
                    entry = row.get("entry")
                    if not isinstance(entry, dict) or int(entry.get("bout_num", -1)) != bout_num:
                        continue
                    recv_ts = _timestamp(row["recv_utc"])
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    audit["invalid"] += 1
                    continue
                first_recv = recv_ts if first_recv is None else first_recv
                file_rows.append((row, entry, recv_ts))
        for row, entry, recv_ts in file_rows:
            key = entry_key(entry)
            if key in seen:
                audit["duplicates"] += 1
                continue
            seen.add(key)
            version = source_version(entry)
            if version is None:
                audit["invalid"] += 1
                continue
            try:
                payload = json.loads(entry.get("log_info") or "{}")
            except json.JSONDecodeError:
                audit["invalid"] += 1
                continue
            # Schema v1 has no poll marker. Its first burst is a historical
            # snapshot; later polls are separated by far more than one second.
            eligible = (bool(row.get("decision_eligible")) if row.get("schema_version") == 2
                        else recv_ts - first_recv >= 1.0)
            if not eligible:
                audit["initial_or_recovery"] += 1
            raw.append((version, recv_ts, eligible, payload))
            audit["events"] += 1
    raw.sort(key=lambda item: item[0])
    current_round, kills = None, []
    for version, recv_ts, eligible, payload in raw:
        kind = str(payload.get("type"))
        if kind == "1":
            try:
                current_round = int((payload.get("round_start") or {}).get("round_num"))
            except (TypeError, ValueError):
                current_round = None
        elif kind == "8" and current_round is not None:
            item = payload.get("kill") or {}
            player = str(item.get("killer_name") or item.get("killer_nick") or "")
            team = player_to_team.get(player.casefold())
            if team is None:
                audit["unknown_killer_team"] += 1
                continue
            kills.append(Kill(version / 1000.0, recv_ts, current_round,
                              team, player, eligible))
    return kills, audit


def find_streaks(kills: list[Kill], min_kills: int = 2,
                 max_gap_seconds: float = 8.0) -> list[list[Kill]]:
    """Return maximal same-team kill chains within one round."""
    chains, current = [], []
    for kill in kills:
        continues = (current and kill.round_num == current[-1].round_num and
                     kill.team == current[-1].team and
                     kill.source_ts - current[-1].source_ts <= max_gap_seconds)
        if continues:
            current.append(kill)
        else:
            if len(current) >= min_kills:
                chains.append(current)
            current = [kill]
    if len(current) >= min_kills:
        chains.append(current)
    return chains


class QuoteSeries:
    def __init__(self, quotes: list[Quote]):
        self.quotes = sorted(quotes, key=lambda quote: quote.ts)
        self.times = [quote.ts for quote in self.quotes]

    def before(self, timestamp: float) -> Quote | None:
        index = bisect_right(self.times, timestamp) - 1
        return self.quotes[index] if index >= 0 else None

    def after(self, timestamp: float) -> Quote | None:
        index = bisect_left(self.times, timestamp)
        return self.quotes[index] if index < len(self.quotes) else None


def load_quotes(path: Path, market: str, outcomes: set[str]) -> tuple[dict[str, QuoteSeries], dict]:
    values = {outcome: [] for outcome in outcomes}
    last = {outcome: None for outcome in outcomes}
    rows = accepted = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            outcome = row.get("outcome")
            if row.get("market") != market or outcome not in outcomes:
                continue
            bid, ask = _price(row.get("best_bid")), _price(row.get("best_ask"))
            # local_ts is epoch seconds, not a probability.
            try:
                ts = float(row.get("local_ts"))
            except (TypeError, ValueError):
                continue
            if bid is None and ask is None:
                continue
            signature = (bid, ask)
            if signature == last[outcome]:
                continue
            values[outcome].append(Quote(ts, bid, ask))
            last[outcome] = signature
            accepted += 1
    return {name: QuoteSeries(quotes) for name, quotes in values.items()}, {
        "market_rows_scanned": rows, "quote_changes_retained": accepted,
        "quote_changes_by_outcome": {name: len(quotes) for name, quotes in values.items()},
    }


def _fee(price: float, rate: float) -> float:
    return rate * price * (1.0 - price)


def study(streaks: list[list[Kill]], quotes: dict[str, QuoteSeries], teams: tuple[str, str],
          *, timing_basis: str, entry_delay: float, pre_seconds: float,
          reaction_threshold: float, horizons: tuple[float, ...], fee_rate: float) -> list[dict]:
    results = []
    for event_id, chain in enumerate(streaks, 1):
        if timing_basis == "strict_receive" and not all(k.decision_eligible for k in chain):
            continue
        streak_team = chain[-1].team
        fade_team = teams[1] if streak_team == teams[0] else teams[0]
        event_start = chain[0].source_ts if timing_basis == "source_update_version" else chain[0].recv_ts
        decision = chain[-1].source_ts if timing_basis == "source_update_version" else chain[-1].recv_ts
        pre = quotes[streak_team].before(event_start - pre_seconds)
        post = quotes[streak_team].after(decision + entry_delay)
        entry = quotes[fade_team].after(decision + entry_delay)
        if not pre or pre.mid is None or not post or post.mid is None or not entry or entry.ask is None:
            continue
        reaction = post.mid - pre.mid
        base = {
            "event_id": event_id, "round_num": chain[-1].round_num,
            "streak_team": streak_team, "fade_team": fade_team, "kill_count": len(chain),
            "event_start_ts": event_start, "decision_ts": decision,
            "entry_quote_ts": entry.ts, "reaction_mid_change": reaction,
            "reaction_threshold": reaction_threshold, "triggered": reaction >= reaction_threshold,
            "entry_ask": entry.ask, "timing_basis": timing_basis,
        }
        for horizon in horizons:
            row = dict(base, horizon_seconds=horizon)
            exit_quote = quotes[fade_team].after(entry.ts + horizon)
            if exit_quote and exit_quote.bid is not None:
                gross = exit_quote.bid - entry.ask
                row.update(exit_quote_ts=exit_quote.ts, exit_bid=exit_quote.bid, gross_pnl=gross,
                           net_pnl=gross-_fee(entry.ask, fee_rate)-_fee(exit_quote.bid, fee_rate))
            else:
                row.update(exit_quote_ts=None, exit_bid=None, gross_pnl=None, net_pnl=None)
            results.append(row)
    return results


def summarize(rows: list[dict], horizons: tuple[float, ...]) -> dict:
    result = {}
    for horizon in horizons:
        selected = [row for row in rows if row["horizon_seconds"] == horizon and
                    row["triggered"] and row["net_pnl"] is not None]
        gross = [row["gross_pnl"] for row in selected]
        pnl = [row["net_pnl"] for row in selected]
        result[str(horizon)] = {"trades": len(pnl),
                                "mean_gross_pnl": mean(gross) if gross else None,
                                "gross_win_rate": mean(value > 0 for value in gross) if gross else None,
                                "mean_net_pnl": mean(pnl) if pnl else None,
                                "median_net_pnl": median(pnl) if pnl else None,
                                "net_win_rate": mean(value > 0 for value in pnl) if pnl else None,
                                "sum_independent_event_net_pnl": sum(pnl) if pnl else None}
    return result


def run(*, fivee_paths: list[Path], market_path: Path, output_dir: Path,
        bout_num: int, market: str, team_a: str, team_b: str,
        team_a_players: list[str], team_b_players: list[str],
        timing_basis: str = "source_update_version", min_kills: int = 2,
        max_gap_seconds: float = 8.0, entry_delay: float = 2.0,
        pre_seconds: float = 5.0, reaction_threshold: float = .03,
        horizons: tuple[float, ...] = (15.0, 30.0, 60.0, 120.0),
        fee_rate: float = .05) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing_overwrite:{output_dir}")
    if timing_basis not in {"source_update_version", "strict_receive"}:
        raise ValueError("invalid_timing_basis")
    if not team_a_players or not team_b_players:
        raise ValueError("both_rosters_required")
    overlap = {name.casefold() for name in team_a_players} & {name.casefold() for name in team_b_players}
    if overlap:
        raise ValueError(f"player_in_both_rosters:{sorted(overlap)}")
    mapping = {name.casefold(): team_a for name in team_a_players}
    mapping.update({name.casefold(): team_b for name in team_b_players})
    kills, event_audit = load_kills(fivee_paths, bout_num, mapping)
    streaks = find_streaks(kills, min_kills, max_gap_seconds)
    quotes, quote_audit = load_quotes(market_path, market, {team_a, team_b})
    rows = study(streaks, quotes, (team_a, team_b), timing_basis=timing_basis,
                 entry_delay=entry_delay, pre_seconds=pre_seconds,
                 reaction_threshold=reaction_threshold, horizons=horizons,
                 fee_rate=fee_rate)
    report = {
        "version": 2,
        "status": "exploratory_event_study" if timing_basis == "source_update_version"
                  else "paper_execution_backtest",
        "live_trading_enabled": False,
        "timing_basis": timing_basis,
        "timing_warning": (
            "source update time is useful for market-response research but was not our receive time; "
            "results are not executable evidence"
            if timing_basis == "source_update_version" else
            "schema-v2 uses explicit normal-poll receive markers; schema-v1 excludes its inferred first burst; "
            "initial and recovery backfills are excluded"
        ),
        "execution": "buy fade outcome at first observed ask; sell at first observed bid after fixed horizon",
        "overlap_warning": "event rows are independent event-study observations, not an additive portfolio",
        "inputs": {"fivee": [str(path) for path in fivee_paths], "market": str(market_path),
                   "bout_num": bout_num, "market_name": market,
                   "teams": {team_a: team_a_players, team_b: team_b_players}},
        "parameters": {"min_kills": min_kills, "max_gap_seconds": max_gap_seconds,
                       "entry_delay": entry_delay, "pre_seconds": pre_seconds,
                       "reaction_threshold": reaction_threshold, "horizons": list(horizons),
                       "fee_rate_assumption": fee_rate},
        "audit": {**event_audit, **quote_audit, "kills": len(kills),
                  "streaks": len(streaks), "matched_event_horizons": len(rows),
                  "matched_events": len({row["event_id"] for row in rows}),
                  "triggered_events": len({row["event_id"] for row in rows if row["triggered"]})},
        "summary_by_horizon": summarize(rows, horizons),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    columns = list(rows[0]) if rows else ["event_id", "horizon_seconds", "net_pnl"]
    with (output_dir / "events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return report


def _split_players(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fivee", action="append", required=True, type=Path)
    parser.add_argument("--market-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bout-num", required=True, type=int)
    parser.add_argument("--market", required=True)
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--team-a-players", required=True, type=_split_players)
    parser.add_argument("--team-b-players", required=True, type=_split_players)
    parser.add_argument("--timing-basis", choices=["source_update_version", "strict_receive"],
                        default="source_update_version")
    parser.add_argument("--min-kills", type=int, default=2)
    parser.add_argument("--max-gap-seconds", type=float, default=8.0)
    parser.add_argument("--entry-delay", type=float, default=2.0)
    parser.add_argument("--pre-seconds", type=float, default=5.0)
    parser.add_argument("--reaction-threshold", type=float, default=.03)
    parser.add_argument("--horizon", action="append", type=float)
    parser.add_argument("--fee-rate", type=float, default=.05)
    args = parser.parse_args(argv)
    horizons = tuple(args.horizon or (15.0, 30.0, 60.0, 120.0))
    report = run(
        fivee_paths=args.fivee, market_path=args.market_log, output_dir=args.output_dir,
        bout_num=args.bout_num, market=args.market, team_a=args.team_a, team_b=args.team_b,
        team_a_players=args.team_a_players, team_b_players=args.team_b_players,
        timing_basis=args.timing_basis, min_kills=args.min_kills,
        max_gap_seconds=args.max_gap_seconds, entry_delay=args.entry_delay,
        pre_seconds=args.pre_seconds, reaction_threshold=args.reaction_threshold,
        horizons=horizons, fee_rate=args.fee_rate)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
