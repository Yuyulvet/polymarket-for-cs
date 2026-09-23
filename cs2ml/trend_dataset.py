"""Phase-2 unified swing-trend dataset builder (frozen protocol cs2_swing_trend_v1).

Turns one audited capture session (5E event log + 5E MQTT states + Polymarket
raw depth) into strictly time-safe decision rows, one row per
(trigger, bout, token side). Triggers are mechanism transitions only:

  post_pistol_freeze        round-start event with round_num == 2
  round_end_economy_update  round-end event (type 2)
  prematch_prior_dislocation first executable market book of the session

Timing chain per protocol:
  info_mono      = local receipt of the trigger (monotonic, same process clock)
  t_exec         = info_mono + MODEL_LATENCY_SECONDS + market seconds_delay
  entry          = first best ask at/after t_exec          (never a quote before)
  exit(h)        = first best bid at/after t_exec + h      (h in 15/30/60/120)
  net(h)         = exit_bid - entry_ask - fee(exit) - fee(entry)

Fail-closed rules: initial/recovery 5E snapshots never trigger and never join;
game state joins are backward-only (recv_mono <= info_mono); rows whose bout
cannot be bound to a market are dropped with a recorded reason; fee and delay
parameters are recorded per row; NaN labels stay NaN (never filled).

MODEL_LATENCY_SECONDS is a documented placeholder (0.0) until phase-5 measures
the real model latency; the frozen protocol entry rule requires replacing it
before any paper-trading evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .trend_protocol import (HORIZONS_SECONDS, PROTOCOL_ID, DECISION_KINDS,
                             protocol_hash)

MODEL_LATENCY_SECONDS = 0.0
EXIT_GRACE_SECONDS = 30.0
DEPTH_LEVELS = 5
HORIZONS = tuple(HORIZONS_SECONDS)


# ---------------------------------------------------------------- market book
@dataclass
class TokenTape:
    """L2 book rebuild for one token plus trade prints."""
    states: list = field(default_factory=list)   # (mono, best_bid, best_ask, bid_depth, ask_depth)
    trades: list = field(default_factory=list)   # (mono, price, size, side)
    _bids: dict = field(default_factory=dict)
    _asks: dict = field(default_factory=dict)

    def _apply_level(self, side: str, price: float, size: float) -> None:
        book = self._bids if side == "bid" else self._asks
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size

    def _commit(self, mono: float) -> None:
        best_bid = max(self._bids) if self._bids else None
        best_ask = min(self._asks) if self._asks else None
        bid_depth = sum(sorted(self._bids.values(), reverse=True)[:DEPTH_LEVELS])
        ask_depth = sum(sorted(self._asks.values())[:DEPTH_LEVELS])
        self.states.append((mono, best_bid, best_ask, bid_depth, ask_depth))

    def apply_snapshot(self, mono: float, bids, asks) -> None:
        self._bids, self._asks = {}, {}
        for level in bids or []:
            price, size = _level(level)
            if price is not None and size and size > 0:
                self._bids[price] = size
        for level in asks or []:
            price, size = _level(level)
            if price is not None and size and size > 0:
                self._asks[price] = size
        self._commit(mono)

    def apply_changes(self, mono: float, changes) -> None:
        touched = False
        for change in changes or []:
            if not isinstance(change, dict):
                continue
            price = _num(change.get("price"))
            size = _num(change.get("size"))
            side = str(change.get("side") or "").lower()
            if price is None or size is None or side not in ("bid", "ask"):
                continue
            self._apply_level(side, price, size)
            touched = True
        if touched:
            self._commit(mono)

    def add_trade(self, mono: float, price, size, side) -> None:
        if price is None or size is None:
            return
        self.trades.append((mono, price, size, str(side or "")))


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _level(level):
    if isinstance(level, dict):
        return _num(level.get("price")), _num(level.get("size"))
    if isinstance(level, (list, tuple)) and len(level) >= 2:
        return _num(level[0]), _num(level[1])
    return None, None


class MarketRebuilder:
    """Rebuild per-token tapes from a raw-depth events.jsonl."""

    def __init__(self, expected_tokens: set[str]):
        self.tapes: dict[str, TokenTape] = {t: TokenTape() for t in expected_tokens}
        self.raw_rows = 0
        self.applied = Counter()

    def feed_row(self, received_mono: float, raw: str) -> None:
        self.raw_rows += 1
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.applied["invalid_json"] += 1
            return
        for message in (payload if isinstance(payload, list) else [payload]):
            if not isinstance(message, dict):
                continue
            self._feed_message(received_mono, message)

    def _feed_message(self, mono: float, message: dict) -> None:
        etype = str(message.get("event_type") or "").lower()
        if "bids" in message and "asks" in message:
            token = str(message.get("asset_id") or "")
            if token in self.tapes:
                self.tapes[token].apply_snapshot(mono, message["bids"], message["asks"])
                self.applied["book_snapshot"] += 1
        elif "price_changes" in message or etype == "price_change":
            touched = False
            for change in message.get("price_changes") or []:
                if not isinstance(change, dict):
                    continue
                token = str(change.get("asset_id") or "")
                tape = self.tapes.get(token)
                if tape is None:
                    continue
                side = {"buy": "bid", "sell": "ask"}.get(
                    str(change.get("side") or "").lower())
                price, size = _num(change.get("price")), _num(change.get("size"))
                if side is None or price is None or size is None:
                    continue
                tape.apply_changes(mono, [{"price": price, "size": size, "side": side}])
                touched = True
            if touched:
                self.applied["price_change"] += 1
        elif etype == "last_trade_price" or "transaction_hash" in message:
            token = str(message.get("asset_id") or "")
            tape = self.tapes.get(token)
            if tape is not None:
                tape.add_trade(mono, _num(message.get("price")),
                               _num(message.get("size")), message.get("side"))
                self.applied["trade"] += 1
        else:
            self.applied[f"other_{etype or 'unknown'}"] += 1

    def first_mono(self) -> float | None:
        starts = [t.states[0][0] for t in self.tapes.values() if t.states]
        return min(starts) if starts else None


def _first_at_or_after(states: list, t: float, index: int, field: int):
    """Scan from cached index forward; states are sorted by mono."""
    i = index
    while i < len(states) and states[i][0] < t:
        i += 1
    if i >= len(states):
        return None, i
    return states[i][field], i


# ---------------------------------------------------------------- 5E triggers
@dataclass
class Trigger:
    kind: str
    bout_num: int
    info_mono: float
    info_utc: str
    round_num: int | None
    raw: dict


def parse_log_info(entry: dict) -> dict:
    try:
        log = json.loads(entry.get("log_info") or "{}")
    except json.JSONDecodeError:
        return {}
    return log if isinstance(log, dict) else {}


def collect_triggers(events_path: Path) -> tuple[list[Trigger], Counter]:
    """Decision-eligible 5E event-log rows only; initial/recovery excluded."""
    triggers: list[Trigger] = []
    stats: Counter = Counter()
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") != "event":
                continue
            if not row.get("decision_eligible"):
                stats["ineligible_event_skipped"] += 1
                continue
            entry = row.get("entry") or {}
            log = parse_log_info(entry)
            try:
                bout = int(entry.get("bout_num"))
            except (TypeError, ValueError):
                stats["missing_bout_num"] += 1
                continue
            kind = None
            round_num = None
            if str(log.get("type")) == "1":
                try:
                    round_num = int((log.get("round_start") or {}).get("round_num"))
                except (TypeError, ValueError):
                    round_num = None
                if round_num == 2:
                    kind = "post_pistol_freeze"
            elif str(log.get("type")) == "2":
                kind = "round_end_economy_update"
                end = log.get("round_end") or {}
                try:
                    round_num = int(end.get("ct_score")) + int(end.get("t_score"))
                except (TypeError, ValueError):
                    round_num = None
            if kind is None:
                stats[f"non_trigger_type_{log.get('type')}"] += 1
                continue
            triggers.append(Trigger(kind=kind, bout_num=bout,
                                    info_mono=float(row.get("recv_mono")),
                                    info_utc=str(row.get("recv_utc")),
                                    round_num=round_num, raw={"entry": entry, "log": log}))
            stats[f"trigger_{kind}"] += 1
    triggers.sort(key=lambda t: t.info_mono)
    return triggers, stats


# ---------------------------------------------------------------- state join
class StateJoin:
    """Backward-only point-in-time join over strict-eligible MQTT snapshots."""

    def __init__(self, states_paths):
        self.rows: list[tuple[float, dict]] = []
        for states_path in states_paths:
            with states_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("record_type") != "state_snapshot":
                        continue
                    strict = (row.get("decision_eligible") is True
                              and row.get("source_transport") == "mqtt"
                              and row.get("timing_quality") == "mqtt_push_receive_time")
                    if not strict:
                        continue
                    self.rows.append((float(row.get("recv_mono")),
                                      row.get("summary") or {}))
        self.rows.sort(key=lambda item: item[0])

    def at(self, mono: float) -> dict | None:
        best = None
        # small tapes: linear scan is fine and stateless-safe
        for when, summary in self.rows:
            if when > mono:
                break
            best = summary
        return best


def bout_state(summary: dict, bout_num: int) -> dict | None:
    for bout in (summary.get("live_bouts") or []):
        if isinstance(bout, dict) and bout.get("bout_num") == bout_num:
            return bout
    return None


def game_features(summary: dict | None, bout_num: int,
                  round_results: list) -> dict:
    if summary is None:
        return {}
    bout = bout_state(summary, bout_num)
    if bout is None:
        return {}
    team1, team2 = bout.get("team1") or {}, bout.get("team2") or {}
    econ1, econ2 = team1.get("economy") or {}, team2.get("economy") or {}
    s1, s2 = team1.get("score"), team2.get("score")
    features = {
        "score_t1": s1, "score_t2": s2,
        "score_gap": (s1 - s2) if s1 is not None and s2 is not None else None,
        "round_num": bout.get("round_number"),
        "side_t1": team1.get("current_side"),
        "alive_t1": econ1.get("alive_count"), "alive_t2": econ2.get("alive_count"),
        "hp_t1": econ1.get("hp_sum"), "hp_t2": econ2.get("hp_sum"),
        "money_t1": econ1.get("money_sum"), "money_t2": econ2.get("money_sum"),
        "bomb_state": bout.get("bomb_state"),
        "pistol_winner_side": None,
        "loss_streak_t1": 0, "loss_streak_t2": 0,
    }
    # backward-safe round results: list of (winner_side_key 't1'/'t2') in order
    streak1 = streak2 = 0
    pistol = None
    for winner in round_results:
        if winner == "t1":
            pistol = pistol or "t1"
            streak1, streak2 = 0, streak2 + 1
        elif winner == "t2":
            pistol = pistol or "t2"
            streak2, streak1 = 0, streak1 + 1
    features["pistol_winner_side"] = pistol
    features["loss_streak_t1"], features["loss_streak_t2"] = streak1, streak2
    return features


def round_results_before(triggers: list[Trigger], bout: int, mono: float) -> list[str]:
    """Ordered bout round winners from round-end triggers (scores sum == round)."""
    results = []
    for trig in triggers:
        if trig.bout_num != bout or trig.info_mono > mono:
            continue
        if trig.kind != "round_end_economy_update":
            continue
        end = trig.raw["log"].get("round_end") or {}
        bout_state_ = trig.raw["entry"]
        try:
            ct, tt = int(end.get("ct_score")), int(end.get("t_score"))
        except (TypeError, ValueError):
            continue
        # side mapping: entry stores ct side roster identity indirectly; the
        # round_end payload carries winner via score delta vs prior round.
        results.append((ct, tt, trig.info_mono))
    winners = []
    prev = (0, 0)
    for ct, tt, _ in sorted(results, key=lambda item: item[2]):
        if ct + tt != prev[0] + prev[1] + 1:
            continue  # gap or duplicate: fail closed, skip
        if ct > prev[0]:
            winners.append("ct_up")
        elif tt > prev[1]:
            winners.append("t_up")
        prev = (ct, tt)
    return winners


# ---------------------------------------------------------------- fee params
def fee_params_for_market(market_meta: dict, fetch_clob=None) -> dict:
    """seconds_delay + symmetric taker fee bps. CLOB lookup when missing."""
    delay = market_meta.get("seconds_delay")
    fee_bps = market_meta.get("taker_fee_bps")
    source = "capture_metadata"
    if (delay is None or fee_bps is None) and fetch_clob is not None:
        try:
            info = fetch_clob(market_meta.get("condition_id"))
            delay = delay if delay is not None else info.get("seconds_delay")
            fee_bps = fee_bps if fee_bps is not None else info.get("taker_base_fee")
            source = "clob_api"
        except Exception:
            source = "default_fallback"
    if delay is None:
        delay = 1
    if fee_bps is None:
        fee_bps = 0
    return {"seconds_delay": float(delay), "taker_fee_bps": float(fee_bps),
            "param_source": source}


# ---------------------------------------------------------------- source res.
def resolve_sources(session_dir: Path, spec: dict, root: Path | None) -> dict:
    """Session-local layout first; spec['sources'] (relative to root) overrides.

    live_session_run 会话：三路文件直接在 session_dir 下。
    旧会话（如 magic–MIBR）：路径在 spec['sources'] 里、相对项目 root。
    """
    session_dir = Path(session_dir)
    root = Path(root) if root else session_dir
    src = spec.get("sources") or {}

    def resolve(path_str):
        path = Path(path_str)
        return path if path.is_absolute() else root / path

    fivee_events = [resolve(p) for p in (src.get("fivee_events") or [])
                    ] or [session_dir / "fivee_events.jsonl"]
    fivee_states = [resolve(p) for p in (src.get("fivee_states") or [])
                    ] or [session_dir / "fivee_states.jsonl"]
    market_src = src.get("market") or {}
    market_events = resolve(market_src["events"]) if market_src.get("events") \
        else session_dir / "market" / "events.jsonl"
    market_meta_path = resolve(market_src["metadata"]) if market_src.get("metadata") \
        else session_dir / "market" / "metadata.json"
    return {"fivee_events": fivee_events, "fivee_states": fivee_states,
            "market_events": market_events, "market_meta": market_meta_path,
            "market_format": market_src.get("format", "market_raw")}


# ---------------------------------------------------------------- row build
def build_rows(session_dir: Path, spec: dict, *, root: Path | None = None,
               fetch_clob=None,
               excluded_bouts: set[int] | None = None) -> tuple[pd.DataFrame, dict]:
    sources = resolve_sources(session_dir, spec, root)
    drop_reasons: Counter = Counter()
    excluded_bouts = excluded_bouts or set()
    market_meta = json.loads(sources["market_meta"].read_text(encoding="utf-8"))
    if market_meta.get("mode") != "read_only_raw_evidence_multi":
        # 旧 normalized_top 采集与 5E recv_mono 不同钟，禁止拼接（fail-closed）
        report = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
                  "protocol_sha256": protocol_hash(),
                  "session_id": spec.get("session_id"),
                  "model_latency_seconds": MODEL_LATENCY_SECONDS,
                  "horizons_seconds": list(HORIZONS),
                  "market_format": market_meta.get("mode"),
                  "drop_reasons": {"market_format_not_time_joinable": 1},
                  "rows": 0, "label_valid_rate": {f"h{h}": 0.0 for h in HORIZONS},
                  "inputs_sha256": _hash_inputs(sources, spec)}
        return pd.DataFrame(), report
    bindings = market_meta.get("markets") or []
    token_to_market: dict[str, dict] = {}
    for binding in bindings:
        for outcome, token in (binding.get("tokens") or {}).items():
            token_to_market[str(token)] = {"binding": binding, "outcome": outcome}
    # bout -> binding via spec maps
    bout_binding = {int(m["bout_num"]): m for m in spec.get("maps") or []}
    market_params: dict[str, dict] = {}
    for binding in bindings:
        meta = {"condition_id": binding.get("condition_id"),
                "seconds_delay": binding.get("seconds_delay"),
                "taker_fee_bps": binding.get("taker_base_fee")}
        params = fee_params_for_market(meta, fetch_clob=fetch_clob)
        market_params[str(binding.get("market_id"))] = params
        binding["_params"] = params

    rebuilder = MarketRebuilder(set(token_to_market))
    with sources["market_events"].open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "market_raw":
                continue
            rebuilder.feed_row(float(row.get("received_monotonic_seconds")),
                               row.get("raw") or "")
    first_book_mono = rebuilder.first_mono()

    triggers: list[Trigger] = []
    trig_stats: Counter = Counter()
    for events_path in sources["fivee_events"]:
        if not events_path.is_file():
            drop_reasons["fivee_events_file_missing"] += 1
            continue
        part, part_stats = collect_triggers(events_path)
        triggers.extend(part)
        trig_stats.update(part_stats)
    triggers.sort(key=lambda t: t.info_mono)
    state_paths = [p for p in sources["fivee_states"] if p.is_file()]
    state_join = StateJoin(state_paths) if state_paths else None

    rows = []
    # prematch trigger: first executable book
    if first_book_mono is not None:
        triggers = [Trigger(kind="prematch_prior_dislocation", bout_num=1,
                            info_mono=first_book_mono, info_utc="",
                            round_num=None, raw={})] + triggers

    state_rows = state_join.rows if state_join else []
    for trig in triggers:
        if trig.bout_num in excluded_bouts:
            # audit v2 R3: bout observed incompletely at the source — its rows
            # must not enter the dataset even where labels look reconstructable
            drop_reasons["bout_excluded_by_audit"] += 1
            continue
        binding_spec = bout_binding.get(trig.bout_num)
        if binding_spec is None:
            drop_reasons["bout_without_market_binding"] += 1
            continue
        binding = next((b for b in bindings
                        if str(b.get("market_id")) == str(binding_spec["market_id"])), None)
        if binding is None:
            drop_reasons["market_binding_missing_in_capture"] += 1
            continue
        params = binding["_params"]
        t_exec = trig.info_mono + MODEL_LATENCY_SECONDS + params["seconds_delay"]
        outcomes = list(binding.get("tokens") or {})
        if len(outcomes) != 2:
            drop_reasons["non_binary_market"] += 1
            continue
        summary = state_join.at(trig.info_mono) if state_join else None
        if trig.kind != "prematch_prior_dislocation" and summary is None:
            drop_reasons["no_state_at_trigger"] += 1
            continue
        results = round_results_before(triggers, trig.bout_num, trig.info_mono)
        # map ct_up/t_up to t1/t2 via first_half_side of current bout
        side_map = _ct_t_to_team_keys(summary, trig.bout_num)
        winners = [w for w in (side_map.get(w) for w in results) if w]
        gf = game_features(summary, trig.bout_num, winners) if summary else {}
        tapes = {outcome: rebuilder.tapes.get(str(binding["tokens"][outcome]))
                 for outcome in outcomes}
        base = {
            "session_id": spec.get("session_id"),
            "bout_num": trig.bout_num,
            "map_name": binding_spec.get("map_name") or "",
            "market_id": str(binding.get("market_id")),
            "decision_kind": trig.kind,
            "trigger_round_num": trig.round_num,
            "info_mono": trig.info_mono,
            "exec_mono": t_exec,
            "seconds_delay": params["seconds_delay"],
            "taker_fee_bps": params["taker_fee_bps"],
            "param_source": params["param_source"],
            "prematch_prior": None,
        }
        base.update({f"game_{k}": v for k, v in gf.items()})
        for outcome in outcomes:
            tape = tapes[outcome]
            other = outcomes[1] if outcomes[0] == outcome else outcomes[0]
            if tape is None or not tape.states:
                drop_reasons["token_without_book"] += 1
                continue
            row = dict(base)
            row["outcome"] = outcome
            row["outcome_other"] = other
            entry_ask, cursor = _first_at_or_after(tape.states, t_exec, 0, 2)
            entry_bid, _ = _first_at_or_after(tape.states, t_exec, 0, 1)
            row["entry_ask"] = entry_ask
            row["entry_bid"] = entry_bid
            row["mid_at_entry"] = (entry_bid + entry_ask) / 2 if (
                entry_bid is not None and entry_ask is not None) else None
            row.update(_market_features(tape, t_exec, rebuilder.tapes.get(
                str(binding["tokens"][other]))))
            fee_rate = params["taker_fee_bps"] / 10_000.0
            for horizon in HORIZONS:
                exit_bid, _ = _first_at_or_after(tape.states, t_exec + horizon, cursor, 1)
                row[f"exit_bid_h{horizon}"] = exit_bid
                if entry_ask is None or exit_bid is None:
                    row[f"net_h{horizon}"] = None
                    row[f"label_valid_h{horizon}"] = False
                else:
                    fee = entry_ask * fee_rate + exit_bid * fee_rate
                    row[f"net_h{horizon}"] = exit_bid - entry_ask - fee
                    row[f"label_valid_h{horizon}"] = True
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["info_mono", "bout_num", "outcome"]
                                  ).reset_index(drop=True)
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_hash(),
        "session_id": spec.get("session_id"),
        "model_latency_seconds": MODEL_LATENCY_SECONDS,
        "exit_grace_seconds": EXIT_GRACE_SECONDS,
        "horizons_seconds": list(HORIZONS),
        "market_applied": dict(rebuilder.applied),
        "trigger_stats": dict(trig_stats),
        "drop_reasons": dict(drop_reasons),
        "state_snapshots_strict": len(state_rows),
        "rows": len(frame),
        "label_valid_rate": {f"h{h}": (float(frame[f"label_valid_h{h}"].mean())
                                       if len(frame) else 0.0) for h in HORIZONS},
        "inputs_sha256": _hash_inputs(sources, spec),
    }
    return frame, report


def _ct_t_to_team_keys(summary: dict | None, bout_num: int):
    """Map 'ct_up'/'t_up' winner markers to team keys t1/t2 for the bout."""
    if summary is None:
        return {}
    bout = bout_state(summary, bout_num)
    if bout is None:
        return {}
    team1 = bout.get("team1") or {}
    team2 = bout.get("team2") or {}
    fh1 = team1.get("first_half_side")
    fh2 = team2.get("first_half_side")
    mapping = {}
    if fh1 == "CT":
        mapping["ct_up"] = "t1"
    elif fh1 == "T":
        mapping["t_up"] = "t1"
    if fh2 == "CT":
        mapping["ct_up"] = "t2"
    elif fh2 == "T":
        mapping["t_up"] = "t2"
    return mapping


def _market_features(tape: TokenTape, t_exec: float, other_tape) -> dict:
    states = tape.states
    idx_now, best_bid, best_ask = 0, None, None
    for i, state in enumerate(states):
        if state[0] > t_exec:
            break
        idx_now = i
        best_bid, best_ask = state[1], state[2]
    out = {"book_bid": best_bid, "book_ask": best_ask,
           "book_spread": (best_ask - best_bid) if (best_bid and best_ask) else None}
    last_mono = states[idx_now][0] if states else None
    for back in (5, 15, 30):
        ref = _mid_at_or_before(states, t_exec - back)
        now = _mid_at_or_before(states, t_exec)
        out[f"mid_ret_{back}s"] = (now - ref) if (ref is not None and now is not None) else None
    mids = [((s[1] + s[2]) / 2) for s in states
            if s[1] is not None and s[2] is not None and s[0] <= t_exec]
    tail = [m for m, s in zip(mids, [s for s in states if s[1] is not None and s[2] is not None and s[0] <= t_exec]) if s[0] >= t_exec - 60]
    out["mid_vol_60s"] = float(np.std(tail)) if len(tail) >= 2 else None
    recent = [s for s in states if s[0] <= t_exec and s[0] >= t_exec - 60]
    if recent:
        out["bid_depth"] = recent[-1][3]
        out["ask_depth"] = recent[-1][4]
        total = recent[-1][3] + recent[-1][4]
        out["depth_imbalance"] = (recent[-1][3] - recent[-1][4]) / total if total else None
    else:
        out["bid_depth"] = out["ask_depth"] = out["depth_imbalance"] = None
    trades = [t for t in tape.trades if t[0] <= t_exec and t[0] >= t_exec - 60]
    signed = sum((t[2] if t[3].lower() in ("buy", "b") else -t[2]) for t in trades)
    out["trade_flow_60s"] = signed if trades else 0.0
    out["trades_60s"] = len(trades)
    mids_window = [((s[1] + s[2]) / 2) for s in states
                   if s[1] is not None and s[2] is not None
                   and t_exec - 10 <= s[0] <= t_exec]
    out["jump_flag"] = bool(mids_window and len(mids_window) >= 2
                            and max(mids_window) - min(mids_window) >= 0.08)
    if other_tape is not None and other_tape.states:
        out["other_mid"] = _mid_at_or_before(other_tape.states, t_exec)
    return out


def _mid_at_or_before(states: list, t: float):
    best = None
    for state in states:
        if state[0] > t:
            break
        if state[1] is not None and state[2] is not None:
            best = (state[1] + state[2]) / 2
    return best


def _hash_inputs(sources: dict, spec: dict) -> dict:
    def digest(path: Path) -> str | None:
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "fivee_events": [digest(p) for p in sources["fivee_events"]],
        "fivee_states": [digest(p) for p in sources["fivee_states"]],
        "market_events": digest(sources["market_events"]),
        "market_meta": digest(sources["market_meta"]),
        "spec": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None,
                        help="spec['sources'] 相对路径的项目根（默认 session-dir）")
    parser.add_argument("--clob-fees", action="store_true",
                        help="build 时从 CLOB 公共接口补查 seconds_delay/费率")
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    fetch = None
    if args.clob_fees:
        from curl_cffi import requests as creq
        cache: dict[str, dict] = {}
        def fetch(condition_id):
            if condition_id not in cache:
                response = creq.get(f"https://clob.polymarket.com/markets/{condition_id}",
                                    impersonate="chrome", timeout=20)
                response.raise_for_status()
                info = response.json()
                cache[condition_id] = {
                    "seconds_delay": info.get("seconds_delay"),
                    "taker_base_fee": info.get("taker_base_fee")}
            return cache[condition_id]
    frame, report = build_rows(args.session_dir, spec, root=args.root, fetch_clob=fetch)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output / "rows.parquet", index=False)
    (args.output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(json.dumps({"rows": len(frame), "session_id": report["session_id"],
                      "label_valid_rate": report["label_valid_rate"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
