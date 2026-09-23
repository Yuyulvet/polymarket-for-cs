"""Phase-5 swing engine: frozen entry/exit/risk rules, paper replay only.

The engine never invents fills: every candidate row carries the protocol's
executable quotes (entry = first ask at/after t_exec, exit = first bid at
t_exec+horizon, symmetric taker fees already netted). The engine only
decides WHETHER to open a paper position on a row and enforces risk caps.
No midpoint, no window-max exit, no backfill rows (protocol prohibitions
are structural in the phase-2 dataset).

Rule gate (all must hold to enter):
  1. decision_kind in rule.mechanisms;
  2. label_valid for the chosen horizon (no exit quote -> fail-closed skip);
  3. p_fair - entry_ask >= min_edge   (edge covers fees + margin);
  4. entry_ask <= max_entry_ask       (longshot cap);
  5. book_spread <= max_spread        (execution quality);
  6. per-(session, bout) exclusivity and session open-position cap.

Exit is always the fixed-horizon executable bid from the row — never a
stop-loss/market exit (those are phase-6+ candidates evaluated separately).

p_fair is pluggable. Built-ins:
  market_prior(df)    p_fair = mid_at_entry -> the no-edge control: with a
                      positive spread it can never clear min_edge, so a
                      market-prior run MUST produce zero trades;
  game_heuristic(df, outcome_is_team1)  frozen documented placeholder fair
                      value from score/economy/pistol until phase-4 models
                      supply calibrated probabilities.

Report: per-mechanism and total trade counts, net mean/median/win-rate,
95% CI lower bound (normal approx, diagnostic only — phase 7 runs the
real acceptance window), skip reasons, rule hash. Paper-only: no wallet,
no orders; ledger rows are deterministic replay records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .map1_market import number
from .trend_dataset import HORIZONS
from .trend_market_baseline import load_rows
from .trend_protocol import PROTOCOL_ID, protocol_hash


@dataclass(frozen=True)
class SwingRule:
    mechanisms: tuple = ("post_pistol_freeze", "round_end_economy_update")
    horizon_seconds: int = 60
    min_edge: float = 0.03
    max_entry_ask: float = 0.90
    max_spread: float = 0.06
    max_open_per_session: int = 3
    one_position_per_bout: bool = True

    def __post_init__(self):
        if self.horizon_seconds not in HORIZONS:
            raise ValueError(f"horizon_not_in_protocol:{self.horizon_seconds}")
        number(self.min_edge, "min_edge", 0, 1)
        number(self.max_entry_ask, "max_entry_ask", 0.01, 1)
        number(self.max_spread, "max_spread", 0.001, 1)
        number(self.max_open_per_session, "max_open_per_session", 1, 1000)

    def rule_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ------------------------------------------------------------- fair value
def market_prior(df: pd.DataFrame) -> np.ndarray:
    """No-edge control: fair = mid. With spread > 0 it never clears min_edge."""
    return df["mid_at_entry"].astype(float).to_numpy()


def game_heuristic(df: pd.DataFrame, outcome_is_team1: pd.Series) -> np.ndarray:
    """Frozen placeholder fair value (NOT calibrated; replaced by phase-4+ models).

    Perspective of the row's token: score gap, economy share and pistol
    result, sign-flipped when the token backs team2. Sigmoid squashes to
    (0, 1). Unknown enums (side labels etc.) are ignored, never guessed.
    """
    team1 = outcome_is_team1.astype(bool).to_numpy()
    sign = np.where(team1, 1.0, -1.0)
    score_gap = df["game_score_gap"].astype(float).to_numpy()
    score_gap = np.nan_to_num(score_gap, nan=0.0) * sign
    money1 = df["game_money_t1"].astype(float).to_numpy()
    money2 = df["game_money_t2"].astype(float).to_numpy()
    total = money1 + money2
    with np.errstate(invalid="ignore", divide="ignore"):
        money_share = np.where(total > 0, money1 / total, np.nan)
    money_edge = np.nan_to_num((money_share - 0.5) * 4.0, nan=0.0) * sign
    winner = df["game_pistol_winner_side"].astype(object).to_numpy()
    pistol_code = np.where(winner == "t1", 1.0, np.where(winner == "t2", -1.0, 0.0))
    pistol_edge = pistol_code * sign * 2.2
    z = 0.25 * score_gap + money_edge + pistol_edge
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- replay
SKIP_KIND = "kind_not_allowed"
SKIP_LABEL = "label_invalid"
SKIP_EDGE = "edge_below_threshold"
SKIP_PRICE = "entry_ask_cap"
SKIP_SPREAD = "spread_cap"
SKIP_BOUT = "bout_position_open"
SKIP_CAP = "session_position_cap"


def replay(rows: pd.DataFrame, rule: SwingRule, p_fair: np.ndarray,
           initial_cash: float = 0.0) -> dict:
    """Deterministic paper replay; returns ledger frame + report."""
    if len(rows) != len(p_fair):
        raise ValueError("p_fair_length_mismatch")
    df = rows.copy()
    df["_p_fair"] = p_fair
    df["_session_start"] = df.groupby("session_id")["info_mono"].transform("min")
    df = df.sort_values(["_session_start", "session_id", "exec_mono", "outcome"],
                        kind="mergesort").reset_index(drop=True)
    h = rule.horizon_seconds
    skips: dict[str, int] = {}
    trades = []
    cash = float(initial_cash)
    open_bouts: set = set()
    open_count: dict[str, int] = {}

    def skip(reason: str) -> None:
        skips[reason] = skips.get(reason, 0) + 1

    for _, row in df.iterrows():
        session = str(row["session_id"])
        if row["decision_kind"] not in rule.mechanisms:
            skip(SKIP_KIND)
            continue
        if not bool(row[f"label_valid_h{h}"]):
            skip(SKIP_LABEL)
            continue
        if float(row["entry_ask"]) > rule.max_entry_ask:
            skip(SKIP_PRICE)
            continue
        spread = row["book_spread"]
        if not np.isfinite(spread) or float(spread) > rule.max_spread:
            skip(SKIP_SPREAD)
            continue
        edge = float(row["_p_fair"]) - float(row["entry_ask"])
        if not np.isfinite(edge) or edge < rule.min_edge:
            skip(SKIP_EDGE)  # NaN p_fair (e.g. no walk-forward prediction) = no trade
            continue
        bout_key = (session, int(row["bout_num"]))
        if rule.one_position_per_bout and bout_key in open_bouts:
            skip(SKIP_BOUT)
            continue
        if open_count.get(session, 0) >= rule.max_open_per_session:
            skip(SKIP_CAP)
            continue
        net = float(row[f"net_h{h}"])
        cash += net
        trades.append({
            "session_id": session, "bout_num": int(row["bout_num"]),
            "decision_kind": row["decision_kind"], "outcome": row["outcome"],
            "info_mono": float(row["info_mono"]),
            "exec_mono": float(row["exec_mono"]),
            "entry_ask": float(row["entry_ask"]),
            "exit_bid": float(row[f"exit_bid_h{h}"]),
            "horizon_seconds": h, "net": net, "p_fair": float(row["_p_fair"]),
            "cash_after": cash,
        })
        open_bouts.add(bout_key)
        open_count[session] = open_count.get(session, 0) + 1

    ledger = pd.DataFrame(trades)
    nets = ledger["net"].to_numpy() if len(ledger) else np.array([])
    n = len(nets)
    ci_lower = None
    if n >= 2:
        ci_lower = float(np.mean(nets) - 1.96 * np.std(nets, ddof=1)
                         / np.sqrt(n))
    report = {
        "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_hash(),
        "rule": asdict(rule), "rule_hash": rule.rule_hash(),
        "p_fair_source": getattr(p_fair, "name", "external") if hasattr(
            p_fair, "name") else "external",
        "n_candidates": int(len(df)), "n_trades": n,
        "skip_reasons": skips,
        "net_total": float(np.sum(nets)) if n else 0.0,
        "net_mean": float(np.mean(nets)) if n else None,
        "net_median": float(np.median(nets)) if n else None,
        "win_rate": float(np.mean(nets > 0)) if n else None,
        "ci95_lower_bound": ci_lower,
        "per_mechanism": (_per_mechanism(ledger) if n else {}),
        "cash_final": cash,
        "mode": "paper_replay_only_no_orders",
    }
    return {"ledger": ledger, "report": report}


def _per_mechanism(ledger: pd.DataFrame) -> dict:
    out = {}
    for kind, group in ledger.groupby("decision_kind"):
        nets = group["net"].to_numpy()
        out[kind] = {"n_trades": int(len(group)),
                     "net_mean": float(np.mean(nets)),
                     "net_total": float(np.sum(nets))}
    return out


# ------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--max-entry-ask", type=float, default=0.90)
    parser.add_argument("--max-spread", type=float, default=0.06)
    parser.add_argument("--max-open", type=int, default=3)
    parser.add_argument("--fair-source", choices=["market_prior", "game_heuristic"],
                        default="market_prior",
                        help="market_prior=无edge对照(应零成交); game_heuristic=占位公平值")
    args = parser.parse_args(argv)
    df = load_rows(args.rows)
    rule = SwingRule(horizon_seconds=args.horizon, min_edge=args.min_edge,
                     max_entry_ask=args.max_entry_ask, max_spread=args.max_spread,
                     max_open_per_session=args.max_open)
    if args.fair_source == "market_prior":
        p_fair = market_prior(df)
        name = "market_prior"
    else:
        # CLI 里没有 outcome->team 映射；game_heuristic 需要注入映射，
        # 命令行默认按 team1 处理仅用于冒烟，正式运行走 Python API 注入。
        p_fair = game_heuristic(df, pd.Series(True, index=df.index))
        name = "game_heuristic_cli_default_team1"
    p_fair = pd.Series(p_fair, index=df.index, name=name)
    result = replay(df, rule, p_fair.to_numpy())
    args.output.mkdir(parents=True, exist_ok=True)
    result["ledger"].to_parquet(args.output / "ledger.parquet", index=False)
    (args.output / "swing_report.json").write_text(
        json.dumps(result["report"], ensure_ascii=False, indent=2,
                   default=str), encoding="utf-8")
    print(json.dumps({"n_trades": result["report"]["n_trades"],
                      "net_total": result["report"]["net_total"],
                      "skip_reasons": result["report"]["skip_reasons"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
