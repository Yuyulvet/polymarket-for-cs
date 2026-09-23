"""Phase-6 scheme comparison and ONE-scheme freeze.

Candidate schemes vary only the pre-registered dimensions:
  fair_source   market_prior (no-edge control) | game_heuristic (frozen
                placeholder) | wf_ridge_exit | wf_xgboost_exit
                (walk-forward predicted exit_bid as fair value — the
                "future executable price model" of the research chain)
  horizon_seconds  one of the protocol horizons
  min_edge / max_entry_ask / max_spread / mechanisms  swing-rule params

Selection criterion is FIXED IN ADVANCE (not chosen after seeing results):
  among candidates with n_trades >= min_trades_floor, pick max net_mean;
  ties broken by net_median, then net_q05, then scheme_hash (deterministic).

The winner is written to frozen_scheme.json with a tamper-evident hash.
Phase-7 forward acceptance may ONLY run the frozen scheme; editing any
rule changes the hash and invalidates the freeze (a new freeze round with
new data is required — never retro-fitting during acceptance).

All replay here is paper-only; no orders, no wallet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .trend_game_increment import build_feature_sets
from .trend_market_baseline import MARKET_FEATURES, session_order
from .trend_protocol import PROTOCOL_ID, protocol_hash
from .trend_swing_engine import (SwingRule, game_heuristic, market_prior,
                                 replay)

FAIR_SOURCES = ("market_prior", "game_heuristic", "wf_ridge_exit",
                "wf_xgboost_exit")


@dataclass(frozen=True)
class CandidateScheme:
    name: str
    fair_source: str = "market_prior"
    horizon_seconds: int = 60
    min_edge: float = 0.03
    mechanisms: tuple = ("post_pistol_freeze", "round_end_economy_update")
    max_entry_ask: float = 0.90
    max_spread: float = 0.06
    features: str = "market_only"  # for wf_* models

    def __post_init__(self):
        if self.fair_source not in FAIR_SOURCES:
            raise ValueError(f"unknown_fair_source:{self.fair_source}")
        SwingRule(horizon_seconds=self.horizon_seconds, min_edge=self.min_edge,
                  max_entry_ask=self.max_entry_ask, max_spread=self.max_spread)

    def scheme_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def default_candidates() -> list:
    out = [CandidateScheme(name="control_market_prior")]
    for horizon in (30, 60, 120):
        out.append(CandidateScheme(name=f"game_heuristic_h{horizon}",
                                   fair_source="game_heuristic",
                                   horizon_seconds=horizon))
        out.append(CandidateScheme(name=f"wf_ridge_h{horizon}",
                                   fair_source="wf_ridge_exit",
                                   horizon_seconds=horizon))
        out.append(CandidateScheme(name=f"wf_xgboost_h{horizon}",
                                   fair_source="wf_xgboost_exit",
                                   horizon_seconds=horizon,
                                   features="market_plus_game"))
    return out


# ------------------------------------------------------- walk-forward fair
def walk_forward_exit_forecast(rows: pd.DataFrame, scheme: CandidateScheme,
                               ) -> np.ndarray:
    """Out-of-sample predicted exit_bid per row (expanding session window).

    Train rows are label-valid only; test rows keep whatever labels they
    have (prediction is a fair value, not a label). Returns an array
    aligned positionally with ``rows``.
    """
    df, feature_sets = build_feature_sets(rows)
    cols = feature_sets["market_plus_game" if scheme.features == "market_plus_game"
                        else "market_only"]
    X = df[cols].astype(float).fillna(0.0)
    h = scheme.horizon_seconds
    y_all = df[f"exit_bid_h{h}"].astype(float).to_numpy()
    valid_all = df[f"label_valid_h{h}"].astype(bool).to_numpy()
    if "session_start" not in df.columns:
        df = df.copy()
        df["session_start"] = df.groupby("session_id")["info_mono"].transform("min")
    sessions = session_order(df)
    pred = np.full(len(df), np.nan)
    for k in range(1, len(sessions)):
        train_mask = (df["session_id"].isin(sessions[:k]) & valid_all).to_numpy()
        test_mask = (df["session_id"] == sessions[k]).to_numpy()
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        X_train, y_train = X[train_mask], y_all[train_mask]
        X_test = X[test_mask]
        if scheme.fair_source == "wf_ridge_exit":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        else:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=200, max_depth=3,
                                 learning_rate=0.05, subsample=0.8,
                                 random_state=7)
        model.fit(X_train, y_train)
        pred[test_mask] = model.predict(X_test)
    return pred


# ------------------------------------------------------------ run/select
def resolve_p_fair(rows: pd.DataFrame, scheme: CandidateScheme,
                   outcome_is_team1: pd.Series | None) -> np.ndarray:
    if scheme.fair_source == "market_prior":
        return market_prior(rows)
    if scheme.fair_source == "game_heuristic":
        if outcome_is_team1 is None:
            raise ValueError("game_heuristic_needs_outcome_is_team1")
        return game_heuristic(rows, outcome_is_team1)
    return walk_forward_exit_forecast(rows, scheme)


def run_candidate(rows: pd.DataFrame, scheme: CandidateScheme,
                  outcome_is_team1: pd.Series | None = None,
                  p_fair_override: np.ndarray | None = None) -> dict:
    rule = SwingRule(horizon_seconds=scheme.horizon_seconds,
                     min_edge=scheme.min_edge, mechanisms=scheme.mechanisms,
                     max_entry_ask=scheme.max_entry_ask,
                     max_spread=scheme.max_spread)
    p_fair = p_fair_override if p_fair_override is not None else \
        resolve_p_fair(rows, scheme, outcome_is_team1)
    return replay(rows, rule, p_fair)


def fit_exit_forecaster(scheme: CandidateScheme, train_rows: pd.DataFrame):
    """Fit the frozen wf_* exit-bid forecaster on PRE-FREEZE rows.

    Returns predict(frame) -> p_fair array aligned positionally. This is
    how walk-forward fair sources survive into the acceptance window:
    the model is trained once on frozen-era data and applied forward,
    never refit on acceptance data.
    """
    if scheme.fair_source not in ("wf_ridge_exit", "wf_xgboost_exit"):
        raise ValueError("not_a_wf_scheme")
    df, feature_sets = build_feature_sets(train_rows)
    cols = feature_sets["market_plus_game" if scheme.features == "market_plus_game"
                        else "market_only"]
    X = df[cols].astype(float).fillna(0.0)
    h = scheme.horizon_seconds
    valid = df[f"label_valid_h{h}"].astype(bool).to_numpy()
    if valid.sum() == 0:
        raise ValueError("no_valid_training_rows")
    X_train = X[valid]
    y_train = df[f"exit_bid_h{h}"].astype(float).to_numpy()[valid]
    if scheme.fair_source == "wf_ridge_exit":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, random_state=7)
    model.fit(X_train, y_train)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        f, fs = build_feature_sets(frame)
        cols_f = fs["market_plus_game" if scheme.features == "market_plus_game"
                   else "market_only"]
        return model.predict(f[cols_f].astype(float).fillna(0.0))

    return predict


SELECTION_CRITERION = ("max net_mean among candidates with "
                       "n_trades >= min_trades_floor; "
                       "ties: net_median, then net_q05, then scheme_hash")


def select_scheme(rows: pd.DataFrame, candidates: list,
                  min_trades_floor: int = 5,
                  outcome_is_team1: pd.Series | None = None) -> dict:
    table = []
    for scheme in candidates:
        result = run_candidate(rows, scheme, outcome_is_team1)
        report = result["report"]
        nets = result["ledger"]["net"].to_numpy() if len(result["ledger"]) \
            else np.array([])
        table.append({
            "name": scheme.name, "scheme_hash": scheme.scheme_hash(),
            "scheme": asdict(scheme),
            "n_trades": report["n_trades"],
            "skip_reasons": report["skip_reasons"],
            "net_mean": float(np.mean(nets)) if len(nets) else None,
            "net_median": float(np.median(nets)) if len(nets) else None,
            "net_q05": float(np.quantile(nets, 0.05)) if len(nets) else None,
            "net_total": report["net_total"],
            "eligible": report["n_trades"] >= min_trades_floor,
        })
    eligible = [t for t in table if t["eligible"] and t["net_mean"] is not None]

    def sort_key(t):
        return (t["net_mean"], t["net_median"], t["net_q05"], t["scheme_hash"])

    winner = max(eligible, key=sort_key) if eligible else None
    return {"criterion": SELECTION_CRITERION, "min_trades_floor": min_trades_floor,
            "candidates": table, "winner": winner,
            "no_eligible_reason": None if winner else
            ("no_candidate_meets_trade_floor:"
             f"n_trades>={min_trades_floor}_required")}


# ---------------------------------------------------------------- freeze
def freeze_payload(selection: dict) -> dict:
    return {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_hash(),
            "selection": selection}


def _canonical_sha(payload: dict) -> str:
    canonical = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def freeze_scheme(selection: dict, path: Path) -> dict:
    if selection.get("winner") is None:
        raise ValueError("cannot_freeze_without_winner")
    payload = freeze_payload(selection)
    payload["sha256"] = _canonical_sha(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return payload


def load_frozen(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if _canonical_sha(payload) != payload.get("sha256"):
        raise ValueError("frozen_scheme_tampered:sha256_mismatch")
    return payload


def run_frozen(rows: pd.DataFrame, frozen: dict,
               outcome_is_team1: pd.Series | None = None,
               p_fair_override: np.ndarray | None = None) -> dict:
    """Phase-7 entry point: replay strictly the frozen scheme."""
    scheme = CandidateScheme(**frozen["selection"]["winner"]["scheme"])
    return run_candidate(rows, scheme, outcome_is_team1,
                         p_fair_override=p_fair_override)


# ------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-trades-floor", type=int, default=5)
    parser.add_argument("--freeze-name", type=Path, default=None,
                        help="写入 frozen_scheme.json 的路径（缺省不冻结）")
    parser.add_argument("--game-heuristic-assume-team1", action="store_true",
                        help="CLI 冒烟：game_heuristic 全部按 team1 视角（正式运行走 Python API 注入映射）")
    args = parser.parse_args(argv)
    from .trend_market_baseline import load_rows
    rows = load_rows(args.rows)
    candidates = default_candidates()
    team1 = None
    if args.game_heuristic_assume_team1:
        team1 = pd.Series(True, index=rows.index)
    else:  # CLI 没有 outcome->team 映射，剔除需要映射的候选并注明
        candidates = [c for c in candidates
                      if c.fair_source != "game_heuristic"]
    selection = select_scheme(rows, candidates,
                              min_trades_floor=args.min_trades_floor,
                              outcome_is_team1=team1)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection_report.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    if args.freeze_name and selection["winner"] is not None:
        freeze_scheme(selection, args.freeze_name)
    print(json.dumps({"winner": selection["winner"]["name"] if selection["winner"] else None,
                      "candidates": len(selection["candidates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
