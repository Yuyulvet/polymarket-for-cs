"""Phase-4 mechanism-signal incremental test (4A/4B/4C).

Question: does game-state information add predictive value OVER the market
baseline on identical samples? Per the frozen protocol this is answered on
three same-sample model versions per mechanism slice:

  market_only       the 15 phase-3 market features
  game_only         encoded game-state features (this module)
  market_plus_game  union

Mechanism slices:
  4A_post_pistol_freeze        post_pistol_freeze rows (pistol economy)
  4B_round_end_economy_update  round_end_economy_update rows (economy break)
  4C_prematch_prior_dislocation prematch rows (prior dislocation)

4C carries no live game state by construction: rows have no joined 5E
summary, so game features are all absent. The 赛前地图先验 (ro-strength)
feature is NOT yet in the dataset — 4C reports
``no_prematch_prior_feature_yet`` and is excluded from deltas until the
prior pipeline lands.

Deltas are computed as (candidate - market_only) on shared metrics; a
negative d_brier / d_logloss / d_bid_mae is an improvement. With the tiny
sample sizes available before real sessions accumulate, all numbers are
diagnostic, never a profitability conclusion (protocol rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .trend_dataset import HORIZONS
from .trend_market_baseline import (MARKET_FEATURES, evaluate, load_rows,
                                    MODELS)
from .trend_protocol import PROTOCOL_ID, protocol_hash

MECHANISMS = {
    "4A_post_pistol_freeze": "post_pistol_freeze",
    "4B_round_end_economy_update": "round_end_economy_update",
    "4C_prematch_prior_dislocation": "prematch_prior_dislocation",
}

GAME_FEATURES = [
    "game_score_t1", "game_score_t2", "game_score_gap", "game_round_num",
    "game_side_t1_code", "game_alive_t1", "game_alive_t2",
    "game_alive_gap", "game_hp_t1", "game_hp_t2", "game_hp_gap",
    "game_money_t1", "game_money_t2", "game_money_gap", "game_money_share_t1",
    "game_pistol_winner_code", "game_loss_streak_t1", "game_loss_streak_t2",
    "game_bomb_state", "game_state_present",
]


def encode_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode raw game_* columns into numeric features (NaN stays NaN).

    Categoricals: side CT=+1 / T=-1; pistol winner t1=+1 / t2=-1.
    ``game_state_present`` marks rows that actually joined a live 5E
    snapshot (prematch rows are 0) so absence is itself a feature.
    """
    out = pd.DataFrame(index=df.index)
    raw = {c: df[c] for c in df.columns}

    def num(col):
        series = raw.get(col)
        return series.astype(float) if series is not None else pd.Series(
            np.nan, index=df.index, dtype=float)

    score1, score2 = num("game_score_t1"), num("game_score_t2")
    out["game_score_t1"] = score1
    out["game_score_t2"] = score2
    out["game_score_gap"] = num("game_score_gap")
    out["game_round_num"] = num("game_round_num")
    side = raw.get("game_side_t1")
    out["game_side_t1_code"] = (side.map({"CT": 1.0, "T": -1.0})
                                .astype(float) if side is not None
                                else np.nan)
    # note: map uses exact labels from fivee_state; unknown -> NaN (recorded)
    alive1, alive2 = num("game_alive_t1"), num("game_alive_t2")
    hp1, hp2 = num("game_hp_t1"), num("game_hp_t2")
    money1, money2 = num("game_money_t1"), num("game_money_t2")
    out["game_alive_t1"], out["game_alive_t2"] = alive1, alive2
    out["game_alive_gap"] = alive1 - alive2
    out["game_hp_t1"], out["game_hp_t2"] = hp1, hp2
    out["game_hp_gap"] = hp1 - hp2
    out["game_money_t1"], out["game_money_t2"] = money1, money2
    out["game_money_gap"] = money1 - money2
    total_money = money1 + money2
    out["game_money_share_t1"] = (money1 / total_money.where(total_money > 0)
                                  ).where(total_money > 0)
    winner = raw.get("game_pistol_winner_side")
    out["game_pistol_winner_code"] = (winner.map({"t1": 1.0, "t2": -1.0})
                                      .astype(float) if winner is not None
                                      else np.nan)
    out["game_loss_streak_t1"] = num("game_loss_streak_t1")
    out["game_loss_streak_t2"] = num("game_loss_streak_t2")
    out["game_bomb_state"] = num("game_bomb_state")
    out["game_state_present"] = score1.notna().astype(float)
    return out


def build_feature_sets(df: pd.DataFrame) -> dict:
    game = encode_game_features(df)
    game_cols = [c for c in GAME_FEATURES if c in game.columns]
    # 编码列与原始 game_* 列同名（直通列），先丢弃原始列再拼接，避免重复标签
    df = df.drop(columns=[c for c in GAME_FEATURES if c in df.columns])
    df = pd.concat([df, game], axis=1)
    return df, {
        "market_only": list(MARKET_FEATURES),
        "game_only": game_cols,
        "market_plus_game": list(MARKET_FEATURES) + game_cols,
    }


DELTA_METRICS = ("brier", "logloss", "bid_mae", "net_mean")


def compute_deltas(slice_report: dict) -> dict:
    """candidate - market_only per (split, horizon, model) shared metrics."""
    deltas: dict = {}
    for split in slice_report.get("splits", []):
        for horizon, fs_results in split["horizons"].items():
            if "market_only" not in fs_results:
                continue
            for fs_name, models in fs_results.items():
                if fs_name == "market_only":
                    continue
                for model, metrics in models.items():
                    base = fs_results["market_only"].get(model) or {}
                    if "brier" not in metrics or "brier" not in base:
                        continue
                    key = (horizon, fs_name, model)
                    entry = deltas.setdefault(key, {m: [] for m in DELTA_METRICS})
                    for metric in DELTA_METRICS:
                        if metric in metrics and metric in base:
                            entry[metric].append(metrics[metric] - base[metric])
    out = {}
    for (horizon, fs_name, model), values in deltas.items():
        out.setdefault(horizon, {}).setdefault(fs_name, {})[model] = {
            f"d_{m}": float(np.mean(v)) for m, v in values.items() if v}
    return out


def evaluate_mechanisms(df: pd.DataFrame) -> dict:
    df, feature_sets = build_feature_sets(df)
    report = {"protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_hash(),
              "mechanisms": {}, "note": "deltas = candidate - market_only; "
                                         "negative is better (diagnostic only)"}
    for name, kind in MECHANISMS.items():
        sliced = df[df["decision_kind"] == kind]
        entry = {"decision_kind": kind, "n_rows": int(len(sliced)),
                 "feature_sets": {k: len(v) for k, v in feature_sets.items()}}
        if len(sliced) == 0:
            entry["skipped"] = "no_rows"
            report["mechanisms"][name] = entry
            continue
        slice_report = evaluate(sliced, feature_sets=feature_sets)
        entry.update({"n_sessions": slice_report["n_sessions"],
                      "insufficient_sessions": slice_report["insufficient_sessions"],
                      "splits": slice_report["splits"],
                      "deltas": compute_deltas(slice_report)})
        if name == "4C_prematch_prior_dislocation":
            entry["prior_feature_status"] = ("no_prematch_prior_feature_yet:"
                                             "赛前地图先验(roster强度)尚未进入数据集;"
                                             "game 列为空属构造使然,不进 delta 结论")
        report["mechanisms"][name] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    df = load_rows(args.rows)
    report = evaluate_mechanisms(df)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mechanism_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {name: {"n_rows": m["n_rows"],
                      "insufficient_sessions": m.get("insufficient_sessions")}
               for name, m in report["mechanisms"].items()}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
