"""Strict-time CS2 post-pistol (round 2 / 14) conversion research.

The target is the immediate round winner after each regulation pistol. Inputs
are restricted to information available at the post-pistol round start plus
strictly prior match history. This module is offline research only; it neither
reads Polymarket prices nor enables trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .map1_data import path_key, roster_key, utc
from .map1_model import probability_metrics
from .pistol_model import (ROSTER, _indexed_roster_features, _roster_history_index,
                           forward_splits, load_event_manifest, matchup_slugs,
                           training_weights)


REFERENCE_REPOSITORY = "https://github.com/TaiZo1/cs2-win-prediction"
REFERENCE_COMMIT = "aba1689661219c085e2a3515d7086c5e65712325"
LOGISTIC_C = .01
ECONOMY = ["a_is_ct", "equipment_gap", "equipment_log_ratio"]
PISTOL_CONTEXT = [
    "pistol_winner_gap", "score_gap_before", "previous_final_margin_gap",
    "previous_stomp_gap", "previous_comeback_gap", "previous_bomb_t_gap",
]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _winner_roster(row) -> str:
    side = str(row.winner_side).upper()
    if side == "CT":
        return roster_key(row.ct_roster)
    if side in {"T", "TERRORIST"}:
        return roster_key(row.t_roster)
    raise ValueError("unknown_round_winner")


def _finite_integer(value, name: str) -> int:
    number = float(value)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError("invalid_" + name)
    return int(number)


def extract_post_pistol_events(rounds: pd.DataFrame, history: pd.DataFrame,
                               labels: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, dict]:
    """Extract R2/R14 targets with only the preceding pistol's completed state."""
    h = history.copy()
    h["map_id"] = h.map_id.map(path_key)
    h["start_at"] = h.start_at.map(utc)
    h["available_at"] = h.available_at.map(utc)
    if h.map_id.duplicated().any() or (h.available_at <= h.start_at).any():
        raise ValueError("invalid_history_identity_or_time")
    meta = h.set_index("map_id").to_dict("index")

    eligible = labels.copy()
    eligible["map_id"] = eligible.map_id.map(path_key)
    eligible = eligible[eligible.complete.eq(True) & eligible.history_check.eq("matched")]
    if eligible.map_id.duplicated().any():
        raise ValueError("duplicate_map_labels")
    allowed = set(eligible.map_id)

    r = rounds.copy()
    r["map_id"] = r.demo_path.map(path_key)
    rejected, records = Counter(), []
    target_slugs = set(manifest["_slugs"])
    for map_id, group in r[r.map_id.isin(allowed)].groupby("map_id", sort=True):
        try:
            map_records = []
            m = meta.get(map_id)
            if m is None:
                raise ValueError("missing_history")
            d = group.sort_values("round_num")
            if d.round_num.tolist() != list(range(1, len(d) + 1)) or len(d) < 2:
                raise ValueError("incomplete_round_ledger")
            a, b = roster_key(m["roster_a"]), roster_key(m["roster_b"])
            if set(d.roster_a.map(roster_key)) != {a} or set(d.roster_b.map(roster_key)) != {b}:
                raise ValueError("roster_identity_mismatch")
            matchup = matchup_slugs(map_id)
            priority = bool(matchup and set(matchup) & target_slugs)
            recent = manifest["_recent_at"] <= m["start_at"] < manifest["_starts_at"]
            for post_round in (2, 14):
                current_rows = d[d.round_num.eq(post_round)]
                if current_rows.empty:
                    continue
                current = current_rows.iloc[0]
                pistol_rows = d[d.round_num.eq(post_round - 1)]
                if len(pistol_rows) != 1:
                    raise ValueError("missing_preceding_pistol")
                pistol = pistol_rows.iloc[0]

                past = d[d.round_num.lt(post_round)]
                past_winners = past.apply(_winner_roster, axis=1)
                score_a = _finite_integer(current.score_a_before, "score_a_before")
                score_b = _finite_integer(current.score_b_before, "score_b_before")
                if score_a != int(past_winners.eq(a).sum()) or score_b != int(past_winners.eq(b).sum()):
                    raise ValueError("pre_round_score_mismatch")

                winner = _winner_roster(current)
                pistol_winner = _winner_roster(pistol)
                if winner not in {a, b} or pistol_winner not in {a, b}:
                    raise ValueError("winner_outside_rosters")
                a_is_ct = bool(current.ct_is_a)
                equipment_a = float(current.ct_equip0 if a_is_ct else current.t_equip0)
                equipment_b = float(current.t_equip0 if a_is_ct else current.ct_equip0)
                margin = float(pistol.final_margin)
                if not np.isfinite([equipment_a, equipment_b, margin]).all() or min(
                        equipment_a, equipment_b) < 0:
                    raise ValueError("invalid_equipment_or_margin")
                pistol_sign = 1.0 if pistol_winner == a else -1.0
                outcome = str(pistol.outcome_type).casefold()
                if outcome not in {"stomp", "comeback", "even"}:
                    raise ValueError("invalid_pistol_outcome_type")
                pistol_a_is_t = not bool(pistol.ct_is_a)
                bomb_t_gap = (1.0 if pistol_a_is_t else -1.0) if bool(pistol.bomb_planted) else 0.0
                map_records.append({
                    "map_id": map_id, "match_id": m["match_id"], "map_name": m["map_name"],
                    "post_round": post_round, "roster_a": a, "roster_b": b,
                    "start_at": m["start_at"], "available_at": m["available_at"],
                    "y": int(winner == a), "a_is_ct": 1.0 if a_is_ct else -1.0,
                    "equipment_gap": equipment_a - equipment_b,
                    "equipment_log_ratio": float(np.log((equipment_a + 100) /
                                                          (equipment_b + 100))),
                    "pistol_winner_gap": pistol_sign,
                    "score_gap_before": float(score_a - score_b),
                    "previous_final_margin_gap": pistol_sign * margin,
                    "previous_stomp_gap": pistol_sign if outcome == "stomp" else 0.0,
                    "previous_comeback_gap": pistol_sign if outcome == "comeback" else 0.0,
                    "previous_bomb_t_gap": bomb_t_gap,
                    "target_event_team": priority, "recent_three_months": bool(recent),
                    "matchup_slugs": "" if matchup is None else "|".join(matchup),
                })
            records.extend(map_records)
        except (ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
    if not records:
        raise ValueError("empty_post_pistol_targets")
    result = pd.DataFrame(records).sort_values(
        ["start_at", "match_id", "map_id", "post_round"]).reset_index(drop=True)
    if result.duplicated(["map_id", "post_round"]).any():
        raise ValueError("duplicate_post_pistol_targets")
    return result, {
        "source_round_maps": int(r.map_id.nunique()), "eligible_label_maps": len(allowed),
        "maps": int(result.map_id.nunique()), "post_pistol_rows": len(result),
        "round_2_rows": int(result.post_round.eq(2).sum()),
        "round_14_rows": int(result.post_round.eq(14).sum()),
        "rejected_maps": dict(rejected),
    }


def build_features(events: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    h = history.copy()
    h["map_id"] = h.map_id.map(path_key)
    h["start_at"] = h.start_at.map(utc)
    h["available_at"] = h.available_at.map(utc)
    by_roster = _roster_history_index(h)
    cache, records = {}, []
    for row in events.to_dict("records"):
        if row["map_id"] not in cache:
            cache[row["map_id"]] = _indexed_roster_features(
                by_roster, row["roster_a"], row["roster_b"], row["map_name"],
                utc(row["start_at"]), row["match_id"])
        prematch, coverage = cache[row["map_id"]]
        records.append({**row, **prematch, **coverage})
    result = pd.DataFrame(records).sort_values(
        ["start_at", "match_id", "map_id", "post_round"]).reset_index(drop=True)
    required = [*ECONOMY, *PISTOL_CONTEXT, *ROSTER]
    if not np.isfinite(result[required].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite_post_pistol_features")
    for row in result.to_dict("records"):
        latest = row.get("max_history_available_at")
        if pd.notna(latest) and utc(latest) >= utc(row["start_at"]):
            raise ValueError("future_roster_history")
    return result


def model_definitions() -> dict[str, list[str]]:
    return {
        "side_only": ["a_is_ct"],
        "pistol_winner_only": ["pistol_winner_gap"],
        "pistol_winner_and_side": ["pistol_winner_gap", "a_is_ct"],
        "economy": ECONOMY,
        "economy_pistol_context": ECONOMY + PISTOL_CONTEXT,
        "economy_context_prematch": ECONOMY + PISTOL_CONTEXT + ROSTER,
    }


def fit_model(train: pd.DataFrame, features: list[str], reference_at: pd.Timestamp):
    if len(train) < 2 or train.y.nunique() < 2:
        raise ValueError("training_requires_both_outcomes")
    x = train[features].to_numpy(dtype=float)
    y = train.y.to_numpy(dtype=int)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_training_features")
    weights = training_weights(train, reference_at)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=LOGISTIC_C, fit_intercept=False, max_iter=2000))
    model.fit(np.concatenate([x, -x]), np.concatenate([y, 1-y]),
              logisticregression__sample_weight=np.tile(weights, 2))
    return model


def _metric_set(frame: pd.DataFrame, columns: dict[str, str]) -> dict:
    return {name: probability_metrics(frame.y, frame[column]) for name, column in columns.items()}


def evaluate(data: pd.DataFrame, min_train: int = 80) -> tuple[pd.DataFrame, dict]:
    d = data.sort_values(["start_at", "match_id", "map_id", "post_round"]).reset_index(drop=True).copy()
    definitions = model_definitions()
    d["p_constant_0_5"] = .5
    for name in definitions:
        d["p_" + name] = np.nan
    d["train_rows"] = 0
    d["fold_id"] = -1
    d["train_max_available_at"] = ""
    folds = []
    for fold_id, (train_index, test_index, test_start) in enumerate(forward_splits(d), 1):
        train, test = d.iloc[train_index], d.iloc[test_index]
        if len(train) < min_train or train.y.nunique() < 2:
            continue
        if train.available_at.map(utc).max() >= test_start:
            raise ValueError("future_training_label")
        if set(train.match_id) & set(test.match_id):
            raise ValueError("series_overlap")
        for name, features in definitions.items():
            model = fit_model(train, features, test_start)
            d.loc[test_index, "p_" + name] = model.predict_proba(
                test[features].to_numpy(dtype=float))[:, 1]
        d.loc[test_index, "train_rows"] = len(train)
        d.loc[test_index, "fold_id"] = fold_id
        d.loc[test_index, "train_max_available_at"] = train.available_at.max().isoformat()
        folds.append({"test_start": test_start.isoformat(), "train_rows": len(train),
                      "train_series": int(train.match_id.nunique()), "test_rows": len(test),
                      "test_series": int(test.match_id.nunique())})
    columns = {"constant_0_5": "p_constant_0_5",
               **{name: "p_" + name for name in definitions}}
    common = d.dropna(subset=list(columns.values()))
    recent = common[common.target_event_team & common.recent_three_months]
    baseline = (common.p_economy-common.y) ** 2
    paired = {name: float((((common[column]-common.y) ** 2)-baseline).mean())
              for name, column in columns.items()}
    fold_metrics = {str(int(fold_id)): _metric_set(frame, columns)
                    for fold_id, frame in common.groupby("fold_id", sort=True)}
    all_metrics = _metric_set(common, columns)
    public_baseline = all_metrics["pistol_winner_and_side"]
    economy_metrics = all_metrics["economy"]
    report = {
        "target_rows": len(d), "scored_rows": len(common),
        "scored_series": int(common.match_id.nunique()), "models": definitions,
        "all_scored_metrics": all_metrics,
        "round_2_metrics": _metric_set(common[common.post_round.eq(2)], columns),
        "round_14_metrics": _metric_set(common[common.post_round.eq(14)], columns),
        "recent_target_rows": len(recent),
        "recent_target_metrics": _metric_set(recent, columns),
        "paired_brier_minus_economy": paired, "fold_metrics": fold_metrics,
        "increment_over_public_pistol_winner_and_side": {
            "auc": economy_metrics["auc"] - public_baseline["auc"],
            "brier_reduction": public_baseline["brier"] - economy_metrics["brier"],
            "accuracy": economy_metrics["accuracy"] - public_baseline["accuracy"],
        },
        "folds": folds,
    }
    return d, report


def run(*, history_path: Path, rounds_path: Path, labels_path: Path,
        manifest_path: Path, output_dir: Path, min_train: int = 80) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing_overwrite:{output_dir}")
    inputs = {"history": Path(history_path), "rounds": Path(rounds_path),
              "labels": Path(labels_path), "manifest": Path(manifest_path)}
    hashes = {name: file_hash(path) for name, path in inputs.items()}
    history, rounds, labels = (pd.read_parquet(inputs[name])
                               for name in ("history", "rounds", "labels"))
    manifest = load_event_manifest(manifest_path)
    raw, audit = extract_post_pistol_events(rounds, history, labels, manifest)
    data = build_features(raw, history)
    predictions, evaluation = evaluate(data, min_train=min_train)
    final_train = data[data.available_at.map(utc).lt(manifest["_starts_at"])]
    models = {name: {"features": features,
                     "model": fit_model(final_train, features, manifest["_starts_at"])}
              for name, features in model_definitions().items()}
    report = {
        "version": 3, "mode": "offline_post_pistol_conversion_research",
        "event": manifest["event"], "event_starts_at": manifest["starts_at"],
        "live_trading_enabled": False, "promoted_model": None, "execution_pnl": None,
        "audit": audit, "evaluation": evaluation,
        "reference": {"repository": REFERENCE_REPOSITORY, "commit": REFERENCE_COMMIT,
                      "license": "MIT", "adopted_concepts": [
                          "separate post-pistol regime", "freeze-end equipment state",
                          "whole-match temporal evaluation"],
                      "copied_source_code": False},
        "feature_scope": {
            "available_now": ["round-start equipment", "score", "starting side",
                              "preceding pistol result/margin/type/bomb plant",
                              "strictly prior roster/map history"],
            "deferred_until_rebuilt": ["cash", "weapon classes", "armor/helmet",
                                       "utility inventory", "saved equipment by roster"],
            "equipment_semantics": "mean current_equip_value over five players at the first post-freeze snapshot",
        },
        "validation": "whole-series forward blocks; results require available_at < block start; mirrored only after split",
        "limitations": [
            "This predicts R2/R14 winner, not Map Winner or Polymarket price movement.",
            "Historical data has already been inspected, so this is development evidence rather than fresh acceptance data.",
            "No synchronized historical order books exist for these maps; profitability is not evaluated.",
            "Round-start equipment is available, but cash, exact weapons and utility require a new demo extraction schema.",
        ],
        "model_stability": {"logistic_C": LOGISTIC_C, "status": "diagnostic candidate only"},
        "inputs": {name: {"path": str(path.resolve()), "sha256": hashes[name]}
                   for name, path in inputs.items()},
    }
    if hashes != {name: file_hash(path) for name, path in inputs.items()}:
        raise ValueError("input_changed_while_running")
    output_dir.mkdir(parents=True, exist_ok=False)
    data.to_parquet(output_dir / "dataset.parquet", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    joblib.dump(models, output_dir / "models.joblib")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = config.DATA_DIR / "map1"
    parser.add_argument("--history", type=Path, default=root / "history.parquet")
    parser.add_argument("--rounds", type=Path,
                        default=root / "rebuild" / "full-20260916" / "round_states.parquet")
    parser.add_argument("--labels", type=Path,
                        default=root / "rebuild" / "full-20260916" / "map_labels.parquet")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("starladder_fall_2026.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=80)
    args = parser.parse_args(argv)
    report = run(history_path=args.history, rounds_path=args.rounds,
                 labels_path=args.labels, manifest_path=args.manifest,
                 output_dir=args.output, min_train=args.min_train)
    print(json.dumps({"output": str(args.output.resolve()), "audit": report["audit"],
                      "evaluation": report["evaluation"]["all_scored_metrics"],
                      "promoted_model": None}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
