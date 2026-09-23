"""Phase-3 market-only baseline models for the swing-trend dataset.

Uses ONLY market-side columns of the phase-2 rows (book/trade context).
Game-state columns are excluded here by construction; phase 4+ measures
their incremental value on identical samples.

Evaluation is strictly time-forward: sessions are ordered by their first
decision time and each split trains on all earlier sessions, tests on the
next one (expanding window, no series crossover). With fewer than 2
sessions the report carries ``insufficient_sessions`` and no scores.

Models per target:
  constant      base rate (classification) / train median (regression)
  random_walk   current price as probability (classification) / exit = entry bid
  logistic      L2-regularized logistic regression on standardized features
  xgboost       gradient boosting (shallow, subsampled)

Targets per horizon h in (15, 30, 60, 120):
  classification  P(net_h > 0)  over rows with a valid h-label
  regression      exit_bid_h    over rows with a valid h-label

Metrics: Brier, logloss, 10-bin calibration, bid MAE, and net-h label
mean/median/5%-quantile on the test session. Probability metrics never
substitute for net-return metrics (protocol rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .trend_dataset import HORIZONS
from .trend_protocol import PROTOCOL_ID, protocol_hash

MARKET_FEATURES = [
    "book_bid", "book_ask", "book_spread", "mid_at_entry", "other_mid",
    "mid_ret_5s", "mid_ret_15s", "mid_ret_30s", "mid_vol_60s",
    "bid_depth", "ask_depth", "depth_imbalance",
    "trade_flow_60s", "trades_60s", "jump_flag",
]
EPS = 1e-4


def load_rows(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths]
    if not frames:
        raise ValueError("no_rows_inputs")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("rows_inputs_empty")
    df["session_start"] = df.groupby("session_id")["info_mono"].transform("min")
    return df


def session_order(df: pd.DataFrame) -> list[str]:
    order = (df.groupby("session_id")["session_start"].first()
             .sort_values(kind="mergesort"))
    return list(order.index)


# ------------------------------------------------------------------ metrics
def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss_score(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not mask.any():
            continue
        out.append({"bin": [round(float(lo), 3), round(float(hi), 3)],
                    "n": int(mask.sum()),
                    "mean_pred": float(p[mask].mean()),
                    "mean_actual": float(y[mask].mean())})
    return out


# ------------------------------------------------------------------- models
def fit_predict(kind: str, model: str, X_train: pd.DataFrame, y_train,
                X_test: pd.DataFrame):
    """Return predictions for X_test. kind in {clf, reg}."""
    if model == "constant":
        if kind == "clf":
            return np.full(len(X_test), float(np.mean(y_train)))
        return np.full(len(X_test), float(np.median(y_train)))
    if model == "random_walk":
        if kind == "clf":
            if "mid_at_entry" in X_test:
                return np.clip(X_test["mid_at_entry"].to_numpy(float), EPS, 1 - EPS)
            return np.full(len(X_test), 0.5)  # no price column -> no-skill
        if "book_bid" in X_test:
            return X_test["book_bid"].to_numpy(float)
        return np.zeros(len(X_test))
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if kind == "clf":
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=1.0, max_iter=1000))
            pipe.fit(X_train, y_train)
            return pipe.predict_proba(X_test)[:, 1]
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(X_train, y_train)
        return pipe.predict(X_test)
    if model == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        if kind == "clf":
            booster = XGBClassifier(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, subsample=0.8,
                                    eval_metric="logloss", random_state=7)
            booster.fit(X_train, y_train)
            return booster.predict_proba(X_test)[:, 1]
        booster = XGBRegressor(n_estimators=200, max_depth=3,
                               learning_rate=0.05, subsample=0.8, random_state=7)
        booster.fit(X_train, y_train)
        return booster.predict(X_test)
    raise ValueError(f"unknown_model:{model}")


MODELS = ("constant", "random_walk", "logistic", "xgboost")


# ---------------------------------------------------------------- evaluation
def _score_models(X_train, X_test, v_train, v_test, y_clf_all, y_reg_all, net_all):
    """Score every model on one (split, horizon); identical masks for all."""
    result = {}
    for model in MODELS:
        if v_train.sum() == 0 or v_test.sum() == 0:
            result[model] = {"n_train": int(v_train.sum()),
                             "n_test": int(v_test.sum()),
                             "skipped": "empty_split"}
            continue
        try:
            p = fit_predict("clf", model, X_train, y_clf_all[v_train], X_test)
            r = fit_predict("reg", model, X_train, y_reg_all[v_train], X_test)
        except ValueError as exc:
            # 退化切片（如单一类别训练集）：记录原因，不中断整份报告
            result[model] = {"n_train": int(v_train.sum()),
                             "n_test": int(v_test.sum()),
                             "skipped": f"fit_error:{type(exc).__name__}"}
            continue
        y_test, net_test = y_clf_all[v_test], net_all[v_test]
        result[model] = {
            "n_train": int(v_train.sum()), "n_test": int(v_test.sum()),
            "brier": brier_score(y_test, p),
            "logloss": logloss_score(y_test, p),
            "calibration": calibration_bins(y_test, p),
            "bid_mae": float(np.mean(np.abs(r - y_reg_all[v_test]))),
            "net_mean": float(np.mean(net_test)),
            "net_median": float(np.median(net_test)),
            "net_q05": float(np.quantile(net_test, 0.05)),
        }
    return result


def evaluate(df: pd.DataFrame, feature_sets: dict | None = None) -> dict:
    """Whole-session forward evaluation.

    feature_sets: optional {name: [columns]}; when given, every horizon
    carries {feature_set: {model: metrics}} on identical row masks
    (phase-4 same-sample comparison). When omitted the phase-3 shape
    {model: metrics} with MARKET_FEATURES is preserved exactly.
    """
    multi = feature_sets is not None
    fs = feature_sets or {"market_only": list(MARKET_FEATURES)}
    if "session_start" not in df.columns:
        df = df.copy()
        df["session_start"] = df.groupby("session_id")["info_mono"].transform("min")
    sessions = session_order(df)
    report = {"protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_hash(),
              "market_features": MARKET_FEATURES, "models": list(MODELS),
              "feature_sets": {k: v for k, v in fs.items()},
              "multi_feature": multi,
              "sessions": sessions, "n_sessions": len(sessions),
              "splits": [], "insufficient_sessions": len(sessions) < 2}
    if len(sessions) < 2:
        report["note"] = ("whole-session forward evaluation needs >= 2 sessions; "
                          "no scores produced")
        return report
    X_by_fs = {name: df[cols].astype(float).fillna(0.0) for name, cols in fs.items()}
    for k in range(1, len(sessions)):
        train_ids, test_id = sessions[:k], sessions[k]
        train_mask = df["session_id"].isin(train_ids)
        test_mask = df["session_id"] == test_id
        split = {"train_sessions": train_ids, "test_session": test_id,
                 "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                 "horizons": {}}
        for horizon in HORIZONS:
            valid_np = df[f"label_valid_h{horizon}"].astype(bool).to_numpy()
            train_np, test_np = train_mask.to_numpy(), test_mask.to_numpy()
            v_train, v_test = valid_np & train_np, valid_np & test_np
            y_clf_all = (df[f"net_h{horizon}"] > 0).astype(int).to_numpy()
            y_reg_all = df[f"exit_bid_h{horizon}"].astype(float).to_numpy()
            net_all = df[f"net_h{horizon}"].astype(float).to_numpy()
            if multi:
                split["horizons"][f"h{horizon}"] = {
                    name: _score_models(X[v_train], X[v_test], v_train, v_test,
                                        y_clf_all, y_reg_all, net_all)
                    for name, X in X_by_fs.items()}
            else:
                X = X_by_fs["market_only"]
                split["horizons"][f"h{horizon}"] = _score_models(
                    X[v_train], X[v_test], v_train, v_test,
                    y_clf_all, y_reg_all, net_all)
        report["splits"].append(split)
    return report


def aggregate(report: dict) -> dict:
    """Mean of per-split metrics across the forward window (diagnostic only)."""
    if report.get("insufficient_sessions"):
        return {}
    out: dict = {}
    for split in report["splits"]:
        for horizon, models in split["horizons"].items():
            for model, metrics in models.items():
                if "brier" not in metrics:
                    continue
                key = (horizon, model)
                out.setdefault(key, []).append(metrics)
    aggregated = {}
    for (horizon, model), entries in out.items():
        aggregated.setdefault(horizon, {})[model] = {
            "brier_mean": float(np.mean([e["brier"] for e in entries])),
            "logloss_mean": float(np.mean([e["logloss"] for e in entries])),
            "bid_mae_mean": float(np.mean([e["bid_mae"] for e in entries])),
            "net_mean_mean": float(np.mean([e["net_mean"] for e in entries])),
            "n_test_total": int(sum(e["n_test"] for e in entries)),
        }
    return aggregated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    df = load_rows(args.rows)
    report = evaluate(df)
    report["aggregated"] = aggregate(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_sessions": report["n_sessions"],
                      "insufficient_sessions": report["insufficient_sessions"],
                      "rows": int(len(df))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
