"""Expanding, whole-series, result-availability-aware Map 1 baseline.

The offline dataset assumes that the map and lineups were known. Its metrics
are diagnostic, not evidence of executable returns. Forward records supply
the missing information timestamps and actual books.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .map1_data import FEATURES, utc


def training_rows(data: pd.DataFrame, at, excluded_matches=()) -> pd.DataFrame:
    cutoff = utc(at)
    return data[data.available_at.lt(cutoff) & data.decision_at.lt(cutoff)
                & ~data.match_id.isin(excluded_matches)].copy()


def fit_model(train: pd.DataFrame, feature_columns=None):
    if train.empty or train.y.nunique() < 2:
        raise ValueError("Training requires both outcomes")
    columns = FEATURES if feature_columns is None else list(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Expected distinct feature columns")
    x = train[columns].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    # Mirror only after splitting; never let a mirrored twin enter validation.
    y = train.y.to_numpy(dtype=int)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=.25, fit_intercept=False, max_iter=2000))
    pipeline.fit(np.concatenate([x, -x]), np.concatenate([y, 1 - y]))
    return pipeline


def walk_forward(data: pd.DataFrame, min_train: int = 60, holdout_fraction: float = .2,
                 feature_columns=None) -> pd.DataFrame:
    if min_train < 2 or not 0 < holdout_fraction < 1:
        raise ValueError("Invalid training size or holdout fraction")
    d = data.sort_values(["decision_at", "match_id"]).reset_index(drop=True).copy()
    columns = FEATURES if feature_columns is None else list(feature_columns)
    if d.duplicated("match_id").any() or not d.map_no.eq(1).all():
        raise ValueError("Expected exactly one Map 1 row per series")
    times = sorted(d.decision_at.unique())
    if not times:
        raise ValueError("No Map 1 rows")
    boundary = times[min(int(len(times) * (1 - holdout_fraction)), len(times) - 1)]
    d["partition"] = np.where(d.decision_at.ge(boundary), "holdout", "development")
    d["p_model"] = np.nan
    d["n_train"] = 0
    d["train_max_available_at"] = ""
    d["reason"] = "insufficient_training_history"
    frozen = training_rows(d, boundary)
    frozen_model = fit_model(frozen, columns) if len(frozen) >= min_train and frozen.y.nunique() == 2 else None
    for at, group in d.groupby("decision_at", sort=True):
        is_holdout = at >= boundary
        train = frozen if is_holdout else training_rows(d, at, group.match_id.tolist())
        if len(train) < min_train or train.y.nunique() < 2:
            continue
        model = frozen_model if is_holdout else fit_model(train, columns)
        if model is None:
            continue
        assert train.available_at.max() < at
        assert not set(train.match_id) & set(group.match_id)
        d.loc[group.index, "p_model"] = model.predict_proba(group[columns].to_numpy())[:, 1]
        d.loc[group.index, "n_train"] = len(train)
        d.loc[group.index, "train_max_available_at"] = train.available_at.max().isoformat()
        d.loc[group.index, "reason"] = "conditional_map_known_diagnostic"
    return d


def probability_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    valid = np.isfinite(p) & np.isin(y, [0, 1])
    y, p = y[valid].astype(int), np.clip(p[valid], 1e-6, 1 - 1e-6)
    if not len(y):
        return {"n": 0}
    calibration = []
    # Integer bins avoid overlapping/gapped boundaries caused by float arange.
    bins = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        low = index / 10
        mask = bins == index
        if mask.any():
            calibration.append({"low": round(float(low), 1), "n": int(mask.sum()),
                                "predicted": float(p[mask].mean()), "actual": float(y[mask].mean())})
    return {"n": len(y), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "accuracy": float(((p >= .5) == y).mean()), "calibration": calibration}


def evaluation_report(predictions: pd.DataFrame, audit: dict) -> dict:
    metrics = {}
    for name, group in predictions.groupby("partition"):
        g = group[group.p_model.notna()]
        metrics[name] = {"model": probability_metrics(g.y, g.p_model),
                         "constant_0_5": probability_metrics(g.y, np.full(len(g), .5))}
    return {"mode": "conditional_map_known_research", "audit": audit, "metrics": metrics,
            "holdout_policy": "last 20% of decision timestamps; model frozen before holdout",
            "execution_pnl": None,
            "execution_status": "not_estimated_without_observed_veto_lineups_and_synchronized_books",
            "market_baseline_status": "requires_forward_records_with_verified_roster_to_outcome_mapping"}


def forward_comparison(records: list[dict], min_train: int = 30) -> dict:
    """One settled forward observation per event, with past-only hybrid fitting.

    Expected fields: event_id, decision_at, resolved_at, payout_a, p_model,
    p_market. Fractional/void payouts belong in accounting, not binary metrics.
    """
    if not records:
        return {"n": 0}
    d = pd.DataFrame(records)
    d["decision_at"] = d.decision_at.map(utc)
    d = d.sort_values("decision_at").drop_duplicates("event_id", keep="first")
    d = d[d.payout_a.isin([0, 1])].copy()
    if d.empty:
        return {"n": 0}
    d["resolved_at"] = d.resolved_at.map(utc)
    d["p_hybrid"] = np.nan
    cols = ["p_model", "p_market"]
    def logits(frame):
        p = np.clip(frame[cols].to_numpy(dtype=float), 1e-5, 1 - 1e-5)
        return np.log(p / (1 - p))
    for index, row in d.iterrows():
        tr = d[d.resolved_at.lt(row.decision_at) & d.decision_at.lt(row.decision_at)
               & d.event_id.ne(row.event_id)]
        if len(tr) < min_train or tr.payout_a.nunique() < 2:
            continue
        lr = LogisticRegression(C=.25, max_iter=2000).fit(logits(tr), tr.payout_a.astype(int))
        d.loc[index, "p_hybrid"] = lr.predict_proba(logits(d.loc[[index]]))[0, 1]
    common = d[d.p_hybrid.notna()]
    return {"n": len(d), "model": probability_metrics(d.payout_a, d.p_model),
            "market": probability_metrics(d.payout_a, d.p_market),
            "hybrid_common_sample": {name: probability_metrics(common.payout_a, common[col])
                                     for name, col in (("model", "p_model"), ("market", "p_market"),
                                                       ("hybrid", "p_hybrid"))}}
