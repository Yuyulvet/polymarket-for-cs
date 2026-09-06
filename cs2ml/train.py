"""Walk-forward training with anti-leakage evaluation.

Usage:
    python -m cs2ml.train            # build features, walk-forward train, save predictions
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURE_COLUMNS, build_features
from .store import connect


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    """Add swapped-orientation copies (anti-symmetry augmentation) for TRAINING only."""
    swapped = df.copy()
    diff_cols = [c for c in FEATURE_COLUMNS if c.endswith("_diff")]
    for c in diff_cols:
        swapped[c] = -swapped[c]
    # h2h_winrate is orientation-dependent but not a "_diff" column: team2's
    # winrate vs team1 is its complement (the 0.5 prior makes the flip exact).
    swapped["h2h_winrate"] = 1 - swapped["h2h_winrate"]
    swapped["y"] = 1 - swapped["y"]
    swapped["p_glicko"] = 1 - swapped["p_glicko"]
    swapped["p_ai"] = 1 - swapped["p_ai"]
    return pd.concat([df, swapped], ignore_index=True)


def _recency_weight(day_ord: np.ndarray, cutoff: float) -> np.ndarray:
    """Exponential decay toward the fold cutoff: newest matches weight ~1.0,
    halving every RECENCY_HALFLIFE_DAYS. Down-weights stale meta (version drift)."""
    age = cutoff - day_ord
    return np.clip(2.0 ** (-age / config.RECENCY_HALFLIFE_DAYS), 0.0, None)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window walk-forward. Fold k: train < T_k, test on [T_k, T_k + 30d)."""
    days = pd.to_datetime(df["start_date"], utc=True, format="ISO8601")
    day0 = days.min()
    day_ord = (days - day0).dt.total_seconds() / 86400.0

    preds: list[pd.DataFrame] = []
    fold_starts = list(range(config.MIN_TRAIN_DAYS, int(day_ord.max()) + 1, config.TEST_STEP_DAYS))
    print(f"rows={len(df)}, folds={len(fold_starts)}")

    for k, t_start in enumerate(fold_starts):
        t_end = t_start + config.TEST_STEP_DAYS
        train_mask = day_ord < t_start
        test_mask = (day_ord >= t_start) & (day_ord < t_end)
        train_df, test_df = df[train_mask], df[test_mask]
        if len(train_df) < 500 or len(test_df) == 0:
            continue

        # sample weights: elite teams (per-row) * recency decay (relative to cutoff)
        train_df = train_df.copy()
        train_df["sample_weight"] = (
            train_df["weight"].to_numpy()
            * _recency_weight(day_ord[train_mask].to_numpy(), t_start)
        )

        tr_aug = _augment(train_df)
        Xtr, ytr = tr_aug[FEATURE_COLUMNS], tr_aug["y"]
        wtr = tr_aug["sample_weight"].to_numpy()
        Xte, yte = test_df[FEATURE_COLUMNS], test_df["y"]

        # Model A: logistic regression (standardized)
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000),
        )
        lr.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
        p_lr = lr.predict_proba(Xte)[:, 1]

        # Model B: LightGBM with isotonic calibration fit INSIDE the training window
        base = LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=60, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=42, verbose=-1,
        )
        lgbm_cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        lgbm_cal.fit(Xtr, ytr, sample_weight=wtr)
        p_lgbm = lgbm_cal.predict_proba(Xte)[:, 1]

        fold_out = test_df[["match_id", "start_date", "team1_id", "team2_id", "y",
                            "p_glicko", "p_ai", "ai_available", "tier", "bo_type",
                            "version_era"]].copy()
        fold_out["p_lr"] = p_lr
        fold_out["p_lgbm"] = p_lgbm
        fold_out["fold"] = k
        preds.append(fold_out)

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def main() -> None:
    t0 = time.time()
    conn = connect()
    df = build_features(conn=conn)
    conn.close()
    if df.empty:
        raise SystemExit("no feature rows - run backfill first")
    print(f"features built: {len(df)} rows in {time.time() - t0:.1f}s")
    print(f"date range: {df['start_date'].min()} .. {df['start_date'].max()}")
    print(f"team1 winrate: {df['y'].mean():.3f}")

    preds = walk_forward(df)
    out_path = config.DATA_DIR / "predictions.parquet"
    preds.to_parquet(out_path, index=False)
    print(f"saved {len(preds)} predictions -> {out_path}")

    features_path = config.DATA_DIR / "features.parquet"
    df.to_parquet(features_path, index=False)
    print(f"saved features -> {features_path}")


if __name__ == "__main__":
    main()
