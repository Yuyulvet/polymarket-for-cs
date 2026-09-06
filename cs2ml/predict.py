"""Predict upcoming matches with the final model.

Usage:
    python -m cs2ml.predict [--date YYYY-MM-DD] [--days N]

Trains on all finished matches (augmented), builds current team states,
fetches upcoming matches from bo3.gg, and writes predictions to
reports/predictions_<date>.csv (plus console summary).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, store
from .bo3gg import Bo3Client
from .features import (
    FEATURE_COLUMNS,
    TeamState,
    _day,
    build_features,
    build_states,
    compute_features,
    load_matches,
)
from .train import _augment, _recency_weight


def _team_names(conn) -> dict[int, str]:
    return {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM teams")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default today UTC)")
    ap.add_argument("--days", type=int, default=2, help="how many upcoming days to scan")
    args = ap.parse_args()

    today = args.date or datetime.utcnow().strftime("%Y-%m-%d")

    conn = store.connect()
    print("building features from finished matches ...")
    df = build_features(conn=conn)
    if df.empty:
        conn.close()
        raise SystemExit("no training data - run backfill first")

    names = _team_names(conn)
    matches = load_matches(conn)
    conn.close()

    # --- final models: trained on ALL data, calibrated in-sample via cv ---
    days = pd.to_datetime(df["start_date"], utc=True, format="ISO8601")
    day_ord = (days - days.min()).dt.total_seconds() / 86400.0
    cutoff = day_ord.max()
    df = df.copy()
    df["sample_weight"] = df["weight"].to_numpy() * _recency_weight(day_ord.to_numpy(), cutoff)
    tr = _augment(df)
    Xtr, ytr = tr[FEATURE_COLUMNS], tr["y"]
    wtr = tr["sample_weight"].to_numpy()

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    lr.fit(Xtr, ytr, logisticregression__sample_weight=wtr)

    lgbm = CalibratedClassifierCV(
        LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=60, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=42, verbose=-1,
        ),
        method="isotonic", cv=3,
    )
    lgbm.fit(Xtr, ytr, sample_weight=wtr)
    print(f"models trained on {len(df)} matches ({df['start_date'].min()[:10]} .. {df['start_date'].max()[:10]})")

    # --- current team states from full history ---
    states = build_states(matches)

    # --- upcoming matches ---
    client = Bo3Client()
    rows = []
    d = datetime.strptime(today, "%Y-%m-%d")
    for i in range(args.days):
        date = (d + timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            payload = client.matches_for_date(date, "upcoming")
            ms, _, _ = Bo3Client.parse_day(payload)
        except Exception as e:  # noqa: BLE001
            print(f"{date}: fetch error {e}")
            continue
        for m in ms:
            if m.get("status") and m["status"] not in ("upcoming", "live", "scheduled", "announced"):
                continue
            day = _day(m.get("start_date") or "")
            if day != day:
                continue
            t1, t2 = m.get("team1_id"), m.get("team2_id")
            if not t1 or not t2 or t1 == t2:
                continue
            s1 = states.setdefault(int(t1), TeamState())
            s2 = states.setdefault(int(t2), TeamState())
            feats = compute_features(s1, s2, day, m)
            if feats is None:
                rows.append({
                    "date": date, "match": f"{names.get(int(t1), t1)} vs {names.get(int(t2), t2)}",
                    "bo": m.get("bo_type"), "tier": m.get("tier"),
                    "p_lgbm": None, "note": "insufficient history (<3 games a side)",
                })
                continue
            X = pd.DataFrame([feats])[FEATURE_COLUMNS]
            p_lr = float(lr.predict_proba(X)[0, 1])
            p_gl = float(feats["p_glicko"])
            p_lg = float(lgbm.predict_proba(X)[0, 1])
            # blend guard: shrink extreme model outputs toward glicko baseline
            p_final = 0.7 * p_lg + 0.3 * p_gl
            rows.append({
                "date": date,
                "match": f"{names.get(int(t1), t1)} vs {names.get(int(t2), t2)}",
                "bo": m.get("bo_type"), "tier": m.get("tier"),
                "p_lr": round(p_lr, 3), "p_glicko": round(p_gl, 3),
                "p_lgbm": round(p_lg, 3), "p_final": round(p_final, 3),
                "note": "",
            })

    out = pd.DataFrame(rows)
    out_path = config.REPORTS_DIR / f"predictions_{today}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n{len(out)} upcoming matches -> {out_path}")
    if not out.empty:
        with pd.option_context("display.max_rows", 200, "display.width", 200):
            print(out.to_string(index=False))


if __name__ == "__main__":
    main()
