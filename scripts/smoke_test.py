"""Smoke test: feature builder on whatever data is currently stored."""
import sys

from cs2ml.store import connect
from cs2ml.features import build_features

conn = connect()
n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
days = conn.execute("SELECT COUNT(*) FROM backfill_days WHERE status IN ('ok','empty')").fetchone()[0]
last = conn.execute("SELECT MAX(date) FROM backfill_days WHERE status IN ('ok','empty')").fetchone()[0]
print(f"stored: {n} matches, {days} days done, latest {last}")

df = build_features(conn=conn)
conn.close()
print(f"feature rows: {len(df)}")
if not df.empty:
    print(df[["start_date", "team1_id", "team2_id", "y", "p_glicko",
              "rating_diff", "form10_diff", "h2h_games", "tier"]].head(8).to_string())
    print("\nteam1 winrate:", round(df["y"].mean(), 4))
    import numpy as np
    y = df["y"].to_numpy()
    p = np.clip(df["p_glicko"].to_numpy(), 1e-6, 1 - 1e-6)
    from sklearn.metrics import log_loss, brier_score_loss
    print("glicko-only in-sample logloss:", round(log_loss(y, p, labels=[0, 1]), 4))
    print("glicko-only in-sample brier:", round(brier_score_loss(y, p), 4))
    print("glicko-only accuracy:", round(((p >= 0.5).astype(int) == y).mean(), 4))
sys.exit(0)
