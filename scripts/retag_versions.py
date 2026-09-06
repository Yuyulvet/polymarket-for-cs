"""Recompute version_era for all stored matches (after editing versions.py)."""
from cs2ml.store import connect
from cs2ml.versions import version_era

conn = connect()
rows = conn.execute("SELECT id, start_date FROM matches").fetchall()
with conn:
    for r in rows:
        conn.execute("UPDATE matches SET version_era=? WHERE id=?",
                     (version_era(r["start_date"]), r["id"]))
print(f"re-tagged {len(rows)} matches")
for era, n in conn.execute(
        "SELECT version_era, COUNT(*) FROM matches GROUP BY version_era ORDER BY 1"):
    print(f"  {era}: {n}")
conn.close()
