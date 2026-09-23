"""Phase 0 gate for cs-net prematch features (paper-only research).

Question: do pretrained cs-net spatial-only per-player features (winrate /
alive_end / future_kill, early & mid round segments) carry MAP-WINNER signal
that beats the per-player form (K/D) baseline (AUC 0.62) under honest
time-ordered evaluation?

Feature families, all roster-mean aggregated:
- F  form gap: log K/D history per player, past maps only (baseline replica).
- C2 history csnet gap: per-player mean of the 6 cs-net features over PAST
  maps only -> true prematch analog of form. THIS is the gate comparison.
- C1 within-map early rounds (round_idx < 6, this map only): spatial read of
  the map's own opening -> diagnostic only (NOT prematch; live territory).

History merges by player NAME (steamid splits across demos). Roster
assignment uses per-map steamid lookup in the cs-net feature rows. Time axis
= demo file mtime (download order follows schedule); train = earliest 70%,
test = latest 30%, single temporal split + bootstrap CIs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "data/csnet_features"
LABELS = ROOT / "data/map1/rebuild/full-20260916/map_labels.parquet"
KILLS = ROOT / "data/player_kills.parquet"
OUT = ROOT / "reports/csnet_phase0"

CSNET_COLS = ["early_winrate", "mid_winrate", "early_alive_end",
              "mid_alive_end", "early_future_kill", "mid_future_kill"]


def norm_key(path: str) -> str:
    """Join key: tail after the last 'demos' segment, separators unified.

    Labels carry real .dem paths; cs-net feature rows carry flattened
    csnet_json slug paths (`_polymarket_data_demos_hltv-..._m1-dust2`).
    Both share the tail `...demos/<match>/<map>.(dem|json.gz)`.
    """
    tail = re.sub(r"\.(dem|json\.gz|parquet)$", "", str(path).lower())
    i = tail.rfind("demos")
    if i >= 0:
        tail = tail[i + len("demos"):]
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def load_csnet() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(FEAT_DIR.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df["demo_path"].map(norm_key)
    df["steamid"] = df["steamid"].astype(str)
    return df


def load_labels(keys: set[str]) -> pd.DataFrame:
    lab = pd.read_parquet(LABELS)
    lab = lab[lab["complete"]].copy()
    lab["key"] = lab["demo_path"].map(norm_key)
    lab = lab[lab["key"].isin(keys)].copy()
    lab["mtime"] = lab["demo_path"].map(lambda p: Path(p).stat().st_mtime)
    lab["y"] = (lab["winner_roster"] == lab["roster_a"]).astype(int)
    lab = lab.sort_values("mtime").reset_index(drop=True)
    for side in ("a", "b"):
        lab[f"roster_{side}_set"] = lab[f"roster_{side}"].map(
            lambda s: set(str(s).split(",")))
    return lab


def kd_table(labels: pd.DataFrame) -> pd.DataFrame:
    kills = pd.read_parquet(KILLS)
    kills["key"] = kills["demo_path"].map(norm_key)
    keys = set(labels["key"])
    kills = kills[kills["key"].isin(keys)]
    k = (kills.dropna(subset=["attacker"])
         .groupby(["key", "attacker"]).size().rename("kills")
         .rename_axis(["key", "steamid"]))
    d = (kills.groupby(["key", "victim"]).size().rename("deaths")
         .rename_axis(["key", "steamid"]))
    kd = pd.concat([k, d], axis=1).fillna(0).reset_index()
    kd["steamid"] = kd["steamid"].astype(str)
    return kd


def build_rows(labels, csnet, kd):
    """Walk maps in time order, past-only history, emit one feature row each."""
    per_map_player = dict(tuple(csnet.groupby("key")))
    kd_by_key = dict(tuple(kd.groupby("key")))
    csnet_hist: dict[str, np.ndarray] = {}
    csnet_n: dict[str, int] = {}
    kd_k: dict[str, float] = {}
    kd_d: dict[str, float] = {}
    rows = []
    for r in labels.itertuples(index=False):
        present = per_map_player.get(r.key)
        lut = (dict(zip(present["steamid"], present["player"]))
               if present is not None else {})

        def name_of(sid):
            return lut.get(str(sid))

        def team_hist(names):
            vals = [csnet_hist[n] for n in names
                    if n is not None and n in csnet_hist]
            return (np.mean(vals, axis=0), len(vals)) if vals else (None, 0)

        def team_form(roster):
            k = sum(kd_k.get(s, 0.0) for s in roster)
            d = sum(kd_d.get(s, 0.0) for s in roster)
            return float(np.log((k + 1.0) / (d + 1.0))) if (k + d) else None

        names_a = [name_of(s) for s in r.roster_a_set]
        names_b = [name_of(s) for s in r.roster_b_set]
        row = {"key": r.key, "y": int(r.y), "mtime": r.mtime}
        fa, fb = team_form(r.roster_a_set), team_form(r.roster_b_set)
        row["form_a"], row["form_b"] = fa, fb
        ca, na = team_hist(names_a)
        cb, nb = team_hist(names_b)
        if ca is not None and cb is not None:
            for i, c in enumerate(CSNET_COLS):
                row[f"c2_{c}_gap"] = float(ca[i] - cb[i])
        row["c2_n_a"], row["c2_n_b"] = na, nb
        # C1: this map's own early rounds (round_idx < 6), per roster player
        if present is not None:
            early = present[present["round_idx"] < 6]

            def team_early(names):
                g = early[early["player"].isin([n for n in names if n])]
                return g[CSNET_COLS].mean().values if len(g) else None
            ea, eb = team_early(names_a), team_early(names_b)
            if ea is not None and eb is not None and not np.isnan(ea).any():
                for i, c in enumerate(CSNET_COLS):
                    row[f"c1_{c}_gap"] = float(ea[i] - eb[i])
        rows.append(row)
        # update history AFTER scoring this map (past-only)
        if present is not None:
            pm = present.groupby("player")[CSNET_COLS].mean()
            for name, vals in pm.iterrows():
                v = vals.values.astype(float)
                cur, n = csnet_hist.get(name), csnet_n.get(name, 0)
                csnet_hist[name] = v if cur is None else (cur * n + v) / (n + 1)
                csnet_n[name] = n + 1
        for kr in kd_by_key.get(r.key, pd.DataFrame()).itertuples(index=False):
            kd_k[kr.steamid] = kd_k.get(kr.steamid, 0.0) + kr.kills
            kd_d[kr.steamid] = kd_d.get(kr.steamid, 0.0) + kr.deaths
    return pd.DataFrame(rows)


def main() -> int:
    import sklearn.linear_model as lm
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    OUT.mkdir(parents=True, exist_ok=True)
    csnet = load_csnet()
    labels = load_labels(set(csnet["key"]))
    print(f"labels matched: {len(labels)} maps")
    rows = build_rows(labels, csnet, kd_table(labels))
    rows.to_csv(OUT / "rows.csv", index=False)
    print(f"rows built: {len(rows)}")

    cutoff = rows.mtime.quantile(0.7)
    train = rows[rows.mtime <= cutoff]
    test = rows[rows.mtime > cutoff]
    print(f"train {len(train)} / test {len(test)}")

    results: dict = {}
    probs: dict[str, pd.Series] = {}

    def evaluate(name, cols):
        tr = train[cols + ["y"]].dropna()
        te = test[cols + ["y"]].dropna()
        if len(te) < 20 or tr.y.nunique() < 2:
            results[name] = {"error": "insufficient data"}
            return
        sc = StandardScaler().fit(tr[cols])
        clf = lm.LogisticRegression(max_iter=2000).fit(sc.transform(tr[cols]), tr.y)
        p = clf.predict_proba(sc.transform(te[cols]))[:, 1]
        probs[name] = pd.Series(p, index=te.index)
        auc = roc_auc_score(te.y, p)
        rng = np.random.default_rng(7)
        yv = te.y.values
        boots = [roc_auc_score(yv[b], p[b]) for b in
                 (rng.integers(0, len(yv), len(yv)) for _ in range(1000))
                 if len(np.unique(yv[b])) == 2]
        ci = np.percentile(boots, [2.5, 97.5]) if boots else [np.nan] * 2
        results[name] = {"auc": round(float(auc), 4),
                         "ci95": [round(float(c), 4) for c in ci],
                         "n_train": int(len(tr)), "n_test": int(len(te))}

    evaluate("F_form_gap", ["form_a", "form_b"])
    c2 = [f"c2_{c}_gap" for c in CSNET_COLS]
    c1 = [f"c1_{c}_gap" for c in CSNET_COLS]
    evaluate("C2_csnet_history_gap", c2)
    evaluate("C1_csnet_earlymap_gap", c1)
    evaluate("C2_plus_F", c2 + ["form_a", "form_b"])

    # paired bootstrap: does C2+F beat F on IDENTICAL test rows?
    if "F_form_gap" in probs and "C2_plus_F" in probs:
        common = probs["F_form_gap"].index.intersection(probs["C2_plus_F"].index)
        if len(common) >= 20:
            yv = rows.loc[common, "y"].values
            pf = probs["F_form_gap"].loc[common].values
            pc = probs["C2_plus_F"].loc[common].values
            rng = np.random.default_rng(11)
            diffs = []
            for _ in range(2000):
                b = rng.integers(0, len(yv), len(yv))
                if len(np.unique(yv[b])) < 2:
                    continue
                diffs.append(roc_auc_score(yv[b], pc[b]) - roc_auc_score(yv[b], pf[b]))
            ci = np.percentile(diffs, [2.5, 97.5])
            results["paired_diff_C2F_minus_F"] = {
                "auc_diff": round(float(roc_auc_score(yv, pc) - roc_auc_score(yv, pf)), 4),
                "ci95": [round(float(c), 4) for c in ci],
                "n": int(len(common)),
                "p_gt0": round(float(np.mean(np.array(diffs) > 0)), 3),
            }

    uni = {}
    for c in c2:
        d = rows[["y", "mtime", c]].dropna()
        if len(d) > 50:
            te = d[d.mtime > cutoff]
            if len(te) > 20 and te.y.nunique() == 2:
                uni[c] = round(float(roc_auc_score(te.y, te[c])), 4)
    results["c2_univariate_test_auc"] = uni
    results["coverage"] = {
        "n_maps": int(len(rows)),
        "form_both_teams": int(rows[["form_a", "form_b"]].notna().all(1).sum()),
        "c2_both_teams": int(rows[c2].notna().all(1).sum()),
        "c1_both_teams": int(rows[c1].notna().all(1).sum()),
    }
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
