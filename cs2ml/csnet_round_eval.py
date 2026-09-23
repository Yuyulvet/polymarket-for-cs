"""Round-level gate for cs-net: can its spatial read beat the economy baseline?

User's correction (2026-09-22): K/D form is a MAP-level signal and is nearly
irrelevant at ROUND level — the tradable question is whether cs-net's read of
the CURRENT round beats equip_gap (round-transition baseline AUC 0.745).

Design:
- Rows: one per (demo, round). y = CT won the round (from cs-net winner col).
- Features: per-round team means of the 6 cs-net features -> CT-T gaps.
  * early_* = freeze+opening (~8s) -> available at the round-start decision
    point (the tradeable set).
  * mid_* added = in-round oracle upper bound.
- equip_gap joined from round_states (ct_equip0 - t_equip0); score gap
  CT-aligned via ct_is_a.
- Time-ordered: demos sorted by mtime, earliest 70% train / latest 30% test;
  bootstrap resamples DEMOS (rounds of a map are not independent).
- Gate: paired AUC increment over equip baseline must be > 0.
- Pistol subset (round_num in {1,13}): economy is flat by construction; the
  one place a spatial read could matter. Directly relevant to the user's
  pistol-swing strategy.
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
ROUND_STATES = ROOT / "data/map1/rebuild/full-20260916/round_states.parquet"
OUT = ROOT / "reports/csnet_round_eval"

ALL6 = ["early_winrate", "mid_winrate", "early_alive_end", "mid_alive_end",
        "early_future_kill", "mid_future_kill"]
EARLY3 = ["early_winrate", "early_alive_end", "early_future_kill"]


def norm_key(path: str) -> str:
    tail = re.sub(r"\.(dem|json\.gz|parquet)$", "", str(path).lower())
    i = tail.rfind("demos")
    if i >= 0:
        tail = tail[i + len("demos"):]
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def load_rounds() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(FEAT_DIR.glob("*.parquet"))]
    cs = pd.concat(frames, ignore_index=True)
    cs["key"] = cs["demo_path"].map(norm_key)
    # per-round team means -> CT-T gaps
    means = (cs.groupby(["key", "round_idx", "team"])[ALL6].mean()
             .unstack("team"))
    means = means.swaplevel(0, 1, axis=1)
    gaps = means["CT"] - means["T"]
    rounds = gaps.reset_index()
    rounds.columns = ["key", "round_idx"] + [
        f"gap_{c}" for c in ALL6]
    win = (cs.groupby(["key", "round_idx"])["winner"].first()
           .map({"CT": 1, "T": 0}))
    rounds["y"] = rounds.set_index(["key", "round_idx"]).index.map(win)
    # demo mtime as time axis
    lab = pd.read_parquet(LABELS, columns=["demo_path", "complete"])
    lab = lab[lab["complete"]]
    lab["key"] = lab["demo_path"].map(norm_key)
    mtime = {r.key: Path(r.demo_path).stat().st_mtime
             for r in lab.itertuples(index=False) if Path(r.demo_path).exists()}
    rounds["mtime"] = rounds["key"].map(mtime)
    # economy + score from round_states
    rs = pd.read_parquet(ROUND_STATES,
                         columns=["demo_path", "round_num", "ct_equip0",
                                  "t_equip0", "score_a_before",
                                  "score_b_before", "ct_is_a", "winner_side"])
    rs["key"] = rs["demo_path"].map(norm_key)
    rs = rs.drop_duplicates(subset=["key", "round_num"])
    rs["round_idx"] = rs["round_num"] - 1
    rs["equip_gap"] = rs["ct_equip0"] - rs["t_equip0"]
    ct_is_a = rs["ct_is_a"].astype(bool)
    rs["score_ct_gap"] = np.where(ct_is_a, rs["score_a_before"] - rs["score_b_before"],
                                  rs["score_b_before"] - rs["score_a_before"])
    rounds = rounds.merge(
        rs[["key", "round_idx", "equip_gap", "score_ct_gap", "round_num"]],
        on=["key", "round_idx"], how="left")
    rounds["pistol"] = rounds["round_num"].isin([1, 13])
    return rounds.dropna(subset=["y", "mtime"]).reset_index(drop=True)


def main() -> int:
    import sklearn.linear_model as lm
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    OUT.mkdir(parents=True, exist_ok=True)
    rounds = load_rounds()
    rounds.to_csv(OUT / "rounds.csv", index=False)
    print(f"rounds: {len(rounds)} | demos: {rounds.key.nunique()} "
          f"| pistol: {int(rounds.pistol.sum())}")

    cutoff = rounds.mtime.quantile(0.7)
    train = rounds[rounds.mtime <= cutoff]
    test = rounds[rounds.mtime > cutoff]
    print(f"train {len(train)} / test {len(test)}")

    results: dict = {}

    def run(tag, df_tr, df_te, cols):
        sc = StandardScaler().fit(df_tr[cols])
        clf = lm.LogisticRegression(max_iter=2000).fit(
            sc.transform(df_tr[cols]), df_tr.y)
        return pd.Series(clf.predict_proba(sc.transform(df_te[cols]))[:, 1],
                         index=df_te.index)

    def paired(base_p, new_p, df_te, n_boot=2000):
        common = base_p.index.intersection(new_p.index)
        yv = df_te.loc[common, "y"].values
        demo = df_te.loc[common, "key"].values
        A, B = base_p.loc[common].values, new_p.loc[common].values
        auc_a, auc_b = roc_auc_score(yv, A), roc_auc_score(yv, B)
        rng = np.random.default_rng(5)
        demo_ids = np.unique(demo)
        diffs = []
        for _ in range(n_boot):
            pick = rng.choice(demo_ids, len(demo_ids), replace=True)
            mask = np.isin(demo, pick)
            if len(np.unique(yv[mask])) < 2:
                continue
            diffs.append(roc_auc_score(yv[mask], B[mask])
                         - roc_auc_score(yv[mask], A[mask]))
        ci = np.percentile(diffs, [2.5, 97.5]) if diffs else [np.nan] * 2
        return {"base_auc": round(float(auc_a), 4),
                "new_auc": round(float(auc_b), 4),
                "diff": round(float(auc_b - auc_a), 4),
                "ci95": [round(float(c), 4) for c in ci],
                "p_gt0": round(float(np.mean(np.array(diffs) > 0)), 3),
                "n_rounds": int(len(common)),
                "n_test_dem": int(pd.Series(demo).nunique())}

    # full-round models (all rounds, non-pistol economy dominated)
    cols_e = ["equip_gap"]
    cols_e3 = [f"gap_{c}" for c in EARLY3]
    cols_all6 = [f"gap_{c}" for c in ALL6]
    for name, cols in [("E_equip", cols_e),
                       ("E_plus_score", cols_e + ["score_ct_gap"]),
                       ("R_early3_only", cols_e3),
                       ("E_plus_R_early3", cols_e + cols_e3),
                       ("E_plus_R_all6", cols_e + cols_all6)]:
        dtr = train[cols + ["y"]].dropna()
        dte = test[cols + ["y"]].dropna()
        p = run(name, dtr, dte, cols)
        results[name] = {"auc": round(float(roc_auc_score(dte.y, p)), 4),
                         "n_test": int(len(dte))}
        if name != "E_equip":
            results[name]["paired_vs_E"] = paired(
                run("E_equip", train[cols_e + ["y"]].dropna(),
                    test[cols_e + ["y"]].dropna(), cols_e), p, test)

    # pistol subset: economy flat -> does the spatial read carry anything?
    pis = rounds[rounds.pistol]
    ptr = pis[pis.mtime <= cutoff]
    pte = pis[pis.mtime > cutoff]
    if len(pte) >= 60 and pte.y.nunique() == 2:
        for name, cols in [("PISTOL_E_equip", cols_e),
                           ("PISTOL_R_early3_only", cols_e3),
                           ("PISTOL_E_plus_R_early3", cols_e + cols_e3)]:
            dtr = ptr[cols + ["y"]].dropna()
            dte = pte[cols + ["y"]].dropna()
            if len(dte) < 40 or len(dtr) < 100:
                results[name] = {"error": "insufficient"}
                continue
            p = run(name, dtr, dte, cols)
            results[name] = {"auc": round(float(roc_auc_score(dte.y, p)), 4),
                             "n_test": int(len(dte))}
        pb = run("pe", ptr[cols_e + ["y"]].dropna(),
                 pte[cols_e + ["y"]].dropna(), cols_e)
        pn = run("pn", ptr[cols_e3 + ["y"]].dropna(),
                 pte[cols_e3 + ["y"]].dropna(), cols_e3)
        results["PISTOL_R_early3_vs_E"] = paired(pb, pn, pte)
    else:
        results["PISTOL"] = {"error": f"test {len(pte)}"}

    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
