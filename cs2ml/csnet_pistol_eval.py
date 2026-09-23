"""Pistol-round-specific evaluation: can anything predict pistol winners?

User's core strategy (cs2-pistol-swing-strategy) buys the form-stronger team
prematch and sells into the sentiment pop after a pistol WIN — so the pistol
outcome is the trigger. This eval measures what is actually predictable:

- P0 prematch: history form + cs-net history features + starting side ->
  decision "which team to buy" before the map starts.
- P1 in-pistol (+8s): adds the pistol round's own early-segment spatial read
  -> decision ~10s into the round (tradeable only with a fast feed).
- Raw stat for the strategy: actual pistol win rate of the form-favored team.

Samples: round 1 and round 13 pistols (sides swap), 593 eligible maps.
Time-ordered 70/30 by demo mtime, demo-level paired bootstrap.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PHASE0_ROWS = ROOT / "reports/csnet_phase0/rows.csv"
ROUND_ROWS = ROOT / "reports/csnet_round_eval/rounds.csv"
ROUND_STATES = ROOT / "data/map1/rebuild/full-20260916/round_states.parquet"
OUT = ROOT / "reports/csnet_pistol_eval"

C2 = ["c2_early_winrate_gap", "c2_mid_winrate_gap",
      "c2_early_alive_end_gap", "c2_mid_alive_end_gap",
      "c2_early_future_kill_gap", "c2_mid_future_kill_gap"]
EARLY3 = ["gap_early_winrate", "gap_early_alive_end", "gap_early_future_kill"]


def norm_key(path: str) -> str:
    tail = re.sub(r"\.(dem|json\.gz|parquet)$", "", str(path).lower())
    i = tail.rfind("demos")
    if i >= 0:
        tail = tail[i + len("demos"):]
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def load_samples() -> pd.DataFrame:
    p0 = pd.read_csv(PHASE0_ROWS)
    rr = pd.read_csv(ROUND_ROWS)
    rs = pd.read_parquet(ROUND_STATES,
                         columns=["demo_path", "round_num", "ct_is_a",
                                  "winner_side"])
    rs["key"] = rs["demo_path"].map(norm_key)
    rs = rs[rs["round_num"].isin([1, 13])].drop_duplicates(
        subset=["key", "round_num"])
    rs["a_won"] = ((rs["winner_side"] == "CT") == rs["ct_is_a"]).astype(int)
    rs["second_half"] = (rs["round_num"] == 13).astype(int)

    pist = rs.merge(p0[["key", "mtime", "form_a", "form_b"] + C2],
                    on="key", how="inner")
    pist["form_gap"] = pist["form_a"] - pist["form_b"]
    # in-pistol early read (rounds.csv gaps are CT-minus-T of that round)
    early = rr[rr["round_num"].isin([1, 13])][
        ["key", "round_num"] + EARLY3]
    pist = pist.merge(early, on=["key", "round_num"], how="left")
    # convert early gaps (CT-T) to roster_a frame
    for c in EARLY3:
        sign = np.where(pist["ct_is_a"].astype(bool), 1.0, -1.0)
        pist[f"a_{c}"] = pist[c] * sign
    return pist.dropna(subset=["a_won", "mtime"]).reset_index(drop=True)


def main() -> int:
    import sklearn.linear_model as lm
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_samples()
    df.to_csv(OUT / "samples.csv", index=False)
    print(f"pistol samples: {len(df)} | demos: {df.key.nunique()}")
    df["second_half_ct"] = np.where(
        df["second_half"] == 1, 1 - df["ct_is_a"].astype(int),
        df["ct_is_a"].astype(int))

    cutoff = df.mtime.quantile(0.7)
    train = df[df.mtime <= cutoff]
    test = df[df.mtime > cutoff]
    print(f"train {len(train)} / test {len(test)}")

    results: dict = {}

    def fitprobs(dtr, dte, cols):
        sc = StandardScaler().fit(dtr[cols])
        clf = lm.LogisticRegression(max_iter=2000).fit(
            sc.transform(dtr[cols]), dtr.a_won)
        return pd.Series(clf.predict_proba(
            sc.transform(dte[cols]))[:, 1], index=dte.index)

    def paired(base_p, new_p, dte, n_boot=2000):
        common = base_p.index.intersection(new_p.index)
        yv = dte.loc[common, "a_won"].values
        demo = dte.loc[common, "key"].values
        A, B = base_p.loc[common].values, new_p.loc[common].values
        rng = np.random.default_rng(5)
        ids = np.unique(demo)
        diffs = []
        for _ in range(n_boot):
            m = np.isin(demo, rng.choice(ids, len(ids), replace=True))
            if len(np.unique(yv[m])) < 2:
                continue
            diffs.append(roc_auc_score(yv[m], B[m]) - roc_auc_score(yv[m], A[m]))
        ci = np.percentile(diffs, [2.5, 97.5]) if diffs else [np.nan] * 2
        return {"base": round(float(roc_auc_score(yv, A)), 4),
                "new": round(float(roc_auc_score(yv, B)), 4),
                "diff": round(float(roc_auc_score(yv, B) - roc_auc_score(yv, A)), 4),
                "ci95": [round(float(c), 4) for c in ci],
                "p_gt0": round(float(np.mean(np.array(diffs) > 0)), 3),
                "n": int(len(common))}

    variants = {
        "SIDE_only": ["second_half_ct"],
        "SIDE_plus_form": ["second_half_ct", "form_gap"],
        "SIDE_plus_c2": ["second_half_ct"] + C2,
        "SIDE_form_c2": ["second_half_ct", "form_gap"] + C2,
        "P1_plus_early3": ["second_half_ct", "form_gap"] + C2 +
                          [f"a_{c}" for c in EARLY3],
    }
    probs = {}
    for name, cols in variants.items():
        dtr = train[cols + ["a_won"]].dropna()
        dte = test[cols + ["a_won"]].dropna()
        if len(dte) < 60 or dtr.a_won.nunique() < 2:
            results[name] = {"error": f"tr{len(dtr)} te{len(dte)}"}
            continue
        probs[name] = fitprobs(dtr, dte, cols)
        results[name] = {"auc": round(float(roc_auc_score(
            dte.a_won, probs[name])), 4), "n_test": int(len(dte))}
    base = probs.get("SIDE_only")
    for name in ("SIDE_plus_form", "SIDE_plus_c2", "SIDE_form_c2",
                 "P1_plus_early3"):
        if name in probs and base is not None:
            results[name]["paired_vs_SIDE"] = paired(base, probs[name], test)

    # raw strategy stat: pistol win rate of the form-favored team
    fav = df.dropna(subset=["form_gap"])
    fav_rate = float((fav.a_won[fav.form_gap > 0]).mean())
    fav_n = int((fav.form_gap > 0).sum())
    # split by time to show it's not an artifact
    fav_tr = fav[fav.mtime <= cutoff]
    fav_te = fav[fav.mtime > cutoff]
    results["strategy_stat_form_favored_pistol_winrate"] = {
        "all": round(fav_rate, 4), "n": fav_n,
        "train": round(float(fav_tr.a_won[fav_tr.form_gap > 0].mean()), 4),
        "test": round(float(fav_te.a_won[fav_te.form_gap > 0].mean()), 4),
        "test_n": int((fav_te.form_gap > 0).sum()),
    }
    # base rates
    results["base_rate_a_won_pistol"] = round(float(df.a_won.mean()), 4)

    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
