"""Pistol win-rate model v2 — the user's spec, fully systematized.

Inputs (all past-only, walk-forward by demo mtime):
  - side (CT=1)                         : map/starting-side prior
  - form: roster-mean log K/D history   : player recent individual state
  - pistol rate: roster historical pistol win rate (shrunk) : team pistol habit
  - pistol cs-net: roster gap of cs-net spatial features aggregated over
    PISTOL ROUNDS ONLY (round_idx 0/12) : team recent pistol tactics
  - general cs-net history as control channel

Labels: every map contributes 2 samples (round 1 and round 13 pistols, sides
swapped at the half). Output metric: AUC (ranking) + TOP-TAIL CALIBRATION
(decile actual win rates) — the strategy only bets the top-confidence games,
so the head of the calibration curve is what matters.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/csnet_features"
LABELS = ROOT / "data/map1/rebuild/full-20260916/map_labels.parquet"
ROUND_STATES = ROOT / "data/map1/rebuild/full-20260916/round_states.parquet"
KILLS = ROOT / "data/player_kills.parquet"
OUT = ROOT / "reports/pistol_model_v2"

CS = ["early_winrate", "mid_winrate", "early_alive_end", "mid_alive_end",
      "early_future_kill", "mid_future_kill"]


def norm_key(path: str) -> str:
    tail = re.sub(r"\.(dem|json\.gz|parquet)$", "", str(path).lower())
    i = tail.rfind("demos")
    if i >= 0:
        tail = tail[i + len("demos"):]
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def load_base():
    lab = pd.read_parquet(LABELS)
    lab = lab[lab["complete"]].copy()
    lab["key"] = lab["demo_path"].map(norm_key)
    lab["mtime"] = lab["demo_path"].map(lambda p: Path(p).stat().st_mtime)
    rs = pd.read_parquet(ROUND_STATES,
                         columns=["demo_path", "round_num", "ct_is_a",
                                  "winner_side"])
    rs["key"] = rs["demo_path"].map(norm_key)
    rs = rs[rs["round_num"].isin([1, 13])].drop_duplicates(
        subset=["key", "round_num"])
    rs["a_won"] = ((rs["winner_side"] == "CT") == rs["ct_is_a"]).astype(int)
    rs["a_ct"] = rs["ct_is_a"].astype(int)
    return lab, rs


def load_csnet():
    frames = [pd.read_parquet(p) for p in sorted(FEAT.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df["demo_path"].map(norm_key)
    df["steamid"] = df["steamid"].astype(str)
    return df


def load_kills(lab):
    kills = pd.read_parquet(KILLS)
    kills["key"] = kills["demo_path"].map(norm_key)
    kills = kills[kills["key"].isin(set(lab["key"]))]
    k = (kills.dropna(subset=["attacker"])
         .groupby(["key", "attacker"]).size().rename("k")
         .rename_axis(["key", "sid"]))
    d = (kills.groupby(["key", "victim"]).size().rename("d")
         .rename_axis(["key", "sid"]))
    kd = pd.concat([k, d], axis=1).fillna(0).reset_index()
    kd["sid"] = kd["sid"].astype(str)
    # map (key,sid) -> name via csnet features
    return kd


def main() -> int:
    import sklearn.linear_model as lm
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    OUT.mkdir(parents=True, exist_ok=True)
    lab, rs = load_base()
    cs = load_csnet()
    name_lut = (cs.drop_duplicates(["key", "steamid"])
                .set_index(["key", "steamid"])["player"].to_dict())
    kd = load_kills(lab)
    kd["name"] = kd.set_index(["key", "sid"]).index.map(name_lut)
    kd = kd.dropna(subset=["name"])

    # per (key, name) per-map aggregates
    cs_all = cs.groupby(["key", "player"])[CS].mean().reset_index()
    pist = cs[cs["round_idx"].isin([0, 12])]
    cs_pist = pist.groupby(["key", "player"])[CS].mean().reset_index()

    hist_cs_all: dict[str, np.ndarray] = {}
    hist_cs_pist: dict[str, np.ndarray] = {}
    hist_k: dict[str, float] = {}
    hist_d: dict[str, float] = {}
    hist_pw: dict[str, float] = {}   # pistol wins
    hist_pn: dict[str, float] = {}   # pistol samples

    rows = []
    for r in lab.sort_values("mtime").itertuples(index=False):
        key = r.key
        rr = rs[rs["key"] == key]
        if not len(rr) or key not in set(cs_all["key"]):
            continue
        present = cs_all[cs_all["key"] == key]
        sub = cs[cs["key"] == key]
        lut = dict(zip(sub["player"], sub["steamid"]))
        ra = set(str(r.roster_a).split(","))
        rb = set(str(r.roster_b).split(","))

        def names_of(roster):
            out = []
            for sid in roster:
                nm = next((n for n, s in lut.items() if s == sid), None)
                out.append(nm)
            return [n for n in out if n]

        def team_feats(names, hist):
            vals = [hist[n] for n in names if n in hist]
            return np.mean(vals, axis=0) if vals else None

        na, nb = names_of(ra), names_of(rb)
        for _, pr in rr.iterrows():
            a_ct = int(pr["a_ct"])
            row = {"key": key, "mtime": r.mtime,
                   "y": int(pr["a_won"]), "is_ct": a_ct}
            # form gap (a - b) in log-kd space
            def form(names):
                k = sum(hist_k.get(n, 0.0) for n in names)
                d = sum(hist_d.get(n, 0.0) for n in names)
                return np.log((k + 1.0) / (d + 1.0)) if (k + d) else None
            fa, fb = form(na), form(nb)
            if fa is not None and fb is not None:
                row["form_gap"] = fa - fb
            # pistol rate gap (shrunk)
            def prate(names):
                w = sum(hist_pw.get(n, 0.0) for n in names)
                n = sum(hist_pn.get(n, 0.0) for n in names)
                return (w + 3.0) / (n + 6.0) if n else None
            pa, pb = prate(na), prate(nb)
            if pa is not None and pb is not None:
                row["pistol_rate_gap"] = pa - pb
                row["pistol_rate_top"] = max(pa, pb)
            # cs-net gaps: all-rounds and pistol-only
            ca, cb = team_feats(na, hist_cs_all), team_feats(nb, hist_cs_all)
            if ca is not None and cb is not None:
                for i, c in enumerate(CS):
                    row[f"call_{c}"] = ca[i] - cb[i]
            cp_a = team_feats(na, hist_cs_pist)
            cp_b = team_feats(nb, hist_cs_pist)
            if cp_a is not None and cp_b is not None:
                for i, c in enumerate(CS):
                    row[f"cpist_{c}"] = cp_a[i] - cp_b[i]
            rows.append(row)
        # update history AFTER scoring this map (past-only)
        for _, pm in present.iterrows():
            v = pm[CS].values.astype(float)
            hist_cs_all[pm["player"]] = v  # latest-map signature (form = recent)
        for _, pm in cs_pist[cs_pist["key"] == key].iterrows():
            v = pm[CS].values.astype(float)
            hist_cs_pist[pm["player"]] = v
        for kr in kd[kd["key"] == key].itertuples(index=False):
            hist_k[kr.name] = hist_k.get(kr.name, 0.0) + kr.k
            hist_d[kr.name] = hist_d.get(kr.name, 0.0) + kr.d
        for _, pr in rr.iterrows():
            a_names, b_names = na, nb
            for nm in a_names:
                hist_pw[nm] = hist_pw.get(nm, 0.0) + int(pr["a_won"])
                hist_pn[nm] = hist_pn.get(nm, 0.0) + 1
            for nm in b_names:
                hist_pw[nm] = hist_pw.get(nm, 0.0) + (1 - int(pr["a_won"]))
                hist_pn[nm] = hist_pn.get(nm, 0.0) + 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "samples.csv", index=False)
    print(f"samples: {len(df)}")
    cutoff = df.mtime.quantile(0.7)
    train, test = df[df.mtime <= cutoff], df[df.mtime > cutoff]
    print(f"train {len(train)} / test {len(test)}")

    results: dict = {}
    variants = {
        "side_only": ["is_ct"],
        "side_form": ["is_ct", "form_gap"],
        "side_pistolrate": ["is_ct", "pistol_rate_gap"],
        "side_form_pistolrate": ["is_ct", "form_gap", "pistol_rate_gap"],
        "full_all": ["is_ct", "form_gap", "pistol_rate_gap"] +
                    [f"call_{c}" for c in CS],
        "full_pistcsnet": ["is_ct", "form_gap", "pistol_rate_gap"] +
                          [f"cpist_{c}" for c in CS],
    }
    for name, cols in variants.items():
        tr = train[cols + ["y"]].dropna()
        te = test[cols + ["y"]].dropna()
        if len(te) < 100 or tr.y.nunique() < 2:
            results[name] = {"error": f"tr{len(tr)} te{len(te)}"}
            continue
        sc = StandardScaler().fit(tr[cols])
        clf = lm.LogisticRegression(max_iter=2000).fit(
            sc.transform(tr[cols]), tr.y)
        p = clf.predict_proba(sc.transform(te[cols]))[:, 1]
        auc = roc_auc_score(te.y, p)
        # top-tail calibration: predicted-prob deciles -> actual win rates
        dec = pd.qcut(pd.Series(p, index=te.index), 10,
                      duplicates="drop")
        cal = (pd.DataFrame({"p": p, "y": te.y.values}, index=te.index)
               .groupby(dec, observed=True)
               .agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")))
        results[name] = {
            "auc": round(float(auc), 4), "n_test": int(len(te)),
            "top20_pred": round(float(cal.pred.iloc[-2:].mean()), 4),
            "top20_actual": round(float(cal.actual.iloc[-2:].mean()), 4),
            "top10_pred": round(float(cal.pred.iloc[-1]), 4),
            "top10_actual": round(float(cal.actual.iloc[-1]), 4),
            "bot10_actual": round(float(cal.actual.iloc[0]), 4),
        }
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
