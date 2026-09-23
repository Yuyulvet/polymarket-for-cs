"""Pistol win-rate model v3 — external-literature features on top of v2.

Web-search driven additions (2026-09-23), all past-only, walk-forward by mtime:
  - side_prior: per-MAP shrunk CT pistol win rate (public sources confirm CT pistols
    favored on ALL maps, degree varies; nobody publishes per-map pistol tables,
    our 593-demo pool computes what the market doesn't have)
  - fk_gap: roster gap of per-player FIRST-KILL rate in pistol rounds
    (opening frag = #2 cited pistol factor)
  - tk_gap: roster gap of per-player TRADE-KILL rate in pistol rounds
    (trading efficiency = #1 cited pistol factor)

Carried over from v2: is_ct, form_gap, pistol_rate_gap, call_* (cs-net all-round).
Pistol-only cs-net (cpist_*) stays OUT (v2: it collapsed the tail).

Phase 1: 70/30 time split, variant ablation, Platt calibration (fit on first 80%
of train, calibrated on last 20%), >=60% subset on test.
Phase 2: expanding-window OOF over ALL samples -> every out-of-sample prediction
with Platt-in-window -> the honest "all matches >=60%" validation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pistol_model_v2 import CS, load_base, load_csnet, norm_key

ROOT = Path(__file__).resolve().parent.parent
KILLS = ROOT / "data/player_kills.parquet"
OUT = ROOT / "reports/pistol_model_v3"

BASE = ["is_ct", "form_gap", "pistol_rate_gap"]
V3NEW = ["side_prior", "fk_gap", "tk_gap"]
VARIANTS = {
    "control_v2best": BASE + [f"call_{c}" for c in CS],
    "v3_nocsnet": BASE + V3NEW,
    "v3_full": BASE + V3NEW + [f"call_{c}" for c in CS],
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(p), float(c - h), float(c + h))


def build_rows():
    lab, rs = load_base()
    cs = load_csnet()
    name_lut = (cs.drop_duplicates(["key", "steamid"])
                .set_index(["key", "steamid"])["player"].to_dict())

    kills = pd.read_parquet(KILLS)
    kills["key"] = kills["demo_path"].map(norm_key)
    keep = set(lab["key"])
    kills = kills[kills["key"].isin(keep)]
    kills["aname"] = kills.set_index(["key", kills["attacker"].astype(str)]).index.map(name_lut)
    kills["vname"] = kills.set_index(["key", kills["victim"].astype(str)]).index.map(name_lut)
    pist = kills[kills["round_num"].isin([1, 13])]

    # per (key, name) aggregates
    kd_k = (kills.dropna(subset=["aname"]).groupby(["key", "aname"]).size())
    kd_d = kills.dropna(subset=["vname"]).groupby(["key", "vname"]).size()
    fk = (pist[pist["is_first_kill"] == True]          # noqa: E712
          .dropna(subset=["aname"]).groupby(["key", "aname"]).size())
    tk = (pist[pist["trade"] == True]                   # noqa: E712
          .dropna(subset=["aname"]).groupby(["key", "aname"]).size())

    cs_all = cs.groupby(["key", "player"])[CS].mean().reset_index()

    hist_cs: dict[str, np.ndarray] = {}
    hist_k: dict[str, float] = {}
    hist_d: dict[str, float] = {}
    hist_pw: dict[str, float] = {}
    hist_pn: dict[str, float] = {}
    hist_fk: dict[str, float] = {}
    hist_tk: dict[str, float] = {}
    map_hist: dict[str, list] = {}   # map_name -> [ct_wins, n]

    rows = []
    for r in lab.sort_values("mtime").itertuples(index=False):
        key, map_name = r.key, r.map_name
        rr = rs[rs["key"] == key]
        if not len(rr) or key not in set(cs_all["key"]):
            continue
        sub = cs[cs["key"] == key]
        lut = dict(zip(sub["player"], sub["steamid"]))

        def names_of(roster):
            out = []
            for sid in str(roster).split(","):
                nm = next((n for n, s in lut.items() if s == sid), None)
                if nm:
                    out.append(nm)
            return out

        def rate(names, num, den):
            vals = [num.get(n, 0.0) / den[n] for n in names
                    if n in den and den[n] > 0]
            return float(np.mean(vals)) if len(vals) == len(names) and vals else None

        na, nb = names_of(r.roster_a), names_of(r.roster_b)
        for _, pr in rr.iterrows():
            a_ct = int(pr["a_ct"])
            w, n = map_hist.get(map_name, [0.0, 0.0])
            prior = (w + 5.5) / (n + 10.0) if n else 0.55
            row = {"key": key, "mtime": r.mtime, "map": map_name,
                   "y": int(pr["a_won"]), "is_ct": a_ct,
                   "side_prior": prior if a_ct else 1.0 - prior}

            def form(names):
                k = sum(hist_k.get(x, 0.0) for x in names)
                d = sum(hist_d.get(x, 0.0) for x in names)
                return np.log((k + 1.0) / (d + 1.0)) if (k + d) else None

            def prate(names):
                w2 = sum(hist_pw.get(x, 0.0) for x in names)
                n2 = sum(hist_pn.get(x, 0.0) for x in names)
                return (w2 + 3.0) / (n2 + 6.0) if n2 else None

            fa, fb = form(na), form(nb)
            if fa is not None and fb is not None:
                row["form_gap"] = fa - fb
            pa, pb = prate(na), prate(nb)
            if pa is not None and pb is not None:
                row["pistol_rate_gap"] = pa - pb
            fka, fkb = rate(na, hist_fk, hist_pn), rate(nb, hist_fk, hist_pn)
            if fka is not None and fkb is not None:
                row["fk_gap"] = fka - fkb
            tka, tkb = rate(na, hist_tk, hist_pn), rate(nb, hist_tk, hist_pn)
            if tka is not None and tkb is not None:
                row["tk_gap"] = tka - tkb
            ta = [hist_cs[x] for x in na if x in hist_cs]
            tb = [hist_cs[x] for x in nb if x in hist_cs]
            if len(ta) == len(na) and len(tb) == len(nb) and na and nb:
                for i, c in enumerate(CS):
                    row[f"call_{c}"] = float(np.mean(ta, 0)[i]
                                             - np.mean(tb, 0)[i])
            rows.append(row)

        # ---- history updates (this map becomes "past" for the next one)
        for _, pm in cs_all[cs_all["key"] == key].iterrows():
            hist_cs[pm["player"]] = pm[CS].values.astype(float)
        for nm, v in kd_k.get(key, pd.Series(dtype=float)).items():
            hist_k[nm] = hist_k.get(nm, 0.0) + float(v)
        for nm, v in kd_d.get(key, pd.Series(dtype=float)).items():
            hist_d[nm] = hist_d.get(nm, 0.0) + float(v)
        fk_m = fk.get(key, pd.Series(dtype=float))
        tk_m = tk.get(key, pd.Series(dtype=float))
        for nm in set(na) | set(nb):
            hist_pn[nm] = hist_pn.get(nm, 0.0) + len(rr)
            hist_pw[nm] = hist_pw.get(nm, 0.0) \
                + sum(int(pr["a_won"]) for _, pr in rr.iterrows()
                      if nm in na) \
                + sum(1 - int(pr["a_won"]) for _, pr in rr.iterrows()
                      if nm in nb)
        for nm, v in fk_m.items():
            hist_fk[nm] = hist_fk.get(nm, 0.0) + float(v)
        for nm, v in tk_m.items():
            hist_tk[nm] = hist_tk.get(nm, 0.0) + float(v)
        w, n = map_hist.get(map_name, [0.0, 0.0])
        for _, pr in rr.iterrows():
            w += float(pr["winner_side"] == "CT")
            n += 1.0
        map_hist[map_name] = [w, n]

    return pd.DataFrame(rows)


def platt_fit(clf, sc, X, y):
    import sklearn.linear_model as lm
    s = clf.decision_function(sc.transform(X))
    return lm.LogisticRegression(max_iter=1000).fit(
        s.reshape(-1, 1), y)


def platt_apply(p, clf, sc, X):
    s = clf.decision_function(sc.transform(X)).reshape(-1, 1)
    return p.predict_proba(s)[:, 1]


def ece(y, p, bins=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["b"] = pd.qcut(df.p, bins, duplicates="drop")
    g = df.groupby("b", observed=True).agg(pm=("p", "mean"),
                                           ym=("y", "mean"), n=("y", "size"))
    return float((g.pm - g.ym).abs().mul(g.n / g.n.sum()).sum())


def subset60(y, p):
    m = p >= 0.60
    if m.sum() == 0:
        return {"n": 0}
    k, n = int(y[m].sum()), int(m.sum())
    lo = wilson(k, n)
    # EV legs: delta~3 limit-entry adjusted (v2 convention) and raw tape legs
    ev_disc = (p[m] * 9.25 - (1 - p[m]) * 8.2).mean()
    ev_raw = (p[m] * 6.25 - (1 - p[m]) * 11.2).mean()
    return {"n": n, "hit": round(k / n, 4), "ci": [round(lo[1], 3), round(lo[2], 3)],
            "mean_p": round(float(p[m].mean()), 4),
            "ev_delta3_pt": round(float(ev_disc), 3),
            "ev_raw_pt": round(float(ev_raw), 3)}


def main() -> int:
    import sklearn.linear_model as lm
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    OUT.mkdir(parents=True, exist_ok=True)
    df = build_rows()
    df.to_csv(OUT / "samples.csv", index=False)
    print(f"samples: {len(df)}  (maps {df.key.nunique()})")

    # ---------------- Phase 1: 70/30 split ablation + Platt
    cutoff = df.mtime.quantile(0.7)
    train, test = df[df.mtime <= cutoff], df[df.mtime > cutoff]
    cal_line = train.mtime.quantile(0.8)
    results = {}
    for name, cols in VARIANTS.items():
        tr = train[cols + ["y"]].dropna()
        te = test[cols + ["y"]].dropna()
        if len(te) < 100:
            results[name] = {"error": f"tr{len(tr)} te{len(te)}"}
            continue
        fit_part = tr[tr.index.isin(train[train.mtime <= cal_line].index)]
        cal_part = tr[~tr.index.isin(fit_part.index)]
        if len(cal_part) < 100:
            fit_part, cal_part = tr, tr
        sc = StandardScaler().fit(fit_part[cols])
        clf = lm.LogisticRegression(max_iter=2000).fit(
            sc.transform(fit_part[cols]), fit_part.y)
        p_raw = clf.predict_proba(sc.transform(te[cols]))[:, 1]
        entry = {"auc": round(float(roc_auc_score(te.y, p_raw)), 4),
                 "n_test": int(len(te)),
                 "ece_raw": round(ece(te.y.values, p_raw), 4)}
        if cal_part.y.nunique() == 2:
            pl = platt_fit(clf, sc, cal_part[cols], cal_part.y)
            p_cal = platt_apply(pl, clf, sc, te[cols])
            entry["ece_platt"] = round(ece(te.y.values, p_cal), 4)
            entry["gte60_test"] = subset60(te.y.values, p_cal)
        results[name] = entry
    (OUT / "results_p1.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # ---------------- Phase 2: expanding-window OOF on ALL samples
    # run BOTH the v3 feature set and the v2 control to attribute any
    # collapse to either the new features or the OOF protocol itself
    p2 = {}
    for tag, cols in [("control_v2best", VARIANTS["control_v2best"]),
                      ("v3_full", VARIANTS["v3_full"])]:
        p2[tag] = run_oof(df, cols)
    (OUT / "results_p2.json").write_text(
        json.dumps(p2, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(p2, ensure_ascii=False, indent=2))
    return 0


def run_oof(df, cols):
    import sklearn.linear_model as lm
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    d = df[cols + ["y", "mtime", "key", "map"]].dropna().copy()
    d = d.sort_values(["mtime", "key"]).reset_index(drop=True)
    blocks = [(mt, g.index.values) for mt, g in d.groupby("mtime", sort=True)]
    oof_p = np.full(len(d), np.nan)
    for i, (mt, idx) in enumerate(blocks):
        tr_idx = np.concatenate([blocks[j][1] for j in range(i)]) \
            if i else np.array([], dtype=int)
        if len(tr_idx) < 300:
            continue
        tr = d.loc[tr_idx]
        cal_line = tr.mtime.quantile(0.8)
        fit_i = tr[tr.mtime <= cal_line].index
        cal_i = tr[tr.mtime > cal_line].index
        if len(cal_i) < 50:
            fit_i, cal_i = tr.index, tr.index
        sc = StandardScaler().fit(d.loc[fit_i, cols])
        clf = lm.LogisticRegression(max_iter=2000).fit(
            sc.transform(d.loc[fit_i, cols]), d.loc[fit_i, "y"])
        pl = platt_fit(clf, sc, d.loc[cal_i, cols], d.loc[cal_i, "y"])
        oof_p[idx] = platt_apply(pl, clf, sc, d.loc[idx, cols])
    d["p_cal"] = oof_p
    d.to_csv(OUT / f"oof_{'v3' if 'fk_gap' in cols else 'control'}.csv",
             index=False)
    ok = d.dropna(subset=["p_cal"])
    hot = ok[ok.p_cal >= 0.60]
    summ = {
        "oof_n": int(len(ok)),
        "oof_auc": round(float(roc_auc_score(ok.y, ok.p_cal)), 4),
        "gte60": subset60(hot.y.values, hot.p_cal.values),
        "gte60_maps": int(hot.key.nunique()),
        "deciles": {},
    }
    if len(hot):
        by_map = hot.groupby("key").agg(y=("y", "max"), p=("p_cal", "mean"))
        k, n = int(by_map.y.sum()), len(by_map)
        summ["gte60_permap"] = {"n_maps": n, "hit": round(k / n, 4),
                                "ci": [round(x, 3) for x in wilson(k, n)[1:]]}
        dec = pd.qcut(ok.p_cal, 10, duplicates="drop")
        cal = ok.groupby(dec, observed=True).agg(
            pred=("p_cal", "mean"), actual=("y", "mean"), n=("y", "size"))
        summ["deciles"] = cal.round(4).reset_index(drop=True).to_dict("records")
    return summ


if __name__ == "__main__":
    raise SystemExit(main())
