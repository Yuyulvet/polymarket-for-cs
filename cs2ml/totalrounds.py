"""Total Rounds O/U 21.5（碾压 vs 胶着）预测 + per-team 细粒度特征。

用户方向：裸比分已被证无用（AUC≈0.5），所以把每队特征单独拆出来，细粒度对比——
队伍强度差、风格(场均回合)、碾压度(场均分差)、边胜率、手枪局胜率，看哪个能抬起来。
全部单图(demo_path)粒度；per-team 特征 LOO by match_id 防泄漏。
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from .round_model import build_round_dataset


def _mapno(p: str) -> int:
    m = re.search(r'-m(\d+)-', p) or re.search(r'map(\d+)', p, re.I)
    return int(m.group(1)) if m else 0


def build_map_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dp, sub in df.groupby("demo_path"):
        sub = sub.sort_values("round_num")
        won = sub[sub["winner_roster"].astype(str) != ""]
        if not len(won):
            continue
        home = sub.iloc[0]["ct_roster"]; away = sub.iloc[0]["t_roster"]
        if not home or not away:
            continue
        vc = won["winner_roster"].value_counts()
        if not len(vc) or vc.index[0] not in (home, away):
            continue
        total = len(won)
        hw = int((won["winner_roster"] == home).sum())
        aw = int((won["winner_roster"] == away).sum())
        rows.append({
            "match_id": sub.iloc[0]["match_id"], "demo_path": dp,
            "map_no": _mapno(dp), "map_name": sub.iloc[0]["map_name"],
            "home": home, "away": away,
            "home_win": int(vc.index[0] == home),
            "total": total, "hw": hw, "aw": aw,
        })
    return pd.DataFrame(rows)


def per_team_features(map_tab: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """每 roster 一组 LOO 特征：win_rate / avg_rounds / avg_margin / ct_rate / t_rate / pistol_rate。

    LOO：用除当前 match_id(系列赛) 外的所有图算。min_n：历史图数 < min_n 时置 NaN。
    """
    # 图级 long 表（每图拆 home/away 两行）
    ml = []
    for _, r in map_tab.iterrows():
        ml.append({"match_id": r["match_id"], "roster": r["home"],
                   "won": int(r["hw"] > r["aw"]), "rw": r["hw"], "total": r["total"]})
        ml.append({"match_id": r["match_id"], "roster": r["away"],
                   "won": int(r["aw"] > r["hw"]), "rw": r["aw"], "total": r["total"]})
    ml = pd.DataFrame(ml)

    # 回合级 long 表（每回合拆 ct/t 两行，用于 ct_rate/t_rate/pistol_rate）
    df_ = build_round_dataset()
    rl = []
    for _, r in df_.iterrows():
        w = r["winner_roster"]
        is_p = r["round_class"] == "pistol"
        rl.append({"match_id": r["match_id"], "roster": r["ct_roster"],
                   "side": "ct", "won_r": int(w == r["ct_roster"]), "pistol": is_p})
        rl.append({"match_id": r["match_id"], "roster": r["t_roster"],
                   "side": "t", "won_r": int(w == r["t_roster"]), "pistol": is_p})
    rl = pd.DataFrame(rl)

    # 图级聚合
    g_ml = (ml.groupby("roster").agg(n=("won", "size"), wins=("won", "sum"),
                                     rw=("rw", "sum"), tr=("total", "sum")).reset_index())
    m_ml = (ml.groupby(["roster", "match_id"]).agg(n=("won", "size"), wins=("won", "sum"),
                                                   rw=("rw", "sum"), tr=("total", "sum")).reset_index())
    gmap = {r.roster: r for r in g_ml.itertuples(index=False)}
    mmap = {(r.roster, r.match_id): r for r in m_ml.itertuples(index=False)}

    # 回合级聚合
    g_rl = (rl.groupby(["roster", "side"]).agg(wn=("won_r", "sum"), n=("won_r", "size")).reset_index())
    m_rl = (rl.groupby(["roster", "match_id", "side"]).agg(wn=("won_r", "sum"), n=("won_r", "size")).reset_index())
    gr = {(r.roster, r.side): r for r in g_rl.itertuples(index=False)}
    mr = {(r.roster, r.match_id, r.side): r for r in m_rl.itertuples(index=False)}

    g_p = (rl[rl["pistol"]].groupby("roster").agg(wn=("won_r", "sum"), n=("won_r", "size")).reset_index())
    m_p = (rl[rl["pistol"]].groupby(["roster", "match_id"]).agg(wn=("won_r", "sum"), n=("won_r", "size")).reset_index())
    gp = {r.roster: r for r in g_p.itertuples(index=False)}
    mp = {(r.roster, r.match_id): r for r in m_p.itertuples(index=False)}

    def feat(roster, mid):
        out = {}
        # 图级
        g = gmap.get(roster); mm = mmap.get((roster, mid))
        n = int(g.n) if g is not None else 0
        n_m = int(mm.n) if mm is not None else 0
        n_loo = n - n_m
        if n_loo < min_n:
            for k in ("win_rate", "avg_rounds", "avg_margin"):
                out[k] = float("nan")
        else:
            w = int(g.wins) - int(mm.wins if mm is not None else 0)
            rw = int(g.rw) - int(mm.rw if mm is not None else 0)
            tr = int(g.tr) - int(mm.tr if mm is not None else 0)
            out["win_rate"] = w / n_loo
            out["avg_rounds"] = tr / n_loo
            out["avg_margin"] = (rw - (tr - rw)) / n_loo
        # 回合级 ct/t
        for side in ("ct", "t"):
            gr_ = gr.get((roster, side)); mr_ = mr.get((roster, mid, side))
            n = int(gr_.n) if gr_ is not None else 0
            n_m = int(mr_.n) if mr_ is not None else 0
            n_loo = n - n_m
            if n_loo < min_n:
                out[f"{side}_rate"] = float("nan")
            else:
                wn = int(gr_.wn) - int(mr_.wn if mr_ is not None else 0)
                out[f"{side}_rate"] = wn / n_loo
        # 手枪局
        gp_ = gp.get(roster); mp_ = mp.get((roster, mid))
        n = int(gp_.n) if gp_ is not None else 0
        n_m = int(mp_.n) if mp_ is not None else 0
        n_loo = n - n_m
        if n_loo < min_n:
            out["pistol_rate"] = float("nan")
        else:
            wn = int(gp_.wn) - int(mp_.wn if mp_ is not None else 0)
            out["pistol_rate"] = wn / n_loo
        return out

    # 应用到每张图
    for prefix, rcol in (("home", "home"), ("away", "away")):
        feats = [feat(r, mid) for r, mid in zip(map_tab[rcol], map_tab["match_id"])]
        for k in ("win_rate", "avg_rounds", "avg_margin", "ct_rate", "t_rate", "pistol_rate"):
            map_tab[f"{prefix}_{k}"] = [f[k] for f in feats]
    return map_tab


def _auc(m: pd.DataFrame, feats: list[str], target: str) -> tuple[float, int]:
    d = m.dropna(subset=feats + [target])
    if len(d) < 40 or d[target].nunique() < 2:
        return float("nan"), len(d)
    X = d[feats].to_numpy(); y = d[target].to_numpy()
    n = min(5, d["match_id"].nunique())
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                          groups=d["match_id"].to_numpy(), cv=GroupKFold(n),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p), len(d)


def evaluate() -> dict:
    df = build_round_dataset()
    mt = build_map_table(df)
    mt = mt[(mt["total"] >= 13) & (mt["total"] <= 24)].copy()
    print(f"有效单图(MR12): {len(mt)}")

    # 边调整：home 首半场=CT、下半场=T。同时算 as-of 早期比分(前6回合)
    home_map = mt.set_index("demo_path")["home"].to_dict()
    side_snaps = {}
    early6 = {}
    for dp, sub in df.groupby("demo_path"):
        sub = sub.sort_values("round_num")
        won = sub[sub["winner_roster"].astype(str) != ""]
        h = home_map.get(dp)
        if not h:
            continue
        ct_win = int(((won["winner_roster"] == h) & (won["round_num"] <= 12)).sum())
        t_win = int(((won["winner_roster"] == h) & (won["round_num"] > 12)).sum())
        side_snaps[dp] = (ct_win, t_win)
        hs = int(((won["winner_roster"] == h) & (won["round_num"] <= 6)).sum())
        aw_ = int(((won["winner_roster"] != h) & (won["round_num"] <= 6)).sum())
        early6[dp] = hs - aw_
    mt["home_ct_half"] = [side_snaps.get(dp, (0, 0))[0] for dp in mt["demo_path"]]
    mt["home_t_half"] = [side_snaps.get(dp, (0, 0))[1] for dp in mt["demo_path"]]
    mt["score_after_6"] = [early6.get(dp, 0) for dp in mt["demo_path"]]

    mt["over21_5"] = (mt["total"] > 21.5).astype(int)
    print(f"over21.5(胶着) 基准率 = {mt['over21_5'].mean():.3f}")

    mt = per_team_features(mt)

    # ---- 每队特征表（单独列出，取历史图数最多的队伍）----
    print("\n" + "=" * 70)
    print("每队特征（LOO，历史图数 top 20）")
    print("=" * 70)
    cnt = mt.groupby("home")["demo_path"].nunique()
    top = cnt.sort_values(ascending=False).head(20).index
    print(f"{'队伍(steamid前8位)':<14}{'图数':>5}{'胜率':>7}{'场均回合':>9}{'场均分差':>9}{'CT率':>7}{'T率':>7}{'手枪率':>7}")
    seen = set()
    for r in top:
        if r in seen:
            continue
        seen.add(r)
        s = mt[mt["home"] == r].iloc[0]
        print(f"{str(r)[:12]:<14}{cnt[r]:>5}{s['home_win_rate']:>7.3f}{s['home_avg_rounds']:>9.2f}"
              f"{s['home_avg_margin']:>9.2f}{s['home_ct_rate']:>7.3f}{s['home_t_rate']:>7.3f}{s['home_pistol_rate']:>7.3f}")

    # ---- 细粒度对比 ----
    mt["str_gap"] = mt["home_win_rate"] - mt["away_win_rate"]
    mt["abs_str_gap"] = mt["str_gap"].abs()
    mt["style_sum"] = mt["home_avg_rounds"] + mt["away_avg_rounds"]
    mt["style_diff"] = mt["home_avg_rounds"] - mt["away_avg_rounds"]
    mt["margin_gap"] = mt["home_avg_margin"] - mt["away_avg_margin"]
    mt["ct_gap"] = mt["home_ct_rate"] - mt["away_ct_rate"]
    mt["pistol_gap"] = mt["home_pistol_rate"] - mt["away_pistol_rate"]

    print("\n" + "=" * 70)
    print("细粒度对比：Total Rounds O/U 21.5（胶着 vs 碾压）预测 AUC")
    print("=" * 70)
    print(f"{'特征':<28}{'AUC':>8}{'n':>6}")
    groups = {
        "early比分 after6": ["score_after_6"],
        "强度差 abs_str_gap": ["abs_str_gap"],
        "强度差(有符号)": ["str_gap"],
        "风格 场均回合和 style_sum": ["style_sum"],
        "风格 场均回合差": ["style_diff"],
        "碾压度差 margin_gap": ["margin_gap"],
        "边胜率差 ct_gap": ["ct_gap"],
        "手枪率差 pistol_gap": ["pistol_gap"],
        "强度差+风格": ["abs_str_gap", "style_sum"],
        "强度差+风格+碾压度": ["abs_str_gap", "style_sum", "margin_gap"],
    }
    for name, feats in groups.items():
        a, n = _auc(mt, feats, "over21_5")
        print(f"  {name:<28}{a:>8.4f}{n:>6}")

    # 方向诊断：强度差越大 → 碾压(under) 还是 胶着(over)？
    print("\n" + "=" * 70)
    print("方向诊断：|强度差| 分桶 → over21.5(胶着) 率")
    print("=" * 70)
    d = mt.dropna(subset=["abs_str_gap", "over21_5"])
    d["bucket"] = pd.qcut(d["abs_str_gap"], 4, labels=["Q1小", "Q2", "Q3", "Q4大"], duplicates="drop")
    for b, s in d.groupby("bucket", observed=True):
        print(f"  {b:<6} n={len(s):>4}  胶着率(over) = {s['over21_5'].mean():.3f}")

    print("\n" + "=" * 70)
    print("同组特征对 map winner 的预测 AUC（对照）")
    print("=" * 70)
    print(f"{'特征':<28}{'AUC':>8}{'n':>6}")
    for name, feats in [("强度差 abs_str_gap", ["abs_str_gap"]), ("强度差(有符号)", ["str_gap"]),
                        ("边胜率差 ct_gap", ["ct_gap"]), ("碾压度差 margin_gap", ["margin_gap"]),
                        ("强度差+边+碾压", ["str_gap", "ct_gap", "margin_gap", "pistol_gap"])]:
        a, n = _auc(mt, feats, "home_win")
        print(f"  {name:<28}{a:>8.4f}{n:>6}")

    return {"n_maps": len(mt), "over_rate": mt["over21_5"].mean()}


if __name__ == "__main__":
    evaluate()
