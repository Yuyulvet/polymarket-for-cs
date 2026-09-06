"""per-player 特征 → map winner：选手「当前竞技状态」重学 map winner。

用户方向：从 roster 级下钻到 player 级，用选手的局内 K/D、伤害量(ADR) 作为
「当前竞技状态」代理，叠加队伍决策信号（首杀/闪光助攻/交易击杀），重测 map winner。

关键纪律（look-ahead）：
  - per-player 局内 K/D/ADR 是**赛后**才知道的，不能用来预测本图结果。
  - 所以赛前模型只能用**跨图 LOO form**（该选手在其他图的平均 K/D、ADR），
    这才是在开赛前就能拿到的「当前竞技状态」。
  - 本图的 per-player 数据只作为「谁打得好」的事后归因，与比分一起看会不会被吸收。

基线（上一轮已测）：roster 级强度 margin_gap(场均分差) 赛前 → map winner AUC 0.60。
本模块回答：per-player form 能否把 0.60 往上抬。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config
from .round_model import build_round_dataset
from .totalrounds import build_map_table, per_team_features


def load_player_features() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_DIR / "player_features.parquet")


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


def build_team_form(pf: pd.DataFrame, mt: pd.DataFrame, min_maps: int = 3) -> pd.DataFrame:
    """把 per-player 特征聚成 map 级「队伍 form」特征（LOO 跨图）。

    pf: player_features（demo_path, steamid, roster_key, kills, deaths, damage, rounds_played...）
    mt: map 表（demo_path, match_id, home, away, home_win, ...）
    返回 mt 增列：home/away 的 form_kd_avg/star_adr 等。
    """
    # 每 demo 的 match_id + 主客 roster
    demo2match = mt.set_index("demo_path")["match_id"].to_dict()
    demo2home = mt.set_index("demo_path")["home"].to_dict()
    demo2away = mt.set_index("demo_path")["away"].to_dict()

    pf = pf.copy()
    pf["match_id"] = pf["demo_path"].map(demo2match)
    pf = pf.dropna(subset=["match_id"])

    # 每选手全局 vs 每 (选手,match_id)：做 LOO
    g = pf.groupby("steamid").agg(n=("demo_path", "nunique"), kills=("kills", "sum"),
                                  deaths=("deaths", "sum"), dmg=("damage", "sum"),
                                  rounds=("rounds_played", "sum"))
    gm = pf.groupby(["steamid", "match_id"]).agg(n=("demo_path", "nunique"), kills=("kills", "sum"),
                                                 deaths=("deaths", "sum"), dmg=("damage", "sum"),
                                                 rounds=("rounds_played", "sum"))
    gd = {idx: row for idx, row in g.iterrows()}
    gmd = {(idx[0], idx[1]): row for idx, row in gm.iterrows()}

    def form(sid, mid):
        gv = gd.get(sid)
        if gv is None:
            return None
        mv = gmd.get((sid, mid))
        n = int(gv.n) - (int(mv.n) if mv is not None else 0)
        if n < min_maps:
            return None
        k = int(gv.kills) - (int(mv.kills) if mv is not None else 0)
        d = int(gv.deaths) - (int(mv.deaths) if mv is not None else 0)
        dmg = float(gv.dmg) - (float(mv.dmg) if mv is not None else 0)
        rnd = int(gv.rounds) - (int(mv.rounds) if mv is not None else 0)
        return {"kd": k / d if d else 0.0, "adr": dmg / rnd if rnd else 0.0, "n": n}

    # 每 map：home/away 各 5 人的 form
    rows = []
    for dp, sub in pf.groupby("demo_path"):
        home = demo2home.get(dp); away = demo2away.get(dp)
        mid = demo2match.get(dp)
        if not home or not away:
            continue
        hf = [f for f in (form(s, mid) for s in sub[sub["roster_key"] == home]["steamid"]) if f]
        af = [f for f in (form(s, mid) for s in sub[sub["roster_key"] == away]["steamid"]) if f]
        if len(hf) < 3 or len(af) < 3:
            continue
        rows.append({
            "demo_path": dp,
            "h_form_kd_avg": np.mean([f["kd"] for f in hf]),
            "h_form_adr_avg": np.mean([f["adr"] for f in hf]),
            "h_form_kd_max": max(f["kd"] for f in hf),
            "h_form_adr_max": max(f["adr"] for f in hf),
            "a_form_kd_avg": np.mean([f["kd"] for f in af]),
            "a_form_adr_avg": np.mean([f["adr"] for f in af]),
            "a_form_kd_max": max(f["kd"] for f in af),
            "a_form_adr_max": max(f["adr"] for f in af),
        })
    form_tab = pd.DataFrame(rows)
    mt = mt.merge(form_tab, on="demo_path", how="left")

    # 主客差分特征
    mt["form_kd_gap"] = mt["h_form_kd_avg"] - mt["a_form_kd_avg"]
    mt["form_adr_gap"] = mt["h_form_adr_avg"] - mt["a_form_adr_avg"]
    mt["form_star_adr_gap"] = mt["h_form_adr_max"] - mt["a_form_adr_max"]
    mt["form_kdmax_gap"] = mt["h_form_kd_max"] - mt["a_form_kd_max"]
    return mt


def evaluate() -> dict:
    df = build_round_dataset()
    mt = build_map_table(df)
    mt = mt[(mt["total"] >= 13) & (mt["total"] <= 24)].copy()

    # 基线：roster 强度（复用 totalrounds 的 per_team_features）
    mt = per_team_features(mt)
    mt["margin_gap"] = mt["home_avg_margin"] - mt["away_avg_margin"]
    mt["str_gap"] = mt["home_win_rate"] - mt["away_win_rate"]

    # per-player form
    pf = load_player_features()
    print(f"player_features: {len(pf)} 选手×地图 / {pf['demo_path'].nunique()} 图 / "
          f"{pf['steamid'].nunique()} 名选手")
    mt = build_team_form(pf, mt)
    n_form = mt.dropna(subset=["form_kd_gap"]).shape[0]
    print(f"赛前 form 覆盖：{n_form}/{len(mt)} 图（双方各有 >=3 名选手有 >=3 图历史）\n")

    print("=" * 70)
    print("赛前 map winner 预测 AUC（GroupKFold by match_id）")
    print("=" * 70)
    print(f"{'特征':<34}{'AUC':>8}{'n':>6}")
    groups = {
        "roster 强度 margin_gap(基线)": ["margin_gap"],
        "roster 强度 str_gap": ["str_gap"],
        "form K/D 差 form_kd_gap": ["form_kd_gap"],
        "form ADR 差 form_adr_gap": ["form_adr_gap"],
        "form 星位ADR 差 star_adr_gap": ["form_star_adr_gap"],
        "form K/D 差+ADR 差": ["form_kd_gap", "form_adr_gap"],
        "roster强度+form": ["margin_gap", "form_kd_gap", "form_adr_gap"],
        "全 form 组合": ["form_kd_gap", "form_adr_gap", "form_star_adr_gap", "form_kdmax_gap"],
    }
    for name, feats in groups.items():
        a, n = _auc(mt, feats, "home_win")
        print(f"  {name:<34}{a:>8.4f}{n:>6}")

    # 事后归因：本图 per-player 打得怎么样 vs 结果（纯诊断，非预测）
    print("\n" + "=" * 70)
    print("事后归因（本图局内 per-player 数据 vs 结果，诊断用）")
    print("=" * 70)
    demo2home = mt.set_index("demo_path")["home"].to_dict()
    demo2win = mt.set_index("demo_path")["home_win"].to_dict()
    pf2 = pf.copy()
    pf2["home_roster"] = pf2["demo_path"].map(demo2home)
    pf2["is_home"] = pf2["roster_key"] == pf2["home_roster"]
    pf2["home_win"] = pf2["demo_path"].map(demo2win)
    # 赢方 vs 输方的本图平均 ADR / K/D
    def _agg(sub):
        w = sub[sub["is_home"] == (sub["home_win"] == 1)]
        l = sub[sub["is_home"] == (sub["home_win"] == 0)]
        return pd.Series({"winner_adr": w["adr"].mean(), "loser_adr": l["adr"].mean(),
                          "winner_kd": w["kd"].mean(), "loser_kd": l["kd"].mean()})
    agg = pf2.groupby("demo_path").apply(_agg, include_groups=False)
    print(f"  赢方 avg ADR = {agg['winner_adr'].mean():.1f}   vs 输方 = {agg['loser_adr'].mean():.1f}")
    print(f"  赢方 avg K/D = {agg['winner_kd'].mean():.2f}   vs 输方 = {agg['loser_kd'].mean():.2f}")

    return {"n_form": n_form}


if __name__ == "__main__":
    evaluate()
