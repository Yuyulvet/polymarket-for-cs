"""structure_form.py —— 回合级结构画像 → 严格时间序 form → map winner。

round_events（分析师 round-level 信号）→ 每队每图聚合「结构画像」：
  - 装备档位胜率（eco / force / full 各档，6 档归 3 大类）
  - RWAFF（首杀 → 回合胜转换率）、首杀率
  - trade 率（5s 复仇）、utility damage/回合
  - 存活总 HP 均值、存活人数均值
  - 手枪局 / 手枪局后胜率、eco 击杀占比（刷 eco 折扣）

再用**严格时间序**（仅更早系列赛）做 form，预测 map winner，对比 form K/D 基线(0.62)。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config
from .market_vs_model import (add_team_form, build_map_table_with_dates, gk_auc,
                              prior_form)
from .mapwin_players import load_player_features

ECO_TIERS = {"eco0", "eco_armor"}
FORCE_TIERS = {"force_low", "force_high"}
FULL_TIERS = {"semi", "full"}

# 结构画像特征（每队每图）
STRUCT_FEATURES = [
    "win_rate",           # 回合胜率（本图内，事后）
    "rwaff",              # 首杀 → 回合胜转换
    "first_rate",         # 首杀率
    "trade_pr",           # trade 击杀/回合
    "util_pr",            # utility damage/回合
    "avg_hp",             # 平均存活总 HP/回合
    "avg_alive",          # 平均存活人数/回合
    "eco_win",            # eco 档回合胜率
    "force_win",          # force 档回合胜率
    "full_win",           # full 档回合胜率
    "pistol_win",         # 手枪局胜率
    "pp_win",             # 手枪局后胜率
    "eco_kill_share",     # eco 击杀占比（刷 eco 折扣）
]


def load_round_events() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_DIR / "round_events.parquet")


def team_round_table(re: pd.DataFrame) -> pd.DataFrame:
    """round_events → 每回合拆 CT/T 两行的 long 表（team 视角，side-agnostic）。"""
    base = ["demo_path", "match_id", "map_name", "round_num", "round_class", "label_ct_win"]

    def _side(prefix: str, won: str):
        cols = {f"{prefix}_roster": "roster_key", f"{prefix}_equip": "equip",
                f"{prefix}_tier6": "tier6", f"{prefix}_alive": "alive", f"{prefix}_hp": "hp",
                f"{prefix}_kills": "kills", f"{prefix}_first": "first", f"{prefix}_trade": "trade",
                f"{prefix}_util": "util", f"{prefix}_eco_kills": "eco_kills"}
        s = re[base + list(cols.keys())].copy()
        s = s.rename(columns=cols)
        s["won"] = won
        return s

    ct = _side("ct", re["label_ct_win"])
    t = _side("t", 1 - re["label_ct_win"])
    tr = pd.concat([ct, t], ignore_index=True)
    # 条件特征（只在对应档位/回合类型上计数）
    tr["is_eco"] = tr["tier6"].isin(ECO_TIERS)
    tr["is_force"] = tr["tier6"].isin(FORCE_TIERS)
    tr["is_full"] = tr["tier6"].isin(FULL_TIERS)
    tr["won_eco"] = tr["won"].where(tr["is_eco"])
    tr["won_force"] = tr["won"].where(tr["is_force"])
    tr["won_full"] = tr["won"].where(tr["is_full"])
    tr["won_pistol"] = tr["won"].where(tr["round_class"] == "pistol")
    tr["won_pp"] = tr["won"].where(tr["round_class"] == "post_pistol")
    tr["won_first"] = tr["won"].where(tr["first"] == 1)
    return tr


def _dates() -> dict[int, pd.Timestamp]:
    import sqlite3
    con = sqlite3.connect(config.DB_PATH)
    dm = pd.read_sql_query("SELECT hltv_match_id, start_date FROM demos", con)
    con.close()
    dm["start_date"] = pd.to_datetime(dm["start_date"], utc=True)
    return {int(r.hltv_match_id): r.start_date for r in dm.itertuples() if pd.notna(r.start_date)}


def team_map_profile(tr: pd.DataFrame) -> pd.DataFrame:
    """long 表 → 每 (demo_path, roster_key) 一行的结构画像（含 start_date）。"""
    dates = _dates()

    def _d(dp):
        m = re.search(r"hltv-(\d+)", dp)
        return dates.get(int(m.group(1))) if m else None

    tr = tr.copy()
    tr["start_date"] = tr["demo_path"].map(_d)

    g = tr.groupby(["demo_path", "roster_key", "match_id", "start_date"])
    prof = g.agg(
        win_rate=("won", "mean"),
        rwaff=("won_first", "mean"),
        first_rate=("first", "mean"),
        trade_pr=("trade", "mean"),
        util_pr=("util", "mean"),
        avg_hp=("hp", "mean"),
        avg_alive=("alive", "mean"),
        eco_win=("won_eco", "mean"),
        force_win=("won_force", "mean"),
        full_win=("won_full", "mean"),
        pistol_win=("won_pistol", "mean"),
        pp_win=("won_pp", "mean"),
        n_rounds=("won", "size"),
        eco_kills=("eco_kills", "sum"),
        kills=("kills", "sum"),
    ).reset_index()
    prof["eco_kill_share"] = (prof["eco_kills"] / prof["kills"].clip(lower=1)).round(4)
    return prof


def structure_prior_form(prof: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """严格时间序（仅更早系列赛）form：每队每图的 prior 结构画像均值。

    返回 (demo_path, roster_key, {f}_prior) 长表。date 分组，同系列(同 date)互不进入 prior。
    """
    prof = prof.sort_values(["roster_key", "start_date"]).copy()
    rows = []
    for roster, g in prof.groupby("roster_key"):
        g = g.sort_values("start_date")
        cum = {f: 0.0 for f in feat_cols}
        cnt = 0
        for date, dg in g.groupby("start_date"):
            if pd.isna(date):
                continue
            prior = {f: (cum[f] / cnt if cnt else np.nan) for f in feat_cols}
            for _, r in dg.iterrows():
                rows.append({"demo_path": r["demo_path"], "roster_key": roster,
                             "prior_n": cnt, **{f"{f}_prior": prior[f] for f in feat_cols}})
            for _, r in dg.iterrows():
                for f in feat_cols:
                    if pd.notna(r[f]):
                        cum[f] += float(r[f])
                cnt += 1
    return pd.DataFrame(rows)


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
    re = load_round_events()
    tr = team_round_table(re)
    prof = team_map_profile(tr)
    print(f"round_events: {len(re)} 回合 / {re['demo_path'].nunique()} 图")
    print(f"结构画像：{len(prof)} 队×图 / {prof['roster_key'].nunique()} 队 / "
          f"有日期 {prof['start_date'].notna().sum()}")

    # 时间序 prior form
    pf = structure_prior_form(prof, STRUCT_FEATURES)
    # 聚成 (demo_path) 级别：home/away 各取 prior 画像
    home_p = pf.rename(columns={f"{f}_prior": f"h_{f}" for f in STRUCT_FEATURES})
    away_p = pf.rename(columns={f"{f}_prior": f"a_{f}" for f in STRUCT_FEATURES})

    mt = build_map_table_with_dates()   # 有 home/away/home_win/start_date

    # 基线：form K/D（严格时间序）
    plf = load_player_features()
    mt = add_team_form(mt, prior_form(plf, mt))

    # join 结构画像 prior 到 home / away
    demo2home = mt.set_index("demo_path")["home"].to_dict()
    demo2away = mt.set_index("demo_path")["away"].to_dict()
    home_p["roster_key"] = home_p["roster_key"]
    hp_join = home_p[["demo_path", "roster_key"] + [f"h_{f}" for f in STRUCT_FEATURES]]
    ap_join = away_p[["demo_path", "roster_key"] + [f"a_{f}" for f in STRUCT_FEATURES]]
    mt = mt.merge(hp_join.rename(columns={"roster_key": "home_roster_check"}),
                  left_on=["demo_path", "home"], right_on=["demo_path", "home_roster_check"],
                  how="left").drop(columns=["home_roster_check"])
    mt = mt.merge(ap_join.rename(columns={"roster_key": "away_roster_check"}),
                  left_on=["demo_path", "away"], right_on=["demo_path", "away_roster_check"],
                  how="left").drop(columns=["away_roster_check"])

    # 主客差
    for f in STRUCT_FEATURES:
        mt[f"{f}_gap"] = mt[f"h_{f}"] - mt[f"a_{f}"]

    print("\n" + "=" * 72)
    print("回合级结构画像 form → map winner（GroupKFold by match，严格时间序）")
    print("=" * 72)
    print(f"{'特征':<40}{'AUC':>8}{'n':>6}")
    rows = [("form K/D 差（基线）", ["form_kd_gap"])]
    for f in STRUCT_FEATURES:
        rows.append((f"{f} 差", [f"{f}_gap"]))
    # 组合
    rows.append(("结构画像全组合", [f"{f}_gap" for f in STRUCT_FEATURES]))
    rows.append(("form K/D + 结构画像", ["form_kd_gap"] + [f"{f}_gap" for f in STRUCT_FEATURES]))
    for name, feats in rows:
        a, n = _auc(mt, feats, "home_win")
        print(f"  {name:<40}{a:>8.4f}{n:>6}")

    # 单特征 top 之外，看每个结构特征的单独增益（叠加到 form_kd_gap 上）
    print("\n" + "=" * 72)
    print("叠加增益：form K/D 基线 + 单个结构特征差")
    print("=" * 72)
    a0, n0 = _auc(mt, ["form_kd_gap"], "home_win")
    print(f"  {'form K/D 单独':<40}{a0:>8.4f}{n0:>6}")
    for f in STRUCT_FEATURES:
        a, n = _auc(mt, ["form_kd_gap", f"{f}_gap"], "home_win")
        print(f"  {'+ ' + f:<40}{a:>8.4f}{n:>6}  (Δ{a - a0:+.4f})")

    return {"n_maps": len(mt)}


if __name__ == "__main__":
    evaluate()
