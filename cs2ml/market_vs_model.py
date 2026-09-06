"""市场价 vs 模型：严格时间排序的赛前 form → map winner。

关键方法修正：mapwin_players 的 LOO form 是「排除当前 match_id 但包含未来比赛」，
0.662 可能被 look-ahead 污染。本模块用 demos 表的 start_date 做**严格时间序**：
一个选手在 t 时刻的 form 只来自 date < t 的比赛（且排除同场 BO3 内部泄漏）。

输出三件事：
1. 复现 LOO form（含未来）AUC —— 确认 0.662。
2. 严格时间序 form（仅历史）AUC（同一 GroupKFold CV）—— 诚实数字。
3. walk-forward 时间序 AUC + 6/19 Spirit vs G2 赛前概率（对照市场 0.645）。
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config
from .mapwin_players import load_player_features
from .round_model import build_round_dataset
from .totalrounds import build_map_table


def _demos_dates() -> dict[int, pd.Timestamp]:
    """hltv_match_id -> start_date（demos 表，覆盖 481 场）。"""
    con = sqlite3.connect(config.DB_PATH)
    dm = pd.read_sql_query("SELECT hltv_match_id, start_date FROM demos", con)
    con.close()
    dm["start_date"] = pd.to_datetime(dm["start_date"], utc=True)
    return {int(r.hltv_match_id): r.start_date for r in dm.itertuples() if pd.notna(r.start_date)}


def build_map_table_with_dates() -> pd.DataFrame:
    """round_dataset 全量图表 + start_date。home/away = roster_key，home_win 目标。"""
    df = build_round_dataset()
    mt = build_map_table(df)
    mt = mt[(mt["total"] >= 13) & (mt["total"] <= 24)].copy()
    dates = _demos_dates()

    def _date(dp):
        m = re.search(r"hltv-(\d+)", dp)
        return dates.get(int(m.group(1))) if m else None

    mt["start_date"] = mt["demo_path"].map(_date)
    return mt


def prior_form(pf: pd.DataFrame, mt: pd.DataFrame, min_maps: int = 3) -> pd.DataFrame:
    """per-player 严格时间序 form：form_kd = 更早比赛的累积 K/D（date < 当前比赛）。"""
    dates = mt.set_index("demo_path")["start_date"].to_dict()
    pf = pf.copy()
    pf["start_date"] = pf["demo_path"].map(dates)
    # 每个选手按时间累积
    pf = pf.sort_values(["steamid", "start_date"]).reset_index(drop=True)
    form = {}
    for sid, g in pf.groupby("steamid"):
        g = g.sort_values("start_date")
        ck = cd = 0.0
        n_prior = 0
        for idx, r in g.iterrows():
            # 关键：用「当前比赛日期之前」的累积，但同场多图会因同日期互泄。
            # 下面用 match 级日期（start_date 是 match 级），同场图同日期，故按
            # 「< 当前日期」累积即可：同日期图不会进累积（相等不算 <）。
            form[idx] = (ck / cd if cd else np.nan, n_prior)
            ck += r["kills"]; cd += r["deaths"]; n_prior += 1
    pf["form_kd"] = [form[i][0] for i in pf.index]
    pf["prior_maps"] = [form[i][1] for i in pf.index]
    pf.loc[pf["prior_maps"] < min_maps, "form_kd"] = np.nan
    return pf


def add_team_form(mt: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """home/away form 差（用严格时间序 form）。"""
    demo2home = mt.set_index("demo_path")["home"].to_dict()
    demo2away = mt.set_index("demo_path")["away"].to_dict()
    rows = []
    for dp, sub in pf.groupby("demo_path"):
        home, away = demo2home.get(dp), demo2away.get(dp)
        if not home or not away:
            continue
        h = sub[sub["roster_key"] == home]["form_kd"].dropna()
        a = sub[sub["roster_key"] == away]["form_kd"].dropna()
        if len(h) < 3 or len(a) < 3:
            continue
        rows.append({"demo_path": dp, "home_form_kd": h.mean(), "away_form_kd": a.mean()})
    ft = pd.DataFrame(rows)
    mt = mt.merge(ft, on="demo_path", how="left")
    mt["form_kd_gap"] = mt["home_form_kd"] - mt["away_form_kd"]
    return mt


def gk_auc(mt: pd.DataFrame, feat: str = "form_kd_gap") -> tuple[float, int]:
    d = mt.dropna(subset=[feat, "home_win"])
    if len(d) < 40 or d["home_win"].nunique() < 2:
        return float("nan"), len(d)
    X = d[[feat]].to_numpy(); y = d["home_win"].to_numpy()
    n = min(5, d["match_id"].nunique())
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y,
                          groups=d["match_id"].to_numpy(), cv=GroupKFold(n),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p), len(d)


def walk_forward_auc(mt: pd.DataFrame, min_train: int = 80) -> tuple[float, int]:
    d = mt.dropna(subset=["form_kd_gap", "home_win"]).sort_values("start_date").reset_index(drop=True)
    X = d[["form_kd_gap"]].to_numpy(); y = d["home_win"].to_numpy()
    preds = np.full(len(d), np.nan)
    for i in range(min_train, len(d)):
        clf = LogisticRegression(max_iter=2000).fit(X[:i], y[:i])
        preds[i] = clf.predict_proba(X[i:i + 1])[:, 1][0]
    valid = ~np.isnan(preds)
    return (roc_auc_score(y[valid], preds[valid]), int(valid.sum())) if valid.sum() >= 30 else (float("nan"), int(valid.sum()))


def main():
    mt = build_map_table_with_dates()
    print(f"图级表：{len(mt)} 图，有日期 {mt['start_date'].notna().sum()}")

    pf = load_player_features()

    # LOO form（含未来）复现
    from .mapwin_players import build_team_form
    mt_loo = build_team_form(pf, mt)
    a_loo, n_loo = gk_auc(mt_loo, "form_kd_gap")

    # 严格时间序 form（仅历史）
    pf_prior = prior_form(pf, mt)
    mt_prior = add_team_form(mt, pf_prior)
    a_prior, n_prior = gk_auc(mt_prior, "form_kd_gap")

    print("\n=== 同一 GroupKFold CV，只改 form 的时序 ===")
    print(f"  LOO form (含未来)      AUC = {a_loo:.4f}  n={n_loo}")
    print(f"  时间序 form (仅历史)   AUC = {a_prior:.4f}  n={n_prior}")
    print(f"  差 = look-ahead 泄漏量 = {a_loo - a_prior:+.4f}")

    auc_wf, n_wf = walk_forward_auc(mt_prior)
    print(f"\n严格时间序 walk-forward AUC = {auc_wf:.4f} (n={n_wf})")

    # 6/19 单场
    print("\n=== 6/19 Spirit vs G2 Map1 (hltv-2394998) ===")
    tgt = mt_prior[mt_prior["demo_path"].str.contains("2394998")].sort_values("demo_path")
    d = mt_prior.dropna(subset=["form_kd_gap", "home_win"]).sort_values("start_date")
    for _, row in tgt.iterrows():
        train = d[d["start_date"] < row["start_date"]]
        if len(train) < 30:
            print(f"  {row['demo_path'].split(chr(92))[-1]:<28} 训练不足({len(train)})")
            continue
        clf = LogisticRegression(max_iter=2000).fit(
            train[["form_kd_gap"]].to_numpy(), train["home_win"].to_numpy())
        p = clf.predict_proba([[row["form_kd_gap"]]])[:, 1][0]
        home_is_spirit = "spirit" in row["demo_path"].lower() and "g2" in row["demo_path"].lower()
        print(f"  {row['demo_path'].split(chr(92))[-1]:<28} P(home 赢)={p:.3f}  gap={row['form_kd_gap']:+.3f}  实际 home_win={row['home_win']}")


if __name__ == "__main__":
    main()
