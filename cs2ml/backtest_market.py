"""系统性回测：时间序 form 模型 vs Polymarket Map Winner 赛前价。

问题：当我的模型概率与市场价分歧 > 阈值时下注，长期是否正收益？

数据流：
  demo 侧  —— round_dataset 图 + demos 表 start_date + team1/team2 名，walk-forward 出 P(home 赢)
  市场侧  —— Gamma 抓已结算 CS2 事件，按队名+日期匹配 Map N Winner，取赛前价
  对齐    —— home = team1 = team_number 2（已验证）
  结算    —— won_map 判定结果，按「分歧>阈值」策略回测 P&L
"""
from __future__ import annotations

import json
import re
import sqlite3
import time

import numpy as np
import pandas as pd
from curl_cffi import requests as creq
from sklearn.linear_model import LogisticRegression

from . import config
from .market_vs_model import build_map_table_with_dates, add_team_form, prior_form
from .mapwin_players import load_player_features
from .team_identity import align_home_win_to_team1

MIN_TRAIN = 80  # walk-forward 最少训练图数
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


# ---------------------------------------------------------------- demo 侧
def build_demo_maps() -> pd.DataFrame:
    """per-map：demo_path, match_id(hltv), start_date, home/away 名, home_win, model_p_home。"""
    mt = build_map_table_with_dates()
    pf = load_player_features()
    pf = prior_form(pf, mt)
    mt = add_team_form(mt, pf)

    # --- 修正队伍身份：home_win / form_kd_gap 从「上半场 CT」视角 → team1 视角 ---
    # 根因见 cs2ml/team_identity.py / [[cs2-homewin-ct-alignment-bug]]
    mt = align_home_win_to_team1(mt)  # home_win 改为 team1 赢；无 raw 的 match 被 drop
    mt["form_kd_gap"] = np.where(mt["team1_is_home"], mt["form_kd_gap"], -mt["form_kd_gap"])

    # 队名：demos 表 hltv_match_id -> team1/team2
    con = sqlite3.connect(config.DB_PATH)
    dm = pd.read_sql_query("SELECT hltv_match_id, team1, team2 FROM demos", con)
    con.close()
    dm["hltv_match_id"] = dm["hltv_match_id"].astype(int)

    def hid(dp):
        m = re.search(r"hltv-(\d+)", dp)
        return int(m.group(1)) if m else None

    mt["hltv_match_id"] = mt["demo_path"].map(hid)
    mt = mt.merge(dm, on="hltv_match_id", how="left")
    mt = mt.dropna(subset=["team1", "team2"])

    # walk-forward 出 P(team1 赢)
    d = mt.dropna(subset=["form_kd_gap", "home_win"]).sort_values("start_date").reset_index(drop=True)
    X = d[["form_kd_gap"]].to_numpy(); y = d["home_win"].to_numpy()
    probs = np.full(len(d), np.nan)
    for i in range(MIN_TRAIN, len(d)):
        clf = LogisticRegression(max_iter=2000).fit(X[:i], y[:i])
        probs[i] = clf.predict_proba(X[i:i + 1])[:, 1][0]
    d["model_p_home"] = probs
    return d[["demo_path", "match_id", "hltv_match_id", "start_date", "map_name",
              "team1", "team2", "home_win", "model_p_home"]]


def _norm(s: str) -> str:
    """队名归一：小写、去 team/esports/the/空格/符号。"""
    s = str(s).lower()
    s = re.sub(r"\bteam\b|\besports\b|\bthe\b|\bgaming\b|\bclub\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def team_match(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    return a == b or a in b or b in a


# ---------------------------------------------------------------- 市场侧
def _gamma(path, **params):
    r = creq.get(f"{GAMMA}{path}", params=params, impersonate="chrome", timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_finished_events(max_events: int = 1200) -> list[dict]:
    evs, off = [], 0
    while len(evs) < max_events:
        batch = _gamma("/events", tag_slug="counter-strike-2", closed="true",
                       limit=50, offset=off, order="volume", ascending="false")
        if not batch:
            break
        evs += batch
        if len(batch) < 50:
            break
        off += 50
        time.sleep(0.2)
    return evs


def parse_event(ev: dict) -> dict | None:
    """事件 -> {team_a, team_b, match_date, markets:{'Map N Winner': {out->token}}}。

    match_date 取 slug 末尾日期（比赛日），不是 startDate（那是市场开盘日）。
    """
    title = ev.get("title") or ""
    m = re.match(r"^Counter-Strike:\s*(.+?)\s+vs\s+(.+?)\s*\(", title)
    if not m:
        return None
    team_a, team_b = m.group(1).strip(), m.group(2).strip()
    # 比赛日期：优先 slug 末尾 2026-06-19；否则 endDate
    md = None
    slug = ev.get("slug") or ""
    sm = re.search(r"(\d{4}-\d{2}-\d{2})$", slug)
    if sm:
        md = pd.to_datetime(sm.group(1), utc=True)
    else:
        ed = ev.get("endDate")
        if ed:
            md = pd.to_datetime(ed, utc=True)
    out = {"team_a": team_a, "team_b": team_b, "match_date": md,
           "markets": {}}
    for mk in ev.get("markets", []):
        q = mk.get("question") or ""
        mm = re.match(r".*Map (\d) Winner$", q)
        if mm and mk.get("sportsMarketType") == "child_moneyline":
            outs = mk.get("outcomes")
            toks = mk.get("clobTokenIds")
            if isinstance(outs, str):
                outs = json.loads(outs)
            if isinstance(toks, str):
                toks = json.loads(toks)
            if outs and toks and len(outs) == len(toks):
                out["markets"][f"Map {mm.group(1)} Winner"] = dict(zip(outs, toks))
    return out if out["markets"] and md is not None else None


def _clob_prices(token: str, start_ts: int, end_ts: int, fidelity: int = 1):
    r = creq.get(f"{CLOB}/prices-history",
                 params={"market": token, "startTs": start_ts, "endTs": end_ts,
                         "fidelity": fidelity}, impersonate="chrome", timeout=60)
    r.raise_for_status()
    return r.json().get("history", [])


def pre_match_price(token: str, match_start: pd.Timestamp) -> float | None:
    """取 match_start 之前最后一个成交价（closing line）。"""
    start = int((match_start - pd.Timedelta(hours=6)).timestamp())
    end = int((match_start + pd.Timedelta(minutes=5)).timestamp())
    try:
        hist = _clob_prices(token, start, end, fidelity=1)
    except Exception:
        return None
    pre = [p for p in hist if p["t"] <= int(match_start.timestamp())]
    if not pre:
        return None
    return pre[-1]["p"]


def match_demo_to_market(dmaps: pd.DataFrame, evs: list[dict]) -> pd.DataFrame:
    """把 demo 比赛（hltv_match_id 唯一）匹配到 Polymarket 事件，返回带市场价的表。"""
    parsed = []
    for ev in evs:
        p = parse_event(ev)
        if p:
            parsed.append(p)

    # 每场比赛一张行（用 Map1 判断），队名 + 日期
    matches = dmaps.drop_duplicates("match_id")[["match_id", "hltv_match_id", "start_date",
                                                  "team1", "team2"]].copy()
    rows = []
    for _, mrow in matches.iterrows():
        for p in parsed:
            a, b = p["team_a"], p["team_b"]
            t1, t2 = mrow["team1"], mrow["team2"]
            # 队名配对（无序）
            ok = (team_match(a, t1) and team_match(b, t2)) or \
                 (team_match(a, t2) and team_match(b, t1))
            if not ok:
                continue
            # 日期接近（slug 比赛日 vs demo 比赛日，±1.5 天）
            ev_dt = p["match_date"]
            if ev_dt is None or abs((ev_dt - mrow["start_date"]).total_seconds()) > 36 * 3600:
                continue
            # 我的 home=team1 对应市场的哪个 out
            if team_match(a, t1):
                rows.append({"match_id": mrow["match_id"], "team1": t1, "team2": t2,
                             "start_date": mrow["start_date"], "mkt_team1": a,
                             "mkt_team2": b, "markets": p["markets"]})
            else:
                rows.append({"match_id": mrow["match_id"], "team1": t1, "team2": t2,
                             "start_date": mrow["start_date"], "mkt_team1": b,
                             "mkt_team2": a, "markets": p["markets"]})
            break
    return pd.DataFrame(rows)


def backtest(dmaps: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """逐场 Map1：模型 P(home) vs 市场 closing line，取价回测。"""
    m1 = dmaps[dmaps["demo_path"].str.contains(r"-m1-", regex=True)].copy()
    res = []
    for _, mm in matched.iterrows():
        if "Map 1 Winner" not in mm["markets"]:
            continue
        # 市场 closing line：mkt_team1（=我的 team1）的 token
        toks = mm["markets"]["Map 1 Winner"]
        token1 = toks.get(mm["mkt_team1"])
        if token1 is None:
            continue
        price = pre_match_price(token1, mm["start_date"])
        if price is None:
            continue
        sub = m1[m1["match_id"] == mm["match_id"]]
        if sub.empty or pd.isna(sub.iloc[0]["model_p_home"]):
            continue
        model_p = sub.iloc[0]["model_p_home"]
        home_win = sub.iloc[0]["home_win"]
        res.append({"match_id": mm["match_id"], "team1": mm["team1"], "team2": mm["team2"],
                    "start_date": mm["start_date"], "model_p_home": model_p,
                    "market_p_home": price, "home_win": home_win})
        time.sleep(0.15)
    return pd.DataFrame(res)


def run_backtest(bt: pd.DataFrame, thresholds=(0.03, 0.05, 0.07, 0.10)) -> None:
    """按分歧阈值下注的 P&L + 校准对比。"""
    if bt.empty:
        print("无回测数据"); return
    bt = bt.copy()
    bt["disagreement"] = bt["model_p_home"] - bt["market_p_home"]

    # 校准（Brier）
    from sklearn.metrics import brier_score_loss
    b_mkt = brier_score_loss(bt["home_win"], bt["market_p_home"])
    b_mod = brier_score_loss(bt["home_win"], bt["model_p_home"])
    print(f"\n=== 校准（越低越好）===")
    print(f"  市场 Brier = {b_mkt:.4f}   模型 Brier = {b_mod:.4f}   (n={len(bt)})")

    print(f"\n=== 分歧下注策略（每注 $1）===")
    print(f"{'阈值':>6}{'注数':>6}{'总P&L':>10}{'每注EV':>10}{'胜率':>8}")
    for thr in thresholds:
        bets = bt[bt["disagreement"].abs() > thr]
        if bets.empty:
            print(f"{thr:>6.2f}{0:>6}{0.0:>10.2f}{0.0:>10.3f}{'—':>8}")
            continue
        pnl = 0.0
        wins = 0
        for _, r in bets.iterrows():
            bet_home = r["disagreement"] > 0
            if bet_home:
                pnl += r["home_win"] - r["market_p_home"]
                wins += r["home_win"]
            else:
                pnl += (1 - r["home_win"]) - (1 - r["market_p_home"])
                wins += (1 - r["home_win"])
        n = len(bets)
        print(f"{thr:>6.2f}{n:>6}{pnl:>10.2f}{pnl/n:>10.3f}{wins/n:>8.1%}")

    # 模型 vs 市场：模型残差是否预测方向
    print(f"\n=== 模型概率 vs 市场价 的残差是否有效 ===")
    print("  （corr(disagreement, outcome) 或 betting on disagreement 的命中率）")
    from scipy.stats import spearmanr
    rho, pv = spearmanr(bt["disagreement"], bt["home_win"])
    print(f"  spearman(分歧, 结果) = {rho:+.3f} (p={pv:.3f})")


if __name__ == "__main__":
    dmaps = build_demo_maps()
    print(f"demo 图：{len(dmaps)}，有模型概率 {dmaps['model_p_home'].notna().sum()}")

    evs = fetch_finished_events()
    print(f"抓取 Polymarket 已结算事件：{len(evs)}")

    matched = match_demo_to_market(dmaps, evs)
    print(f"匹配到事件：{len(matched)} 场")

    bt = backtest(dmaps, matched)
    print(f"取到赛前价并回测：{len(bt)} 场")
    run_backtest(bt)

    # 保存回测表供后续分析
    out = config.DATA_DIR / "backtest_map1.parquet"
    bt.to_parquet(out)
    print(f"\n已保存 {out}")
