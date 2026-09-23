"""inplay_data.py —— #34 盘中数据：Polymarket 盘中价格 + demo 回合时间线对齐。

用户要「低买高卖」：预测回合级趋势（经济信号 0.745），在 Polymarket Map Winner
盘中价上买低卖高。这里做两件事：

  1. 抓全量已结算 CS2 事件（缓存 data/polymarket_events.json，断点续传）。
  2. 对匹配到 demo 的事件，抓 Map N Winner 的**盘中**逐分钟价格序列（fidelity=1），
     并与 demo 回合 wall-clock 时间线对齐。

对齐：demos.start_date = 比赛开始 wall-clock；round_states.round_len_sec 累积 →
      回合 k 的 wall-clock 起止。盘中价序列按分钟采样，回合内约 1-2 个价格点。
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from curl_cffi import requests as creq

from . import config
from .backtest_market import (GAMMA, CLOB, parse_event, match_demo_to_market,
                              build_demo_maps, team_match)

EVENTS_CACHE = config.DATA_DIR / "polymarket_events.json"
INPLAY_CACHE = config.DATA_DIR / "polymarket_inplay.parquet"


# ---------------------------------------------------------------- 事件抓取（缓存+续传）
def _gamma(path, **params):
    r = creq.get(f"{GAMMA}{path}", params=params, impersonate="chrome", timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_all_events(max_events: int = 3000) -> list[dict]:
    """抓全量已结算 CS2 事件，断点续传 + 指数退避。"""
    evs, off = [], 0
    # Gamma offset 分页上限 2050（"offset too large"）；按 volume 降序取前 2050 即可，
    # 缺的是最低量事件（不是 S 级 demo 匹配目标）。
    while len(evs) < max_events and off < 2050:
        batch = None
        for attempt in range(8):
            try:
                batch = _gamma("/events", tag_slug="counter-strike-2", closed="true",
                               limit=50, offset=off, order="volume", ascending="false")
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2.0 * (attempt + 1) + np.random.uniform(0, 1))
        if not batch:
            break
        evs += batch
        if len(batch) < 50:
            break
        off += 50
        time.sleep(0.3)
    return evs


def load_events(refresh: bool = False) -> list[dict]:
    if not refresh and EVENTS_CACHE.exists():
        return json.loads(EVENTS_CACHE.read_text(encoding="utf-8"))
    evs = fetch_all_events()
    EVENTS_CACHE.write_text(json.dumps(evs, ensure_ascii=False), encoding="utf-8")
    return evs


# ---------------------------------------------------------------- 盘中价抓取
def _clob_prices(token: str, start_ts: int, end_ts: int, fidelity: int = 1) -> list[dict]:
    r = creq.get(f"{CLOB}/prices-history",
                 params={"market": token, "startTs": start_ts, "endTs": end_ts,
                         "fidelity": fidelity}, impersonate="chrome", timeout=60)
    r.raise_for_status()
    return r.json().get("history", [])


def fetch_inplay_series(token: str, t0: pd.Timestamp, t1: pd.Timestamp,
                        fidelity: int = 1) -> pd.DataFrame:
    """逐分钟价格序列，t0→t1。返回 {t(秒), p}。"""
    hist = _clob_prices(token, int(t0.timestamp()), int(t1.timestamp()), fidelity)
    if not hist:
        return pd.DataFrame(columns=["t", "p"])
    df = pd.DataFrame(hist)
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df[["t", "p"]].sort_values("t").reset_index(drop=True)


# ---------------------------------------------------------------- demo 回合 wall-clock 时间线
def build_round_timeline() -> pd.DataFrame:
    """每回合一行：demo_path, match_id, round_num, round_start_utc, round_end_utc, winner_side。

    round_start_utc = demos.start_date + 前面回合 round_len_sec 累积。
    """
    rs = pd.read_parquet(config.DATA_DIR / "round_states.parquet")
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    keep = ["demo_path", "round_num", "round_len_sec", "winner_side"]
    r = rs[keep].merge(rd[["demo_path", "round_num", "match_id", "map_name"]],
                       on=["demo_path", "round_num"], how="inner")
    r["round_num"] = r["round_num"].astype(int)

    con = sqlite3.connect(config.DB_PATH)
    dm = pd.read_sql_query("SELECT hltv_match_id, start_date FROM demos", con)
    con.close()
    dm["hltv_match_id"] = dm["hltv_match_id"].astype(int)

    def hid(dp):
        m = re.search(r"hltv-(\d+)", dp)
        return int(m.group(1)) if m else None

    r["hltv_match_id"] = r["demo_path"].map(hid)
    r = r.merge(dm, on="hltv_match_id", how="left")
    r["start_date"] = pd.to_datetime(r["start_date"], utc=True)

    rows = []
    for dp, sub in r.groupby("demo_path"):
        sub = sub.sort_values("round_num").reset_index(drop=True)
        start = sub["start_date"].iloc[0]
        if pd.isna(start):
            continue
        t = start
        for _, cur in sub.iterrows():
            d = float(cur["round_len_sec"])
            if pd.isna(d):
                d = 115.0  # 平均回合时长兜底
            rows.append({
                "demo_path": dp, "match_id": cur["match_id"],
                "map_name": cur["map_name"], "round_num": int(cur["round_num"]),
                "round_start_utc": t, "round_end_utc": t + pd.Timedelta(seconds=d),
                "winner_side": cur["winner_side"],
            })
            t = t + pd.Timedelta(seconds=d)
    return pd.DataFrame(rows)


def align_rounds_to_prices(timeline: pd.DataFrame, series: pd.DataFrame,
                           lag_sec: int = 0) -> pd.DataFrame:
    """把每个回合对齐到「回合开始时刻」最近的一个盘中价（lag_sec 秒后）。

    返回：round 级表 + price_start（回合开始时的价格）+ price_end（回合结束时的价格）。
    """
    out = []
    for _, rw in timeline.iterrows():
        t_start = rw["round_start_utc"] + pd.Timedelta(seconds=lag_sec)
        t_end = rw["round_end_utc"]
        s = series
        # 找 <= t_start 的最近价（成交价快照）
        pre = s[s["t"] <= t_start]
        if pre.empty:
            continue
        p_start = pre.iloc[-1]["p"]
        post = s[s["t"] >= t_end]
        p_end = post.iloc[0]["p"] if not post.empty else s.iloc[-1]["p"]
        out.append({
            "demo_path": rw["demo_path"], "match_id": rw["match_id"],
            "map_name": rw["map_name"], "round_num": rw["round_num"],
            "round_start_utc": rw["round_start_utc"], "round_end_utc": rw["round_end_utc"],
            "winner_side": rw["winner_side"],
            "price_start": p_start, "price_end": p_end,
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 原始盘中价缓存
RAW_CACHE = config.DATA_DIR / "polymarket_inplay_raw.parquet"


def fetch_raw_series(matched: pd.DataFrame, refresh: bool = False,
                     window_hours: float = 5.0,
                     lead_days: float = 7.0) -> pd.DataFrame:
    """对每场匹配事件的每个 Map N Winner，抓两队的盘中价序列，缓存为原始 parquet。

    列：match_id, team1, team2, map_market, outcome, token, t, p。

    修复（2026-09-16）：原先把抓取窗口起点写死为 `start_date - 5 分钟`，
    导致整份 RAW_CACHE 只有开赛前 5 分钟的价格（278/278 场无一例外），
    而 CLOB 其实能返回赛前十几小时到几天的分钟级序列（实测 Feb 场 1186 个赛前点、
    最早距开赛 19.8h）。这让所有"开盘价/软市场/早期定价"的分析都建立在
    被自己截断的数据上。现在起点改为 `start_date - lead_days`。

    另注意 API 契约：`interval=1m` 配 fidelity=1 会 HTTP 400；
    必须用显式 `startTs`/`endTs` + fidelity=1（见 fetch_inplay_series）。
    """
    if not refresh and RAW_CACHE.exists():
        return pd.read_parquet(RAW_CACHE)

    rows = []
    total = sum(len(mm["markets"]) for _, mm in matched.iterrows())
    done = 0
    for _, mm in matched.iterrows():
        for map_key, toks in mm["markets"].items():
            # 每个 outcome（队名）一个 token，抓两队的序列
            for outcome, token in toks.items():
                done += 1
                t0 = mm["start_date"] - pd.Timedelta(days=lead_days)
                t1 = mm["start_date"] + pd.Timedelta(hours=window_hours)
                try:
                    series = fetch_inplay_series(token, t0, t1)
                except Exception:
                    continue
                if series.empty:
                    continue
                series = series.copy()
                series["match_id"] = mm["match_id"]
                series["team1"] = mm["team1"]
                series["team2"] = mm["team2"]
                series["map_market"] = map_key
                series["outcome"] = outcome
                series["token"] = token
                rows.append(series)
                time.sleep(0.12)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(RAW_CACHE)
    return df


def acquire(refresh_events: bool = False, refresh_price: bool = False) -> dict:
    """抓事件 → 匹配 → 抓原始盘中价 → 缓存。"""
    dmaps = build_demo_maps()
    evs = load_events(refresh=refresh_events)
    print(f"demo 图(有模型概率): {int(dmaps['model_p_home'].notna().sum())}  "
          f"| 唯一比赛 {int(dmaps['match_id'].nunique())}")
    print(f"Polymarket 已结算事件: {len(evs)}")
    matched = match_demo_to_market(dmaps, evs)
    print(f"匹配到事件: {len(matched)} 场")
    if matched.empty:
        return {"matched": 0, "with_price": 0}
    raw = fetch_raw_series(matched, refresh=refresh_price)
    n_maps = raw["map_market"].nunique() if not raw.empty else 0
    n_matches = raw["match_id"].nunique() if not raw.empty else 0
    n_points = len(raw)
    return {"matched": len(matched), "with_price": n_points,
            "matches_with_price": n_matches, "map_markets": n_maps}


if __name__ == "__main__":
    rep = acquire(refresh_events=False, refresh_price=True)
    print("结果:", rep)
