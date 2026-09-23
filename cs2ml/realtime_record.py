"""realtime_record.py —— 实时录价：Polymarket CLOB WebSocket 订阅 CS2 盘口。

用于 latency edge 测试：把 Map Winner 盘口的盘中价以秒级(及以上)频率落库，
配合实时读 demo/观赛，测「我先于市场读经济」的可捕捉价差。

协议（2026-09 验证）：
  wss://ws-subscriptions-clob.polymarket.com/ws/market   （公开 market 频道，无需鉴权）
  订阅消息: {"assets_ids": [token...], "type": "market"}
  事件类型: book(快照) / price_change(增量) / last_trade_price(成交) / tick_size_change / best_bid_ask

用法：
  python -m cs2ml.realtime_record --probe            # 连接并打印前 N 条消息（验证协议/格式）
  python -m cs2ml.realtime_record --event 944152     # 订阅该场全部盘口，落库
  python -m cs2ml.realtime_record --match G2 Falcons --hours 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests as creq

from . import config

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA = "https://gamma-api.polymarket.com"
OUT_DIR = config.DATA_DIR / "realtime"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 盘口发现
def discover_events(query: str, limit: int = 10) -> list[dict]:
    r = creq.get(f"{GAMMA}/public-search", params={"q": query, "limit": limit},
                 impersonate="chrome", timeout=40)
    r.raise_for_status()
    d = r.json()
    return d.get("events", d if isinstance(d, list) else [])


def event_details(event_id: int) -> dict:
    r = creq.get(f"{GAMMA}/events/{event_id}", impersonate="chrome", timeout=40)
    r.raise_for_status()
    return r.json()


def tokens_from_event(e: dict) -> dict:
    """Event payload -> {market_name: {outcome: token_id}}."""
    out = {}
    for mk in e.get("markets", []):
        name = mk.get("groupItemTitle") or mk.get("question")
        ct = mk.get("clobTokenIds")
        outcomes = mk.get("outcomes") or []
        if not ct or not outcomes:
            continue
        toks = json.loads(ct) if isinstance(ct, str) else ct
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        out[name] = dict(zip(outcomes, toks))
    return out


def event_tokens(event_id: int) -> dict:
    """event_id -> {market_name: {outcome: token_id}}."""
    return tokens_from_event(event_details(event_id))


# ---------------------------------------------------------------- WebSocket 录价
class RealtimeRecorder:
    def __init__(self, tokens: dict[str, dict[str, str]], out_file: Path,
                 probe_limit: int = 0, event_id: str | None = None):
        self.tokens = tokens                      # market -> {outcome: token}
        self.tok2info = {}                        # token -> (market, outcome)
        self.assets = []
        for m, o2t in tokens.items():
            for outcome, tok in o2t.items():
                self.tok2info[tok] = (m, outcome)
                self.assets.append(tok)
        self.out_file = out_file
        self.probe_limit = probe_limit
        self.event_id = None if event_id is None else str(event_id)
        self.n_events = 0
        self.n_msgs = 0
        self.best = {}                            # token -> {"bid","ask","last","ts"}
        self.last_write = 0.0

    # ---- 连接 ----
    def _connect(self):
        import websocket  # websocket-client（同步）
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        return ws

    def _on_open(self, ws):
        sub = {"assets_ids": self.assets, "type": "market"}
        ws.send(json.dumps(sub))
        print(f"[{self._now()}] connected, subscribed {len(self.assets)} tokens", flush=True)

    def _on_error(self, ws, err):
        print(f"[{self._now()}] WS error: {err}", flush=True)

    def _on_close(self, ws, code, msg):
        print(f"[{self._now()}] WS closed code={code} msg={msg}", flush=True)

    # ---- 消息处理 ----
    @staticmethod
    def _f(v):
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    def _on_message(self, ws, raw):
        self.n_msgs += 1
        try:
            msg = json.loads(raw)
        except Exception:
            return
        # 心跳响应（服务端可能发文本 ping）
        if isinstance(msg, str):
            if msg == "PING":
                ws.send("PONG")
            return
        if not isinstance(msg, dict):
            return
        if self.probe_limit:
            print(f"[PROBE] {json.dumps(msg, ensure_ascii=False)[:400]}", flush=True)
            self.n_events += 1
            if self.n_events >= self.probe_limit:
                ws.close()
            return
        self._handle(msg)

    def _handle(self, msg):
        local_ts = time.time()
        source_ts = self._source_ts(msg)
        # 1) price_changes：数组，每项带 asset_id + best_bid/best_ask
        pcs = msg.get("price_changes")
        if isinstance(pcs, list):
            for pc in pcs:
                if not isinstance(pc, dict):
                    continue
                aid = str(pc.get("asset_id"))
                info = self.tok2info.get(aid)
                if info is None:
                    continue
                market, outcome = info
                bb = self._f(pc.get("best_bid"))
                ba = self._f(pc.get("best_ask"))
                mid = (bb + ba) / 2 if (bb is not None and ba is not None) else None
                self._write({
                    "type": "pc", "market": market, "outcome": outcome, "token": aid,
                    "price": self._f(pc.get("price")), "size": self._f(pc.get("size")),
                    "side": pc.get("side"), "best_bid": bb, "best_ask": ba, "mid": mid,
                    "local_ts": local_ts, "source_ts": source_ts,
                })
            return
        # 2) book 快照：asset_id + bids/asks
        if "asset_id" in msg and ("bids" in msg or "asks" in msg):
            aid = str(msg.get("asset_id"))
            info = self.tok2info.get(aid)
            if info is None:
                return
            market, outcome = info
            bb = self._top(msg.get("bids"), "bids")
            ba = self._top(msg.get("asks"), "asks")
            mid = (bb + ba) / 2 if (bb is not None and ba is not None) else None
            self._write({
                "type": "book", "market": market, "outcome": outcome, "token": aid,
                "best_bid": bb, "best_ask": ba, "mid": mid, "local_ts": local_ts,
                "source_ts": source_ts,
            })
            return
        # 3) last_trade_price 成交
        if "asset_id" in msg and "price" in msg and ("side" in msg or "size" in msg):
            aid = str(msg.get("asset_id"))
            info = self.tok2info.get(aid)
            if info is None:
                return
            market, outcome = info
            self._write({
                "type": "trade", "market": market, "outcome": outcome, "token": aid,
                "price": self._f(msg.get("price")), "size": self._f(msg.get("size")),
                "side": msg.get("side"), "local_ts": local_ts, "source_ts": source_ts,
            })
            return

    @staticmethod
    def _top(levels, kind):
        if not isinstance(levels, list) or not levels:
            return None

    @staticmethod
    def _source_ts(message):
        try:
            value = float(message.get("timestamp"))
        except (AttributeError, TypeError, ValueError):
            return None
        return value / 1000.0 if value > 10_000_000_000 else value
        try:
            prices = []
            for lv in levels:
                p = lv.get("price") if isinstance(lv, dict) else lv[0]
                prices.append(float(p))
            if not prices:
                return None
            return max(prices) if kind == "bids" else min(prices)
        except Exception:
            return None

    def _write(self, row):
        # 追加 JSONL；每行一个事件，保秒级时间戳
        if self.event_id is not None:
            row = {"event_id": self.event_id, **row}
        row["recv_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        line = json.dumps(row, ensure_ascii=False)
        with open(self.out_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.n_events += 1

    @staticmethod
    def _now():
        return dt.datetime.now().strftime("%H:%M:%S")

    # ---- 主循环 ----
    def run(self, duration_hours: float = 6.0):
        import websocket
        deadline = time.time() + duration_hours * 3600
        while time.time() < deadline:
            ws = self._connect()
            try:
                ws.run_forever(ping_interval=15, ping_timeout=10)
            except Exception as e:
                print(f"[{self._now()}] run_forever exited: {e}", flush=True)
            if self.probe_limit and self.n_events >= self.probe_limit:
                break  # probe 达到目标即停，不重连
            if time.time() >= deadline:
                break
            print(f"[{self._now()}] reconnect in 3s ...", flush=True)
            time.sleep(3)
        print(f"[{self._now()}] done. events={self.n_events} msgs={self.n_msgs} "
              f"-> {self.out_file}", flush=True)


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="连接并打印前 N 条消息")
    ap.add_argument("--probe-limit", type=int, default=8)
    ap.add_argument("--event", type=int, default=None)
    ap.add_argument("--match", nargs="+", default=None, help="队名关键字，如 G2 Falcons")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--market", action="append", default=[],
                    help="只录指定盘口，可重复，例如 --market 'Map 2 Winner'")
    args = ap.parse_args()

    if args.probe:
        # 用今晚的盘口做 probe
        event_id = args.event or 944152
        tokens = event_tokens(event_id)
        flat = {m: o2t for m, o2t in tokens.items()}
        rec = RealtimeRecorder(flat, OUT_DIR / "probe.jsonl",
                               probe_limit=args.probe_limit)
        rec.run(duration_hours=0.05)  # ~3 分钟上限，实际 probe_limit 触发即停
        return

    # 正常录价
    if args.event:
        event_id = args.event
    elif args.match:
        q = " ".join(args.match)
        evs = discover_events(q)
        evs = [e for e in evs if not e.get("closed")]
        if not evs:
            print("未找到未结算赛事:", q)
            return
        e = evs[0]
        print(f"选中: {e.get('title')} (id={e.get('id')}, start={e.get('startDate')})")
        event_id = e["id"]
    else:
        print("需要 --event 或 --match")
        return

    event = event_details(event_id)
    tokens = tokens_from_event(event)
    if args.market:
        wanted = {name.casefold() for name in args.market}
        tokens = {name: values for name, values in tokens.items()
                  if name.casefold() in wanted}
        missing = wanted - {name.casefold() for name in tokens}
        if missing:
            raise ValueError(f"markets_not_found:{sorted(missing)}")
    for m, o2t in tokens.items():
        print(f"  {m}: {list(o2t)}")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"event_{event_id}_{stamp}.jsonl"
    selected = []
    for market in event.get("markets", []):
        name = market.get("groupItemTitle") or market.get("question")
        if name not in tokens:
            continue
        selected.append({
            "market": name, "market_id": str(market.get("id")),
            "condition_id": market.get("conditionId"),
            "outcomes": list(tokens[name]), "tokens": tokens[name],
            "seconds_delay": market.get("secondsDelay"),
            "fees_enabled": market.get("feesEnabled"),
            "fee_type": market.get("feeType"),
            "fee_schedule": market.get("feeSchedule"),
        })
    meta = {
        "schema_version": 2, "event_id": str(event_id),
        "event_title": event.get("title"), "event_slug": event.get("slug"),
        "ws_url": WS_URL, "timing_basis": "local_receive_time",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "markets": selected,
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    rec = RealtimeRecorder(tokens, out, event_id=str(event_id))
    rec.run(duration_hours=args.hours)


if __name__ == "__main__":
    main()
