"""paper_trade.py —— P4 paper-trading 状态机（虚拟，不碰真钱）。

把三样东西接成一个闭环，落盘全量日志：
  状态源：cs2ml.live_state（bo3.gg live：地图比分 + 庄家赔率）
  价源：Polymarket CLOB book（REST 轮询，取 mid）
  对齐：cs2ml.realtime_record.discover_events / event_tokens（队名 -> Map Winner token）
  信号：可插拔。v1 = 跨盘领先：庄家 de-vig 概率 vs Polymarket 价分歧 > 阈值则买便宜侧。

用法：
  python -m cs2ml.paper_trade --list                     # 列出 live 比赛 + 匹配到的盘口
  python -m cs2ml.paper_trade --team1 'G2' --team2 'Falcons' --minutes 60
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，中文输出会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from curl_cffi import requests as creq

from . import config
from .live_state import live_snapshot
from .realtime_record import discover_events, event_tokens

CLOB = "https://clob.polymarket.com"
OUT_DIR = config.DATA_DIR / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_THR = 0.05  # 分歧阈值：庄家概率 - Polymarket 价


# ---------------------------------------------------------------- 工具
def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\bteam\b|\besports\b|\bthe\b|\bgaming\b|\bclub\b|\bacademy\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def de_vig(c1: float, c2: float) -> tuple[float, float]:
    """十进制赔率 -> de-vig 概率对（去掉庄家 water）。"""
    inv = 1.0 / c1 + 1.0 / c2
    return (1.0 / c1) / inv, (1.0 / c2) / inv


def _mid(token: str) -> float | None:
    """CLOB book 顶端 bid/ask 的中点价。"""
    r = creq.get(f"{CLOB}/book", params={"token_id": token},
                 impersonate="chrome", timeout=20)
    r.raise_for_status()
    b = r.json()
    bids = b.get("bids") or []
    asks = b.get("asks") or []
    if not bids or not asks:
        return None
    bb = max(float(x["price"]) for x in bids)
    ba = min(float(x["price"]) for x in asks)
    return (bb + ba) / 2


# ---------------------------------------------------------------- 对齐
def align_market(team1: str, team2: str) -> dict | None:
    """队名 -> {event_id, market, team1_token, team2_token}（取第一张 Map Winner）。"""
    evs = discover_events(f"{team1} {team2}", limit=10)
    for ev in evs:
        if ev.get("closed"):
            continue
        eid = ev.get("id")
        if not eid:
            continue
        try:
            toks = event_tokens(eid)
        except Exception:
            continue
        n1, n2 = _norm(team1), _norm(team2)
        # 候选：两个 outcome 分别匹配 team1/team2（Winner 和 Handicap 都含两队名）
        cands = []
        for market, o2t in toks.items():
            outs = list(o2t)
            t1_out = next((o for o in outs if n1 and n1 in _norm(o)), None)
            t2_out = next((o for o in outs if n2 and n2 in _norm(o)), None)
            if t1_out and t2_out and t1_out != t2_out:
                cands.append((market, o2t, t1_out, t2_out))
        if not cands:
            continue
        # 优先 Winner 市场（bo3.gg coeff 是系列赛胜赔 → 优先 Match Winner）
        order = ["match winner", "map 1 winner", "map 2 winner", "map 3 winner",
                 "map 4 winner", "map 5 winner"]

        def rank(c):
            m = c[0].lower()
            for i, k in enumerate(order):
                if k in m:
                    return i
            return 99
        market, o2t, t1_out, t2_out = min(cands, key=rank)
        return {"event_id": eid, "market": market,
                "team1_token": o2t[t1_out], "team2_token": o2t[t2_out],
                "team1_out": t1_out, "team2_out": t2_out}
    return None


# ---------------------------------------------------------------- 状态机
class PaperTrader:
    """虚拟持仓 + 交易日志。position: None 或 dict(token, side, entry_price, entry_ts)。"""

    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.position: dict | None = None
        self.trades: list[dict] = []
        self._log = open(self.log_file, "a", encoding="utf-8")

    def _emit(self, row: dict) -> None:
        self._log.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._log.flush()

    def on_price(self, ts: float, state: dict, p1: float | None) -> None:
        """主循环每拍调用一次：state=bo3.gg 快照字段，p1=team1 token mid 价。"""
        if p1 is None:
            return
        action = signal(state, p1)  # 'buy1' | 'buy2' | 'sell' | 'hold'
        pos = self.position
        if pos is None:
            if action in ("buy1", "buy2"):
                token = state["team1_token"] if action == "buy1" else state["team2_token"]
                side = 1 if action == "buy1" else -1
                self.position = {"token": token, "side": side,
                                 "entry_price": p1, "entry_ts": ts,
                                 "reason": state.get("signal_reason", "")}
                self._emit({"ts": ts, "type": "open", "action": action,
                            "token": token, "price": p1, "reason": self.position["reason"]})
        else:
            exit_price = p1
            if action == "sell" or state.get("map_over"):
                pnl = (exit_price - pos["entry_price"]) * pos["side"]
                self.trades.append({**pos, "exit_price": exit_price, "exit_ts": ts,
                                    "pnl": pnl})
                self._emit({"ts": ts, "type": "close", "token": pos["token"],
                            "entry": pos["entry_price"], "exit": exit_price,
                            "pnl": pnl, "side": pos["side"]})
                self.position = None

    def summary(self) -> dict:
        n = len(self.trades)
        if not n:
            return {"n": 0, "pnl": 0.0, "win": 0.0}
        pnl = sum(t["pnl"] for t in self.trades)
        win = sum(1 for t in self.trades if t["pnl"] > 0) / n
        return {"n": n, "pnl": pnl, "win": win, "avg": pnl / n}


# ---------------------------------------------------------------- 信号（v1 跨盘领先）
def signal(state: dict, p1: float) -> str:
    """庄家 de-vig 概率 vs Polymarket 价。分歧 > 阈值则买被 Polymarket 低估的一侧。

    state 需含 team1_odds / team2_odds。p1 = team1 token 价。收盘触发卖出。
    """
    c1, c2 = state.get("team1_odds"), state.get("team2_odds")
    state["signal_reason"] = ""
    if not c1 or not c2:
        return "hold"
    q1, _ = de_vig(float(c1), float(c2))
    div = q1 - p1
    state["signal_reason"] = f"div={div:+.3f} (bk={q1:.3f} pm={p1:.3f})"
    if abs(div) < _THR:
        return "sell" if div < 0 else "sell"  # 分歧收敛即平仓
    return "buy1" if div > 0 else "buy2"


# ---------------------------------------------------------------- 主循环
def run(team1: str, team2: str, minutes: float = 60.0) -> Path:
    al = align_market(team1, team2)
    if al is None:
        raise SystemExit(f"未匹配到 Polymarket 盘口: {team1} vs {team2}")
    print(f"盘口: {al['market']}  {al['team1_out']} vs {al['team2_out']}")

    trader = PaperTrader(OUT_DIR / f"paper_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl")
    deadline = time.time() + minutes * 60
    print(f"paper-trading {team1} vs {team2} ({minutes}min) ...")
    while time.time() < deadline:
        # 状态：找这场 bo3.gg live 快照
        state = None
        for r in live_snapshot():
            if _norm(r["team1"] or "") == _norm(team1) and _norm(r["team2"] or "") == _norm(team2):
                state = r
                break
        if state is None:
            time.sleep(2)
            continue
        # 价：team1 token mid
        try:
            p1 = _mid(al["team1_token"])
        except Exception:
            p1 = None
        state.update(al)
        state["map_over"] = False  # v1 不判结束，只靠信号平仓
        trader.on_price(time.time(), state, p1)
        time.sleep(2)

    # 强制平仓（若有持仓）
    if trader.position:
        # 用最后一次 mid 收
        try:
            p1 = _mid(al["team1_token"])
        except Exception:
            p1 = trader.position["entry_price"]
        trader.on_price(time.time(), {**al, "team1_odds": None, "team2_odds": None,
                                     "map_over": True}, p1)
    s = trader.summary()
    print(f"\n=== 结果 ===  n={s['n']}  总P&L={s['pnl']:+.4f}  每注={s.get('avg', 0):+.5f}  "
          f"胜率={s.get('win', 0):.1%}")
    print(f"日志 -> {trader.log_file}")
    return trader.log_file


def list_markets() -> None:
    print("live 比赛 -> Polymarket 盘口匹配：")
    for r in live_snapshot():
        t1, t2 = r["team1"], r["team2"]
        if not t1 or not t2:
            continue
        try:
            al = align_market(t1, t2)
        except Exception:
            al = None
        tag = al["market"] if al else "（无盘口）"
        print(f"  {t1} vs {t2}  ->  {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--team1", default=None)
    ap.add_argument("--team2", default=None)
    ap.add_argument("--minutes", type=float, default=60.0)
    args = ap.parse_args()

    if args.list:
        list_markets()
    elif args.team1 and args.team2:
        run(args.team1, args.team2, args.minutes)
    else:
        ap.print_help()
