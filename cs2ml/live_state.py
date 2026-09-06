"""live_state.py —— 实时比赛状态轮询（第一版数据源：bo3.gg live）。

数据源决策（2026-09-06 实测）：
  - HLTV scorebot（scorebot-lb.hltv.org）确实有回合级比分/经济/击杀，但前端反爬混淆：
    data-href 编码块、HTTP 全 502（只走 WebSocket）、JS bundle 里无 scorebot 字样。
    直接接是坑、脆弱、且有 ToS 灰色地带——第一版不碰它。
  - bo3.gg /api/v2/matches/live 是干净 JSON、项目已有 client，给出 series 地图比分 +
    庄家实时赔率(coeff)。回合级经济/击杀等 GRID Open Access（已申请）到位后再接。

本模块：轮询 bo3.gg live，把 (地图比分 + 赔率) 压成结构化快照落盘，供 P3 信号引擎 /
P4 paper-trading 状态机消费。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from . import config
from .bo3gg import Bo3Client

OUT_DIR = config.DATA_DIR / "live"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def live_snapshot(client: Bo3Client | None = None) -> list[dict]:
    """当前所有 live 比赛的压缩快照：地图比分 + 庄家赔率 + 队名。"""
    client = client or Bo3Client()
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    payload = client.matches_for_date(today, "live")
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: list[dict] = []
    for m in payload.get("data", []):
        bu = m.get("bet_updates") or {}
        t1 = (bu.get("team_1") or {}).get("name")
        t2 = (bu.get("team_2") or {}).get("name")
        rows.append({
            "ts": ts,
            "match_id": m.get("id"),
            "team1": t1,
            "team2": t2,
            "team1_score": m.get("team1_score"),
            "team2_score": m.get("team2_score"),
            "team1_odds": (bu.get("team_1") or {}).get("coeff"),
            "team2_odds": (bu.get("team_2") or {}).get("coeff"),
            "status": m.get("status"),
            "parsed_status": m.get("parsed_status"),
        })
    return rows


def poll(match_id: int | None = None, interval_sec: float = 2.0,
         duration_min: float = 120.0, out_file: Path | None = None) -> Path:
    """轮询 live 快照并追加 JSONL。match_id 给定时只记录该场。返回输出路径。"""
    client = Bo3Client()
    if out_file is None:
        out_file = OUT_DIR / f"live_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + duration_min * 60
    n = 0
    print(f"polling bo3.gg live -> {out_file} (match_id={match_id})")
    while time.time() < deadline:
        try:
            rows = live_snapshot(client)
        except Exception as e:  # noqa: BLE001
            print(f"[{dt.datetime.now():%H:%M:%S}] poll error: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(interval_sec)
            continue
        for r in rows:
            if match_id is not None and r["match_id"] != match_id:
                continue
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        time.sleep(interval_sec)
    print(f"done. {n} rows -> {out_file}")
    return out_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="只打印一次快照")
    ap.add_argument("--match-id", type=int, default=None)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--minutes", type=float, default=120.0)
    args = ap.parse_args()

    if args.once:
        rows = live_snapshot()
        print(f"{len(rows)} live match(es):")
        for r in rows:
            print(f"  [{r['match_id']}] {r['team1']} {r['team1_score']} - {r['team2_score']} "
                  f"{r['team2']}  odds={r['team1_odds']}/{r['team2_odds']} "
                  f"({r['status']}/{r['parsed_status']})")
        return

    poll(match_id=args.match_id, interval_sec=args.interval,
         duration_min=args.minutes)


if __name__ == "__main__":
    main()
