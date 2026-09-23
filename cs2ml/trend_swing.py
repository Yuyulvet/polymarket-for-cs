"""Trend-swing replay — 用户的「吃优势趋势波段」策略 tape 级回测.

用户策略 (2026-09-23): 手枪结束后跟市场跳变买下手枪胜者, 持有过连锁经济
优势窗口 (anti-eco 回合), 在第一个长枪局前离场. 实时 feed 慢于市场 → 以
跳变为入场信号, 吃「胜方经济趋势」的后续漂移.

与已测版本的区别: v1 pistol replay 是跳变前入场吃 pop; inplay backtest 是
均势点读经济预测下一回合. 本策略 = 跳变后入场, 持有 R2(±R3), 新的持有窗.

Tape: data/live_sessions 的秒级 book + fivee 回合事件 (pistol_replay 同款).
诚实性: 入场吃 ask、出场吃 bid (付点差); 窗口用 plateau 中位数 (与
pistol_replay 一致); 样本仅 ~12 图, 结论作方向参考.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from pistol_replay import (Book, load_book, load_rounds, load_tokens,
                           price_jump)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SESS = ROOT / "data/live_sessions"
OUT = ROOT / "reports/trend_swing_20260923"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sess in sorted(SESS.iterdir()):
        if not (sess / "fivee_events.jsonl").exists():
            continue
        spec_p = sess / "session_spec.json"
        if not spec_p.exists():
            continue
        tokens = load_tokens(sess)
        if not tokens:
            continue
        by_asset = load_book(sess, tokens)
        bouts = load_rounds(sess)
        for bout, rb in sorted(bouts.items()):
            rends = [(t, w, c, s) for (t, w, c, s) in rb["rends"]
                     if w in ("CT", "T")]
            if [c + s for (_, _, c, s) in rends[:2]] != [1, 2]:
                continue
            teams = {t2 for (b2, t2) in tokens if b2 == bout}
            if len(teams) != 2:
                continue
            ta, tb = sorted(teams)
            if (bout, ta) not in tokens or (bout, tb) not in tokens:
                continue
            ba, bb = Book(by_asset[tokens[(bout, ta)]]), \
                Book(by_asset[tokens[(bout, tb)]])
            t_p1 = rends[0][0]
            rstarts = [t for t in rb["rstarts"]]
            # 需要 R3/R4 开始时间作为离场点
            if len(rstarts) < 4:
                continue
            t_r3s, t_r4s = rstarts[2], rstarts[3]
            t_r2e = rends[1][0]
            win1, conf = price_jump(ba, bb, t_p1)
            if win1 is None:
                continue
            fav = ta if win1 == "a" else tb
            bf = ba if fav == ta else bb

            def zone(t0, t1):
                got = bf.plateau(t0, t1)
                if got is None:
                    return None
                return got  # (bid, ask, mid, n)

            # 入场: 跳变后 plateau (t_p1+30s, +150s) — 跟单可成交窗口
            ent = zone(t_p1 + 30, t_p1 + 150)
            if ent is None:
                continue
            entry_ask = ent[1]
            # 离场窗口 (各持有期)
            exits = {
                "r2_end": zone(t_r2e - 30, t_r2e + 30),       # 持有一整个anti-eco
                "pre_r3": zone(t_r3s - 90, t_r3s - 10),       # 长枪局(R3)前
                "pre_r4": zone(t_r4s - 90, t_r4s - 10),       # R4 前 (连环深时)
                "map_end": zone(rends[-1][0] - 60,
                                rends[-1][0] + 300),          # 对照: 拿到图尾
            }
            score_end = rends[-1][2] + rends[-1][3]
            fav_won_map = None
            if score_end >= 13:
                # 用跳变归属的队名近似地图胜者需比分绑定 — fivee 比分不绑队,
                # 用终盘 plateau 方向定 (赢图方 mid→~1)
                z = zone(rends[-1][0] + 30, rends[-1][0] + 300)
            row = {"session": sess.name, "bout": bout, "fav": fav,
                   "entry_ask": round(entry_ask, 3),
                   "conf1": round(conf, 4)}
            for k, z in exits.items():
                if z is None:
                    row[f"{k}_pnl"] = np.nan
                else:
                    row[f"{k}_pnl"] = round(z[0] - entry_ask, 4)  # 出场吃bid
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trades.csv", index=False)
    print(f"bouts replayed: {len(df)}")
    if not len(df):
        return 0
    summary = {}
    for k in ("r2_end", "pre_r3", "pre_r4", "map_end"):
        s = df[f"{k}_pnl"].dropna()
        summary[k] = {
            "n": int(len(s)),
            "pnl_mean_pt": round(float(s.mean()) * 100, 2),
            "pnl_median_pt": round(float(s.median()) * 100, 2),
            "win_rate": round(float((s > 0).mean()), 3),
            "pnl_total_pt": round(float(s.sum()) * 100, 2),
        }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nper-trade (entry = post-jump ask, pnl in pt):")
    cols = ["session", "bout", "fav", "entry_ask", "r2_end_pnl",
            "pre_r3_pnl", "pre_r4_pnl", "map_end_pnl"]
    print((df[cols].assign(**{c: (df[c] * 100).round(1)
                              for c in cols if c.endswith("_pnl")}))
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
