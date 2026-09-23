"""Forward PAPER ledger for the pistol-swing strategy (user reads, not model).

Why user picks and not model picks: pistol_model_v3 Phase-2 OOF verdict
(reports/pistol_model_v3/) — expanding-window OOF, BOTH the v2 control and the
v3 literature features land at AUC ~0.48 with >=60% predictions hitting only
~42% (n=28). The model has no out-of-sample pistol edge; the only untested
variable left in this strategy is the user's own read quality. This ledger
measures it, with the market as benchmark (CLV).

FROZEN RULES (written at ledger creation 2026-09-23; do not edit mid-ledger):
  ENTRY   limit buy at (pre-pistol plateau ASK - 2pt), paper fill only if the
          tape later shows a trade at <= limit inside the entry window
          [t_p1-420, t_p1-240]; else mark NO_FILL.
  EXIT-W  pistol won  -> sell at plateau BID [t_p1-60, t_p1+60] (first plateau,
          sell into the sentiment pop; the +13pt Mirage tail is not the norm).
  EXIT-L  pistol lost -> sell at plateau BID [t_r2-60, t_r2+60] (mechanical:
          waiting for a recovery is a 44% coin, R2 result known -> out).
  CLV     close_mid = median mid of [t_p1-900, t_p1-600]; clv_pt = close-entry.
  Honesty: record (user pick + timestamp) BEFORE t_p1; settlement reads only
          the tape; spreads paid both ways (entry ask, exit bid).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pistol_replay import (Book, load_book, load_rounds, load_tokens,  # noqa: E402
                           price_jump)

ROOT = Path(__file__).resolve().parent.parent
SESS = ROOT / "data/live_sessions"
LEDGER = ROOT / "reports/forward_ledger/ledger.csv"
FIELDS = ["trade_id", "recorded_at_utc", "session", "bout", "map",
          "team_a", "team_b", "user_pick", "entry_window_ask", "entry_limit",
          "filled", "close_mid", "clv_pt", "pistol_won", "exit_rule",
          "exit_bid", "pnl_pt", "status", "notes"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_csv(LEDGER, dtype=str)
    return pd.DataFrame(columns=FIELDS)


def _save(df: pd.DataFrame) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEDGER, index=False)


def cmd_list(_args) -> int:
    for sess in sorted(SESS.iterdir()):
        if not (sess / "market/events.jsonl").exists():
            continue
        spec_p = sess / "session_spec.json"
        label = sess.name
        if spec_p.exists():
            spec = json.load(open(spec_p, encoding="utf-8"))
            label += f"  {spec.get('team_a','?')} vs {spec.get('team_b','?')}"
        print(f"\n== {label}")
        tokens = load_tokens(sess)
        for bout in sorted({b for (b, _) in tokens}):
            teams = sorted({t2 for (b2, t2) in tokens if b2 == bout})
            print(f"  bout {bout}: {' vs '.join(teams)}")
    return 0


def cmd_record(args) -> int:
    sess = SESS / args.session
    tokens = load_tokens(sess)
    teams = sorted({t for (b, t) in tokens if b == args.bout})
    if args.pick not in teams:
        print(f"pick must be one of {teams}")
        return 1
    df = _load()
    tid = f"{args.session}-b{args.bout}"
    if (df.trade_id == tid).any():
        print(f"{tid} already recorded")
        return 1
    row = {c: "" for c in FIELDS}
    row.update({"trade_id": tid, "recorded_at_utc": _now(),
                "session": args.session, "bout": str(args.bout),
                "user_pick": args.pick, "team_a": teams[0],
                "team_b": teams[1], "status": "recorded",
                "notes": args.notes or ""})
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    print(f"recorded {tid}: pick={args.pick} at {row['recorded_at_utc']} "
          f"(must be BEFORE pistol end to be valid)")
    return 0


def cmd_settle(args) -> int:
    sess = SESS / args.session
    df = _load()
    tid = f"{args.session}-b{args.bout}"
    m = df.trade_id == tid
    if not m.any():
        print(f"{tid} not in ledger; record it first")
        return 1
    tokens = load_tokens(sess)
    teams = sorted({t for (b, t) in tokens if b == args.bout})
    ta, tb = teams[0], teams[1]
    by_asset = load_book(sess, tokens)
    bouts = load_rounds(sess)
    rb = bouts[args.bout]
    rends = [(t, w, c, s) for (t, w, c, s) in rb["rends"] if w in ("CT", "T")]
    if [c + s for (_, _, c, s) in rends[:2]] != [1, 2]:
        print("rounds 1-2 not cleanly anchored; cannot settle honestly")
        return 1
    ba, bb = Book(by_asset[tokens[(args.bout, ta)]]), \
        Book(by_asset[tokens[(args.bout, tb)]])
    t_p1, t_r2 = rends[0][0], rends[1][0]
    pre_a, pre_b = ba.plateau(t_p1 - 420, t_p1 - 240), \
        bb.plateau(t_p1 - 420, t_p1 - 240)
    if pre_a is None or pre_b is None:
        print("no pre-pistol book; cannot fill entry honestly")
        return 1
    win1, conf = price_jump(ba, bb, t_p1)
    if win1 is None:
        print(f"pistol attribution ambiguous (conf={conf:.4f}); not settled")
        return 1

    pick = df.loc[m, "user_pick"].iloc[0]
    fav_book = ba if pick == ta else bb
    entry_ask = (pre_a if pick == ta else pre_b)[1]
    entry_mid = (pre_a if pick == ta else pre_b)[2]
    limit = round(entry_ask - 0.02, 4)
    # paper fill: did the ask touch the limit inside the entry window?
    win_lo, win_hi = t_p1 - 420, t_p1 - 240
    touched = fav_book.df[(fav_book.df.recv >= win_lo)
                          & (fav_book.df.recv <= win_hi)
                          & (fav_book.df.ask <= limit)]
    filled = len(touched) > 0
    pistol_won = ((pick == ta) == (win1 == "a"))
    exit_rule = "EXIT-W" if pistol_won else "EXIT-L"
    t_exit = t_p1 if pistol_won else t_r2
    ex = fav_book.plateau(t_exit - 60, t_exit + 60)
    close = fav_book.plateau(t_p1 - 900, t_p1 - 600)
    updates = {
        "map": json.load(open(sess / "session_spec.json", encoding="utf-8")
                         ).get("maps", [{}])[args.bout - 1].get("map_name", "")
        if (sess / "session_spec.json").exists() else "",
        "entry_window_ask": f"{entry_ask:.4f}",
        "entry_limit": f"{limit:.4f}",
        "filled": str(filled),
        "close_mid": f"{close[2]:.4f}" if close else "",
        "clv_pt": f"{(close[2] - entry_mid) * 100:.2f}" if close else "",
        "pistol_won": str(pistol_won),
        "exit_rule": exit_rule,
        "exit_bid": f"{ex[0]:.4f}" if ex else "",
        "status": "settled" if ex else "pending",
    }
    if ex and filled:
        updates["pnl_pt"] = f"{(ex[0] - min(entry_ask, limit)) * 100:.2f}"
    df.loc[m, list(updates)] = list(updates.values())
    _save(df)
    print(json.dumps({"trade_id": tid, **updates}, ensure_ascii=False,
                     indent=1))
    return 0


def cmd_report(_args) -> int:
    df = _load()
    st = df[df.status == "settled"]
    print(f"recorded {len(df)} | settled {len(st)}")
    if len(st):
        f = st[st.filled == "True"]
        print(f"filled {len(f)}/{len(st)}")
        if len(f):
            pnl = f.pnl_pt.astype(float)
            print(f"pnl: mean {pnl.mean():.2f}pt  sum {pnl.sum():.2f}pt  "
                  f"win {(pnl > 0).mean():.2f}")
            print(f"pistol hit rate: {f.pistol_won.astype(str).eq('True').mean():.3f}")
            print(f"CLV mean: {f.clv_pt.astype(float).mean():.2f}pt")
        bench = st[st.filled != "True"]
        if len(bench):
            print("(unfilled rows excluded from pnl)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    r = sub.add_parser("record")
    r.add_argument("--session", required=True)
    r.add_argument("--bout", type=int, required=True)
    r.add_argument("--pick", required=True)
    r.add_argument("--notes", default="")
    r.set_defaults(fn=cmd_record)
    s = sub.add_parser("settle")
    s.add_argument("--session", required=True)
    s.add_argument("--bout", type=int, required=True)
    s.set_defaults(fn=cmd_settle)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
