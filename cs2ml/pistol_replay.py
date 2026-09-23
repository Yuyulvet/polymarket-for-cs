"""Paper replay of the user's pistol-swing strategy on recorded tapes.

User rule (mechanized): buy the stronger team BEFORE the pistol round; sell
into the post-pistol sentiment pop. Arms:
  A "safest":   exit +15s after pistol ends (all cases)
  B "before R3": exit just before round 3 starts (all cases)
  C "full rule": pistol WON -> exit before R3; pistol LOST -> hold through the
                 first rifle round, exit +15s after round 3 ends (stop/recovery)
  D baseline:   hold to last book of the map

SELECTION ARMS (user's correction 2026-09-22): "stronger" is NOT the market
favorite. It must come from team-specific, map-specific recent pistol win
rates (and eventually playstyle):
  M0 market: favorite = higher pre-pistol mid (reference arm, NOT the strategy)
  M1 roster pistol rate: past pistol win rate of the matched roster (all maps,
     Beta(3,3)-shrunk, past-only vs session start time)
  M2 map-blended: 50/50 blend of roster rate and map-specific pistol rate
Team name -> roster via demo-path slug + modal 5-steamid lineup (no a/b side
orientation needed); roster attribution by Jaccard overlap >= 0.6. Missing
history -> falls back to M0 and is flagged.

Fills: entry at best ASK, exits at best BID (spread paid honestly); mids too.
Pistol/round-3 winner attribution: team whose mid jumps at round-end (price
tape is self-contained). Bouts with ambiguous attribution are skipped.
Universe: every session bout with tape coverage from round 1 (12 maps).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SESS = ROOT / "data/live_sessions"
LABELS = ROOT / "data/map1/rebuild/full-20260916/map_labels.parquet"
ROUND_STATES = ROOT / "data/map1/rebuild/full-20260916/round_states.parquet"
OUT = ROOT / "reports/pistol_replay_20260922"
MAX_BOOK_AGE = 240.0


def _ts(iso: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def norm_key(path: str) -> str:
    tail = re.sub(r"\.(dem|json\.gz|parquet)$", "", str(path).lower())
    i = tail.rfind("demos")
    if i >= 0:
        tail = tail[i + len("demos"):]
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def norm_team(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower().replace("gaming", ""))


# ---------------------------------------------------------------- history
def load_pistol_history():
    """Per-roster pistol samples from the rebuild pool, time-ordered."""
    lab = pd.read_parquet(LABELS)
    lab = lab[lab["complete"]].copy()
    lab["key"] = lab["demo_path"].map(norm_key)
    lab["mtime"] = lab["demo_path"].map(lambda p: Path(p).stat().st_mtime)
    rs = pd.read_parquet(ROUND_STATES,
                         columns=["demo_path", "round_num", "ct_is_a",
                                  "winner_side"])
    rs["key"] = rs["demo_path"].map(norm_key)
    rs = rs[rs["round_num"].isin([1, 13])].drop_duplicates(
        subset=["key", "round_num"])
    rs["a_won"] = ((rs["winner_side"] == "CT") == rs["ct_is_a"]).astype(int)
    rows = []
    for r in rs.merge(lab[["key", "map_name", "mtime",
                           "roster_a", "roster_b"]], on="key").itertuples():
        ra = frozenset(str(r.roster_a).split(","))
        rb = frozenset(str(r.roster_b).split(","))
        rows.append({"roster": ra, "map": r.map_name, "mtime": r.mtime,
                     "won": int(r.a_won)})
        rows.append({"roster": rb, "map": r.map_name, "mtime": r.mtime,
                     "won": 1 - int(r.a_won)})
    return pd.DataFrame(rows)


def team_roster_lut(lab: pd.DataFrame) -> dict[str, frozenset]:
    """team slug -> modal 5-steamid roster (from demo path team names)."""
    lut: dict[str, Counter] = {}
    for r in lab.itertuples(index=False):
        base = Path(str(r.demo_path)).stem.lower()
        m = re.match(r"([a-z0-9.-]+?)-vs-([a-z0-9.-]+?)-m\d", base)
        if not m:
            continue
        ra = frozenset(str(r.roster_a).split(","))
        rb = frozenset(str(r.roster_b).split(","))
        if len(ra) == 5:
            lut.setdefault(norm_team(m.group(1)), Counter())[ra] += 1
        if len(rb) == 5:
            lut.setdefault(norm_team(m.group(2)), Counter())[rb] += 1
    return {t: c.most_common(1)[0][0] for t, c in lut.items() if c}


def roster_rate(hist, roster, map_name, t_now, map_weight=0.0):
    """Shrunk pistol win rate of rosters overlapping `roster` before t_now."""
    def overlap(r):
        return len(r & roster) / max(len(r | roster), 1)
    h = hist[(hist.mtime < t_now)]
    h = h[h.roster.map(overlap) >= 0.6]
    if not len(h):
        return None
    w, n = h.won.sum(), len(h)
    overall = (w + 1.5) / (n + 3.0)
    if map_weight <= 0:
        return overall, int(n)
    hm = h[h["map"] == map_name]
    if len(hm) >= 3:
        wm, nm = hm.won.sum(), len(hm)
        mrate = (wm + 1.5) / (nm + 3.0)
    else:
        mrate = overall
    return (1 - map_weight) * overall + map_weight * mrate, int(n)


# ---------------------------------------------------------------- tapes
def load_tokens(sess: Path) -> dict[tuple[int, str], str]:
    meta = json.load(open(sess / "market/metadata.json", encoding="utf-8"))
    out = {}
    for m in meta.get("markets", []):
        mt = re.search(r"Map\s+(\d+)", m.get("market_name") or "")
        if not mt:
            continue
        for team, asset in (m.get("tokens") or {}).items():
            out[(int(mt.group(1)), str(team).replace(" Gaming", ""))] = str(asset)
    return out


def load_book(sess, tokens):
    assets = set(tokens.values())
    rows: dict[str, list] = {a: [] for a in assets}
    with open(sess / "market/events.jsonl", encoding="utf-8") as h:
        for line in h:
            row = json.loads(line)
            if row.get("type") != "market_raw":
                continue
            recv = _ts(row["received_at"])
            try:
                payload = json.loads(row.get("raw") or "null")
            except json.JSONDecodeError:
                continue
            for msg in (payload if isinstance(payload, list) else [payload]):
                if not isinstance(msg, dict) or "bids" not in msg:
                    continue
                a = str(msg.get("asset_id"))
                if a not in assets:
                    continue
                bids = [float(b["price"]) for b in msg["bids"] if b]
                asks = [float(x["price"]) for x in msg["asks"] if x]
                if bids and asks:
                    rows[a].append((recv, max(bids), min(asks)))
    out = {}
    for a, r in rows.items():
        if r:
            df = pd.DataFrame(r, columns=["recv", "bid", "ask"]).drop_duplicates()
            df["mid"] = (df.bid + df.ask) / 2
            out[a] = df.sort_values("recv").reset_index(drop=True)
    return out


def load_rounds(sess: Path) -> dict[int, dict]:
    bouts: dict[int, dict] = {}
    last_bout = None
    with open(sess / "fivee_events.jsonl", encoding="utf-8") as h:
        for line in h:
            row = json.loads(line)
            if row.get("record_type") != "event":
                continue
            e = row.get("entry") or {}
            try:
                bout = int(e.get("bout_num"))
                last_bout = bout
            except (TypeError, ValueError):
                bout = last_bout
            if bout is None:
                continue
            try:
                log = json.loads(e.get("log_info") or "{}")
            except json.JSONDecodeError:
                continue
            t = str(log.get("type"))
            b = bouts.setdefault(bout, {"rstarts": [], "rends": []})
            if t == "1":
                b["rstarts"].append(_ts(row["recv_utc"]))
            elif t == "2":
                end = log.get("round_end") or {}
                b["rends"].append((_ts(row["recv_utc"]),
                                   end.get("winner") or end.get("winner_side"),
                                   int(end.get("ct_score") or 0),
                                   int(end.get("t_score") or 0)))
    for b in bouts.values():
        b["rstarts"].sort()
        b["rends"].sort()
    return bouts


class Book:
    def __init__(self, df):
        self.df = df

    def at(self, t):
        i = self.df.recv.searchsorted(t, side="right") - 1
        if i < 0:
            return None
        r = self.df.iloc[i]
        if t - r.recv > MAX_BOOK_AGE:
            return None
        return float(r.bid), float(r.ask), float(r.mid), t - r.recv

    def plateau(self, t0, t1, min_updates=3):
        """Median bid/ask/mid over a window (market-time fill zone)."""
        w = self.df[(self.df.recv >= t0) & (self.df.recv <= t1)]
        if len(w) < min_updates:
            return None
        return (float(w.bid.median()), float(w.ask.median()),
                float(w.mid.median()), len(w))


def price_jump(book_a, book_b, t):
    """Pistol winner from plateau drift: post-pistol level vs pre-pistol
    level per team. Robust to the market leading the 5E poll by 1-2 min
    (point-in-time jumps around t are NOT reliable)."""
    def m(book):
        pre = book.plateau(t - 420, t - 240)
        post = book.plateau(t - 60, t + 60)
        if pre is None or post is None:
            return None
        return post[2] - pre[2]
    da, db = m(book_a), m(book_b)
    if da is None or db is None:
        return None, 0.0
    conf = abs(da - db)
    if conf < 0.005:
        return None, conf
    return ("a" if da > db else "b"), conf


# ---------------------------------------------------------------- replay
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hist = load_pistol_history()
    lab = pd.read_parquet(LABELS, columns=["demo_path", "roster_a", "roster_b"])
    lut = team_roster_lut(lab)
    print(f"history pistol samples: {len(hist)} | team lut size: {len(lut)}")

    trades = []
    for sess in sorted(SESS.iterdir()):
        if not (sess / "fivee_events.jsonl").exists():
            continue
        spec_p = sess / "session_spec.json"
        if not spec_p.exists():
            continue
        spec = json.load(open(spec_p, encoding="utf-8"))
        map_names = {int(m["bout_num"]): m.get("map_name")
                     for m in spec.get("maps", [])}
        try:
            t_now = _ts(json.load(open(sess / "market/metadata.json",
                                       encoding="utf-8"))["started_at"])
        except (KeyError, FileNotFoundError):
            t_now = 1e18  # no start time -> use all history
        tokens = load_tokens(sess)
        if not tokens:
            continue
        by_asset = load_book(sess, tokens)
        bouts = load_rounds(sess)
        for bout, rb in sorted(bouts.items()):
            rends = [(t, w, c, s) for (t, w, c, s) in rb["rends"]
                     if w in ("CT", "T")]
            # need rounds 1-2 present with sequential scores; round-3 anchor
            # used when available, else a fixed +6min proxy for the C leg
            if [c + s for (_, _, c, s) in rends[:2]] != [1, 2]:
                continue
            teams = {t2 for (b2, t2) in tokens if b2 == bout}
            if len(teams) != 2:
                continue
            ta, tb = sorted(teams)
            ba = Book(by_asset[tokens[(bout, ta)]])
            bb = Book(by_asset[tokens[(bout, tb)]])
            t_p1 = rends[0][0]          # pistol end (score reaches 1)
            t_r2 = rends[1][0]          # round 2 end
            r3_ok = ([c + s for (_, _, c, s) in rends[:3]] == [1, 2, 3])
            t_r3e = rends[2][0] if r3_ok else t_p1 + 360.0
            t_end = rends[-1][0]
            # fills are medians over market-time windows: the tape shows the
            # market learns round outcomes ~1-2 min BEFORE the 5E poll reports
            # them, so pre-pop entry sits well before t_p1 and post-pop exits
            # straddle the 5E timestamp.
            pre_a = ba.plateau(t_p1 - 420, t_p1 - 240)
            pre_b = bb.plateau(t_p1 - 420, t_p1 - 240)
            if pre_a is None or pre_b is None or pre_a[2] == pre_b[2]:
                continue  # no fresh book -> cannot fill entry honestly
            win1, conf1 = price_jump(ba, bb, t_p1)
            if win1 is None:
                continue
            win3, _ = price_jump(ba, bb, t_r3e)
            map_name = map_names.get(bout)

            # selections
            sel = {}
            sel["M0_market"] = ta if pre_a[2] > pre_b[2] else tb
            rates = {}
            for tname in (ta, tb):
                r = lut.get(norm_team(tname))
                rates[tname] = (roster_rate(hist, r, map_name, t_now, 0.0)
                                if r else None)
            ok = {t: v for t, v in rates.items() if v is not None}
            if len(ok) == 2:
                sel["M1_pistol_rate"] = max(ok, key=lambda t: ok[t][0])
                ok2 = {}
                for tname in (ta, tb):
                    r = lut.get(norm_team(tname))
                    ok2[tname] = roster_rate(hist, r, map_name, t_now, 0.5)
                sel["M2_map_blend"] = max(ok2, key=lambda t: ok2[t][0])
            elif len(ok) == 1:
                only = next(iter(ok))
                sel["M1_pistol_rate"] = only  # other team unknown -> pick known
                sel["M2_map_blend"] = only
            for sname, fav in sel.items():
                bf = ba if fav == ta else bb
                entry_bid, entry_ask, entry_mid, _ = pre_a if fav == ta else pre_b
                pistol_won = ((fav == ta) == (win1 == "a"))
                r3_won = None if win3 is None else ((fav == ta) == (win3 == "a"))

                def exit_plateau(t0, t1):
                    got = bf.plateau(t0, t1)
                    if got is None:
                        return (np.nan, np.nan)
                    return got[0], got[2]  # median bid, median mid

                exits = {
                    "A": exit_plateau(t_p1 - 60, t_p1 + 60),
                    "B": exit_plateau(t_r2 - 60, t_r2 + 60),
                    "C": (exit_plateau(t_p1 - 60, t_p1 + 60) if pistol_won
                          else exit_plateau(t_r3e - 60, t_r3e + 60)),
                    "D": exit_plateau(t_end - 60, t_end + 300),
                }
                row = {"session": sess.name, "bout": bout, "map": map_name,
                       "selection": sname, "favorite": fav,
                       "other": tb if fav == ta else ta,
                       "entry_bid": entry_bid, "entry_ask": entry_ask,
                       "entry_mid": entry_mid,
                       "hist_n": (rates.get(fav) or (None, 0))[1],
                       "pistol_won": pistol_won, "r3_won": r3_won,
                       "conf1": round(conf1, 4)}
                for arm, (xbid, xmid) in exits.items():
                    row[f"{arm}_pnl_spread"] = xbid - entry_ask
                    row[f"{arm}_pnl_mid"] = xmid - entry_mid
                trades.append(row)
    df = pd.DataFrame(trades)
    df.to_csv(OUT / "trades.csv", index=False)
    print(f"trade rows: {len(df)} ({df[['session','bout']].drop_duplicates().shape[0]} maps x selections)")

    summary = {}
    for sname, g in df.groupby("selection"):
        entry = {"maps": int(g[["session", "bout"]].drop_duplicates().shape[0]),
                 "pistol_hit_rate": round(float(g.pistol_won.mean()), 3)}
        for arm in "ABCD":
            s = g[f"{arm}_pnl_spread"].dropna()
            entry[f"{arm}_spread_mean"] = round(float(s.mean()), 4)
            entry[f"{arm}_spread_total"] = round(float(s.sum()), 4)
            entry[f"{arm}_winrate"] = round(float((s > 0).mean()), 3)
        summary[sname] = entry
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    cols = ["session", "bout", "map", "selection", "favorite", "hist_n",
            "pistol_won", "A_pnl_mid", "B_pnl_mid", "C_pnl_mid", "C_pnl_spread"]
    print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
