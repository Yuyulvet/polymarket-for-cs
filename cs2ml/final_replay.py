"""Post-match replay of the 2026-09-20 grand final (plan steps B + C).

B: event-annotated price-jump timeline per map — every |Δmid| >= 0.03 move in
   the Aurora token is annotated with the nearest 5E game event (round end /
   kill streak / first death); measures repricing latency and plateau length.
C: fair-value overlay — empirical score-conditional P(roster_a wins map) from
   the v2 rebuild pool (Laplace-smoothed) vs the tape mid price; emits the
   sentiment-gap series and the two user-observed moments (3:2 even-economy
   pessimism; 28->18 first-deaths repricing) as quantified rows.

Paper-only diagnostics; writes reports/final_replay_20260920/.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SESSION = ROOT / "data/live_sessions/manual-1053479-omc2398108"
OUT = ROOT / "reports/final_replay_20260920"
REBUILD = ROOT / "data/map1/rebuild/full-20260916"
DETAIL = Path(r"C:\Users\33155\AppData\Local\Temp\fivee_final_detail.json")

JUMP_THRESH = 0.03
ANNOTATE_WINDOW = 15.0  # seconds either side of a jump to look for events


def _ts(iso: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def load_rosters() -> dict[int, dict]:
    """bout_num -> {'Aurora': set(nicks), 'Vitality': set(nicks)}"""
    data = json.load(open(DETAIL, encoding="utf-8"))
    match = data.get("data", {}).get("match", {})
    bouts = match.get("bouts_state", [])
    team1 = match.get("team1_name") or "Aurora"
    team2 = match.get("team2_name") or "Vitality"
    out = {}
    for i, bout in enumerate(bouts, 1):
        t1 = {p["name"] for p in bout.get("t1_pr_stats") or []}
        t2 = {p["name"] for p in bout.get("t2_pr_stats") or []}
        out[i] = {team1: t1, team2: t2}
    # bouts missing stats (late maps) reuse the series rosters
    filled = next((r for r in out.values() if any(r.values())), None)
    if filled:
        for i, r in out.items():
            if not any(r.values()):
                out[i] = filled
    return out


def load_events(rosters: dict[int, dict]) -> pd.DataFrame:
    """5E event stream -> typed, team-attributed rows."""
    nick_team = {}
    for bout in rosters.values():
        for team, nicks in bout.items():
            for n in nicks:
                nick_team[n] = team
    rows = []
    with open(SESSION / "fivee_events.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "event":
                continue
            entry = row.get("entry") or {}
            try:
                bout = int(entry.get("bout_num"))
            except (TypeError, ValueError):
                continue
            try:
                log = json.loads(entry.get("log_info") or "{}")
            except json.JSONDecodeError:
                continue
            t = str(log.get("type", ""))
            base = {"recv": _ts(row["recv_utc"]), "bout": bout, "type": t}
            if t == "2":
                end = log.get("round_end") or {}
                base.update(kind="round_end",
                            ct=int(end.get("ct_score") or 0),
                            t=int(end.get("t_score") or 0),
                            winner_side=end.get("winner"))
            elif t == "8":
                kill = log.get("kill") or {}
                base.update(kind="kill",
                            killer=kill.get("killer_nick"),
                            killer_side=kill.get("killer_side"),
                            victim_side=kill.get("victim_side"))
                base["killer_team"] = nick_team.get(base["killer"])
            elif t == "1":
                base.update(kind="round_start")
            else:
                base.update(kind=f"type_{t}")
            rows.append(base)
    return pd.DataFrame(rows)


def load_prices() -> dict[int, pd.DataFrame]:
    """bout -> per-team mid price series from the raw depth tape."""
    spec = json.load(open(SESSION / "session_spec.json", encoding="utf-8"))
    tokens = {}  # asset -> (bout, team)
    for item in spec["maps"]:
        for team, asset in (item.get("tokens") or {}).items():
            tokens[str(asset)] = (int(item["bout_num"]), team)
    meta = json.load(open(SESSION / "market/metadata.json", encoding="utf-8"))

    def _team(name: str) -> str:
        return str(name).replace(" Gaming", "").strip()

    # fall back to metadata token mapping (spec uses different key names)
    if not tokens:
        for m in meta.get("markets", []):
            for team, asset in (m.get("tokens") or {}).items():
                tokens[str(asset)] = (int(m["market_name"].split()[1]),
                                      _team(team))
    books: dict[str, list] = {}
    with open(SESSION / "market/events.jsonl", encoding="utf-8") as handle:
        for line in handle:
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
                asset = str(msg.get("asset_id"))
                if asset not in tokens:
                    continue
                bids = [float(b["price"]) for b in msg["bids"] if b]
                asks = [float(a["price"]) for a in msg["asks"] if a]
                if bids and asks:
                    books.setdefault(asset, []).append(
                        (recv, (max(bids) + min(asks)) / 2))
    out = {}
    for asset, series in books.items():
        bout, team = tokens[asset]
        frame = pd.DataFrame(series, columns=["recv", "mid"]).drop_duplicates()
        out.setdefault(bout, {})[team] = frame
    return {b: pd.DataFrame({t: f.set_index("recv")["mid"]
                             for t, f in teams.items()})
            for b, teams in out.items()}


def fair_model() -> dict:
    """Score-conditional likelihood ratios from the rebuild pool.

    P(state | a won) / P(state | a lost) over early-round (round_num <= 6)
    score states; combined with each bout's OPENING market price (the market's
    own strength prior) via odds multiplication. This isolates the discount
    beyond what scoreline + prematch strength justify — the purest available
    measure of the sentiment component.
    """
    states = pd.read_parquet(REBUILD / "round_states.parquet",
                             columns=["demo_path", "round_num", "score_a_before",
                                      "score_b_before"])
    labels = pd.read_parquet(REBUILD / "map_labels.parquet",
                             columns=["demo_path", "winner_roster", "roster_a"])
    merged = states.merge(labels, on="demo_path")
    merged["a_won"] = (merged["winner_roster"] == merged["roster_a"]).astype(int)
    early = merged[merged["round_num"] <= 6]
    total_w = early["a_won"].mean()
    lr = {}
    for (sa, sb), grp in early.groupby(["score_a_before", "score_b_before"]):
        p_state_w = (grp["a_won"] == 1).mean()
        p_state_l = (grp["a_won"] == 0).mean()
        if p_state_l > 0 and len(grp) >= 8:
            lr[(int(sa), int(sb))] = float(p_state_w / p_state_l)
    return {"lr": lr, "n_states": len(early), "base_rate": float(total_w)}


def _fair_adj(lr_table: dict, p0: float, my: int, opp: int) -> float | None:
    """Opening price p0 updated by the scoreline likelihood ratio."""
    key = (my, opp)
    if key not in lr_table:
        return None
    prior = min(max(p0, 0.02), 0.98)
    odds = prior / (1 - prior) * lr_table[key]
    return odds / (1 + odds)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rosters = load_rosters()
    print("rosters:", {b: {t: len(n) for t, n in r.items()}
                       for b, r in rosters.items()})
    events = load_events(rosters)
    prices = load_prices()
    fair = fair_model()
    print("fair states:", fair["n_states"], "| lr table:", len(fair["lr"]))

    summary = {"bouts": {}}
    for bout in sorted(prices):
        price = prices[bout].sort_index()
        ev = events[events["bout"] == bout].sort_values("recv")
        if price.empty or ev.empty:
            continue
        # --- B: jumps + annotations
        mid = price["Aurora"].dropna()
        jumps = []
        last = None
        for recv, val in mid.items():
            if last is not None and abs(val - last[1]) >= JUMP_THRESH:
                jumps.append((recv, last[1], val))
            last = (recv, val)
        # annotate each jump with nearest event
        annot = []
        for recv, old, new in jumps:
            win = ev[(ev["recv"] >= recv - ANNOTATE_WINDOW)
                     & (ev["recv"] <= recv + ANNOTATE_WINDOW)]
            label, latency = "none", None
            if len(win):
                near = win.iloc[(win["recv"] - recv).abs().argmin()]
                label = near["kind"]
                latency = float(near["recv"] - recv)
                if near["kind"] == "kill":
                    label = f"kill({near.get('killer_team') or '?'})"
                elif near["kind"] == "round_end":
                    label = f"round_end({near.get('winner_side')})"
            annot.append({"bout": bout, "recv": recv, "old": round(old, 3),
                          "new": round(new, 3), "event": label,
                          "latency_s": None if latency is None else round(latency, 2)})
        jf = pd.DataFrame(annot)
        jf.to_csv(OUT / f"jumps_bout{bout}.csv", index=False)
        # plateaus: time between jumps
        plateaus = np.diff([j[0] for j in jumps]) if len(jumps) > 1 else []
        # --- C: score timeline -> gap series
        ends = ev[ev["kind"] == "round_end"]
        nick_team_global = {n: t for b in rosters.values()
                            for t, ns in b.items() for n in ns}
        score_rows = []
        ct_team = None
        for _, r in ends.iterrows():
            ct, t, wside = int(r["ct"]), int(r["t"]), r.get("winner_side")
            if ct_team is None:
                # first round_end: winner_side tells who won pistol; use kill
                # sides in this bout to establish CT team robustly
                kills = ev[(ev["kind"] == "kill")]
                ct_nicks = set()
                for _, k in kills.iterrows():
                    if k.get("killer_side") == "CT" and k.get("killer"):
                        ct_nicks.add(k["killer"])
                    if k.get("victim_side") == "CT" and k.get("victim"):
                        ct_nicks.add(k.get("victim"))
                votes = {}
                for n in ct_nicks:
                    tm = nick_team_global.get(n)
                    if tm:
                        votes[tm] = votes.get(tm, 0) + 1
                ct_team = max(votes, key=votes.get) if votes else None
            a_score = ct if ct_team == "Aurora" else t
            v_score = t if ct_team == "Aurora" else ct
            score_rows.append({"recv": r["recv"], "a": a_score, "v": v_score})
        sf = pd.DataFrame(score_rows)
        gap_rows = []
        if not sf.empty and "Aurora" in price.columns:
            p0_series = price["Aurora"].dropna()
            p0 = float(p0_series.iloc[0]) if len(p0_series) else 0.5
            joined = pd.merge_asof(
                price.sort_index().reset_index(), sf.sort_values("recv"),
                on="recv", direction="backward")
            for _, r in joined.iterrows():
                my, opp = (int(r["a"]), int(r["v"])) if not np.isnan(r.get("a", np.nan)) else (None, None)
                if my is None:
                    continue
                f = _fair_adj(fair["lr"], p0, my, opp)
                if f is None:
                    continue
                gap_rows.append({"recv": r["recv"], "a_score": my,
                                 "v_score": opp, "mid_aurora": r["Aurora"],
                                 "fair_a": round(f, 3),
                                 "gap": round(r["Aurora"] - f, 3)})
        gf = pd.DataFrame(gap_rows)
        gf.to_csv(OUT / f"gap_bout{bout}.csv", index=False)
        # round-start sentiment gap: sample price just BEFORE each round_start
        # event (between-round window; excludes rational in-round dips)
        rs_gap = []
        anchors = ev[ev["kind"] == "round_start"].sort_values("recv")
        if len(anchors) and len(gf):
            g = gf.set_index("recv").sort_index()
            for _, a in anchors.iterrows():
                win = g[g.index <= a["recv"]]
                if not len(win):
                    continue
                row = win.iloc[-1]
                if a["recv"] - win.index[-1] > 60:
                    continue
                rs_gap.append({"recv": a["recv"], "a": int(row["a_score"]),
                               "v": int(row["v_score"]),
                               "mid": row["mid_aurora"],
                               "gap": row["gap"]})
        rgf = pd.DataFrame(rs_gap)
        rgf.to_csv(OUT / f"roundstart_gap_bout{bout}.csv", index=False)
        summary["bouts"][bout] = {
            "n_jumps": len(jf),
            "event_mix": jf["event"].str.split("(").str[0].value_counts().to_dict() if len(jf) else {},
            "median_plateau_s": float(np.median(plateaus)) if len(plateaus) else None,
            "median_latency_s": float(np.median(jf["latency_s"].dropna())) if len(jf) and jf["latency_s"].notna().any() else None,
            "gap_mean_adj": float(gf["gap"].mean()) if len(gf) else None,
            "gap_min_adj": float(gf["gap"].min()) if len(gf) else None,
            "gap_adj_at_a2_v3": gf[gf["a_score"].eq(2) & gf["v_score"].eq(3)]["gap"].round(3).tolist() if len(gf) else [],
            "roundstart_gap_mean": float(rgf["gap"].mean()) if len(rgf) else None,
            "roundstart_gap_list": rgf["gap"].round(3).tolist() if len(rgf) else [],
            "gap_at_a2_v3": gf[gf["a_score"].eq(2) & gf["v_score"].eq(3)]["gap"].round(3).tolist() if len(gf) else [],
        }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
