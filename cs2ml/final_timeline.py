"""Raw price-vs-game-event timeline for the 2026-09-20 grand final (v2).

No analysis: per map, a chronological round-by-round log pairing every deduped
kill and round boundary with the actual token prices at that moment (last book
update, staleness-flagged). For human reading. Writes
reports/final_replay_20260920/timeline_bout<N>.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .final_replay import SESSION, OUT, load_prices, load_rosters, _ts

DETAIL = Path(r"C:\Users\33155\AppData\Local\Temp\fivee_final_detail.json")
STALE_SECONDS = 20.0


def load_full_events(rosters: dict[int, dict]) -> pd.DataFrame:
    nick_team = {n: t for b in rosters.values() for t, ns in b.items()
                 for n in ns}
    rows = []
    last_bout = None
    with open(SESSION / "fivee_events.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "event":
                continue
            entry = row.get("entry") or {}
            try:
                bout = int(entry.get("bout_num"))
                last_bout = bout
            except (TypeError, ValueError):
                bout = last_bout  # some rows (round_start) lack bout_num
            if bout is None:
                continue
            try:
                log = json.loads(entry.get("log_info") or "{}")
            except json.JSONDecodeError:
                continue
            t = str(log.get("type", ""))
            base = {"recv": _ts(row["recv_utc"]), "bout": bout}
            if t == "1":
                base["kind"] = "round_start"
            elif t == "2":
                end = log.get("round_end") or {}
                base.update(kind="round_end", ct=int(end.get("ct_score") or 0),
                            t_score=int(end.get("t_score") or 0),
                            winner_side=end.get("winner_side") or end.get("winner"),
                            win_type=end.get("win_type"))
            elif t == "8":
                k = log.get("kill") or {}
                base.update(kind="kill",
                            killer=k.get("killer_nick") or k.get("killer_name"),
                            killer_team=nick_team.get(k.get("killer_nick")
                                                      or k.get("killer_name")),
                            killer_side=k.get("killer_side"),
                            victim=k.get("victim_nick") or k.get("victim_name"),
                            victim_team=nick_team.get(k.get("victim_nick")
                                                      or k.get("victim_name")),
                            victim_side=k.get("victim_side"),
                            weapon=k.get("weapon"),
                            headshot=bool(k.get("head_shot")))
            else:
                base["kind"] = f"type_{t}"
            rows.append(base)
    events = pd.DataFrame(rows)
    # 5E poll can deliver the same kill multiple times: dedup by identity,
    # keeping the first receipt
    if len(events):
        kills = events["kind"] == "kill"
        dup = (events[kills]
               .assign(_k=events[kills]["killer"].astype(str) + ">"
                            + events[kills]["victim"].astype(str))
               .sort_values("recv")
               .drop_duplicates(subset=["bout", "_k"], keep="first"))
        events = pd.concat([events[~kills], dup.drop(columns="_k")])
    return events.sort_values("recv")


def fmt_time(epoch: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%H:%M:%S")


class PriceTracker:
    """Forward-filled last-book prices with staleness flags."""

    def __init__(self, sa: pd.Series, sv: pd.Series):
        self.sa, self.sv = sa, sv

    def at(self, t: float):
        def last(s):
            if s is None or s.empty:
                return None, None
            idx = s.index.searchsorted(t, side="right") - 1
            if idx < 0:
                return None, None
            return float(s.iloc[idx]), t - s.index[idx]
        pa, age_a = last(self.sa)
        pv, age_v = last(self.sv)
        def fmt(p, age):
            if p is None:
                return "  -- "
            mark = "~" if age > STALE_SECONDS else ""
            return f"{p:.2f}{mark}"
        return f"Aurora {fmt(pa, age_a)} / Vita {fmt(pv, age_v)}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rosters = load_rosters()
    events = load_full_events(rosters)
    prices = load_prices()
    data = json.load(open(DETAIL, encoding="utf-8"))
    map_names = {i + 1: (b.get("map_name") or "?")
                 for i, b in enumerate(data["data"]["match"]["bouts_state"])}

    for bout in (1, 2, 3, 4):
        if bout not in prices:
            continue
        price = prices[bout].sort_index()
        ev = events[events["bout"] == bout].sort_values("recv")
        sa = (price["Aurora"].dropna() if "Aurora" in price
              else pd.Series(dtype=float))
        sv = (price["Vitality"].dropna() if "Vitality" in price
              else pd.Series(dtype=float))
        tracker = PriceTracker(sa, sv)

        lines = [f"# 图{bout}（{map_names.get(bout, '?')}）价格-事件对照",
                 "（价格=最近一次盘口更新，~ = 该价格已超 20 秒未更新；时间 UTC）",
                 ""]
        round_no = 0
        score = "0:0"
        first_round_start = (ev[ev["kind"] == "round_start"]["recv"].min()
                             if (ev["kind"] == "round_start").any() else None)
        for _, e in ev.iterrows():
            if first_round_start is not None and e["recv"] < first_round_start:
                continue  # inter-map tail events, not part of this map
            if e["kind"] == "round_start":
                round_no += 1
                lines.append(
                    f"\n## 第{round_no}回合开始 {fmt_time(e['recv'])} 比分{score}  "
                    f"| {tracker.at(e['recv'])}")
            elif e["kind"] == "kill":
                hs = " 爆头" if e.get("headshot") else ""
                weapon = e.get("weapon") or "?"
                lines.append(
                    f"- {fmt_time(e['recv'])} "
                    f"{e.get('killer') or '?'}({e.get('killer_team') or '?'}|{e.get('killer_side') or '?'}) "
                    f"[{weapon}{hs}] → "
                    f"{e.get('victim') or '?'}({e.get('victim_team') or '?'}|{e.get('victim_side') or '?'}) "
                    f"| {tracker.at(e['recv'])}")
            elif e["kind"] == "round_end":
                score = f"{int(e['ct'])}:{int(e['t_score'])}"
                lines.append(
                    f"**回合结束 {fmt_time(e['recv'])}  "
                    f"CT {int(e['ct'])} : {int(e['t_score'])} T  胜 {e.get('winner_side')} "
                    f"({e.get('win_type') or '?'}) | {tracker.at(e['recv'])}**")
        (OUT / f"timeline_bout{bout}.md").write_text(
            "\n".join(lines), encoding="utf-8")
        print(f"bout {bout}: {len(lines)} lines, "
              f"{int((ev['kind'] == 'kill').sum())} kills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
