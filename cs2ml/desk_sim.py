"""Desk sim protocol — 历史回放模拟测试的出题/结算工具.

模拟测试的纪律: 报告生成前绝不接触比赛结果. 本工具的三个模式都遵守:
  pick   扫描候选 (大赛队伍、接近均价的 Map1、有录价), 只打印赛前信息
  prep   为选定场次产出 desk CLI 参数 (token→队用 slug 解析, 阵营用 demo
         ct_roster, 入场价 = 手枪前 3 分钟最后的赛前价), 不触碰 winner
  settle 赛后结算: 读 demo 真实结果, 按修正腿 (+4.4/−6.6pt) 填 sim_ledger

用法:
  python -m cs2ml.desk_sim pick
  python -m cs2ml.desk_sim prep --key <demo_key>
  python -m cs2ml.desk_sim settle --key <demo_key> --pick a|b --note "..."
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from pistol_replay import norm_team, team_roster_lut  # noqa: E402

from cs2ml import map1_data as md
from cs2ml import pistol_strategy_backtest as pb

ROOT = Path(__file__).resolve().parent.parent
SIM_LEDGER = ROOT / "reports/match_desk/sim_ledger.csv"
SIM_PREP = ROOT / "reports/match_desk/sim_prep.json"
LEG_WIN, LEG_LOSS = 4.4, 6.6

BIG = {"vitality", "furia", "g2", "falcons", "natusvincere", "spirit",
       "heroic", "faze", "mouz", "astralis", "pain", "big"}


# ------------------------------------------------------- outcome-free prep
def load_market():
    px = pd.read_parquet(pb.PX, columns=["t", "p", "match_id", "map_market",
                                         "outcome", "token"])
    px = px[px["map_market"] == "Map 1 Winner"]
    px_wide, px_tokens = {}, {}
    for mid, g in px.groupby("match_id"):
        wide = g.pivot_table(index="t", columns="outcome", values="p",
                             aggfunc="last")
        if wide.shape[1] == 2 and len(wide) >= 30:
            px_wide[mid] = wide
            px_tokens[mid] = (g.drop_duplicates("outcome")
                              .set_index("outcome")["token"].to_dict())
    pm = pd.read_parquet(pb.PM, columns=["t", "p", "token"])
    pm_series = {tok: g.set_index("t")["p"].sort_index()
                 for tok, g in pm.groupby("token")}
    return px_wide, px_tokens, pm_series


def demo_slugs(demo_path: str) -> tuple[str, str] | None:
    m = re.match(r"([a-z0-9.-]+?)-vs-([a-z0-9.-]+?)-m\d",
                 Path(str(demo_path)).stem.lower())
    return (m.group(1), m.group(2)) if m else None


def ct_team(key: str, lut: dict) -> str | None:
    """首回合 CT 侧队伍 slug (demo ct_roster, 与胜负无关)"""
    rs = pd.read_parquet(pb.ROUND_STATES,
                         columns=["demo_path", "round_num", "ct_roster"])
    r1 = rs[(rs["demo_path"].map(pb.norm_key) == key) & (rs.round_num == 1)]
    if not len(r1):
        return None
    ct = frozenset(str(r1.iloc[0]["ct_roster"]).split(","))
    best, bo = None, 0.6
    for slug, ros in lut.items():
        o = len(ct & ros) / max(len(ct | ros), 1)
        if o >= bo:
            best, bo = slug, o
    return best


def entry_prices(mid, wide, tokens, pm_series, t_pistol):
    """两 token 在手枪前 3 分钟的最后赛前价; slug 绑定 (无结算泄漏)"""
    cutoff = t_pistol - pd.Timedelta(minutes=3)
    out = {}
    for outcome, tok in tokens[mid].items():
        s = pm_series.get(tok)
        if s is not None:
            pre = s[s.index <= cutoff]
            if len(pre):
                out[outcome] = (pre.index[-1], float(pre.iloc[-1]))
                continue
        early = wide[mid][outcome].dropna()
        early = early[early.index <= cutoff]
        if len(early):
            out[outcome] = (early.index[0], float(early.iloc[0]))
    return out


def candidates():
    labels = pb.load_labels()
    lab = pd.read_parquet(pb.LABELS)
    lut = team_roster_lut(lab[lab["complete"]])
    px_wide, px_tokens, pm_series = load_market()
    tl = pb.build_round_timeline()
    tl["key"] = tl["demo_path"].map(pb.norm_key)
    r1 = tl[tl["round_num"] == 1][["key", "round_end_utc"]]

    rows = []
    for r in labels.itertuples(index=False):
        slugs = demo_slugs(r.demo_path)
        if not slugs:
            continue
        sa, sb = norm_team(slugs[0]), norm_team(slugs[1])
        if sa not in BIG or sb not in BIG:
            continue
        if r.match_id not in px_wide:
            continue
        sub = r1[r1["key"] == r.key]
        if not len(sub) or pd.isna(sub.iloc[0]["round_end_utc"]):
            continue
        wide = px_wide[r.match_id]
        side_of = {}
        for outcome in wide.columns:
            s = pb.resolve_outcome(outcome, sa, sb)
            if s:
                side_of[s] = outcome
        if set(side_of) != {"a", "b"}:
            continue
        ends = tl[tl["key"] == r.key]
        ends = ends[ends["round_num"] <= 10]["round_end_utc"].tolist()
        if len(ends) < 3:
            continue
        dm = pb.estimate_delta(wide, ends, sub.iloc[0]["round_end_utc"])
        if dm is None:
            continue
        t_pistol = sub.iloc[0]["round_end_utc"] + pd.Timedelta(minutes=dm)
        ep = entry_prices(r.match_id, px_wide, px_tokens, pm_series, t_pistol)
        if set(ep) != set(side_of.values()):
            continue
        pa = ep[side_of["a"]][1]
        if not (0.40 <= pa <= 0.65):
            continue
        rows.append({
            "key": r.key, "match_id": r.match_id, "map": r.map_name,
            "a": sa, "b": sb, "price_a": round(pa, 3),
            "price_b": round(ep[side_of["b"]][1], 3),
            "at": ep[side_of["a"]][0].isoformat(),
            "ct_first": ct_team(r.key, lut), "delta_min": dm,
        })
    return pd.DataFrame(rows)


def cmd_prep(args) -> int:
    lut = team_roster_lut(pd.read_parquet(pb.LABELS))
    c = candidates()
    row = c[c["key"] == args.key]
    if not len(row):
        raise SystemExit(f"key 不在候选池: {args.key}")
    r = row.iloc[0].to_dict()
    r["team_a"] = r["a"]
    r["team_b"] = r["b"]
    r["note"] = ("阵营(赛前公开): 首CT = " + str(r["ct_first"])
                 if r["ct_first"] else "阵营未知")
    SIM_PREP.write_text(json.dumps(r, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print("\n下一步:")
    print(f"  python -m cs2ml.match_desk --team-a {r['a']} --team-b {r['b']} "
          f"--map {str(r['map']).replace('de_','')} --price-a {r['price_a']} "
          f"--price-b {r['price_b']} --at {r['at']} --sim --record "
          f'--pick <a|b|none> --note "理由"')
    return 0


# ------------------------------------------------------------- settle
def cmd_settle(args) -> int:
    lab = pd.read_parquet(pb.LABELS)
    lut = team_roster_lut(lab[lab["complete"]])
    prep = json.loads(SIM_PREP.read_text(encoding="utf-8"))
    if prep["key"] != args.key:
        raise SystemExit(f"prep 缓存是 {prep['key']}, 与 --key {args.key} 不符")
    rs = pd.read_parquet(pb.ROUND_STATES,
                         columns=["demo_path", "round_num", "winner_side",
                                  "ct_roster"])
    m = rs[rs["demo_path"].map(pb.norm_key) == args.key]
    if not len(m):
        raise SystemExit("demo 不在 round_states")
    r1 = m[m.round_num == 1].iloc[0]
    ct_slug = prep["ct_first"]
    winner_slug = ct_slug if r1["winner_side"] == "CT" else (
        prep["team_b"] if ct_slug == prep["team_a"] else prep["team_a"])
    a_won_pistol = winner_slug == prep["team_a"]
    labels_row = lab[lab["demo_path"].map(pb.norm_key) == args.key]
    map_a = None
    if len(labels_row):
        wr = frozenset(str(labels_row.iloc[0]["winner_roster"]).split(","))
        ra = lut.get(prep["team_a"], frozenset())
        map_a = len(wr & ra) / max(len(wr | ra), 1) >= 0.6 if ra else None
    # 修正腿 (与 match_desk 的 EV 读数一致): 0.55-0.65 价格桶 +4.4/−6.6pt
    pnl = (LEG_WIN if (args.pick == "a") == a_won_pistol
           else -LEG_LOSS) if args.pick in ("a", "b") else 0.0
    result = (f"pistol_{'W' if a_won_pistol else 'L'}"
              f"_map_{'W' if map_a else 'L' if map_a is not None else '?'}")
    # 若 match_desk --record 已写入同场决策行, 就地补结算列而非追加新行
    if SIM_LEDGER.exists():
        df = pd.read_csv(SIM_LEDGER, dtype=str).fillna("")
        m = (df["ts"] == prep["at"]) & (df["team_a"] == prep["team_a"]) \
            & (df["team_b"] == prep["team_b"]) & (df["map"] == prep["map"]) \
            & (df["result_a"] == "")
        if m.any():
            df.loc[m, "result_a"] = result
            df.loc[m, "settled_pnl"] = str(round(pnl, 2))
            df.to_csv(SIM_LEDGER, index=False)
            print(f"[settle] 手枪局 {'A' if a_won_pistol else 'B'} 胜 | 地图 "
                  f"{'A' if map_a else 'B' if map_a is not None else '?'} 胜 | "
                  f"pick={df.loc[m, 'pick'].iloc[0]} → pnl {pnl:+.2f}pt (就地更新)")
            return 0
    row = {
        "ts": prep["at"], "team_a": prep["team_a"], "team_b": prep["team_b"],
        "map": prep["map"], "p_pistol_a_ct": prep["price_a"],
        "p_pistol_a_t": prep["price_b"], "p_mapwin_a": "",
        "price_a": prep["price_a"], "price_b": prep["price_b"],
        "ev_buy_a_pt": "", "ev_buy_b_pt": "",
        "pick": args.pick, "note": "【模拟】" + args.note,
        "result_a": result, "settled_pnl": round(pnl, 2),
    }
    SIM_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    exists = SIM_LEDGER.exists()
    with open(SIM_LEDGER, "a", encoding="utf-8", newline="") as h:
        if not exists:
            h.write(",".join(row) + "\n")
        h.write(",".join('"' + str(v).replace('"', "'") + '"'
                         if "," in str(v) else str(v)
                         for v in row.values()) + "\n")
    print(f"[settle] 手枪局 {'A' if a_won_pistol else 'B'} 胜 | 地图 "
          f"{'A' if map_a else 'B' if map_a is not None else '?'} 胜 | "
          f"pick={args.pick} → pnl {pnl:+.2f}pt")
    print(f"[ledger] {SIM_LEDGER} (+1)")
    return 0


# ---------------------------------------------------- mid-match scenarios
# 抄底/中途进场场景: 前提 = 热门刚输手枪 (题面给定, 不构成泄漏);
# 必须保密的是地图结果. dip 用分钟级 px (真实闪跌更深, 题面注明 caveat).
SCEN_PREP = ROOT / "reports/match_desk/scenario_prep.json"


def scenario_candidates():
    labels = pb.load_labels()
    lab = pd.read_parquet(pb.LABELS)
    lut = team_roster_lut(lab[lab["complete"]])
    px_wide, px_tokens, pm_series = load_market()
    tl = pb.build_round_timeline()
    tl["key"] = tl["demo_path"].map(pb.norm_key)
    rs = pd.read_parquet(pb.ROUND_STATES,
                         columns=["demo_path", "round_num", "winner_side",
                                  "ct_roster"])

    out = []
    for r in labels.itertuples(index=False):
        slugs = demo_slugs(r.demo_path)
        if not slugs:
            continue
        sa, sb = norm_team(slugs[0]), norm_team(slugs[1])
        if sa not in BIG or sb not in BIG or r.match_id not in px_wide:
            continue
        sub = tl[(tl["key"] == r.key) & (tl["round_num"] == 1)]
        if not len(sub) or pd.isna(sub.iloc[0]["round_end_utc"]):
            continue
        wide = px_wide[r.match_id]
        side_of = {}
        for outcome in wide.columns:
            s = pb.resolve_outcome(outcome, sa, sb)
            if s:
                side_of[s] = outcome
        if set(side_of) != {"a", "b"}:
            continue
        ends = tl[tl["key"] == r.key]
        ends = ends[ends["round_num"] <= 10]["round_end_utc"].tolist()
        if len(ends) < 3:
            continue
        dm = pb.estimate_delta(wide, ends, sub.iloc[0]["round_end_utc"])
        if dm is None:
            continue
        t_p = sub.iloc[0]["round_end_utc"] + pd.Timedelta(minutes=dm)
        ep = entry_prices(r.match_id, px_wide, px_tokens, pm_series, t_p)
        if set(ep) != set(side_of.values()):
            continue
        pa, pb_ = ep[side_of["a"]][1], ep[side_of["b"]][1]
        if abs(pa - pb_) < 0.06:      # 抄底场景要有多方热门
            continue
        fav, other = ("a", "b") if pa > pb_ else ("b", "a")
        # 手枪结果 (题面前提; 仅用于筛选场景, 出题时作为已知条件给出)
        r1 = rs[(rs["demo_path"].map(pb.norm_key) == r.key)
                & (rs["round_num"] == 1)]
        if not len(r1):
            continue
        ct_slug = ct_team(r.key, lut)
        if ct_slug is None:
            continue
        win_slug = ct_slug if r1.iloc[0]["winner_side"] == "CT" else (
            sb if ct_slug == sa else sa)
        if win_slug != (sa if fav == "a" else sb):
            continue                    # 只要热门输手枪的场
        # dip: 热门 token 在手枪后 1-4 分钟窗口的最低价 (分钟级)
        s = wide[side_of[fav]].dropna()
        win = s[(s.index >= t_p + pd.Timedelta(minutes=1))
                & (s.index <= t_p + pd.Timedelta(minutes=4))]
        if len(win) < 2:
            continue
        dip_t, dip = win.index[0], float(win.iloc[0])
        # 地图结果 (保密, 仅存档不打印)
        lr = lab[lab["demo_path"].map(pb.norm_key) == r.key]
        fav_map = None
        if len(lr):
            wr = frozenset(str(lr.iloc[0]["winner_roster"]).split(","))
            fr = lut.get(sa if fav == "a" else sb, frozenset())
            fav_map = len(wr & fr) / max(len(wr | fr), 1) >= 0.6 if fr else None
        out.append({
            "key": r.key, "map": r.map_name,
            "fav": sa if fav == "a" else sb,
            "other": sb if fav == "a" else sa,
            "entry": round(max(pa, pb_), 3), "dip": round(dip, 3),
            "dip_t": dip_t.isoformat(), "t_pistol": t_p.isoformat(),
            "pistol_loser_ct": ct_slug == (sa if fav == "a" else sb),
            "fav_map_won": fav_map,
        })
    return pd.DataFrame(out)


def cmd_scenario_pick() -> int:
    import random
    c = scenario_candidates()
    c = c[(c["dip"] <= 0.62) & (c["entry"] - c["dip"] >= 0.04)
          & (c["dip"] >= 0.30)]
    if not len(c):
        raise SystemExit("无满足条件的抄底场景")
    r = random.choice(c.to_dict("records"))
    SCEN_PREP.write_text(json.dumps(r, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[scenario] {r['fav']} (热门 {r['entry']:.2f}) 刚输手枪给 "
          f"{r['other']}, 价格闪跌至 {r['dip']:.2f} ({r['map']}, "
          f"{'热门当时为CT' if r['pistol_loser_ct'] else '热门当时为T'})")
    print("地图结果已封存, 不打印。")
    return 0


def cmd_scenario_settle(args) -> int:
    r = json.loads(SCEN_PREP.read_text(encoding="utf-8"))
    dip = r["dip"]
    pnl = ((1 - dip) if r["fav_map_won"] else -dip) \
        if args.pick == "fav" else (((1 - (1 - dip)) if not r["fav_map_won"]
                                     else -(1 - dip))
                                    if args.pick == "other" else 0.0)
    res = (f"抄底场景 {r['fav']}@{dip:.2f}: "
           f"{'赢' if r['fav_map_won'] else '输'}图")
    row = {
        "ts": r["dip_t"], "team_a": r["fav"], "team_b": r["other"],
        "map": r["map"], "p_pistol_a_ct": "", "p_pistol_a_t": "",
        "p_mapwin_a": "", "price_a": dip, "price_b": round(1 - dip, 3),
        "ev_buy_a_pt": "", "ev_buy_b_pt": "",
        "pick": args.pick, "note": "【模拟·抄底】" + args.note,
        "result_a": res, "settled_pnl": round(pnl * 100, 2),
    }
    exists = SIM_LEDGER.exists()
    with open(SIM_LEDGER, "a", encoding="utf-8", newline="") as h:
        if not exists:
            h.write(",".join(row) + "\n")
        h.write(",".join('"' + str(v).replace('"', "'") + '"'
                         if "," in str(v) else str(v)
                         for v in row.values()) + "\n")
    print(f"[settle] {res} | pick={args.pick} → pnl {pnl * 100:+.2f}pt")
    print(f"[ledger] {SIM_LEDGER} (+1)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="desk sim protocol")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pick")
    p2 = sub.add_parser("prep")
    p2.add_argument("--key", required=True)
    p3 = sub.add_parser("settle")
    p3.add_argument("--key", required=True)
    p3.add_argument("--pick", choices=["a", "b", "none"], required=True)
    p3.add_argument("--note", default="")
    sub.add_parser("scenario_pick")
    p4 = sub.add_parser("scenario_settle")
    p4.add_argument("--pick", choices=["fav", "other", "none"], required=True)
    p4.add_argument("--note", default="")
    args = ap.parse_args()
    if args.cmd == "pick":
        c = candidates()
        print(c.sort_values("price_a").to_string(index=False))
        print(f"\n{len(c)} candidates")
        return 0
    if args.cmd == "prep":
        return cmd_prep(args)
    if args.cmd == "scenario_pick":
        return cmd_scenario_pick()
    if args.cmd == "scenario_settle":
        return cmd_scenario_settle(args)
    return cmd_settle(args)


if __name__ == "__main__":
    raise SystemExit(main())
