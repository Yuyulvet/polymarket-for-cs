"""赛前选品分析台 (match desk) — 人机分工的读局工具.

用户的选品原则: 只挑大赛 + 实力差距大/有把握的对阵; 市场定价可能含粉丝情感
资金, 价格或偏离合理概率. 本工具算两件事返给用户:
  (a) 结合近期状态 + 当下地图的合理手枪局胜率 (v3 特征管道 in-sample 评分);
  (b) 两队在该地图可能打法的概率统计 (手枪局画像: 下包/节奏/首杀/补枪/侵略性).
用户看报告后自行决定买卖, --record 把决策写进前向账本.

诚实性约束 (写进报告头, 不可省略):
  - 模型置信无增量 (v3 OOF AUC~0.48, form-strong 天花板 53.4%) — 桌面是读局
    辅助不是预测器, p 值只作基准参照, 最终判断权在人;
  - 所有统计 past-only (available_at < decision_at, 90 天半衰);
  - in-sample 拟合有乐观偏差; 样本 <8 的桶显示原始 n.

CLI:
  python -m cs2ml.match_desk --team-a g2 --team-b falcons --map mirage \
      [--price-a 0.62] [--price-b 0.41] [--at 2026-09-23T17:00Z] \
      [--record --pick a --note "理由"]
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)          # pm3/pistol_replay use bare imports
import pistol_model_v2 as pm2
import pistol_model_v3 as pm3
from pistol_replay import norm_team, team_roster_lut

from . import map1_data as md
from . import map1_model as mm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/match_desk"
HISTORY_PQ = ROOT / "data/map1/history.parquet"
FEATURES_PQ = ROOT / "data/map1/features.parquet"
V3_SAMPLES = ROOT / "reports/pistol_model_v3/samples.csv"
INROUND_PQ = ROOT / "data/map1/rebuild/full-20260916/inround_events.parquet"
LEDGER = OUT / "ledger.csv"
LABELS, ROUND_STATES, KILLS = pm2.LABELS, pm2.ROUND_STATES, pm2.KILLS

V3_FULL = pm3.VARIANTS["v3_full"]
# 修正腿(pt): 0.55-0.65 价格桶的真实赢跳/输跌 (用户纠正后的分桶统计)
LEG_WIN_POP, LEG_LOSS_DROP = 4.4, 6.6
OOF_AUC = 0.48          # v3 expanding-window OOF (reports/pistol_model_v3)
CEILING = 0.534         # form-strong 手枪命中率天花板
MIN_N = 8               # 低于此样本量显示原始 n, 不装精确


# ----------------------------------------------------------------- caches
def _src_mtime() -> float:
    mt = max(p.stat().st_mtime for p in
             (LABELS, ROUND_STATES, KILLS, V3_SAMPLES))
    for p in sorted((ROOT / "data/csnet_features").glob("*.parquet")):
        mt = max(mt, p.stat().st_mtime)
    return mt


def build_states_cache() -> dict:
    """Replay pm3.build_rows' history loop, snapshotting state BEFORE each map.

    Snapshots let us score an arbitrary new matchup at any decision time with
    exactly the past-only state pm3 itself would have used. Formulas kept
    identical to pm3.build_rows on purpose.
    """
    lab, rs = pm3.load_base()
    cs = pm3.load_csnet()
    name_lut = (cs.drop_duplicates(["key", "steamid"])
                .set_index(["key", "steamid"])["player"].to_dict())
    kills = pd.read_parquet(KILLS)
    kills["key"] = kills["demo_path"].map(pm3.norm_key)
    keep = set(lab["key"])
    kills = kills[kills["key"].isin(keep)]
    kills["aname"] = kills.set_index(
        ["key", kills["attacker"].astype(str)]).index.map(name_lut)
    kills["vname"] = kills.set_index(
        ["key", kills["victim"].astype(str)]).index.map(name_lut)
    pist = kills[kills["round_num"].isin([1, 13])]
    kd_k = kills.dropna(subset=["aname"]).groupby(["key", "aname"]).size()
    kd_d = kills.dropna(subset=["vname"]).groupby(["key", "vname"]).size()
    fk = (pist[pist["is_first_kill"] == True]          # noqa: E712
          .dropna(subset=["aname"]).groupby(["key", "aname"]).size())
    tk = (pist[pist["trade"] == True]                   # noqa: E712
          .dropna(subset=["aname"]).groupby(["key", "aname"]).size())
    cs_all = cs.groupby(["key", "player"])[pm3.CS].mean().reset_index()

    hist_cs: dict = {}
    hist_k: dict = {}
    hist_d: dict = {}
    hist_pw: dict = {}
    hist_pn: dict = {}
    hist_fk: dict = {}
    hist_tk: dict = {}
    map_hist: dict = {}
    snapshots: list = []
    cs_keys = set(cs_all["key"])

    for r in lab.sort_values("mtime").itertuples(index=False):
        key = r.key
        rr = rs[rs["key"] == key]
        if not len(rr) or key not in cs_keys:
            continue
        sub = cs[cs["key"] == key]
        lut = dict(zip(sub["player"], sub["steamid"]))

        def names_of(roster):
            out = []
            for sid in str(roster).split(","):
                nm = next((n for n, s in lut.items() if s == sid), None)
                if nm:
                    out.append(nm)
            return out

        na, nb = names_of(r.roster_a), names_of(r.roster_b)
        snapshots.append((float(r.mtime), {
            "hist_cs": copy.deepcopy(hist_cs),
            "hist_k": dict(hist_k), "hist_d": dict(hist_d),
            "hist_pw": dict(hist_pw), "hist_pn": dict(hist_pn),
            "hist_fk": dict(hist_fk), "hist_tk": dict(hist_tk),
            "map_hist": copy.deepcopy(map_hist),
        }))

        for _, pm_row in cs_all[cs_all["key"] == key].iterrows():
            hist_cs[pm_row["player"]] = pm_row[pm3.CS].values.astype(float)
        for nm, v in kd_k.get(key, pd.Series(dtype=float)).items():
            hist_k[nm] = hist_k.get(nm, 0.0) + float(v)
        for nm, v in kd_d.get(key, pd.Series(dtype=float)).items():
            hist_d[nm] = hist_d.get(nm, 0.0) + float(v)
        fk_m = fk.get(key, pd.Series(dtype=float))
        tk_m = tk.get(key, pd.Series(dtype=float))
        for nm in set(na) | set(nb):
            hist_pn[nm] = hist_pn.get(nm, 0.0) + len(rr)
            hist_pw[nm] = hist_pw.get(nm, 0.0) \
                + sum(int(pr["a_won"]) for _, pr in rr.iterrows() if nm in na) \
                + sum(1 - int(pr["a_won"]) for _, pr in rr.iterrows() if nm in nb)
        for nm, v in fk_m.items():
            hist_fk[nm] = hist_fk.get(nm, 0.0) + float(v)
        for nm, v in tk_m.items():
            hist_tk[nm] = hist_tk.get(nm, 0.0) + float(v)
        w, n = map_hist.get(r.map_name, [0.0, 0.0])
        for _, pr in rr.iterrows():
            w += float(pr["winner_side"] == "CT")
            n += 1.0
        map_hist[r.map_name] = [w, n]

    # modal player name per steamid (for roster listing; steamid splits are
    # merged by name upstream — see cs2-player-steamid-split)
    sid_names = (cs.drop_duplicates(["steamid", "player"])
                 .groupby("steamid")["player"]
                 .agg(lambda s: s.value_counts().index[0]).to_dict())
    return {"src_mtime": _src_mtime(), "snapshots": snapshots,
            "sid_names": sid_names}


def load_states() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "pm3_states_cache.pkl"
    mt = _src_mtime()
    if cache.exists():
        blob = pickle.load(open(cache, "rb"))
        if blob.get("src_mtime") == mt:
            return blob
    blob = build_states_cache()
    blob["src_mtime"] = mt
    with open(cache, "wb") as h:
        pickle.dump(blob, h)
    return blob


def state_at(snapshots: list, at_ts: float) -> dict | None:
    best = None
    for mt, st in snapshots:
        if mt < at_ts:
            best = st
        else:
            break
    return best


# --------------------------------------------------------------- scoring
def _form(names, hk, hd):
    k = sum(hk.get(x, 0.0) for x in names)
    d = sum(hd.get(x, 0.0) for x in names)
    return float(np.log((k + 1.0) / (d + 1.0))) if (k + d) else None


def _prate(names, hpw, hpn):
    w = sum(hpw.get(x, 0.0) for x in names)
    n = sum(hpn.get(x, 0.0) for x in names)
    return (w + 3.0) / (n + 6.0) if n else None


def _rate(names, num, den):
    vals = [num.get(n, 0.0) / den[n] for n in names
            if n in den and den[n] > 0]
    return float(np.mean(vals)) if len(vals) == len(names) and vals else None


def score_pistol_row(st: dict, na: list, nb: list, arena: str,
                     a_ct: int) -> dict:
    """v3_full feature row for a NEW matchup at snapshot st (pm3 formulas)."""
    row: dict = {"is_ct": a_ct}
    fa, fb = _form(na, st["hist_k"], st["hist_d"]), \
        _form(nb, st["hist_k"], st["hist_d"])
    if fa is not None and fb is not None:
        row["form_gap"] = fa - fb
    pa, pb = _prate(na, st["hist_pw"], st["hist_pn"]), \
        _prate(nb, st["hist_pw"], st["hist_pn"])
    if pa is not None and pb is not None:
        row["pistol_rate_gap"] = pa - pb
    fka, fkb = _rate(na, st["hist_fk"], st["hist_pn"]), \
        _rate(nb, st["hist_fk"], st["hist_pn"])
    if fka is not None and fkb is not None:
        row["fk_gap"] = fka - fkb
    tka, tkb = _rate(na, st["hist_tk"], st["hist_pn"]), \
        _rate(nb, st["hist_tk"], st["hist_pn"])
    if tka is not None and tkb is not None:
        row["tk_gap"] = tka - tkb
    w, n = st["map_hist"].get(arena, [0.0, 0.0])
    prior = (w + 5.5) / (n + 10.0) if n else 0.55
    row["side_prior"] = prior if a_ct else 1.0 - prior
    ta = [st["hist_cs"][x] for x in na if x in st["hist_cs"]]
    tb = [st["hist_cs"][x] for x in nb if x in st["hist_cs"]]
    if len(ta) == len(na) and len(tb) == len(nb) and na and nb:
        for i, c in enumerate(pm3.CS):
            row[f"call_{c}"] = float(np.mean(ta, 0)[i]
                                     - np.mean(tb, 0)[i])
    return row


def pistol_model():
    """In-sample logistic on the cached v3 samples (optimism noted in report)."""
    import sklearn.linear_model as lm
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(V3_SAMPLES)
    d = df[V3_FULL + ["y"]].dropna()
    sc = StandardScaler().fit(d[V3_FULL])
    clf = lm.LogisticRegression(max_iter=2000).fit(
        sc.transform(d[V3_FULL]), d.y)
    return sc, clf


def mapwin_model():
    import sklearn.linear_model as lm
    from sklearn.preprocessing import StandardScaler
    df = pd.read_parquet(FEATURES_PQ, columns=mm.FEATURES + ["y"])
    d = df.dropna(subset=mm.FEATURES + ["y"])
    sc = StandardScaler().fit(d[mm.FEATURES])
    clf = lm.LogisticRegression(max_iter=2000).fit(
        sc.transform(d[mm.FEATURES]), d.y)
    return sc, clf


# ----------------------------------------------------------------- teams
def resolve_team(name: str, lut: dict) -> str:
    key = norm_team(name)
    aliases = {"navi": "natusvincere"}     # 常见简称 -> demo 池 slug
    key = aliases.get(key, key)
    if key in lut:
        return key
    exact = [k for k in lut if k == key.replace("team", "")]
    if exact:
        return exact[0]
    cands = difflib.get_close_matches(key, list(lut), n=5, cutoff=0.55)
    sub = [k for k in lut if key and key in k] or \
          [k for k in lut if k and k in key]
    pool = list(dict.fromkeys(cands + sub))
    raise SystemExit(f"队伍名无法解析: {name!r} (规范化={key!r}). "
                     f"近似候选: {pool[:8]} | 池中示例: {sorted(lut)[:20]}")


def roster_names(roster: frozenset, sid_names: dict) -> list:
    return [sid_names.get(s, s) for s in sorted(roster)]


# -------------------------------------------------------------- profiles
def attribute_pistols(pist: pd.DataFrame, ros_a: frozenset,
                      ros_b: frozenset) -> pd.DataFrame:
    """Per-round side attribution, each team independently.

    side_a: A 在该回合是 CT/T (Jaccard>=0.6, 对任意对手); None = 无法归属.
    side_b: 同理. 两队各自的手枪局画像应计入其所有比赛, 不只是相互交手.
    """
    def side_of(row, ros):
        ct = frozenset(str(row.ct_roster).split(","))
        t = frozenset(str(row.t_roster).split(","))
        o_ct = len(ct & ros) / max(len(ct | ros), 1)
        o_t = len(t & ros) / max(len(t | ros), 1)
        if o_ct >= 0.6 and o_ct >= o_t:
            return "CT"
        if o_t >= 0.6:
            return "T"
        return None

    out = pist.copy()
    out["side_a"] = [side_of(r, ros_a) for r in pist.itertuples(index=False)]
    out["side_b"] = [side_of(r, ros_b) for r in pist.itertuples(index=False)]
    out["both"] = out["side_a"].notna() & out["side_b"].notna()
    return out


def side_profile(sub: pd.DataFrame, side: str) -> dict:
    n = len(sub)
    d: dict = {"n": n}
    if not n:
        return d
    won = sub["winner_side"] == side
    d["win"] = float(won.mean())
    d["fk"] = float((sub["first_kill_side"] == side).mean())
    if side == "T":
        d["plant"] = float(sub["bomb_planted"].mean())
        fk_win = sub[sub["first_kill_side"] == "T"]
        d["fk_win"] = float((fk_win["winner_side"] == "T").mean()) if len(fk_win) else None
        d["len_s"] = float(sub["round_len_sec"].mean())
        d["outcome"] = sub["outcome_type"].value_counts(normalize=True).round(3).to_dict()
    else:
        noplant = sub[~sub["bomb_planted"]]
        d["noplant_win"] = float((noplant["winner_side"] == "CT").mean()) \
            if len(noplant) else None
    return d


def fmt_rate(v, n, digits=1):
    if v is None:
        return "  —  "
    s = f"{v * 100:.{digits}f}%" if abs(v) <= 1.5 else f"{v:+.3f}"
    return s + (f"(n={n})" if n < MIN_N else "")


def fmt_gap(v):
    return f"{v:+.3f}" if v is not None else "  —  "


# ----------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser(description="赛前选品分析台")
    ap.add_argument("--team-a", required=True)
    ap.add_argument("--team-b", required=True)
    ap.add_argument("--map", required=True, dest="arena")
    ap.add_argument("--price-a", type=float, default=None)
    ap.add_argument("--price-b", type=float, default=None)
    ap.add_argument("--at", default=None, help="决策时刻 ISO; 默认 now(UTC)")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--sim", action="store_true",
                    help="模拟模式: 写入 sim_ledger.csv (历史回放测试, 非真实前向)")
    ap.add_argument("--pick", choices=["a", "b", "none"], default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    at = pd.Timestamp(args.at) if args.at else pd.Timestamp.now(tz="UTC")
    if at.tzinfo is None:
        at = at.tz_localize("UTC")
    at = at.tz_convert("UTC")
    arena = md.map_name(args.arena)
    at_ts = at.timestamp()

    # ---- shared loads
    lab = pd.read_parquet(LABELS, columns=["demo_path", "roster_a",
                                               "roster_b", "complete"])
    lut = team_roster_lut(lab[lab["complete"]])
    ka = resolve_team(args.team_a, lut)
    kb = resolve_team(args.team_b, lut)
    ros_a, ros_b = lut[ka], lut[kb]
    blob = load_states()
    sid_names = blob["sid_names"]
    names_a = roster_names(ros_a, sid_names)
    names_b = roster_names(ros_b, sid_names)

    rs = pd.read_parquet(ROUND_STATES)
    pist = rs[rs["round_num"].isin([1, 13])].copy()
    pist["arena_norm"] = pist["map_name"].map(
        lambda m: md.map_name(m) if re.fullmatch(r"[A-Za-z0-9_ ]+", str(m)) else m)
    att = attribute_pistols(pist, ros_a, ros_b)

    history = pd.read_parquet(HISTORY_PQ)
    feats, cov = md.features_at(history, ",".join(sorted(ros_a)),
                                ",".join(sorted(ros_b)), arena, at)
    L: list = []
    w = L.append

    title = f"{ka} vs {kb} @ {arena}"
    w(f"# 选品分析台: {title}")
    w(f"决策时刻 {at.isoformat()} | 生成 {pd.Timestamp.now(tz='UTC').isoformat()}")
    w("")
    w("> **诚实性**: 模型置信无增量 (v3 OOF AUC≈0.48, form-strong 天花板 "
      f"{CEILING:.1%}) — 本报告是读局辅助不是预测器, p 值只作基准参照, "
      "**最终判断权在你**。所有统计 past-only (90 天半衰); in-sample p 有乐观偏差; "
      f"n<{MIN_N} 的桶显示原始 n。")
    w("")

    # ---- A 队伍识别
    w("## A. 队伍识别")
    w(f"- **A = {ka}**: {', '.join(names_a)}")
    w(f"- **B = {kb}**: {', '.join(names_b)}")
    meet = att[att["both"]]
    if len(meet):
        a_ct = (meet["side_a"] == "CT").mean()
        w(f"- 历史交手 (demo 池, 任一手枪局样本): n={len(meet)}, "
          f"A 首CT 占比 {a_ct:.0%} → 本场首枪倾向: "
          f"{'A 大概率首 CT' if a_ct >= 0.5 else 'A 大概率首 T'}")
        a_ct_hint = a_ct >= 0.5
    else:
        w("- 历史交手: demo 池中两队无共同手枪局样本")
        a_ct_hint = None
    w("")

    # ---- B 近期状态
    w("## B. 近期状态 (decision_at 前, 90 天半衰)")
    order = ["win", "round", "kd", "adr", "opening", "trade", "pistol", "ct"]
    lbl = {"win": "胜率", "round": "回合率", "kd": "logK/D", "adr": "ADR",
           "opening": "首杀率", "trade": "补枪率", "pistol": "手枪率",
           "ct": "CT 回合率"}
    for scope, name in (("all", "全部地图"), ("map", arena)):
        covk = f"history_{scope}_a"
        na_, nb_ = cov.get(f"history_{scope}_a", 0), cov.get(f"history_{scope}_b", 0)
        cells = " | ".join(f"{lbl[k]} {fmt_gap(feats.get(f'{scope}_{k}_gap'))}"
                           for k in order)
        w(f"- **{name}** (样本 A={na_} 图/B={nb_} 图): {cells}")
    for tag, ros in (("A", ros_a), ("B", ros_b)):
        h = history[(history.roster_a == ",".join(sorted(ros)))
                    | (history.roster_b == ",".join(sorted(ros)))]
        h = h[h.start_at < at].sort_values("start_at", ascending=False).head(10)
        if not len(h):
            w(f"- {tag} 近 10 图: 无历史")
            continue
        items = []
        for r in h.itertuples(index=False):
            mine_a = r.roster_a == ",".join(sorted(ros))
            s_a = r.round_wins_a if mine_a else r.round_wins_b
            s_b = r.round_wins_b if mine_a else r.round_wins_a
            res = "W" if (r.y == 1) == mine_a else "L"
            items.append(f"{r.start_at:%m-%d} {r.map_name.replace('de_', '')} "
                         f"{res}{s_a}-{s_b}")
        w(f"- {tag} 近 {len(h)} 图: " + "; ".join(items))
    w("")

    # ---- C 地图上下文
    w("## C. 地图上下文")
    m_all = rs.copy()
    m_all["arena_norm"] = m_all["map_name"].map(
        lambda m: md.map_name(m) if re.fullmatch(r"[A-Za-z0-9_ ]+", str(m)) else m)
    g = pist.groupby("arena_norm")["winner_side"]
    m_ct = g.apply(lambda s: (s == "CT").mean())
    m_n = g.size()
    g2 = m_all.groupby("arena_norm")["winner_side"]
    r_ct = g2.apply(lambda s: (s == "CT").mean())
    this = pist[pist.arena_norm == arena]
    w(f"- **{arena}** 手枪局: CT 胜率 {fmt_rate(float(m_ct.get(arena, np.nan)), len(this))}"
      f" | 全部回合 CT 胜率 {fmt_rate(float(r_ct.get(arena, np.nan)), int(m_all.arena_norm.eq(arena).sum()))}")
    w(f"- 全池手枪局 CT 胜率 (对照): {fmt_rate(float((pist.winner_side == 'CT').mean()), len(pist))}")
    for tag, ros in (("A", ros_a), ("B", ros_b)):
        rk = ",".join(sorted(ros))
        hm = history[(history.map_name == arena)
                     & ((history.roster_a == rk) | (history.roster_b == rk))
                     & (history.start_at < at)]
        if len(hm):
            wins = sum((r.y == 1) if r.roster_a == rk else (1 - r.y)
                       for r in hm.itertuples(index=False))
            w(f"- {tag} 在 {arena}: {wins}/{len(hm)} 图 "
              f"({wins / len(hm):.0%})" + (f" (n={len(hm)})" if len(hm) < MIN_N else ""))
        else:
            w(f"- {tag} 在 {arena}: 无历史样本")
    w("")

    # ---- D 手枪局打法画像
    w("## D. 手枪局打法画像 (demo 池, Jaccard≥0.6 归属)")
    w("句式: 该队 T 手枪 = 下包率/均时/首杀率/先首杀胜率; CT 手枪 = 首杀率/无下包守成率")
    profiles: dict = {}
    for tag, side_col in (("A", "side_a"), ("B", "side_b")):
        mine = att[att[side_col].notna()]
        prof: dict = {}
        for scope, sub in (("all", mine), ("map", mine[mine.arena_norm == arena])):
            prof[scope] = {side: side_profile(sub[sub[side_col] == side], side)
                           for side in ("CT", "T")}
        profiles[tag] = prof
        for scope, sname in (("all", "全部地图"), ("map", arena)):
            p = prof[scope]
            ct, t = p["CT"], p["T"]
            t_len = f"{t['len_s'] / 60:.1f}分" if t.get("len_s") else "—"
            w(f"- **{tag} {sname}** CT手枪: 胜率 {fmt_rate(ct.get('win'), ct['n'])} "
              f"首杀 {fmt_rate(ct.get('fk'), ct['n'])} "
              f"无下包守成 {fmt_rate(ct.get('noplant_win'), ct['n'])} || "
              f"T手枪: 胜率 {fmt_rate(t.get('win'), t['n'])} "
              f"下包 {fmt_rate(t.get('plant'), t['n'])} "
              f"均时 {t_len} "
              f"首杀 {fmt_rate(t.get('fk'), t['n'])} "
              f"先首杀胜率 {fmt_rate(t.get('fk_win'), t['n'])}"
              + (f" 结构 {t.get('outcome')}" if t.get("outcome") and t["n"] < 40 else ""))
    # 侵略性 (contact) from inround kill events
    ie = pd.read_parquet(INROUND_PQ,
                         columns=["demo_path", "round_num", "event_type",
                                  "ct_contact", "t_contact"])
    ie = ie[(ie.round_num.isin([1, 13])) & (ie.event_type == "kill")]
    ie["key"] = ie["demo_path"].map(pm3.norm_key)
    key_idx = att["demo_path"].map(pm3.norm_key)
    for tag, col in (("A", "side_a"), ("B", "side_b")):
        side_map = att.set_index([key_idx, "round_num"])[col]
        ie[col] = ie.set_index(["key", "round_num"]).index.map(side_map)
    for tag, col in (("A", "side_a"), ("B", "side_b")):
        parts = []
        for side_is, fld in (("CT", "ct_contact"), ("T", "t_contact")):
            mine = ie[ie[col] == side_is]
            if len(mine):
                parts.append(f"{side_is}侧 {mine[fld].mean():.0f} "
                             f"(n={mine['key'].nunique()}图)")
        w(f"- {tag} 侵略性 (手枪 kill 事件 contact 均值): "
          + (" | ".join(parts) if parts else "无样本"))
    w("")

    # ---- E 概率合成
    w("## E. 概率合成")
    st = state_at(blob["snapshots"], at_ts)
    sc_p, clf_p = pistol_model()
    p_rows: dict = {}
    decomp = None
    if st is None:
        w("- **手枪局 p: 历史快照不足, 无法评分**")
    else:
        missing = [n for n in names_a + names_b
                   if n not in st["hist_k"]]
        for a_ct in (1, 0):
            row = score_pistol_row(st, names_a, names_b, arena, a_ct)
            have = [c for c in V3_FULL if c in row]
            if len(have) < len(V3_FULL):
                w(f"- **手枪局 p (A{'首CT' if a_ct else '首T'}): 特征缺失 "
                  f"{sorted(set(V3_FULL) - set(row))} 不可评分**"
                  + (f"; 无历史选手: {missing[:5]}" if missing else ""))
                continue
            x = sc_p.transform(pd.DataFrame([row])[V3_FULL])
            p = float(clf_p.predict_proba(x)[0, 1])
            p_rows[a_ct] = p
            if decomp is None:
                decomp = row
        for a_ct, p in p_rows.items():
            w(f"- **手枪局 p (in-sample, 乐观)**: A {'首CT' if a_ct else '首T'} "
              f"→ A 胜 {p:.1%} / B 胜 {1 - p:.1%}")
        if decomp is not None:
            w(f"- 特征分解 (A−B): form {fmt_gap(decomp.get('form_gap'))} | "
              f"pistol_rate {fmt_gap(decomp.get('pistol_rate_gap'))} | "
              f"fk {fmt_gap(decomp.get('fk_gap'))} | tk {fmt_gap(decomp.get('tk_gap'))} | "
              f"side_prior {decomp['side_prior']:.3f} (该图 CT 手枪先验)")
        w(f"- 警示: OOF AUC≈{OOF_AUC}, form-strong 天花板 {CEILING:.1%}; "
          "高置信子集历史上反指, 别把 p 当优势")
    sc_m, clf_m = mapwin_model()
    fx = np.array([[feats.get(c, np.nan) for c in mm.FEATURES]])
    if not np.isnan(fx).any():
        p_map = float(clf_m.predict_proba(sc_m.transform(
            pd.DataFrame(fx, columns=mm.FEATURES)))[0, 1])
        w(f"- **地图胜率 p_mapwin (in-sample, 乐观)**: A {p_map:.1%} / B {1 - p_map:.1%}")
    else:
        p_map = None
        w("- **地图胜率 p_mapwin: 特征缺失, 不可评分**")
    ev_a = ev_b = None
    if args.price_a is not None:
        p_ref = p_rows.get(1, p_map)
        if p_ref is not None:
            ev_a = p_ref * LEG_WIN_POP - (1 - p_ref) * LEG_LOSS_DROP
            if args.price_b is not None:
                ev_b = (1 - p_ref) * LEG_WIN_POP - p_ref * LEG_LOSS_DROP
            be = LEG_LOSS_DROP / (LEG_WIN_POP + LEG_LOSS_DROP)
            dev = p_ref - args.price_a
            w(f"- **市场价**: A {args.price_a:.2f}"
              + (f" / B {args.price_b:.2f}" if args.price_b is not None else ""))
            w(f"- 盈亏线 (修正腿 +{LEG_WIN_POP}/−{LEG_LOSS_DROP}pt): 需命中率 "
              f"{be:.1%}; 模型 p − 价格 = {dev:+.1%} "
              f"({'|偏差|≥5pt, 情感资金假设观测窗' if abs(dev) >= 0.05 else '偏差<5pt'})")
            w(f"- EV 读数: 买 A 每份期望 {ev_a:+.2f}pt"
              + (f" | 买 B 每份期望 {ev_b:+.2f}pt" if ev_b is not None else ""))
    w("")
    w("## F. 决策 (人工)")
    w("看过以上统计后自行决定。`--record --pick a|b|none --note \"理由\"` 写入前向账本。")

    text = "\n".join(L)
    print("\n" + text)
    OUT.mkdir(parents=True, exist_ok=True)
    safe = lambda s: re.sub(r"[^a-z0-9]+", "_", s)  # noqa: E731
    rp = OUT / f"{safe(ka)}_vs_{safe(kb)}_{arena}_{at:%Y%m%dT%H%M}.md"
    rp.write_text(text, encoding="utf-8")
    print(f"\n[saved] {rp}")

    if args.record:
        if not args.pick:
            raise SystemExit("--record 需要 --pick a|b|none")
        row = {
            "ts": at.isoformat(), "team_a": ka, "team_b": kb, "map": arena,
            "p_pistol_a_ct": round(p_rows.get(1, np.nan), 4),
            "p_pistol_a_t": round(p_rows.get(0, np.nan), 4),
            "p_mapwin_a": None if p_map is None else round(p_map, 4),
            "price_a": args.price_a, "price_b": args.price_b,
            "ev_buy_a_pt": None if ev_a is None else round(ev_a, 3),
            "ev_buy_b_pt": None if ev_b is None else round(ev_b, 3),
            "pick": args.pick, "note": args.note,
            "result_a": "", "settled_pnl": "",
        }
        ledger = OUT / "sim_ledger.csv" if args.sim else LEDGER
        exists = ledger.exists()
        with open(ledger, "a", encoding="utf-8", newline="") as h:
            if not exists:
                h.write(",".join(row) + "\n")
            h.write(",".join('"' + str(v).replace('"', "'") + '"'
                             if "," in str(v) else str(v)
                             for v in row.values()) + "\n")
        print(f"[ledger] {ledger} (+1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
