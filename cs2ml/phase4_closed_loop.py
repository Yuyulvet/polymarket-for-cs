"""phase4_closed_loop.py —— Phase 4：操作/风格特征的预测闭环检验（先两支队伍）。

把 Phase 1–3 学到的「战术层」——谁 opener、首杀强度、队伍侵略性、主狙存在——
做成 roster 级的 **walk-forward 签名**（只用 hltv 时间戳严格之前的 demo），
join 到回合上，检验它能否在 equip_gap 之上预测「下一回合胜方」。

闸门（round_transition.py 全局基线）：equip_gap = ct_equip0 - t_equip0 → AUC 0.745。

本脚本做两件事：
  1. 统计检验：round 级 GroupKFold(match_id)，baseline(equip_gap) vs +战术签名，比 AUC。
  2. 可读闭环：选两支队伍（默认 Spirit vs Vitality），把它们的 H2H 每张图按时间序
     排出「预测胜方（out-of-fold 回合多数票） vs 实际胜方」，让人能逐图核对。

签名特征（per roster，walk-forward，只用 hltv 之前的回合累计）：
  opener_opk   = 队内 max(首杀/回合) —— 突破手杀伤力
  opener_duel  = 突破手首杀成功率
  first_contact= 队均 first_contact —— 队伍侵略性
  awp_rate     = 队内 max(awp击杀/回合) —— 主狙存在
→ 特征 = ct 签名 - t 签名（4 个差值）+ equip_gap。
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config
from .team_style import cluster_rosters, label_teams

_TWO_TEAMS = ("spirit", "vitality")


# ---------------------------------------------------------------- 数据准备

def _hltv_id(demo_path: str) -> int:
    m = Path(demo_path).parent.name
    g = re.search(r"hltv-(\d+)", m)
    return int(g.group(1)) if g else -1


def load_rounds(behavior_path=None) -> pd.DataFrame:
    """player_behavior -> 回合级表：每行 = 一回合，含 ct/t roster、winner、equip、hltv 时间。"""
    beh = pd.read_parquet(behavior_path or config.DATA_DIR / "player_behavior.parquet")
    beh["hltv"] = beh["demo_path"].astype(str).map(_hltv_id)
    beh["rk"] = beh["roster_key"].astype(str)

    rows = []
    for (dp, rn), sub in beh.groupby(["demo_path", "round_num"]):
        sub = sub.dropna(subset=["winner_side"])
        if sub.empty:
            continue
        winner = sub["winner_side"].mode()
        if winner.empty:
            continue
        winner = str(winner.iloc[0])
        ct = sub[sub["side"] == "CT"]
        tt = sub[sub["side"] == "T"]
        if ct.empty or tt.empty:
            continue
        ct_rk = ct["rk"].mode().iloc[0]
        t_rk = tt["rk"].mode().iloc[0]
        rows.append({
            "demo_path": dp, "round_num": int(rn),
            "hltv": int(sub["hltv"].iloc[0]),
            "winner_side": winner, "label_ct_win": int(winner == "CT"),
            "ct_roster": ct_rk, "t_roster": t_rk,
            "equip_gap": float(sub["ct_equip0"].mean() - sub["t_equip0"].mean()),
        })
    r = pd.DataFrame(rows)
    return r


def roster_to_team(rounds: pd.DataFrame) -> dict[str, str]:
    """roster_key -> team 名（复用 Phase 3 聚类+命名）。"""
    beh = pd.read_parquet(config.DATA_DIR / "player_behavior.parquet")
    beh["rk"] = beh["roster_key"].astype(str)
    beh["dp"] = beh["demo_path"].astype(str)

    roster_demos: dict[str, set[str]] = defaultdict(set)
    demo_sides: dict[str, dict[str, str]] = defaultdict(dict)
    for r, dp, side in beh[["rk", "dp", "side"]].drop_duplicates().itertuples(index=False):
        roster_demos[r].add(dp)
        demo_sides[dp][side] = r

    rosters = sorted(beh["rk"].unique())
    clusters = cluster_rosters(rosters)
    labels = label_teams(clusters, roster_demos, demo_sides)

    r2t: dict[str, str] = {}
    for root, rs in clusters.items():
        for r in rs:
            r2t[r] = labels[root]
    return r2t


# ---------------------------------------------------------------- walk-forward 签名

def build_prior_stats(behavior_path=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """每个 steamid 按 hltv 时间累计的签名中间量，返回 (hltv_sorted, cumsum_matrix)。

    matrix 列 = [opening_kill, opening_death, first_contact, awp_kills, n_rounds]。
    """
    beh = pd.read_parquet(behavior_path or config.DATA_DIR / "player_behavior.parquet")
    beh["hltv"] = beh["demo_path"].astype(str).map(_hltv_id)
    beh = beh.sort_values("hltv").reset_index(drop=True)

    ok = beh["opening_kill"].fillna(0).astype(float).to_numpy()
    od = beh["opening_death"].fillna(0).astype(float).to_numpy()
    fc = beh["first_contact"].fillna(0).astype(float).to_numpy()
    awp = beh["awp_kills"].fillna(0).astype(float).to_numpy()
    nr = np.ones(len(beh))
    M = np.column_stack([ok, od, fc, awp, nr])
    h = beh["hltv"].to_numpy()
    sid = beh["steamid"].astype(str).to_numpy()

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in np.unique(sid):
        m = sid == s
        cum = np.cumsum(M[m], axis=0)
        out[s] = (h[m], cum)
    return out


def roster_sig(steamids: list[str], t: int,
               prior: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float, float]:
    """roster 在时间 t 之前（hltv < t）的签名 (opener_opk, opener_duel, first_contact, awp_rate)。"""
    best_opk = best_duel = best_awp = 0.0
    fc_sum = 0.0
    nr_sum = 0.0
    for s in steamids:
        if s not in prior:
            continue
        h, cum = prior[s]
        idx = int(np.searchsorted(h, t, side="left"))  # h[i] < t 的个数
        if idx == 0:
            continue
        row = cum[idx - 1]  # t 之前最后一个累计
        ok, od, fc, awp, nr = row
        opk = ok / max(nr, 1.0)
        if opk > best_opk:
            best_opk = opk
            best_duel = ok / max(ok + od, 1.0)
        best_awp = max(best_awp, awp / max(nr, 1.0))
        fc_sum += fc
        nr_sum += nr
    return best_opk, best_duel, (fc_sum / max(nr_sum, 1.0)), best_awp


def add_signature(rounds: pd.DataFrame,
                  prior: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """rounds 上加 4 个战术差值特征（ct 签名 - t 签名）。"""
    cols = ["f_opener_opk", "f_opener_duel", "f_first_contact", "f_awp_rate"]

    def one(row):
        ct = str(row["ct_roster"]).split(",")
        tt = str(row["t_roster"]).split(",")
        cs = roster_sig(ct, int(row["hltv"]), prior)
        ts = roster_sig(tt, int(row["hltv"]), prior)
        return pd.Series([cs[i] - ts[i] for i in range(4)], index=cols)

    rounds[cols] = rounds.apply(one, axis=1)
    return rounds


# ---------------------------------------------------------------- 检验

def _auc(d: pd.DataFrame, feats: list[str]) -> tuple[float, int]:
    d = d.dropna(subset=feats + ["label_ct_win"]).reset_index(drop=True)
    if len(d) < 100 or d["label_ct_win"].nunique() < 2:
        return float("nan"), len(d)
    y = d["label_ct_win"].to_numpy()
    n = min(5, d["hltv"].nunique())
    clf = LogisticRegression(max_iter=3000)
    p = cross_val_predict(clf, d[feats].to_numpy(), y,
                          groups=d["hltv"].to_numpy(), cv=GroupKFold(n),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p), len(d)


def main() -> None:
    rounds = load_rounds()
    print(f"回合级样本：{len(rounds)} 回合 / {rounds['hltv'].nunique()} 场(hltv) / "
          f"{rounds['demo_path'].nunique()} 图")
    r2t = roster_to_team(rounds)
    rounds["ct_team"] = rounds["ct_roster"].map(r2t)
    rounds["t_team"] = rounds["t_roster"].map(r2t)
    print(f"roster→team 映射：{len(r2t)} 个 roster / {len(set(r2t.values()))} 队")

    prior = build_prior_stats()
    print(f"walk-forward 先验：{len(prior)} 名选手")
    rounds = add_signature(rounds, prior)

    print("\n" + "=" * 72)
    print("下一回合胜方：equip_gap 基线 vs +战术签名（GroupKFold by match）")
    print("=" * 72)
    sig = ["f_opener_opk", "f_opener_duel", "f_first_contact", "f_awp_rate"]
    a0, n0 = _auc(rounds, ["equip_gap"])
    a1, n1 = _auc(rounds, ["equip_gap"] + sig)
    print(f"  {'equip_gap 基线':<40}{a0:>8.4f}{n0:>7}")
    print(f"  {'equip_gap + 战术签名':<40}{a1:>8.4f}{n1:>7}")
    print(f"  {'增量':<40}{a1 - a0:>+8.4f}")

    # 只保留两队相关回合，看增量是否还在
    print("\n--- 限定两队（Spirit / Vitality 出战的所有回合）---")
    two = rounds[(rounds["ct_team"].isin(_TWO_TEAMS)) | (rounds["t_team"].isin(_TWO_TEAMS))]
    a0t, n0t = _auc(two, ["equip_gap"])
    a1t, n1t = _auc(two, ["equip_gap"] + sig)
    print(f"  两队回合 n={n1t}")
    print(f"  {'equip_gap 基线':<40}{a0t:>8.4f}{n0t:>7}")
    print(f"  {'equip_gap + 战术签名':<40}{a1t:>8.4f}{n1t:>7}")
    print(f"  {'增量':<40}{a1t - a0t:>+8.4f}")

    _h2h_closed_loop(rounds, r2t, sig)
    _map_level(rounds, sig)


def _map_level(rounds: pd.DataFrame, sig: list[str]) -> None:
    """地图级：walk-forward 战术签名能否预测 map 胜方（团队身份的自然归宿）。"""
    # 每图一行：ct 赢的回合占比 + 该图 walk-forward 签名差值
    g = rounds.dropna(subset=sig).groupby("demo_path")
    maps = pd.DataFrame({
        "hltv": g["hltv"].first(),
        "label_ct_win": (g["label_ct_win"].mean() > 0.5).astype(int),
        "ct_team": g["ct_team"].first(),
        "t_team": g["t_team"].first(),
        "equip_gap": g["equip_gap"].mean(),
    })
    for f in sig:
        maps[f] = g[f].mean()

    print("\n" + "=" * 72)
    print("地图级：walk-forward 战术签名 → map 胜方（GroupKFold by match）")
    print("=" * 72)
    print(f"{'特征':<40}{'AUC':>8}{'n':>7}")
    a0, n0 = _auc(maps, ["equip_gap"])
    a1, n1 = _auc(maps, sig)
    a2, n2 = _auc(maps, ["equip_gap"] + sig)
    print(f"  {'equip_gap(平均) 基线':<38}{a0:>8.4f}{n0:>7}")
    print(f"  {'战术签名 only':<38}{a1:>8.4f}{n1:>7}")
    print(f"  {'equip_gap + 战术签名':<38}{a2:>8.4f}{n2:>7}")


def _h2h_closed_loop(rounds: pd.DataFrame, r2t: dict[str, str], sig: list[str]) -> None:
    """可读闭环：Spirit vs Vitality 的 H2H 每张图，**真 walk-forward** 预测 vs 实际。

    对每张 H2H 图，用 hltv 时间严格更早的所有回合训练 logistic，再预测该图每回合，
    多数票 → 预测胜方，与实际胜方对照。这是无泄漏的真预测。
    """
    a, b = _TWO_TEAMS
    h2h = rounds[((rounds["ct_team"] == a) & (rounds["t_team"] == b))
                 | ((rounds["ct_team"] == b) & (rounds["t_team"] == a))].copy()
    if h2h.empty:
        print(f"\n（无 {a} vs {b} H2H 图）")
        return
    h2h = h2h.dropna(subset=sig).reset_index(drop=True)
    if len(h2h) < 50:
        print(f"\n（{a} vs {b} H2H 回合太少，跳过闭环）")
        return

    allr = rounds.dropna(subset=sig).reset_index(drop=True)
    feats = ["equip_gap"] + sig

    print("\n" + "=" * 72)
    print(f"闭环：{a} vs {b} H2H 逐图（真 walk-forward，只用更早回合训练）")
    print("=" * 72)
    print(f"{'图':<34}{'实际':>8}{'预测':>8}{'对错':>5}{'回':>4}")
    acc = n_maps = 0
    ordered = h2h.sort_values("hltv")
    for dp, g in ordered.groupby("demo_path"):
        t = int(g["hltv"].iloc[0])
        train = allr[allr["hltv"] < t]
        if len(train) < 100 or train["label_ct_win"].nunique() < 2:
            continue
        clf = LogisticRegression(max_iter=3000)
        clf.fit(train[feats].to_numpy(), train["label_ct_win"].to_numpy())
        g = g.copy()
        g["p_ct"] = clf.predict_proba(g[feats].to_numpy())[:, 1]
        actual = "CT" if g["label_ct_win"].mean() > 0.5 else "T"
        pred = "CT" if (g["p_ct"] > 0.5).mean() > 0.5 else "T"
        act_team = g["ct_team"].iloc[0] if actual == "CT" else g["t_team"].iloc[0]
        pred_team = g["ct_team"].iloc[0] if pred == "CT" else g["t_team"].iloc[0]
        ok = "✓" if act_team == pred_team else "✗"
        acc += int(act_team == pred_team)
        n_maps += 1
        mapname = Path(dp).name.replace(".dem", "")
        print(f"  {mapname:<34}{act_team:>8}{pred_team:>8}{ok:>5}{len(g):>4}")
    print(f"\n  图级命中：{acc}/{n_maps}")


if __name__ == "__main__":
    main()
