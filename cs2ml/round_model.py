"""round 级验证（路线 B 第一步）：回合起始状态 → 回合胜方。

回答的关键问题：只用「边(CT/T) + 地图 + 回合序号(手枪/经济/长枪) + 双方经济档位」，
能否在回合开始时预测本回合胜方（AUC > 0.5）？
  - 若成立 → 叠 roster 对阵 + ②③战术 + 30s 中局状态，再经济仿真推到地图/总回合。
  - 若 ≈0.5 → 说明回合起始信息不够，"30s 后" 的中局状态才是信号所在。

纪律：GroupKFold by match（防同场泄漏）；标签 = round_end 的胜方(CT/T)。
产物：data/round_dataset.parquet（每回合一行，供后续所有回合级实验复用）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .demo_features import (PISTOL_ROUNDS, POST_PISTOL_ROUNDS, _classify_buy,
                            _group_roster, _group_side_first_half, _steamid_to_group,
                            _winner_group)
from .duel import discover_demos

FIRST_HALF_ROUNDS = 12


def _round_class(r: int) -> str:
    if r in PISTOL_ROUNDS:
        return "pistol"
    if r in POST_PISTOL_ROUNDS:
        return "post_pistol"
    return "regular"


def _flip(side: str) -> str:
    return "T" if side == "CT" else "CT"


def build_round_dataset(dem_paths: list[Path] | None = None) -> pd.DataFrame:
    demos = dem_paths or discover_demos()
    cache = config.DATA_DIR / "round_dataset.parquet"
    if dem_paths is None and cache.exists():
        return pd.read_parquet(cache)

    rows: list[dict] = []
    for dp in demos:
        try:
            parser = DemoParser(str(dp))
            header = parser.parse_header()
            map_name = header.get("map_name", dp.stem)
            player_info = parser.parse_player_info()
            if player_info is None or not len(player_info):
                continue  # 截断 demo（空 player_info）
            round_end = parser.parse_event("round_end")
            freeze = parser.parse_event("round_freeze_end")
            if round_end is None or not len(round_end):
                continue
            sid2group = _steamid_to_group(player_info)
            group_roster = _group_roster(player_info)
            groups = sorted(set(sid2group.values()))
            roster_keys = {g: ",".join(group_roster[g]) for g in groups}
            if len(groups) != 2:
                continue

            freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())
            start_ticks = [t + 1 for t in freeze_ticks]
            tick2round = {t: i + 1 for i, t in enumerate(start_ticks)}

            ticks_start = parser.parse_ticks(["team_name"], ticks=[start_ticks[0]])
            first_half_side = _group_side_first_half(ticks_start, sid2group)

            equip = parser.parse_ticks(["current_equip_value"], ticks=start_ticks)
            if equip is None or not len(equip):
                continue
            equip["group"] = equip["steamid"].astype(str).map(sid2group)
            equip = equip.dropna(subset=["group"])
            equip["group"] = equip["group"].astype(int)
            equip["round"] = equip["tick"].map(tick2round)
            eq = (equip.groupby(["round", "group"])["current_equip_value"]
                      .mean().reset_index())
            eq_by_round = {(int(r.round), int(r.group)): float(r.current_equip_value)
                           for r in eq.itertuples(index=False)}

            def side_of(g: int, r: int) -> str:
                s = str(first_half_side.get(g, "")).upper()
                return s if r <= FIRST_HALF_ROUNDS else _flip(s)

            re_ = round_end[round_end["winner"].notna()].copy()
            re_["round"] = re_["round"].astype(int)
            for rr in re_.itertuples(index=False):
                r = int(rr.round)
                winner_side = str(rr.winner).upper()
                ct_g = next((g for g in groups if side_of(g, r) == "CT"), None)
                t_g = next((g for g in groups if side_of(g, r) == "T"), None)
                if ct_g is None or t_g is None:
                    continue
                ct_equip = eq_by_round.get((r, ct_g), np.nan)
                t_equip = eq_by_round.get((r, t_g), np.nan)
                if pd.isna(ct_equip) or pd.isna(t_equip):
                    continue
                wg = _winner_group(winner_side, r, first_half_side)
                rows.append({
                    "match_id": dp.parent.name,
                    "demo_path": str(dp),
                    "map_name": map_name,
                    "round_num": r,
                    "ct_roster": roster_keys[ct_g],
                    "t_roster": roster_keys[t_g],
                    "ct_equip": round(ct_equip, 1),
                    "t_equip": round(t_equip, 1),
                    "ct_tier": _classify_buy(ct_equip),
                    "t_tier": _classify_buy(t_equip),
                    "round_class": _round_class(r),
                    "winner_side": winner_side,
                    "winner_roster": roster_keys.get(wg, ""),
                    "label_ct_win": int(winner_side == "CT"),
                })
        except Exception as e:
            print(f"跳过 {dp.name}: {e}")
            continue

    df = pd.DataFrame(rows)
    if dem_paths is None and not df.empty:  # 只在全量运行时写缓存（部分运行不覆盖）
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def _round_auc(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["ct_equip", "t_equip"]).copy()
    if len(d) < 100:
        return {"error": f"样本不足：{len(d)} 回合"}
    y = d["label_ct_win"].to_numpy()
    groups = d["match_id"].to_numpy()

    cat = ["map_name", "round_class", "ct_tier", "t_tier"]
    num = ["ct_equip", "t_equip", "round_num"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    n_splits = min(5, d["match_id"].nunique())
    proba = cross_val_predict(model, d[cat + num], y, groups=groups,
                              cv=GroupKFold(n_splits), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)

    out = {"n_rounds": len(d), "n_maps": int(d["match_id"].nunique()),
           "ct_base_rate": round(float(y.mean()), 4), "auc": round(auc, 4)}

    # 分回合类型看
    d["p_ct"] = proba
    out["by_class"] = {}
    for cls, sub in d.groupby("round_class"):
        if len(sub) < 50 or sub["label_ct_win"].nunique() < 2:
            continue
        out["by_class"][cls] = {
            "n": len(sub),
            "ct_rate": round(float(sub["label_ct_win"].mean()), 3),
            "auc": round(roc_auc_score(sub["label_ct_win"], sub["p_ct"]), 3),
        }
    return out


def _map_accuracy(df: pd.DataFrame) -> dict:
    """聚合到地图：每 roster 期望回合数 = Σ P(CT胜)+Σ(1-P) 按边归属，预测地图胜者。"""
    d = df.dropna(subset=["ct_equip", "t_equip"]).copy()
    cat = ["map_name", "round_class", "ct_tier", "t_tier"]
    num = ["ct_equip", "t_equip", "round_num"]
    y = d["label_ct_win"].to_numpy()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    n_splits = min(5, d["match_id"].nunique())
    d["p_ct"] = cross_val_predict(model, d[cat + num], y, groups=d["match_id"].to_numpy(),
                                  cv=GroupKFold(n_splits), method="predict_proba")[:, 1]

    rows = []
    for mid, sub in d.groupby("match_id"):
        exp = {}
        act = {}
        for r in sub.itertuples(index=False):
            exp[r.ct_roster] = exp.get(r.ct_roster, 0.0) + r.p_ct
            exp[r.t_roster] = exp.get(r.t_roster, 0.0) + (1 - r.p_ct)
            if r.winner_roster:
                act[r.winner_roster] = act.get(r.winner_roster, 0) + 1
        if len(exp) != 2 or len(act) != 2:
            continue
        pred_winner = max(exp, key=exp.get)
        true_winner = max(act, key=act.get)
        rows.append((int(pred_winner == true_winner), len(sub)))
    if not rows:
        return {"n_maps": 0}
    acc = float(np.mean([r[0] for r in rows]))
    return {"n_maps": len(rows), "map_acc": round(acc, 4),
            "avg_rounds": round(float(np.mean([r[1] for r in rows])), 1)}


def evaluate() -> dict:
    df = build_round_dataset()
    print(f"回合数据集：{len(df)} 回合 / {df['match_id'].nunique()} 图 / "
          f"{df['map_name'].nunique()} 地图")
    print(f"CT 胜率基准：{round(float(df['label_ct_win'].mean()), 4)}")

    res = _round_auc(df)
    print("\n回合级（GroupKFold by match，逻辑回归）：")
    print(f"  AUC = {res.get('auc')}  n_rounds = {res.get('n_rounds')}  "
          f"n_maps = {res.get('n_maps')}  CT_base = {res.get('ct_base_rate')}")
    if "by_class" in res:
        print("  分回合类型：")
        for cls, v in res["by_class"].items():
            print(f"    {cls:<12} n={v['n']:<5} CT率={v['ct_rate']:<5} AUC={v['auc']}")

    mp = _map_accuracy(df)
    print(f"\n地图级（期望回合数聚合）：map_acc = {mp.get('map_acc')}  "
          f"n_maps = {mp.get('n_maps')}")
    return {"round": res, "map": mp}


if __name__ == "__main__":
    evaluate()
