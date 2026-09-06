"""strategy_prior.py —— 把队伍级策略记忆接进预测闭环（校正→再预测）。

测：per-team 策略签名（out-of-fold）能否提升「回合胜方」预测。

两个设置（对应两个决策场景）：
  1. 有 equip（盘口已看到买枪）：签名应冗余（equip 已含策略结果）。
  2. 无 equip（盘口未定价买枪，或你提前于买枪知道）：签名是唯一能
     「提前猜到这队怎么买」的信息 → 这是「先市场一步」的 latency edge 测试。

方法：GroupKFold by match（5 折），每折用**训练场次**的队伍回合算签名
      （global-mean 收缩兜底），注入测试场次的回合特征。防同场泄漏。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .round_transition import _classify6, _round_class
from .team_strategy import team_signature, _score_bucket

SIG_KEYS = ["force_when_behind", "eco_when_behind", "full_when_ahead",
            "post_pistol_loss_force", "full_when_even"]
_MIN_ROUNDS = 20


def build_round_level() -> pd.DataFrame:
    """回合级：roster + equip(6档) + 比分 + 标签。"""
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    rs = pd.read_parquet(config.DATA_DIR / "round_states.parquet")
    keep = ["demo_path", "round_num", "outcome_type", "ct_equip0", "t_equip0"]
    m = rd.merge(rs[keep], on=["demo_path", "round_num"], how="inner")
    m["round_num"] = m["round_num"].astype(int)
    m["label_ct_win"] = (m["winner_side"] == "CT").astype(int)
    m = m.sort_values(["demo_path", "round_num"]).reset_index(drop=True)

    rows = []
    for dp, sub in m.groupby("demo_path"):
        sub = sub.sort_values("round_num").reset_index(drop=True)
        winners = sub["winner_side"].tolist()
        score_ct = 0
        score_t = 0
        for i in range(len(sub)):
            cur = sub.iloc[i]
            if i > 0 and winners[i - 1] == "CT":
                score_ct += 1
            elif i > 0:
                score_t += 1
            rows.append({
                "match_id": cur["match_id"], "demo_path": dp,
                "map_name": cur["map_name"], "round_num": int(cur["round_num"]),
                "round_class": _round_class(int(cur["round_num"])),
                "score_gap": score_ct - score_t,
                "ct_score_bucket": _score_bucket(score_ct - score_t),
                "ct_roster": str(cur["ct_roster"]), "t_roster": str(cur["t_roster"]),
                "ct_tier6": _classify6(cur["ct_equip0"]),
                "t_tier6": _classify6(cur["t_equip0"]),
                "equip_gap": cur["ct_equip0"] - cur["t_equip0"],
                "label_ct_win": int(cur["label_ct_win"]),
            })
    return pd.DataFrame(rows)


def _signature_map(team_rounds: pd.DataFrame) -> dict[str, dict[str, float]]:
    """roster → 签名（含 global-mean 收缩兜底）。"""
    sig = {}
    for roster, sub in team_rounds.groupby("roster"):
        if len(sub) >= _MIN_ROUNDS:
            sig[roster] = team_signature(sub)
    # 用全体队伍签名的中位做兜底（缺样本 / NaN 时收缩到全局中位）
    all_sig = [team_signature(sub) for _, sub in team_rounds.groupby("roster")]
    fallback = {k: float(np.nanmedian([s[k] for s in all_sig if not pd.isna(s[k])]))
                for k in SIG_KEYS}
    return sig, fallback


def _add_sig_features(rounds: pd.DataFrame, sig: dict, fallback: dict) -> pd.DataFrame:
    """给回合行加 10 个签名特征（CT 队 5 + T 队 5），用 fallback 兜底。"""
    r = rounds.copy()
    for side in ("ct", "t"):
        col = f"{side}_roster"
        for k in SIG_KEYS:
            vals = []
            for roster in r[col]:
                s = sig.get(roster, {})
                v = s.get(k, None)
                vals.append(fallback[k] if (v is None or pd.isna(v)) else v)
            r[f"{side}_{k}"] = vals
    return r


def _fit_predict(tr, te, cat, num):
    """在 tr 上训练，te 上预测概率。"""
    tr = tr.dropna(subset=cat + num + ["label_ct_win"])
    te = te.dropna(subset=cat + num + ["label_ct_win"])
    y = tr["label_ct_win"].to_numpy()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
    clf.fit(tr[cat + num], y)
    return clf.predict_proba(te[cat + num])[:, 1]


def test_round_winner() -> dict:
    rounds = build_round_level()
    rounds = rounds.dropna(subset=["ct_roster", "t_roster", "equip_gap"])
    # 队伍回合（用于算签名）需要 buy_tier，从 team_strategy 复用
    from .team_strategy import build_team_rounds
    team_rounds = build_team_rounds()
    team_rounds = team_rounds.dropna(subset=["buy_tier"])

    base_cat = ["map_name", "round_class"]
    sig_cat = ["map_name", "round_class"]
    sig_num = [f"{side}_{k}" for side in ("ct", "t") for k in SIG_KEYS]

    gkf = GroupKFold(n_splits=5)
    y_all = rounds["label_ct_win"].to_numpy()
    groups = rounds["match_id"].to_numpy()

    # 收集每折预测
    preds = {name: np.full(len(rounds), np.nan)
             for name in ["base", "base_sig", "nop_eq", "nop_eq_sig"]}
    for tr_idx, te_idx in gkf.split(rounds, groups=groups):
        tr = rounds.iloc[tr_idx]
        te = rounds.iloc[te_idx]
        tr_matches = set(tr["match_id"])
        # 用训练场次算签名
        tr_team = team_rounds[team_rounds["match_id"].isin(tr_matches)]
        sig, fallback = _signature_map(tr_team)
        te_sig = _add_sig_features(te, sig, fallback)

        # 1) 有 equip 基线（比分+经济）
        preds["base"][te_idx] = _fit_predict(tr, te, base_cat,
                                             ["score_gap", "equip_gap"])
        # 2) 有 equip + 签名
        preds["base_sig"][te_idx] = _fit_predict(
            _add_sig_features(tr, sig, fallback), te_sig, sig_cat,
            ["score_gap", "equip_gap"] + sig_num)
        # 3) 无 equip 基线（只比分+回合类）
        preds["nop_eq"][te_idx] = _fit_predict(tr, te, base_cat, ["score_gap"])
        # 4) 无 equip + 签名（latency edge）
        preds["nop_eq_sig"][te_idx] = _fit_predict(
            _add_sig_features(tr, sig, fallback), te_sig, sig_cat,
            ["score_gap"] + sig_num)

    print("=" * 76)
    print("per-team 策略签名 → 回合胜方（GroupKFold by match, out-of-fold 签名）")
    print("=" * 76)
    print(f"  {'设置':<34}{'AUC':>8}{'n':>7}")
    for name in ["base", "base_sig", "nop_eq", "nop_eq_sig"]:
        p = preds[name]
        mask = ~np.isnan(p)
        a = roc_auc_score(y_all[mask], p[mask])
        print(f"  {name:<34}{a:>8.4f}{mask.sum():>7}")

    # 分层：队伍差异最大的处境
    print("\n--- 分层（无 equip 基线 vs +签名，latency edge）---")
    for label, mask in [
        ("post_pistol", rounds["round_class"] == "post_pistol"),
        ("CT落后(behind)", rounds["ct_score_bucket"].isin(["behind1-2", "behind3+"])),
        ("regular", rounds["round_class"] == "regular"),
    ]:
        p0 = preds["nop_eq"][mask]
        p1 = preds["nop_eq_sig"][mask]
        a0 = roc_auc_score(y_all[mask], p0) if (~np.isnan(p0)).sum() > 0 else float("nan")
        a1 = roc_auc_score(y_all[mask], p1) if (~np.isnan(p1)).sum() > 0 else float("nan")
        print(f"  {label:<24} 无equip={a0:.4f}  +签名={a1:.4f}  Δ={a1-a0:+.4f}  n={mask.sum()}")
    return {}


if __name__ == "__main__":
    test_round_winner()
