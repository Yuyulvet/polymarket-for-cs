"""map 级 LOO 评估（路线 A 效果实验）。

把 demo 派生特征（① duel skill / ② 阵型打点 / ③ 手枪经济）作为"队伍历史
as-of 特征"预测地图胜者，评估无泄漏 AUC。① 单独已知 ≈0.51，看 ②③ 是否加分。

关键纪律：
  - 队伍身份 = roster_key（5 steamid 排序元组），与 map_rosters / rounds 一致。
  - LOO：队伍在某地图的特征 = 该 roster 在【其他比赛】的均值（leave-one-match-out，
    与 ① 的 player_skill_loo 同款；同一场 BO3 的其他地图也剔除，最保守）。
  - 标签 = won_map，预测"特征更高的一方赢"。

产物：
  - data/map_features.parquet   ③ 逐队逐图原始特征（compute_map_features 落盘）
  - reports/map_loo_eval.csv    逐特征 LOO AUC / 正确率
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .demo_features import compute_map_features
from .duel import discover_demos

# ② 阵型 / 打点特征列（rounds.py 输出，逐队逐图）
_FORMATION_FEATURES = ["avg_spread", "avg_n_places", "stack_rate"]
_EXECUTE_FEATURES = ["execute_rate", "cluster_rate", "avg_site_max",
                     "avg_contact_spread", "avg_contact_maxpair"]
# ③ 击杀效率 / 手枪 / 经济 / 道具特征列（compute_map_features，逐队逐图）
_DEMO_FEATURES = [
    "kpr", "kd", "headshot_rate", "opening_kill_rate", "awp_share",
    "thrusmoke_rate", "flash_assist_rate", "trade_rate",
    "pistol_win_rate", "post_pistol_win_rate",
    "eco_win_rate", "force_win_rate", "full_win_rate", "avg_equip_value",
    "flashes_per_round", "smokes_per_round", "mollies_per_round", "he_per_round",
]
# 需要 leave-one-match-out 的原始特征（② + ③ + won_map 作为历史胜率）
_RAW_FEATURES = _FORMATION_FEATURES + _EXECUTE_FEATURES + _DEMO_FEATURES
# 已经 LOO 过的特征（① roster_skill），直接使用不重复 LOO
_PRE_LOO_FEATURES = ["roster_skill"]


# --- ③ 落盘：compute_map_features -> 逐队逐图特征表 ---

def build_map_features(dem_paths: list[Path] | None = None) -> pd.DataFrame:
    """跑 compute_map_features，逐 (demo_path, roster_key) 落盘全量特征。

    （map_rosters 表只存了 roster/胜者，没存 ③ 的击杀/经济/道具特征，这里补齐。）
    """
    demos = dem_paths or discover_demos()
    cache = config.DATA_DIR / "map_features.parquet"
    if dem_paths is None and cache.exists():
        return pd.read_parquet(cache)  # 已落盘则复用，避免重复解析 366 张 demo
    rows: list[dict] = []
    for dp in demos:
        try:
            res = compute_map_features(dp)
        except Exception as e:  # 空 player_info 等截断 demo 跳过，不阻断
            print(f"跳过 {dp.name}: {e}")
            continue
        winner = res.get("winner_roster")
        map_name = res.get("map_name")
        for key, entry in res.get("teams", {}).items():
            r = {
                "demo_path": str(dp),
                "match_id": dp.parent.name,
                "roster_key": key,
                "map_name": map_name,
                "won_map": int(key == winner),
                "complete": int(bool(res.get("complete"))),
            }
            for f, v in entry.items():
                if f in ("team_number", "side_first_half", "roster"):
                    continue
                r[f] = v
            # 派生经济胜率（dataclass 只有分子分母）
            r["eco_win_rate"] = (entry["eco_wins"] / entry["eco_rounds"]
                                 if entry["eco_rounds"] else 0.0)
            r["force_win_rate"] = (entry["force_wins"] / entry["force_rounds"]
                                   if entry["force_rounds"] else 0.0)
            r["full_win_rate"] = (entry["full_wins"] / entry["full_rounds"]
                                  if entry["full_rounds"] else 0.0)
            rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.DATA_DIR / "map_features.parquet", index=False)
    return df


# --- 拼接 ①②③ 成一张宽表 ---

def load_consolidated() -> pd.DataFrame:
    """以 ③ map_features.parquet 为枢纽，接上 ①（roster_skill，已 LOO）和 ②（阵型/打点）。"""
    mf = pd.read_parquet(config.DATA_DIR / "map_features.parquet")

    skill = pd.read_csv(config.REPORTS_DIR / "roster_skill_eval.csv")
    skill = skill[["match_id", "map_name", "roster_key", "roster_skill", "coverage"]]
    df = mf.merge(skill, on=["match_id", "map_name", "roster_key"], how="left")

    fmt = pd.read_csv(config.REPORTS_DIR / "round_formation.csv")
    exe = pd.read_csv(config.REPORTS_DIR / "round_execute.csv")
    fmt_cols = ["demo_path", "roster"] + _FORMATION_FEATURES
    exe_cols = ["demo_path", "roster"] + _EXECUTE_FEATURES
    df = df.merge(fmt[fmt_cols], left_on=["demo_path", "roster_key"],
                  right_on=["demo_path", "roster"], how="left").drop(columns="roster")
    df = df.merge(exe[exe_cols], left_on=["demo_path", "roster_key"],
                  right_on=["demo_path", "roster"], how="left").drop(columns="roster")
    return df


def loo_match_mean(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """对每个 value_col，算"同 roster 在其他比赛"的均值（leave-one-match-out）。

    用 groupby.transform 的 sum/count 一次算完，O(n)；无其他比赛 => NaN。
    """
    out = df.copy()
    roster = df["roster_key"]
    match = df["match_id"]
    for c in value_cols:
        s = df[c].astype(float)
        r_sum = s.groupby(roster).transform("sum")
        r_cnt = s.groupby(roster).transform("count")
        m_sum = s.groupby([roster, match]).transform("sum")
        m_cnt = s.groupby([roster, match]).transform("count")
        loo_sum = r_sum - m_sum
        loo_cnt = r_cnt - m_cnt
        out[f"{c}_loo"] = loo_sum / loo_cnt
    return out


# --- 评估：逐特征 AUC + 组合逻辑回归 AUC ---

def _pair_scores(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """每张地图两方 -> (label=1 胜者 / 0 负者, score)。跳过缺失。"""
    rows: list[tuple[int, float]] = []
    for (mid, mn), sub in df.groupby(["match_id", "map_name"]):
        if len(sub) != 2:
            continue
        w = sub[sub["won_map"] == 1]
        l = sub[sub["won_map"] == 0]
        if len(w) != 1 or len(l) != 1:
            continue
        ws, ls = w[score_col].iloc[0], l[score_col].iloc[0]
        if pd.isna(ws) or pd.isna(ls):
            continue
        rows.append((1, float(ws)))
        rows.append((0, float(ls)))
    return pd.DataFrame(rows, columns=["label", "score"])


def _pair_auc_acc(pair: pd.DataFrame) -> tuple[float, float]:
    if not len(pair) or pair["label"].nunique() < 2:
        return float("nan"), float("nan")
    auc = roc_auc_score(pair["label"], pair["score"])
    # 正确率 = 胜者 score > 负者 score 的地图占比
    w = pair[pair["label"] == 1]["score"].to_numpy()
    l = pair[pair["label"] == 0]["score"].to_numpy()
    acc = float((w > l).mean())
    return round(auc, 4), round(acc, 4)


def _combined_auc(df: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    """组合逻辑回归：每地图取两方，按 (iloc[0] 一方) 定向差分，GroupKFold(by match)。"""
    rows, ys, grps = [], [], []
    for mid, sub in df.groupby("match_id"):
        # 一场比赛内可能有多个地图，逐图配对
        for _, mp in sub.groupby("map_name"):
            if len(mp) != 2:
                continue
            r1, r2 = mp.iloc[0], mp.iloc[1]
            d = [float(r1[c]) - float(r2[c]) for c in cols]
            if any(pd.isna(x) for x in d):
                continue
            rows.append(d)
            ys.append(int(r1["won_map"]))
            grps.append(mid)
    if len(rows) < 20 or len(set(ys)) < 2:
        return float("nan"), float("nan")
    X = np.asarray(rows, dtype=float)
    y = np.asarray(ys)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    proba = cross_val_predict(model, X, y, groups=np.asarray(grps),
                              cv=GroupKFold(3), method="predict_proba")[:, 1]
    acc = float(((proba > 0.5).astype(int) == y).mean())
    return round(roc_auc_score(y, proba), 4), round(acc, 4)


def evaluate(root: Path | None = None) -> dict:
    mf = build_map_features(root and discover_demos(root) or None)
    print(f"③ map_features：{len(mf)} 队·图（{mf['roster_key'].nunique()} 个 roster）")

    df = load_consolidated()
    df = loo_match_mean(df, ["won_map"] + _RAW_FEATURES)
    loo_cols = ["won_map_loo"] + [f"{c}_loo" for c in _RAW_FEATURES]
    all_cols = loo_cols + _PRE_LOO_FEATURES

    print(f"拼接后：{len(df)} 队·图 / {df['match_id'].nunique()} 图 / "
          f"{df['roster_key'].nunique()} roster")

    # 覆盖率：有多少地图的两方都有跨场历史（LOO 非空）
    both = df.dropna(subset=["won_map_loo"])
    both = both.groupby(["match_id", "map_name"]).filter(lambda g: len(g) == 2)
    print(f"两队都有跨场历史的地图：{both['match_id'].nunique()} / {df['match_id'].nunique()}")

    print("\n逐特征 LOO（预测地图胜者，无泄漏）：")
    print(f"{'feature':<24}{'AUC':>8}{'acc':>8}{'n_maps':>8}")
    per: list[dict] = []
    for c in all_cols:
        pair = _pair_scores(df, c)
        auc, acc = _pair_auc_acc(pair)
        per.append({"feature": c, "auc": auc, "acc": acc, "n_maps": len(pair) // 2})
        print(f"{c:<24}{auc:>8}{acc:>8}{len(pair) // 2:>8}")

    auc_c, acc_c = _combined_auc(df, all_cols)
    print(f"\n组合（①+②+③ 全部特征，逻辑回归 GroupKFold by match）：AUC={auc_c} acc={acc_c}")

    auc_23, acc_23 = _combined_auc(df, loo_cols)
    print(f"组合（仅 ②+③，去掉 roster_skill）：            AUC={auc_23} acc={acc_23}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per).to_csv(config.REPORTS_DIR / "map_loo_eval.csv", index=False)
    print("已写 reports/map_loo_eval.csv")
    return {"per_feature": per, "combined_all": (auc_c, acc_c),
            "combined_23": (auc_23, acc_23)}


if __name__ == "__main__":
    evaluate()
