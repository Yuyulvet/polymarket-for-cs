"""交火（duel）级学习引擎：击杀可能性 + per-player 学习权重。

用户的核心洞察："击杀效率"应细化为"交火前胜率预测"——在交火发生前，
用（位置 + 个人状态 + 阵型 + 信息差 + 道具覆盖）预测这一方赢下这次交火的概率。
用大量样本堆出"不同选手在不同位置的胜率"（per-player 权重自动学出，不手标）。

数据构造（对称标签，见 engagements.py）：
  每次击杀给出两个有标签样本——
    (killer 交火前状态, label=1) 与 (victim 交火前状态, label=0)。
  特征取 kill_tick - pre_ticks 的快照，此时双方都活着、结果未定，无标签泄漏。

两个输出：
  1. per-player × per-place 胜率表（收缩估计，少样本向 0.5 收缩）——"不同选手在不同位置的胜率"
  2. P(win duel | 全状态) 的可预测性（分组 CV 的 AUC）——"击杀可能性"到底能不能学出来
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss

from . import config
from .engagements import extract_engagements

# 模型输入：数值特征（交火前双方状态 + 阵型 + 信息差）
_NUM_FEATURES = [
    "subject_health", "subject_armor", "subject_equip",
    "opponent_health", "opponent_armor", "opponent_equip",
    "distance", "man_diff", "subject_nearby", "opponent_nearby",
    "subject_alive", "opponent_alive",
    "subject_facing", "opponent_facing", "subject_flashed", "opponent_flashed",
    "subject_has_flash", "subject_has_smoke", "subject_has_he", "subject_has_molly",
    "opponent_has_flash", "opponent_has_smoke", "opponent_has_he", "opponent_has_molly",
]
# 分类特征（player_id 是关键：让模型自动学 per-player 权重）
# subject_weapon/opponent_weapon = 双方各自手持武器（active_weapon_name，已修正泄漏）
_CAT_FEATURES = ["player_id", "subject_place", "opponent_place",
                 "subject_weapon", "opponent_weapon"]

# 手持武器归到粗粒度家族（active_weapon_name 有几十种刀/枪，归并降噪）
_WEAPON_FAMILIES = [
    ("sniper", ["awp", "ssg", "scar20", "g3sg1", "scout"]),
    ("lmg", ["m249", "negev"]),
    ("rifle", ["ak-47", "ak47", "m4a1", "m4a4", "aug", "sg 556", "sg556", "galil", "famas"]),
    ("smg", ["mp5", "mp7", "mp9", "mac-10", "mac10", "p90", "ump", "bizon"]),
    ("shotgun", ["nova", "xm1014", "mag-7", "mag7", "sawed"]),
    ("pistol", ["glock", "usp", "p2000", "hkp2000", "p250", "five-seven", "fiveseven",
                "tec-9", "tec9", "cz75", "deagle", "desert eagle", "revolver", "r8",
                "elite", "dual berettas"]),
    ("knife", ["knife", "karambit", "bayonet", "butterfly", "talon", "flip", "gut",
               "huntsman", "classic", "skeleton", "navaja", "stiletto", "ursus", "nomad",
               "survival", "paracord", "falchion", "bowie", "shadow", "m9"]),
]


def _weapon_family(w: str) -> str:
    s = (w or "").lower()
    for fam, keys in _WEAPON_FAMILIES:
        if any(k in s for k in keys):
            return fam
    return "nade"

# 收缩估计的伪样本数（越大越保守，向先验 0.5 收缩）
_PRIOR_DUELS = 5.0


def discover_demos(root: Path | None = None) -> list[Path]:
    """递归发现已解压的 .dem（含 manifest 未记录的旧目录）。"""
    root = Path(root) if root else (config.DATA_DIR / "demos")
    return sorted(root.rglob("*.dem"))


def load_player_names(dem_paths: list[Path]) -> dict[str, str]:
    """steamid -> 玩家名（解析每张 demo 的 player_info，取最后一次出现的名字）。"""
    from demoparser2 import DemoParser

    names: dict[str, str] = {}
    for dp in dem_paths:
        try:
            info = DemoParser(str(dp)).parse_player_info()
        except Exception:
            continue
        if info is None or not len(info):
            continue
        for r in info.itertuples(index=False):
            sid = str(getattr(r, "steamid", None))
            nm = str(getattr(r, "name", "") or "")
            if sid and sid != "None" and nm:
                names[sid] = nm
    return names


def build_duel_dataset(dem_paths: list[Path], pre_ticks: int = 64) -> pd.DataFrame:
    """把一批 .dem 转成交火级对称样本（每击杀两行：killer=1 / victim=0）。

    返回列：
      match_id（demo 名，用于分组 CV 防泄漏）
      player_id（subject 的 steamid）
      subject_*（subject 交火前状态） / opponent_*（对方状态）
      distance, man_diff（subject 视角的人数差）, subject_nearby, opponent_nearby
      subject_alive, opponent_alive, weapon, subject_place, opponent_place
      label（subject 是否赢下这次交火）
    """
    frames: list[pd.DataFrame] = []
    for dp in dem_paths:
        try:
            eng = extract_engagements(str(dp), pre_ticks=pre_ticks)
        except Exception as e:  # 单张 demo 解析失败不阻断整批（截断/异常 demo）
            print(f"  跳过 {dp.name}（交火解析失败）: {e}")
            continue
        if eng is None or not len(eng):
            continue
        # killer 样本（label=1）
        k = pd.DataFrame({
            "match_id": dp.stem,
            "player_id": eng["att_steamid"],
            "subject_health": eng["att_health"], "subject_armor": eng["att_armor"],
            "subject_equip": eng["att_equip"], "subject_place": eng["att_place"],
            "subject_nearby": eng["att_nearby"], "subject_alive": eng["att_alive"],
            "opponent_health": eng["vic_health"], "opponent_armor": eng["vic_armor"],
            "opponent_equip": eng["vic_equip"], "opponent_place": eng["vic_place"],
            "opponent_nearby": eng["vic_nearby"], "opponent_alive": eng["vic_alive"],
            "distance": eng["distance"], "man_diff": eng["man_diff"],
            "subject_weapon": eng["att_weapon"].map(_weapon_family),
            "opponent_weapon": eng["vic_weapon"].map(_weapon_family),
            "subject_facing": eng["att_facing"].astype(int),
            "opponent_facing": eng["vic_facing"].astype(int),
            "subject_flashed": eng["att_flashed"].astype(int),
            "opponent_flashed": eng["vic_flashed"].astype(int),
            "subject_has_flash": eng["att_has_flash"].astype(int),
            "subject_has_smoke": eng["att_has_smoke"].astype(int),
            "subject_has_he": eng["att_has_he"].astype(int),
            "subject_has_molly": eng["att_has_molly"].astype(int),
            "opponent_has_flash": eng["vic_has_flash"].astype(int),
            "opponent_has_smoke": eng["vic_has_smoke"].astype(int),
            "opponent_has_he": eng["vic_has_he"].astype(int),
            "opponent_has_molly": eng["vic_has_molly"].astype(int),
            "label": 1,
        })
        # victim 样本（label=0，视角翻转）
        v = pd.DataFrame({
            "match_id": dp.stem,
            "player_id": eng["vic_steamid"],
            "subject_health": eng["vic_health"], "subject_armor": eng["vic_armor"],
            "subject_equip": eng["vic_equip"], "subject_place": eng["vic_place"],
            "subject_nearby": eng["vic_nearby"], "subject_alive": eng["vic_alive"],
            "opponent_health": eng["att_health"], "opponent_armor": eng["att_armor"],
            "opponent_equip": eng["att_equip"], "opponent_place": eng["att_place"],
            "opponent_nearby": eng["att_nearby"], "opponent_alive": eng["att_alive"],
            "distance": eng["distance"], "man_diff": -eng["man_diff"],
            "subject_weapon": eng["vic_weapon"].map(_weapon_family),
            "opponent_weapon": eng["att_weapon"].map(_weapon_family),
            "subject_facing": eng["vic_facing"].astype(int),
            "opponent_facing": eng["att_facing"].astype(int),
            "subject_flashed": eng["vic_flashed"].astype(int),
            "opponent_flashed": eng["att_flashed"].astype(int),
            "subject_has_flash": eng["vic_has_flash"].astype(int),
            "subject_has_smoke": eng["vic_has_smoke"].astype(int),
            "subject_has_he": eng["vic_has_he"].astype(int),
            "subject_has_molly": eng["vic_has_molly"].astype(int),
            "opponent_has_flash": eng["att_has_flash"].astype(int),
            "opponent_has_smoke": eng["att_has_smoke"].astype(int),
            "opponent_has_he": eng["att_has_he"].astype(int),
            "opponent_has_molly": eng["att_has_molly"].astype(int),
            "label": 0,
        })
        frames.extend([k, v])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # 丢弃快照数据缺口的样本（整行 NaN），保证下游 Logistic 等不接受 NaN
    df = df.dropna(subset=["subject_health", "opponent_health",
                           "subject_equip", "opponent_equip"])
    df["label"] = df["label"].astype(int)
    return df


# --- per-player × per-place 胜率（收缩估计）---

def _shrunken_rate(n: float, wins: float, prior: float = 0.5, k: float = _PRIOR_DUELS) -> float:
    """贝叶斯收缩：少量样本时向先验 0.5 靠拢，样本越多越接近经验值。"""
    return round((wins + k * prior) / (n + k), 4)


# --- per-player 学习权重（Bradley-Terry / L2 logistic，Glicko 式）---

def player_rating(df: pd.DataFrame, C: float = 1.0) -> pd.DataFrame:
    """学出来的 per-player skill 评分（log-odds，越高越强）。

    以 player_id 为主效应，同时控制 位置/经济/武器/人数差 这些"情境"协变量，
    于是 skill 系数 = 剥离了位置与装备优势后的"纯个人能力"。
    L2 正则天然把少样本玩家向 0 收缩（= Glicko 式的先验收缩）。
    """
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_cols = ["player_id", "subject_place", "opponent_place", "subject_weapon", "opponent_weapon"]
    num_cols = ["subject_equip", "opponent_equip", "man_diff"]

    enc = OneHotEncoder(handle_unknown="ignore")
    X_cat = enc.fit_transform(df[cat_cols].astype(str))
    # 数值协变量标准化，避免 equip(数百~数千) 与 man_diff(±几) 量纲差异导致 L2 不均、不收敛
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols].to_numpy(dtype=float))
    X = sparse.hstack([X_cat, sparse.csr_matrix(X_num)]).tocsr()
    y = df["label"].to_numpy()

    clf = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    clf.fit(X, y)

    players = [str(p) for p in enc.categories_[0]]
    n = len(players)
    skill = clf.coef_[0][:n]

    counts = df.groupby("player_id")["label"].size().reindex(players).fillna(0).astype(int)
    wins = df.groupby("player_id")["label"].sum().reindex(players).fillna(0).astype(int)

    out = pd.DataFrame({"player_id": players, "skill": skill,
                        "n_duels": counts.values, "wins": wins.values})
    out["raw_win_rate"] = (out["wins"] / out["n_duels"].clip(lower=1)).round(4)
    out["shrunken_rate"] = [_shrunken_rate(n, w) for n, w in zip(out["n_duels"], out["wins"])]
    out = out.sort_values("skill", ascending=False).reset_index(drop=True)
    return out


def player_skill(df: pd.DataFrame) -> pd.DataFrame:
    """per-player 汇总 + per-player×per-place 胜率表。

    返回 (players, player_place)：players 每行一个 steamid（n_duels, win_rate 收缩后）；
    player_place 是长表（steamid × place -> n, win_rate 收缩后）。
    """
    if not len(df):
        return pd.DataFrame(), pd.DataFrame()
    g = df.groupby("player_id").agg(n_duels=("label", "size"), wins=("label", "sum")).reset_index()
    g["win_rate"] = [_shrunken_rate(n, w) for n, w in zip(g["n_duels"], g["wins"])]

    pp = df.groupby(["player_id", "subject_place"]).agg(
        n=("label", "size"), wins=("label", "sum")).reset_index()
    pp["win_rate"] = [_shrunken_rate(n, w) for n, w in zip(pp["n"], pp["wins"])]
    return g, pp


# --- 击杀可能性可预测性（分组 CV）---

def evaluate_predictability(df: pd.DataFrame, n_splits: int = 4) -> dict:
    """P(win duel | 全状态) 的可预测性：按 match 分组 CV，避免同一场比赛泄漏。

    返回 AUC / log-loss / 特征重要性。
    """
    if len(df) < 50 or df["match_id"].nunique() < 2:
        return {"error": f"样本不足：{len(df)} 行 / {df['match_id'].nunique()} 场比赛"}

    X = df[_NUM_FEATURES + _CAT_FEATURES].copy()
    y = df["label"].to_numpy()
    groups = df["match_id"].to_numpy()

    # 分类列转 str（HGB 的原生分类需要字符串或有序整数编码）
    for c in _CAT_FEATURES:
        X[c] = X[c].astype(str)
    cat_idx = [X.columns.get_loc(c) for c in _CAT_FEATURES]

    model = HistGradientBoostingClassifier(
        categorical_features=cat_idx,
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=20, early_stopping=False, random_state=0,
    )
    cv = GroupKFold(n_splits=min(n_splits, df["match_id"].nunique()))
    proba = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    ll = log_loss(y, proba)

    # 特征重要性：全体拟合 + permutation importance（子样本加速，仅解释用）
    model.fit(X, y)
    sub = X.sample(min(800, len(X)), random_state=0)
    r = permutation_importance(model, sub, df["label"].loc[sub.index].to_numpy(),
                               n_repeats=5, random_state=0, n_jobs=-1)
    imp = sorted(zip(X.columns, r.importances_mean), key=lambda t: -t[1])
    return {"auc": round(auc, 4), "log_loss": round(ll, 4), "n": len(df),
            "n_matches": int(df["match_id"].nunique()),
            "base_rate": round(float(y.mean()), 4),
            "feature_importance": imp[:15]}


def run(root: Path | None = None) -> dict:
    demos = discover_demos(root)
    print(f"发现 {len(demos)} 张 .dem")
    names = load_player_names(demos)
    df = build_duel_dataset(demos)
    print(f"交火级对称样本：{len(df)} 行（{df['match_id'].nunique()} 张地图，"
          f"{df['player_id'].nunique()} 名选手）")
    if not len(df):
        return {}
    # 先落盘交火样本，下游（skill/rating/可预测性）失败不丢这次昂贵解析
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATA_DIR / "duel_dataset.parquet", index=False)

    _, pp = player_skill(df)
    pp = pp.copy()
    pp["name"] = pp["player_id"].map(names)

    players = player_rating(df)
    players["name"] = players["player_id"].map(names)

    print(f"\nper-player 学习权重（L2-logistic 调整后，Top 15 by skill）：")
    cols = ["name", "skill", "n_duels", "wins", "raw_win_rate", "shrunken_rate"]
    print(players[cols].head(15).to_string(index=False))

    print(f"\nper-player × per-place 胜率（n>=8，按 n 排序前 20）：")
    pp_f = pp[pp["n"] >= 8].sort_values("n", ascending=False)[["name", "subject_place", "n", "wins", "win_rate"]].head(20)
    print(pp_f.to_string(index=False) if len(pp_f) else "  （样本不足，无 n>=8 条目）")

    print("\n击杀可能性可预测性：")
    res = evaluate_predictability(df)
    if "error" in res:
        print("  ", res["error"])
    else:
        print(f"  AUC={res['auc']}  log_loss={res['log_loss']}  n={res['n']}  "
              f"matches={res['n_matches']}  base_rate={res['base_rate']}")
        print("  特征重要性 Top：")
        for name, w in res["feature_importance"]:
            print(f"    {name:<20} {w:.4f}")

    # 落盘：选手权重 + 位置胜率 + 交火样本，供下游 match 级模型复用
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    players.to_csv(config.REPORTS_DIR / "player_skill.csv", index=False)
    pp.to_csv(config.REPORTS_DIR / "player_place_winrate.csv", index=False)
    print(f"\n已写：reports/player_skill.csv, reports/player_place_winrate.csv, data/duel_dataset.parquet")
    return {"players": players, "player_place": pp, "eval": res}


if __name__ == "__main__":
    run()
