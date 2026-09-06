"""roster 级 skill 回灌（方向①）：把 per-player 学习权重作用到地图/比赛胜负。

用户洞察：`player_id` 是交火胜负的头号信号（AUC 0.68 里权重第一），但不能
停在描述层——要把它变成 match 级预测的特征。核心是"roster skill"：
一队 5 名选手各自 as-of 收缩胜率的均值。

关键纪律（防泄漏）：per-player skill 必须 as-of——
  - 生产：预测比赛 M（start_date=D）时，只用 start_date < D 的 duel 样本。
  - 现在（demo 全同一天）：退化为 leave-one-match-out——预测某场比赛时，
    剔除该场自身的所有 duel 样本（GroupKFold 同款，杜绝同场泄漏）。

流程：
  1. 从每张 demo 抽取 roster（5 steamid）+ 地图胜者（复用 demo_features 的
     roster/winner 逻辑），落 `map_rosters` 表。
  2. duel 样本打 start_date（join demos manifest）。
  3. per-roster as-of skill = 5 名选手收缩胜率（剔除本场）的均值；
     预测地图胜者 = skill 更高的一方；评估正确率 / AUC。

产物：reports/roster_skill_eval.csv（每张地图两方 skill + 胜负）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, store
from .demo_features import compute_map_features
from .duel import _shrunken_rate, build_duel_dataset, discover_demos

_PRIOR_SKILL = 0.5  # 无 duel 样本选手的 skill 先验（= 收缩先验）


def _manifest_dates() -> dict[int, str]:
    """hltv_match_id -> start_date（从 demos manifest）。"""
    conn = store.connect()
    rows = conn.execute("SELECT hltv_match_id, start_date FROM demos").fetchall()
    conn.close()
    return {int(r["hltv_match_id"]): r["start_date"] for r in rows if r["hltv_match_id"]}


def _date_for_demo(dem_path: Path, md: dict[int, str]) -> str | None:
    """目录名 `hltv-{match_id}-{demo_id}` -> hltv_match_id -> start_date。"""
    m = re.search(r"hltv-(\d+)", dem_path.parent.name)
    return md.get(int(m.group(1))) if m else None


def extract_map_rosters(dem_paths: list[Path] | None = None,
                        force: bool = False) -> pd.DataFrame:
    """每张地图 -> 两方 roster + 胜者（复用 demo_features 的 roster/winner 逻辑）。

    结果持久化到 `map_rosters` 表；表里已含全部 demo 时直接读表（避免重复解析）。
    """
    demos = dem_paths or discover_demos()
    demo_paths = {str(dp) for dp in demos}

    conn = store.connect()
    existing = {r["demo_path"] for r in conn.execute("SELECT demo_path FROM map_rosters")}
    conn.close()
    if not force and demo_paths and demo_paths <= existing:
        conn = store.connect()
        df = pd.read_sql_query("SELECT * FROM map_rosters WHERE demo_path != ''", conn)
        conn.close()
        df["steamids"] = df["steamids"].map(json.loads)
        return df

    md = _manifest_dates()
    rows: list[dict] = []
    for dp in demos:
        try:
            res = compute_map_features(dp)
        except Exception as e:  # 单图失败不阻断
            print(f"跳过 {dp.name}: {e}")
            continue
        winner = res.get("winner_roster")
        for key, entry in res.get("teams", {}).items():
            rows.append({
                "demo_path": str(dp),
                "match_id": dp.parent.name,
                "start_date": _date_for_demo(dp, md),
                "map_name": res.get("map_name"),
                "roster_key": key,
                "team_number": entry.get("team_number"),
                "steamids": entry.get("roster", []),
                "won_map": int(key == winner),
                "complete": int(bool(res.get("complete"))),
            })
    df = pd.DataFrame(rows)
    _persist(df)
    return df


def _persist(df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn = store.connect()
    with conn:
        for r in df.itertuples(index=False):
            conn.execute(
                """
                INSERT INTO map_rosters (demo_path, roster_key, match_id, start_date, map_name,
                    team_number, steamids, won_map, complete)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(demo_path, roster_key) DO UPDATE SET
                    match_id=excluded.match_id, start_date=excluded.start_date,
                    map_name=excluded.map_name, team_number=excluded.team_number,
                    steamids=excluded.steamids, won_map=excluded.won_map,
                    complete=excluded.complete
                """,
                (r.demo_path, r.roster_key, r.match_id, r.start_date, r.map_name,
                 r.team_number, json.dumps(r.steamids), r.won_map, r.complete),
            )
    conn.close()


def tag_duel_dates(duel_df: pd.DataFrame, dem_paths: list[Path]) -> pd.DataFrame:
    """给 duel 样本打上 match 分组（父目录名）与 start_date。"""
    md = _manifest_dates()
    stem2dir = {dp.stem: dp.parent.name for dp in dem_paths}
    stem2date = {dp.stem: _date_for_demo(dp, md) for dp in dem_paths}
    out = duel_df.copy()
    out["match_group"] = out["match_id"].map(stem2dir).fillna(out["match_id"])
    out["start_date"] = out["match_id"].map(stem2date)
    return out


def load_duel_dataset(dem_paths: list[Path]) -> pd.DataFrame:
    """加载 duel 样本：优先用 duel.py 缓存的 data/duel_dataset.parquet（避免重解析）。"""
    cache = config.DATA_DIR / "duel_dataset.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return tag_duel_dates(df, dem_paths)
    return tag_duel_dates(build_duel_dataset(dem_paths), dem_paths)


def player_skill_loo(duel_df: pd.DataFrame, held_out_match: str) -> pd.DataFrame:
    """per-player 收缩胜率，剔除 held_out_match 的样本（leave-one-match-out）。

    返回 player_id / skill（收缩后，向 0.5 收缩）/ n_duels。
    """
    sub = duel_df[duel_df["match_group"] != held_out_match]
    if not len(sub):
        return pd.DataFrame(columns=["player_id", "skill", "n_duels"])
    g = sub.groupby("player_id").agg(n=("label", "size"), wins=("label", "sum")).reset_index()
    g["skill"] = [_shrunken_rate(n, w) for n, w in zip(g["n"], g["wins"])]
    return g.rename(columns={"n": "n_duels"})[["player_id", "skill", "n_duels"]]


def roster_skill(steamids: list[str], skill_df: pd.DataFrame) -> tuple[float, int]:
    """roster 的 as-of skill = 5 名选手 skill 均值；缺的选手按先验 0.5。

    返回 (skill, coverage)：coverage = 有样本的选手数（0..5）。
    """
    have = skill_df[skill_df["player_id"].isin(steamids)]
    if not len(have):
        return _PRIOR_SKILL, 0
    skill = have["skill"].sum() + _PRIOR_SKILL * (5 - len(have))
    return float(skill / 5.0), int(len(have))


def evaluate_roster_skill(dem_paths: list[Path] | None = None) -> dict:
    """leave-one-match-out：roster as-of skill -> 地图胜者，评估正确率 / AUC。"""
    demos = dem_paths or discover_demos()
    roster_df = extract_map_rosters(demos)
    duel_df = load_duel_dataset(demos)
    if roster_df.empty or duel_df.empty:
        return {"error": "无 roster 或 duel 数据"}

    preds: list[dict] = []
    for match, g in roster_df.groupby("match_id"):
        skill = player_skill_loo(duel_df, match)
        for _, row in g.iterrows():
            s, cov = roster_skill(row["steamids"], skill)
            preds.append({
                "match_id": match, "map_name": row["map_name"],
                "roster_key": row["roster_key"], "won_map": row["won_map"],
                "roster_skill": round(s, 4), "coverage": cov,
            })
    pred_df = pd.DataFrame(preds)
    acc, n_eval, auc = _eval_pair(pred_df)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(config.REPORTS_DIR / "roster_skill_eval.csv", index=False)
    return {"n_matches": int(roster_df["match_id"].nunique()),
            "n_map_rows": len(roster_df),
            "n_players": int(duel_df["player_id"].nunique()),
            "cross_match_overlap": _cross_overlap(duel_df),
            "accuracy": acc, "n_decided": n_eval, "auc": auc,
            "pred_df": pred_df}


def _eval_pair(pred_df: pd.DataFrame) -> tuple[float, int, float]:
    """每张地图两方 roster_skill（winner=1/loser=0），评估正确率与 AUC。"""
    pair_rows: list[dict] = []
    n_correct = n_eval = 0
    for (match, map_name), sub in pred_df.groupby(["match_id", "map_name"]):
        if len(sub) != 2:
            continue
        w = sub[sub["won_map"] == 1]
        l = sub[sub["won_map"] == 0]
        if not len(w) or not len(l):
            continue
        ws, ls = float(w["roster_skill"].iloc[0]), float(l["roster_skill"].iloc[0])
        pair_rows.append({"label": 1, "score": ws})
        pair_rows.append({"label": 0, "score": ls})
        if ws != ls:
            n_eval += 1
            n_correct += int(ws > ls)
    pair = pd.DataFrame(pair_rows)
    auc = (roc_auc_score(pair["label"], pair["score"])
           if len(pair) and pair["label"].nunique() > 1 else float("nan"))
    acc = round(n_correct / n_eval, 4) if n_eval else float("nan")
    return acc, n_eval, round(auc, 4)


def _cross_overlap(duel_df: pd.DataFrame) -> float:
    """出现在 >1 场比赛的选手占比（per-player 跨场可学性的先决条件）。"""
    if not len(duel_df):
        return 0.0
    cnt = duel_df.groupby("player_id")["match_group"].nunique()
    return round(float((cnt > 1).mean()), 4)


def sanity_roster_skill(duel_df: pd.DataFrame, roster_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """in-sample sanity：用全量 duel 的 per-player skill（含本场）预测地图胜者。

    这【不是】无泄漏预测，只验证"学出的 skill 确实能把胜者排更高"——信号存在性。
    真正无泄漏的 as-of/LOO 见 evaluate_roster_skill（需要跨场选手重叠，靠采集）。
    """
    g = duel_df.groupby("player_id").agg(n=("label", "size"), wins=("label", "sum")).reset_index()
    g["skill"] = [_shrunken_rate(n, w) for n, w in zip(g["n"], g["wins"])]
    skill = g[["player_id", "skill"]]
    preds: list[dict] = []
    for _, row in roster_df.iterrows():
        s, cov = roster_skill(row["steamids"], skill)
        preds.append({"match_id": row["match_id"], "map_name": row["map_name"],
                      "roster_key": row["roster_key"], "won_map": row["won_map"],
                      "roster_skill": round(s, 4), "coverage": cov})
    pred_df = pd.DataFrame(preds)
    acc, n_eval, auc = _eval_pair(pred_df)
    return pred_df, {"accuracy": acc, "n_decided": n_eval, "auc": auc}


def run() -> dict:
    res = evaluate_roster_skill()
    if "error" in res:
        print(res["error"])
        return res
    print("roster 级 skill 回灌：")
    print(f"  数据：{res['n_matches']} 场比赛 / {res['n_map_rows']} 行 roster 快照 / "
          f"{res['n_players']} 名选手")
    print(f"  跨场选手重叠（per-player 可学性先决条件）：{res['cross_match_overlap']*100:.0f}%")
    print(f"  [无泄漏 LOO] 预测地图胜者：正确 {res['n_decided']} 场中 {res['accuracy']}，AUC={res['auc']}")

    demos = discover_demos()
    _, san = sanity_roster_skill(load_duel_dataset(demos), extract_map_rosters(demos))
    print(f"  [in-sample sanity] 全量 skill 预测地图胜者：正确 {san['n_decided']} 场中 "
          f"{san['accuracy']}，AUC={san['auc']}")
    print("  已写 reports/roster_skill_eval.csv")
    print(res["pred_df"].sort_values(["match_id", "map_name"]).to_string(index=False))
    return res


if __name__ == "__main__":
    run()
