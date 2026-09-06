"""bo3.gg 选手 score（外部评分）→ 叠加到 per-player form，重测 map winner。

用户方向：demo 数据（局内 K/D、ADR）+ 外部选手数据（HLTV/bo3.gg 评分）结合。
bo3.gg team_rankings 带 roster_players(nickname + score)，nickname 与 demo 选手名一致，
映射链 nickname -> demo name -> steamid 成立，无需爬 HLTV。

注意（look-ahead 诚实地讲）：bo3.gg score 是**当前**快照。对历史 demo 回测，
它是"事后评分"，会高估（对最老 demo 是泄漏、对最新 demo 近似可用）。所以这里
bo3.gg score 的 AUC 是**上界**；真正无泄漏的赛前信号仍是 demo 跨图 form(0.66)。
"""
from __future__ import annotations

import difflib
import numpy as np
import pandas as pd

from . import config
from .bo3gg import Bo3Client
from .mapwin_players import build_team_form, load_player_features, _auc
from .round_model import build_round_dataset
from .totalrounds import build_map_table, per_team_features


def fetch_player_scores(pages: int = 4, per_page: int = 100) -> dict[str, float]:
    """返回 nickname_lower -> score 的全局映射（跨所有上榜队伍）。"""
    c = Bo3Client()
    scores: dict[str, float] = {}
    for page in range(1, pages + 1):
        try:
            r = c.team_rankings(page=page, per_page=per_page)
        except Exception as e:
            print(f"  bo3.gg rankings page {page} 失败: {e}")
            break
        data = r.get("data") or []
        if not data:
            break
        for team in data:
            for p in team.get("roster_players", []):
                nick = str(p.get("nickname") or "").strip().lower()
                sc = p.get("score")
                if nick and sc is not None:
                    scores[nick] = float(sc)
    return scores


def map_player_scores(pf: pd.DataFrame, scores: dict[str, float]) -> pd.DataFrame:
    """给 player_features 的每名选手挂 bo3.gg score（按 nickname 匹配 + 模糊兜底）。"""
    all_nicks = list(scores.keys())
    name2score: dict[str, float] = {}

    def lookup(name: str) -> float | None:
        n = str(name).strip().lower()
        if n in scores:
            return scores[n]
        # 模糊兜底（处理 sh1ro/SH1R0 这类 0/o 差异，ratio 0.8）
        m = difflib.get_close_matches(n, all_nicks, n=1, cutoff=0.8)
        return scores[m[0]] if m else None

    pf = pf.copy()
    pf["bo3_score"] = pf["name"].map(lookup)
    return pf


def evaluate() -> dict:
    df = build_round_dataset()
    mt = build_map_table(df)
    mt = mt[(mt["total"] >= 13) & (mt["total"] <= 24)].copy()
    mt = per_team_features(mt)
    mt["margin_gap"] = mt["home_avg_margin"] - mt["away_avg_margin"]

    pf = load_player_features()
    scores = fetch_player_scores()
    print(f"bo3.gg 上榜选手 {len(scores)} 名")
    pf = map_player_scores(pf, scores)
    cov = pf["bo3_score"].notna().mean()
    print(f"demo 选手 -> bo3.gg score 匹配率：{cov:.1%}（{pf['bo3_score'].notna().sum()}/{len(pf)} 选手×地图）")

    mt = build_team_form(pf, mt)  # form 用全 pf（bo3_score 列不参与 form）

    # 每 map：主客队 bo3.gg score avg / max
    demo2home = mt.set_index("demo_path")["home"].to_dict()
    demo2away = mt.set_index("demo_path")["away"].to_dict()
    pf2 = pf.copy()
    pf2["home_roster"] = pf2["demo_path"].map(demo2home)
    pf2["is_home"] = pf2["roster_key"] == pf2["home_roster"]
    agg = pf2.dropna(subset=["bo3_score"]).groupby(["demo_path", "is_home"])["bo3_score"].agg(
        ["mean", "max"]).unstack()
    agg.columns = ["away_score_avg", "home_score_avg", "away_score_max", "home_score_max"]
    mt = mt.merge(agg.reset_index(), on="demo_path", how="left")
    mt["score_avg_gap"] = mt["home_score_avg"] - mt["away_score_avg"]
    mt["score_star_gap"] = mt["home_score_max"] - mt["away_score_max"]

    print("\n" + "=" * 70)
    print("赛前 map winner AUC：demo form + bo3.gg score")
    print("=" * 70)
    print(f"{'特征':<34}{'AUC':>8}{'n':>6}")
    groups = {
        "form K/D 差（无泄漏基线）": ["form_kd_gap"],
        "bo3.gg score avg 差": ["score_avg_gap"],
        "bo3.gg score star 差": ["score_star_gap"],
        "form + bo3.gg score": ["form_kd_gap", "score_avg_gap"],
        "form + score + roster强度": ["form_kd_gap", "score_avg_gap", "margin_gap"],
    }
    for name, feats in groups.items():
        a, n = _auc(mt, feats, "home_win")
        print(f"  {name:<34}{a:>8.4f}{n:>6}")

    return {"cov": cov}


if __name__ == "__main__":
    evaluate()
