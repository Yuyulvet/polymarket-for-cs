"""赛前 form + 赛中比分 → map winner（真实交易场景：读局）。

核心问题：赛前 per-player form（0.66，无泄漏）在赛中能否叠在比分之上给增量？
比分已知主导（after3 0.72 / after6 0.82 / after12 0.90），form 早期可能有用、
后期可能被吸收。逐 horizon 对比 裸比分 vs 比分+form。
"""
from __future__ import annotations

import pandas as pd

from .mapwin_players import build_team_form, load_player_features, _auc
from .round_model import build_round_dataset
from .totalrounds import build_map_table, per_team_features


def add_score_features(mt: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """给 map 表加 home 视角的 as-of 比分（after 3/6/12/全部）。"""
    home_map = mt.set_index("demo_path")["home"].to_dict()
    away_map = mt.set_index("demo_path")["away"].to_dict()
    snaps: dict[str, dict[int, int]] = {}
    for dp, sub in df.groupby("demo_path"):
        home = home_map.get(dp); away = away_map.get(dp)
        if not home or not away:
            continue
        sub = sub.sort_values("round_num")
        won = sub[sub["winner_roster"].astype(str) != ""]
        hs = 0; aw = 0
        snap = {}
        for r in won.itertuples(index=False):
            if r.winner_roster == home:
                hs += 1
            elif r.winner_roster == away:
                aw += 1
            n = hs + aw
            if n in (3, 6, 12):
                snap[n] = hs - aw
        snaps[dp] = snap
    for h in (3, 6, 12):
        mt[f"score_after_{h}"] = [snaps.get(dp, {}).get(h, float("nan")) for dp in mt["demo_path"]]
    return mt


def evaluate() -> dict:
    df = build_round_dataset()
    mt = build_map_table(df)
    mt = mt[(mt["total"] >= 13) & (mt["total"] <= 24)].copy()
    mt = per_team_features(mt)
    mt["margin_gap"] = mt["home_avg_margin"] - mt["away_avg_margin"]
    pf = load_player_features()
    mt = build_team_form(pf, mt)
    mt = add_score_features(mt, df)

    print("=" * 70)
    print("赛中 map winner AUC：裸比分 vs 比分+赛前form")
    print("=" * 70)
    print(f"{'horizon':<10}{'裸比分':>10}{'比分+form':>12}{'Δ':>8}{'n':>6}")
    for h in (3, 6, 12):
        s = f"score_after_{h}"
        a_score, n = _auc(mt, [s], "home_win")
        a_both, _ = _auc(mt, [s, "form_kd_gap"], "home_win")
        print(f"{'after '+str(h):<10}{a_score:>10.4f}{a_both:>12.4f}{a_both-a_score:>+8.4f}{n:>6}")

    # 赛前 form 单独（对照）
    a_form, nf = _auc(mt, ["form_kd_gap"], "home_win")
    print(f"\n赛前 form 单独：AUC = {a_form:.4f} (n={nf})")
    print("（早期比分+form 的增量 = form 是否在比分局势不明朗时仍提供信息）")


if __name__ == "__main__":
    evaluate()
