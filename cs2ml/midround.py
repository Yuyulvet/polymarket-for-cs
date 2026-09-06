"""30s 中局模型：回合 freeze_end + 30s 的战术状态 → 本回合胜方。

这是「半职业读局」的量化：在回合进行到 30s 时，谁有人数优势、谁拿到首杀、
谁阵型压点——这些是比分/经济里没有的、市场看不到的信息。

提取（每回合 freeze+30s 快照）：
  - ct_alive / t_alive：双方存活人数（5 - 30s 前阵亡数）
  - ct_deaths30 / t_deaths30：30s 内双方阵亡数
  - first_kill_side：首杀是哪边拿的（CT/T/none）

纪律：与 round_model 一致，GroupKFold by match_id（= 一场 BO3）。
评估：baseline（经济+边+地图）vs + 30s 状态，看 AUC 能否抬过 0.75。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .demo_features import _group_side_first_half, _steamid_to_group
from .duel import discover_demos
from .round_model import FIRST_HALF_ROUNDS, build_round_dataset

SECONDS = 30
TICKS = 64


def _flip(side: str) -> str:
    return "T" if side == "CT" else "CT"


def compute_midround(dp: Path) -> pd.DataFrame | None:
    """一张 demo -> 每回合 30s 中局状态行。"""
    parser = DemoParser(str(dp))
    player_info = parser.parse_player_info()
    if player_info is None or not len(player_info):
        return None
    sid2group = _steamid_to_group(player_info)

    freeze = parser.parse_event("round_freeze_end")
    if freeze is None or not len(freeze):
        return None
    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())

    kills = parser.parse_event("player_death")
    if kills is None or not len(kills):
        kills = pd.DataFrame(columns=["tick", "attacker_steamid", "user_steamid"])

    ticks_start = parser.parse_ticks(["team_name"], ticks=[freeze_ticks[0] + 1])
    first_half_side = _group_side_first_half(ticks_start, sid2group)

    def side_of(g, r: int) -> str:
        s = str(first_half_side.get(g, "")).upper()
        return s if r <= FIRST_HALF_ROUNDS else _flip(s)

    k = kills.copy()
    k["attacker_steamid"] = k["attacker_steamid"].astype(str)
    k["user_steamid"] = k["user_steamid"].astype(str)
    k["ag"] = k["attacker_steamid"].map(sid2group)
    k["vg"] = k["user_steamid"].map(sid2group)
    k = k.dropna(subset=["ag", "vg"])

    # 归回合（与 demo_features._round_of 一致）
    def _round_of(tick: int) -> int:
        r = 1
        for i, t in enumerate(freeze_ticks):
            if tick >= t:
                r = i + 1
            else:
                break
        return r

    k["round"] = k["tick"].astype(int).map(_round_of)
    k["tick"] = k["tick"].astype(int)

    rows: list[dict] = []
    for i, ft in enumerate(freeze_ticks):
        r = i + 1
        mark30 = ft + SECONDS * TICKS
        rd = k[k["round"] == r]
        before30 = rd[rd["tick"] <= mark30]
        ct_deaths = t_deaths = 0
        fk_side = "none"
        if len(before30):
            for row in before30.itertuples(index=False):
                vs = side_of(int(row.vg), r)
                if vs == "CT":
                    ct_deaths += 1
                elif vs == "T":
                    t_deaths += 1
        # 首杀（本回合第一个击杀的击杀者边）
        if len(rd):
            fk = rd.sort_values("tick").iloc[0]
            fk_side = side_of(int(fk.ag), r)
        rows.append({
            "demo_path": str(dp),
            "match_id": dp.parent.name,
            "round_num": r,
            "ct_deaths30": ct_deaths,
            "t_deaths30": t_deaths,
            "ct_alive": 5 - ct_deaths,
            "t_alive": 5 - t_deaths,
            "first_kill_side": fk_side,
        })
    return pd.DataFrame(rows)


def build_midround(dem_paths: list[Path] | None = None) -> pd.DataFrame:
    demos = dem_paths or discover_demos()
    cache = config.DATA_DIR / "midround.parquet"
    if dem_paths is None and cache.exists():
        return pd.read_parquet(cache)
    frames: list[pd.DataFrame] = []
    for dp in demos:
        try:
            f = compute_midround(dp)
            if f is not None and len(f):
                frames.append(f)
        except Exception as e:
            print(f"跳过 {dp.name}: {e}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if dem_paths is None and not df.empty:
        df.to_parquet(cache, index=False)
    return df


CAT = ["map_name", "round_class", "ct_tier", "t_tier"]
NUM_BASE = ["ct_equip", "t_equip", "round_num"]
NUM_30 = ["ct_alive", "t_alive"]
CAT_30 = ["first_kill_side"]


def _auc(df: pd.DataFrame, features: list[str]) -> float:
    y = df["label_ct_win"].to_numpy()
    n_splits = min(5, df["match_id"].nunique())
    cat = [c for c in features if c in (CAT + CAT_30)]
    num = [c for c in features if c not in (CAT + CAT_30)]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    proba = np.empty(len(df))
    gkf = GroupKFold(n_splits)
    for tr, te in gkf.split(df, groups=df["match_id"].to_numpy()):
        model.fit(df.iloc[tr][features], df.iloc[tr]["label_ct_win"])
        proba[te] = model.predict_proba(df.iloc[te][features])[:, 1]
    return roc_auc_score(y, proba)


def evaluate() -> dict:
    rd = build_round_dataset()
    mr = build_midround()
    df = rd.merge(mr, on=["demo_path", "round_num"], how="inner", suffixes=("", "_m"))
    df = df.dropna(subset=["ct_equip", "t_equip"]).copy()
    df["ct_equip"] = df["ct_equip"].astype(float)
    df["t_equip"] = df["t_equip"].astype(float)
    print(f"合并后：{len(df)} 回合 / {df['match_id'].nunique()} series "
          f"(round_dataset {len(rd)} -> 有30s快照 {len(df)})")

    # 30s 状态本身的先验强度：有人数优势时 CT 胜率
    adv = df[df["ct_alive"] != df["t_alive"]]
    if len(adv):
        ct_adv = adv[adv["ct_alive"] > adv["t_alive"]]["label_ct_win"].mean()
        t_adv = adv[adv["t_alive"] > adv["ct_alive"]]["label_ct_win"].mean()
        even = df[df["ct_alive"] == df["t_alive"]]["label_ct_win"].mean()
        print(f"\n30s 人数差 → CT 胜率：CT多{round(ct_adv,3)} / 均势{round(even,3)} / T多{round(t_adv,3)}")
        print(f"首杀边 → CT 胜率：CT首杀{round(df[df.first_kill_side=='CT'].label_ct_win.mean(),3)} "
              f"/ T首杀{round(df[df.first_kill_side=='T'].label_ct_win.mean(),3)} "
              f"/ 无首杀{round(df[df.first_kill_side=='none'].label_ct_win.mean(),3)}")

    base = _auc(df, CAT + NUM_BASE)
    print(f"\nbaseline（经济+边+地图）：AUC = {base:.4f}")

    mid = _auc(df, CAT + NUM_BASE + NUM_30 + CAT_30)
    print(f"+ 30s 状态（存活+首杀）：AUC = {mid:.4f}")

    only_adv = _auc(df, CAT + NUM_BASE + ["ct_alive", "t_alive"])
    print(f"+ 仅存活人数：AUC = {only_adv:.4f}")
    return {"base": base, "mid": mid, "only_adv": only_adv}


if __name__ == "__main__":
    evaluate()
