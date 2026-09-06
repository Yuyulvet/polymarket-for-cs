"""team_identity.py —— 队伍身份的唯一可靠来源：team1 的 roster_key。

背景（[[cs2-homewin-ct-alignment-bug]] 根因）：
  demo 侧没有「team1」这个稳定概念，只有「边」（CT/T）和「roster」（5 个 steamid）。
  - team_number 是 demoparser 内部索引，跨图翻转、同图 11% 与 CT 相反（不可靠）。
  - 上半场 CT（round-1 ct_roster）会因 BO3 换边在 team1/team2 之间翻转（不可靠）。
  - 唯一稳定的是 roster（5 个排序 steamid 元组）。

team1 的定义 = 市场（Polymarket）的 team1 列 = 文件名首队。要拿到 team1 的 roster，
用 raw 的 Map1 结算反推：
  Map1 里若 team1 赢 → winner_roster(Map1) 就是 team1 的 roster；
  否则 team1 = Map1 里输的那支队（ct/t 里不是 winner 的那支）。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

_MAP_NUM = re.compile(r"Map (\d)")
_DEMO_MAP_NUM = re.compile(r"-m(\d)-")


def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\bteam\b|\besports\b|\bthe\b|\bgaming\b|\bclub\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def build_team1_roster_map() -> dict[str, str]:
    """match_id -> team1 的 roster_key（5 个排序 steamid 逗号串）。

    覆盖所有「有 raw Map1 干净结算」的 match（约 259 场）。无 raw 的 match 不在 map 里。
    """
    rd = pd.read_parquet(config.DATA_DIR / "round_dataset.parquet")
    raw = pd.read_parquet(config.DATA_DIR / "polymarket_inplay_raw.parquet")

    # demo 侧：每 (match,map) 的 winner_roster + round1 ct/t
    dd = rd.copy()
    dd["map_num"] = dd["demo_path"].map(lambda d: int(m.group(1)) if (m := _DEMO_MAP_NUM.search(str(d))) else None)
    dd = dd.dropna(subset=["map_num"]).copy()
    dd["map_num"] = dd["map_num"].astype(int)
    dd = dd.sort_values("round_num")
    winr = (dd.groupby(["match_id", "map_num"])
            .apply(lambda g: g["winner_roster"].value_counts().index[0], include_groups=False)
            .rename("winner_roster").reset_index())
    r1 = (dd.groupby(["match_id", "map_num"])
          .agg(ct_roster=("ct_roster", "first"), t_roster=("t_roster", "first")).reset_index())

    # raw 侧：Map1 里 team1 是否结算赢
    r = raw.copy()
    r["map_num"] = r["map_market"].str.extract(r"Map (\d)").astype(int)
    r["is_home"] = r["outcome"].astype(str).map(_norm) == r["team1"].astype(str).map(_norm)
    hwin = (r[r["is_home"]].sort_values("t").groupby(["match_id", "map_num"])["p"].last()
            .rename("team1_last").reset_index())
    hwin["team1_won"] = (hwin["team1_last"] >= 0.9).astype(int)

    m1 = (winr[winr["map_num"] == 1]
          .merge(r1[r1["map_num"] == 1], on=["match_id", "map_num"])
          .merge(hwin[hwin["map_num"] == 1], on=["match_id", "map_num"]))
    # 只保留结算干净的（一方 >=0.9 或 <=0.1）
    m1 = m1[(m1["team1_last"] >= 0.9) | (m1["team1_last"] <= 0.1)].copy()
    # 防御：winner 必须是两支之一
    m1 = m1[(m1["winner_roster"] == m1["ct_roster"]) | (m1["winner_roster"] == m1["t_roster"])].copy()

    def _t1(row) -> str:
        w, c, t, won = row["winner_roster"], row["ct_roster"], row["t_roster"], row["team1_won"]
        return w if won else (t if w == c else c)

    m1["team1_roster"] = m1.apply(_t1, axis=1)
    return m1.set_index("match_id")["team1_roster"].to_dict()


def align_home_win_to_team1(mt: pd.DataFrame) -> pd.DataFrame:
    """把「home_win = 上半场 CT 赢」修正为「home_win = team1 赢」。

    输入 mt 需含 home(=CT-starter roster)/away(=T-starter roster)/home_win 列。
    返回加了 team1_roster 列、且 home_win 已改为 team1 口径的表；无 raw 的 match 会被 drop。
    """
    t1 = build_team1_roster_map()
    mt = mt.copy()
    mt["team1_roster"] = mt["match_id"].map(t1)
    mt = mt.dropna(subset=["team1_roster"]).copy()
    mt["team1_is_home"] = mt["team1_roster"] == mt["home"]
    mt["home_win"] = np.where(mt["team1_is_home"], mt["home_win"], 1 - mt["home_win"]).astype(int)
    return mt
