"""fade_favorite.py —— 热门-冷门偏差交易策略回测（已修正数据对齐 bug）。

结合「模型」+「市场规律」做「怎么买能相对盈利」的回测。核心结论（修正后）：

  **冷门偏差是真的，但比之前量的弱，且只在 Map1 弱存在，Map2 方向反了。**

历史 bug（本文件修复）：
  之前的 home_win 来自 build_demo_maps()，它是「上半场 CT 那队赢没赢」(=ct_roster 视角)，
  不是「team1（市场主队）赢没赢」。BO3 换边导致 Map2 的 CT 队经常变成 team2，于是
  ~31% 的 Map1 场次 home_win 被反了，把冷门偏差虚高约 2 倍（+0.118 → 真实 ~+0.05）。

修正方案：**完全用 raw 结算**（Polymarket 里 token 最后成交价 >=0.9 即结算为赢），
这与真实 P&L 一致（下注按市场结算赔付），彻底绕开 demo 的 CT/team1 对齐问题。

入场价 = raw 的 min-t 首价（≈ CLOB 收盘价，corr 0.987）。结算 = raw 的 max-t 末价。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

_MAP_NUM = re.compile(r"Map (\d)")


def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\bteam\b|\besports\b|\bthe\b|\bgaming\b|\bclub\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


# ---------------------------------------------------------------- 纯 raw 的入场 + 结算

def build_fade_backtest() -> pd.DataFrame:
    """每 (match_id, map_num) 一行：home_p(team1 首价)、home_won(team1 是否结算赢)。

    home = raw 自己的 team1 列（同一 match 内 team1 唯一，已验证 278/278）。
    home_won = team1 的 token 末价 >= 0.9（Polymarket 结算口径，与真实 P&L 一致）。
    只保留结算干净的场次（一方 >=0.9 一方 <=0.1）。
    """
    raw = pd.read_parquet(config.DATA_DIR / "polymarket_inplay_raw.parquet")
    r = raw.copy()
    r["map_num"] = r["map_market"].str.extract(r"Map (\d)").astype(int)
    r["is_home"] = r["outcome"].astype(str).map(_norm) == r["team1"].astype(str).map(_norm)

    g = r.sort_values("t").groupby(["match_id", "map_num", "is_home"])
    first = g["p"].first().unstack("is_home").rename(columns={True: "home_p", False: "away_p"})
    last = g["p"].last().unstack("is_home").rename(columns={True: "home_last", False: "away_last"})
    bt = first.join(last).reset_index()

    clean = ((bt["home_last"] >= 0.9) & (bt["away_last"] <= 0.1)) | \
            ((bt["home_last"] <= 0.1) & (bt["away_last"] >= 0.9))
    bt = bt[clean].copy()
    bt["home_won"] = (bt["home_last"] >= 0.9).astype(int)
    return bt


# ---------------------------------------------------------------- 冷门视角

def _fade_frame(bt: pd.DataFrame) -> pd.DataFrame:
    df = bt.copy()
    p = df["home_p"].to_numpy()
    hw = df["home_won"].astype(int).to_numpy()
    fav_home = p >= 0.5
    fav_price = np.where(fav_home, p, 1 - p)
    df["fav_home"] = fav_home
    df["fav_price"] = fav_price
    df["dog_price"] = 1 - fav_price
    df["fav_won"] = np.where(fav_home, hw, 1 - hw)
    df["dog_won"] = 1 - df["fav_won"]
    df["dog_ev"] = df["dog_won"] * (1 - df["dog_price"]) - (1 - df["dog_won"]) * df["dog_price"]
    df["fav_ev"] = -df["dog_ev"]  # 零和两 token（忽略手续费）
    return df


def _show(df: pd.DataFrame, label: str, mask: np.ndarray) -> None:
    if mask.sum() == 0:
        print(f"  {label:<36} n=   0")
        return
    ev = df.loc[mask, "dog_ev"].mean()
    win = df.loc[mask, "dog_won"].mean()
    n = int(mask.sum())
    print(f"  {label:<36} n={n:>4}  dogEV={ev:+.4f}  dogWin={win:.3f}")


def fade_report(df: pd.DataFrame, label: str) -> None:
    df = _fade_frame(df)
    n = len(df)
    print(f"\n=== {label}  (n={n}) ===")
    bias = df["fav_won"].mean() - df["fav_price"].mean()
    print(f"  热门偏差(实际胜率-隐含) = {bias:+.4f}   热门实际胜率={df['fav_won'].mean():.3f} "
          f"隐含={df['fav_price'].mean():.3f}")
    zone = (df["fav_price"] >= 0.50) & (df["fav_price"] < 0.65)
    _show(df, "无条件押冷门", np.ones(n, bool))
    _show(df, "市场热门 50-65% 押冷门", zone)
    _show(df, "市场 65-70% 押冷门(真热门区)", (df["fav_price"] >= 0.65) & (df["fav_price"] < 0.70))
    _show(df, "市场热门 50-65% 押热门(对照)", zone)


def main() -> None:
    bt = build_fade_backtest()
    print(f"纯 raw 回测表：{len(bt)} 行 / {bt['match_id'].nunique()} 场")
    print("各图覆盖：")
    print(bt["map_num"].value_counts().to_string())

    for mn in [1, 2, 3]:
        sub = bt[bt["map_num"] == mn]
        if len(sub):
            fade_report(sub, f"Map {mn}")

    m12 = bt[bt["map_num"].isin([1, 2])]
    fade_report(m12, "Map1+Map2 合并")
    fade_report(bt, "Map1+Map2+Map3 合并")


if __name__ == "__main__":
    main()
