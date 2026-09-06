"""execution_layer.py —— Phase 4.5：执行脚本层 pilot（Spirit 假打/真打自动分类）。

用户问题：模型学不到「假打/真打、战术怎么执行」吗？这里用 demo 里真实存在的原料
重建每一回合的「执行脚本」，并自动分出假打 vs 真打：

  原料（都在盘上，Phase 1 已落库）：
    player_util   —— 每次投掷的 family(烟/闪/火/雷) + rel_tick(相对 freeze 秒数) + x/y/z(位置)
    player_kills  —— 首杀方/时机(rel_tick)
    player_behavior —— 每个 (demo_path, round_num, steamid) 的 side(CT/T)

  方法：
    1. 把投掷位置按地图聚成 K 个「区域」（≈ A/mid/B）。
    2. 每回合取 T 侧烟雾，按 rel_tick 排序，看「第一颗烟落在哪」vs「最后一颗烟落在哪」。
    3. 首烟区 ≠ 末烟区 → 执行过程中换点了 = 假打；首末同区 → 真打/直扑。

  输出：Spirit T 侧每回合的 (first_area, last_area, fake?, first_kill 时机/方)。
  这是「执行脚本」的第一层，只证明可学；能不能变现交给 phase4 lead test。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import config

N_AREAS = 3  # ≈ A / mid / B


# ---------------------------------------------------------------- 数据

def _is_spirit(dp: str) -> bool:
    return "spirit" in str(dp).lower()


def load_t_throws() -> pd.DataFrame:
    """Spirit T 侧投掷：join player_util 与 player_behavior 拿 side，只留 T。"""
    util = pd.read_parquet(config.DATA_DIR / "player_util.parquet")
    beh = pd.read_parquet(config.DATA_DIR / "player_behavior.parquet")
    util = util[util["demo_path"].map(_is_spirit)].copy()
    side = beh[["demo_path", "round_num", "steamid", "side"]].drop_duplicates()
    m = util.merge(side, on=["demo_path", "round_num", "steamid"], how="left")
    m = m[m["side"] == "T"].dropna(subset=["x", "y"]).copy()
    return m


def load_kills() -> pd.DataFrame:
    kills = pd.read_parquet(config.DATA_DIR / "player_kills.parquet")
    return kills[kills["demo_path"].map(_is_spirit)].copy()


# ---------------------------------------------------------------- 区域聚类

def area_model(throws: pd.DataFrame, n_areas: int = N_AREAS) -> dict[str, KMeans]:
    """每个地图一个 KMeans，把 (x,y) 聚成 n_areas 个区域。"""
    models = {}
    for mp, sub in throws.groupby("map_name_x"):
        if len(sub) < n_areas * 3:
            continue
        X = sub[["x", "y"]].to_numpy()
        models[mp] = KMeans(n_clusters=n_areas, n_init=10, random_state=0).fit(X)
    return models


def assign_area(df: pd.DataFrame, models: dict[str, KMeans]) -> pd.Series:
    """给每行分配区域标签（没有模型的图标 -1）。"""
    out = pd.Series(-1, index=df.index)
    for mp, km in models.items():
        m = df["map_name_x"] == mp
        if m.sum() == 0:
            continue
        out[m] = km.predict(df.loc[m, ["x", "y"]].to_numpy())
    return out


# ---------------------------------------------------------------- 执行脚本 + 分类

def build(throws: pd.DataFrame, kills: pd.DataFrame) -> pd.DataFrame:
    """每回合一行：首烟区/末烟区/假打判定/首杀时机与方。"""
    models = area_model(throws)
    throws = throws.copy()
    throws["area"] = assign_area(throws, models)
    sm = throws[throws["family"] == "smoke"]

    # 首杀：每回合第一次击杀的 rel_tick + attacker_side
    fk = kills[kills["is_first_kill"]].groupby(["demo_path", "round_num"]).agg(
        first_kill_tick=("rel_tick", "first"),
        first_kill_att=("attacker_side", "first"),
    ).reset_index()

    rows = []
    for (dp, rn), g in sm.groupby(["demo_path", "round_num"]):
        g = g[g["area"] >= 0].sort_values("rel_tick")
        if g.empty:
            continue
        first_area = int(g["area"].iloc[0])
        last_area = int(g["area"].iloc[-1])
        first_tick = float(g["rel_tick"].iloc[0])
        last_tick = float(g["rel_tick"].iloc[-1])
        # 假打 = 首末烟落不同区域（执行中换点）
        fake = int(first_area != last_area)
        # 烟的数量 + 时间跨度
        n_smoke = len(g)
        span = last_tick - first_tick
        # 首杀信息
        fkrow = fk[(fk["demo_path"] == dp) & (fk["round_num"] == rn)]
        fk_tick = float(fkrow["first_kill_tick"].iloc[0]) if len(fkrow) else np.nan
        fk_att = str(fkrow["first_kill_att"].iloc[0]) if len(fkrow) else ""
        rows.append({
            "demo_path": dp, "round_num": int(rn), "map_name": g["map_name_x"].iloc[0],
            "n_smoke": n_smoke, "span_ticks": span,
            "first_area": first_area, "last_area": last_area, "fake": fake,
            "first_smoke_tick": first_tick, "last_smoke_tick": last_tick,
            "first_kill_tick": fk_tick, "first_kill_att": fk_att,
        })
    return pd.DataFrame(rows)


def main() -> None:
    throws = load_t_throws()
    kills = load_kills()
    print(f"Spirit T 侧投掷：{len(throws)} 次 / {throws['demo_path'].nunique()} 图")
    print(f"Spirit 击杀：{len(kills)} 次")

    ex = build(throws, kills)
    print(f"\n执行脚本回合数：{len(ex)} / {ex['demo_path'].nunique()} 图")

    print("\n=== 假打率（首末烟不同区）===")
    print(f"  整体 fake 率 = {ex['fake'].mean():.3f}  ({int(ex['fake'].sum())}/{len(ex)})")
    for mp, g in ex.groupby("map_name"):
        print(f"  {mp:<12} fake={g['fake'].mean():.3f}  n={len(g):>3}")

    print("\n=== 假打 vs 真打：首杀时机 & 时间跨度 ===")
    cmp = ex.groupby("fake").agg(
        n=("round_num", "size"),
        span_s=("span_ticks", lambda s: (s / 64).mean()),
        first_kill_tick=("first_kill_tick", lambda s: s.mean()),
    ).reset_index()
    cmp["fake"] = cmp["fake"].map({1: "假打", 0: "真打"})
    print(cmp.to_string(index=False))

    out = config.DATA_DIR / "spirit_execution.csv"
    ex.to_csv(out, index=False)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
