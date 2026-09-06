"""Per-player-per-map 特征（学习引擎的「选手局内状态」粒度）。

用户方向：从 roster 级下钻到 player 级。选手的「当前竞技状态」最直观的代理就是
局内 k/d 与伤害量（ADR）。本模块把 demo_features 里按队伍聚合的击杀/伤害，改为
**按 steamid（选手）聚合**，输出 data/player_features.parquet。

每行 = 一张地图里的一名选手：
  demo_path, map_name, steamid, name, team_number, roster_key
  kills, deaths, kd, damage, adr, rounds_played
  headshots, headshot_rate, opening_kills, opening_deaths
  flash_assist_kills, awp_kills, thrusmoke_kills, trade_kills

team_number 在同一张地图内稳定、跨地图会翻转（主/客队摇号），队伍身份用
roster_key（5 个 steamid 排序拼接）保证跨 demo 稳定，可 join 回 round_dataset 的
ct_roster/t_roster。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

FIRST_HALF_ROUNDS = 12
TRADE_TICKS = 5 * 64  # 5s @ 64 tick/s


def _safe_rate(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def compute_player_features(dem_path: str | Path) -> pd.DataFrame:
    """解析一张 .dem，返回每名选手一行（10 行）。"""
    dem_path = Path(dem_path)
    parser = DemoParser(str(dem_path))

    header = parser.parse_header()
    map_name = header.get("map_name", dem_path.stem)

    player_info = parser.parse_player_info()          # steamid, name, team_number
    kills = parser.parse_event("player_death")        # 击杀明细
    hurts = parser.parse_event("player_hurt")         # 伤害明细
    round_end = parser.parse_event("round_end")       # 回合结束（算 rounds_played）
    freeze = parser.parse_event("round_freeze_end")   # 回合开始 tick（算首杀）

    sid2group = {str(r.steamid): int(r.team_number) for r in player_info.itertuples(index=False)}
    name_map = {str(r.steamid): str(r.name or "") for r in player_info.itertuples(index=False)}
    group_roster: dict[int, tuple[str, ...]] = {}
    for g, sub in player_info.groupby("team_number"):
        group_roster[int(g)] = tuple(sorted(str(s) for s in sub["steamid"]))
    roster_key = {g: ",".join(sids) for g, sids in group_roster.items()}

    sids = list(sid2group.keys())
    st = {s: defaultdict(int) for s in sids}

    n_rounds = 0
    if round_end is not None and len(round_end):
        n_rounds = int(round_end["winner"].notna().sum())

    # ---- 击杀：attacker -> kills/headshot/awp/thrusmoke/flash，user -> deaths ----
    if kills is not None and len(kills):
        k = kills.copy()
        k["att"] = k["attacker_steamid"].astype(str)
        k["vic"] = k["user_steamid"].astype(str)
        k = k[k["att"].isin(st) | k["vic"].isin(st)]
        for att, sub in k.groupby("att"):
            if att not in st:
                continue
            s = st[att]
            s["kills"] += len(sub)
            s["headshots"] += int((sub["headshot"] == True).sum())
            s["awp_kills"] += int((sub["weapon"].astype(str).str.lower() == "awp").sum())
            s["thrusmoke_kills"] += int((sub["thrusmoke"] == True).sum())
            s["flash_assist_kills"] += int((sub["assistedflash"] == True).sum())
        for vic, sub in k.groupby("vic"):
            if vic in st:
                st[vic]["deaths"] += len(sub)

        # 交易击杀（队友 5s 内阵亡后的复仇）：抱团清点/补枪协调信号
        k_sorted = k.sort_values("tick").reset_index(drop=True)
        ticks_arr = k_sorted["tick"].to_numpy(dtype=float)
        att_sid = k_sorted["att"].to_numpy()
        vic_sid = k_sorted["vic"].to_numpy()
        vg = k_sorted["vic"].map(sid2group).to_numpy()
        ag = k_sorted["att"].map(sid2group).to_numpy()
        trade = np.zeros(len(k_sorted), dtype=bool)
        for i in range(len(k_sorted)):
            trade[i] = ((att_sid == vic_sid[i]) & (vg == ag[i])
                        & (ticks_arr >= ticks_arr[i] - TRADE_TICKS)
                        & (ticks_arr < ticks_arr[i])).any()
        for i in range(len(k_sorted)):
            if trade[i]:
                a = att_sid[i]
                if a in st:
                    st[a]["trade_kills"] += 1

        # 首杀（每回合第一个击杀）
        freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())

        def round_of(tick: int) -> int:
            r = 1
            for i, t in enumerate(freeze_ticks):
                if tick >= t:
                    r = i + 1
                else:
                    break
            return r

        k["round"] = k["tick"].astype(int).map(round_of)
        first = k.sort_values("tick").groupby("round").head(1)
        for row in first.itertuples(index=False):
            if row.att in st:
                st[row.att]["opening_kills"] += 1
            if row.vic in st:
                st[row.vic]["opening_deaths"] += 1

    # ---- 伤害：attacker -> 累计 dmg_health + dmg_armor（ADR = 伤害/回合）----
    if hurts is not None and len(hurts):
        h = hurts.copy()
        h["att"] = h["attacker_steamid"].astype(str)
        h = h[h["att"].isin(st)]
        h["dmg"] = (h["dmg_health"].astype(float) + h["dmg_armor"].astype(float))
        dmg = h.groupby("att")["dmg"].sum()
        for sid, v in dmg.items():
            st[sid]["damage"] = float(v)

    rows = []
    for sid in sids:
        s = st[sid]
        g = sid2group[sid]
        k = s["kills"]; d = s["deaths"]
        rows.append({
            "demo_path": str(dem_path),
            "map_name": map_name,
            "steamid": sid,
            "name": name_map.get(sid, ""),
            "team_number": g,
            "roster_key": roster_key[g],
            "kills": k,
            "deaths": d,
            "kd": _safe_rate(k, d),
            "damage": round(s["damage"], 1),
            "adr": _safe_rate(s["damage"], max(n_rounds, 1)),
            "rounds_played": n_rounds,
            "headshots": s["headshots"],
            "headshot_rate": _safe_rate(s["headshots"], k),
            "opening_kills": s["opening_kills"],
            "opening_deaths": s["opening_deaths"],
            "flash_assist_kills": s["flash_assist_kills"],
            "awp_kills": s["awp_kills"],
            "thrusmoke_kills": s["thrusmoke_kills"],
            "trade_kills": s["trade_kills"],
        })
    return pd.DataFrame(rows)


def build(dem_paths: list[str] | None = None) -> pd.DataFrame:
    """解析一批 .dem，返回 per-player 特征长表，写 data/player_features.parquet。"""
    if dem_paths is None:
        rd = pd.read_parquet(__import__("cs2ml.config", fromlist=["config"]).DATA_DIR / "round_dataset.parquet")
        dem_paths = sorted(rd["demo_path"].dropna().unique().tolist())

    frames: list[pd.DataFrame] = []
    n_fail = 0
    for i, dp in enumerate(dem_paths):
        try:
            frames.append(compute_player_features(dp))
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  跳过 {Path(dp).name}（解析失败）: {e}")
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(dem_paths)}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    from . import config
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATA_DIR / "player_features.parquet", index=False)
    print(f"完成：{len(df)} 名选手×地图 / {df['demo_path'].nunique()} 张图 / "
          f"{df['steamid'].nunique()} 名选手（跳过 {n_fail} 张失败）")
    return df


if __name__ == "__main__":
    build()
