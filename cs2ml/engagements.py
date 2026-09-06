"""交火（engagement）级特征：每次击杀前，重构交火双方的状态。

用户的洞察：'击杀效率'应细化为'交火前胜率预测'——分析交火发生前双方的
位置 + 个人状态（血/甲/装备）+ 阵型 + 信息差，学 P(一方赢下这次交火)。

单位 = 一次击杀（player_death），锚定 kill tick，往前取 `pre_ticks` tick 的快照。
每行输出一次交火，含 attacker / victim 两方各自 pre-fight 的状态与阵型上下文。

对称标签（供下游学 duel 胜率）：
  一次击杀同时给出两个有标签样本——
    (killer 方交火前状态, label=win) 与 (victim 方交火前状态, label=loss)。
  这样每场 demo 能产出平衡的正负样本，用来堆'不同选手在不同位置的胜率'。

tick rate：CS2 GOTV 为 64 tick（round 1 实测 3836 tick ≈ 60s）。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from demoparser2 import DemoParser

TICKS_PER_SECOND = 64

# parse_ticks 需要取的字段
# last_place_name = nav mesh 命名区域（社区"位置 token"方案，编码墙/房间结构）
# active_weapon_name = 手上拿的武器名（修正 weapon 泄漏：双方各自武器）
# yaw / flash_duration = 朝向 / 被闪致盲，用于"信息差"特征
_TICK_FIELDS = [
    "X", "Y", "Z", "health", "armor_value", "current_equip_value",
    "is_alive", "team_name", "last_place_name", "steamid", "name",
    "active_weapon_name", "yaw", "flash_duration", "inventory",
]

NEARBY_RADIUS = 400.0  # 单位（CS2 距离单位，~400 内算"抱团/同区域"）

# 朝向判定阈值：与"朝向对方"的水平夹角 < 45° 算"正看着对方"
# 实测 forward=(cos(yaw), sin(yaw))：86.6% 的击杀者在此锥内朝向受害者
_FACING_COS = math.cos(math.radians(45.0))


def _dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))


def _facing(x1: float, y1: float, yaw1: float, x2: float, y2: float) -> bool:
    """player1 是否朝 player2 看（水平面，45° 锥内）。yaw 单位度，forward=(cos,sin)。"""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return False
    r = math.radians(yaw1)
    fx, fy = math.cos(r), math.sin(r)
    return (fx * dx + fy * dy) / length > _FACING_COS


def _inv_has(inventory, *keys: str) -> bool:
    """inventory 里是否含某类道具（keys 是小写子串，如 'flash'/'smoke'）。"""
    if inventory is None:
        return False
    s = " ".join(str(i).lower() for i in inventory)
    return any(k in s for k in keys)


def extract_engagements(dem_path: str | Path, pre_ticks: int = 64,
                        nearby_radius: float = NEARBY_RADIUS) -> pd.DataFrame:
    """每行一次击杀，含交火前（kill_tick - pre_ticks）双方状态与阵型特征。

    返回列：
      tick, round, weapon, headshot, thru_smoke, distance
      att_health, att_armor, att_equip, att_x, att_y, att_z
      vic_health, vic_armor, vic_equip, vic_x, vic_y, vic_z
      att_alive, vic_alive, man_diff            # 少打多/多打少
      att_nearby, vic_nearby                    # 抱团程度
    """
    parser = DemoParser(str(dem_path))
    kills = parser.parse_event("player_death")
    freeze = parser.parse_event("round_freeze_end")

    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())

    def _round_of(tick: int) -> int:
        r = 1
        for i, t in enumerate(freeze_ticks):
            if tick >= t:
                r = i + 1
            else:
                break
        return r

    # 有效击杀（双方 steamid 都存在）。原来每击杀一次 parse_ticks，现在整图一次性批量
    # 取所有交火前快照 tick，再按 tick 索引——把 O(击杀数) 次全量 tick 加载降为 1 次。
    valid = kills[kills["attacker_steamid"].notna() & kills["user_steamid"].notna()]
    if not len(valid):
        return pd.DataFrame()
    snap_ticks = sorted({int(kt) - pre_ticks for kt in valid["tick"] if int(kt) >= pre_ticks})
    snap = parser.parse_ticks(_TICK_FIELDS, ticks=snap_ticks)
    snap["steamid"] = snap["steamid"].astype(str)
    snap["team_name"] = snap["team_name"].astype(str)
    snap["is_alive"] = snap["is_alive"].astype(bool)
    by_tick = {int(t): g for t, g in snap.groupby("tick")}

    rows: list[dict] = []
    for kr in valid.itertuples(index=False):
        kt = int(kr.tick)
        snap_t = by_tick.get(kt - pre_ticks)
        if snap_t is None:
            continue
        att = snap_t[snap_t["steamid"] == str(kr.attacker_steamid)]
        vic = snap_t[snap_t["steamid"] == str(kr.user_steamid)]
        if not len(att) or not len(vic):
            continue
        att, vic = att.iloc[0], vic.iloc[0]
        # 快照数据缺口（整行 NaN 的 degenerate tick）则丢弃，避免污染下游模型
        if not (math.isfinite(float(att.X)) and math.isfinite(float(vic.X))
                and math.isfinite(float(att.health)) and math.isfinite(float(vic.health))):
            continue
        att_team = str(att.team_name)
        vic_team = str(vic.team_name)

        # 双方各自存活人数（交火瞬间的阵型，只数活着的）
        alive_mask = snap_t["is_alive"].to_numpy()
        att_alive = int(((snap_t["team_name"] == att_team).to_numpy() & alive_mask).sum())
        vic_alive = int(((snap_t["team_name"] == vic_team).to_numpy() & alive_mask).sum())

        # 抱团程度：交火双方身边同队队友数（半径内）
        att_x, att_y = float(att.X), float(att.Y)
        vic_x, vic_y = float(vic.X), float(vic.Y)
        x = snap_t["X"].to_numpy(dtype=float)
        y = snap_t["Y"].to_numpy(dtype=float)
        att_near = int(((snap_t["team_name"] == att_team)
                        & (snap_t["steamid"] != str(kr.attacker_steamid))
                        & (np.hypot(x - att_x, y - att_y) <= nearby_radius)).sum())
        vic_near = int(((snap_t["team_name"] == vic_team)
                        & (snap_t["steamid"] != str(kr.user_steamid))
                        & (np.hypot(x - vic_x, y - vic_y) <= nearby_radius)).sum())

        # 信息差：双方各自手持武器、被闪致盲、是否朝向对方
        att_facing = _facing(att_x, att_y, float(att.yaw), vic_x, vic_y)
        vic_facing = _facing(vic_x, vic_y, float(vic.yaw), att_x, att_y)
        att_flashed = float(att.flash_duration) > 0.0
        vic_flashed = float(vic.flash_duration) > 0.0

        # 道具覆盖：双方各自是否持有 flash/smoke/he/molly
        att_has_flash = _inv_has(att.inventory, "flash")
        att_has_smoke = _inv_has(att.inventory, "smoke")
        att_has_he = _inv_has(att.inventory, "high explosive")
        att_has_molly = _inv_has(att.inventory, "molotov", "incendiary")
        vic_has_flash = _inv_has(vic.inventory, "flash")
        vic_has_smoke = _inv_has(vic.inventory, "smoke")
        vic_has_he = _inv_has(vic.inventory, "high explosive")
        vic_has_molly = _inv_has(vic.inventory, "molotov", "incendiary")

        rows.append({
            "tick": kt,
            "round": _round_of(kt),
            "att_steamid": str(kr.attacker_steamid),
            "vic_steamid": str(kr.user_steamid),
            "weapon": str(kr.weapon),
            "att_weapon": str(att.active_weapon_name),
            "vic_weapon": str(vic.active_weapon_name),
            "headshot": bool(kr.headshot),
            "thru_smoke": bool(kr.thrusmoke),
            "distance": _dist2d(att.X, att.Y, vic.X, vic.Y),
            "att_place": str(att.last_place_name),
            "vic_place": str(vic.last_place_name),
            # 信息差
            "att_facing": att_facing, "vic_facing": vic_facing,
            "att_info_adv": att_facing and not vic_facing,
            "att_flashed": att_flashed, "vic_flashed": vic_flashed,
            # 道具覆盖
            "att_has_flash": att_has_flash, "att_has_smoke": att_has_smoke,
            "att_has_he": att_has_he, "att_has_molly": att_has_molly,
            "vic_has_flash": vic_has_flash, "vic_has_smoke": vic_has_smoke,
            "vic_has_he": vic_has_he, "vic_has_molly": vic_has_molly,
            # 交火前个人状态
            "att_health": float(att.health), "att_armor": float(att.armor_value),
            "att_equip": float(att.current_equip_value),
            "att_x": float(att.X), "att_y": float(att.Y), "att_z": float(att.Z),
            "vic_health": float(vic.health), "vic_armor": float(vic.armor_value),
            "vic_equip": float(vic.current_equip_value),
            "vic_x": float(vic.X), "vic_y": float(vic.Y), "vic_z": float(vic.Z),
            # 阵型
            "att_alive": att_alive, "vic_alive": vic_alive,
            "man_diff": att_alive - vic_alive,
            "att_nearby": att_near, "vic_nearby": vic_near,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    df = extract_engagements(sys.argv[1])
    print(df.to_string())
