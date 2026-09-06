"""Demo -> per-team-per-map 聚合特征（学习引擎的输入）。

优先级（来自用户的半职业视角）：
  1. 击杀效率/击杀可能性 —— 射击游戏的根本，权重最大
  2. 手枪局 + 经济 —— MR12 的决定性因素
  3. 战术/站位/道具覆盖 —— 深层理解（Phase 2，需要 tick 级位置 + 投掷物）

一张地图的 .dem -> 两队各一条特征向量，存成聚合特征（原始 .dem/.rar 解析后删除，
按用户指示）。所有特征都按"队伍"聚合：队伍身份跨半场稳定，CT/T 边在半场（第 13 回合）互换。

正确性关键点：
  - 队伍身份用 player_info.team_number 分组（2/3 两组），跨半场稳定；
  - CT/T 边在第 13 回合（半场）互换，因此 round_end 的 winner("CT"/"T") 必须按半场映射回队伍；
  - 边用 parse_ticks 的 team_name（"CT"/"TERRORIST"）直接读取，与 steamid 关联，避开
    player_info.team_number 与 parse_ticks.team_num 编号不一致的坑；
  - 回合边界用 round_freeze_end 的 tick。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demoparser2 import DemoParser

# 经济分类阈值（current_equip_value，粗略的 CS2 购买档位）
ECO_MAX = 1500.0    # 纯手枪/裸奔
FORCE_MAX = 4000.0  # 半起（冲锋枪 / 甲 + 单枪）
# > FORCE_MAX => 全起（长枪 + 满道具）

FIRST_HALF_ROUNDS = 12  # MR12：前 12 回合上半场，13 起半场换边

# 手枪局 + 手枪局后的小分区间（MR12 下每半场各一次手枪局）
PISTOL_ROUNDS = {1, 13}
POST_PISTOL_ROUNDS = {2, 3, 14, 15}


@dataclass
class TeamMapFeatures:
    """一队一张地图的聚合特征。"""
    kills: int = 0
    deaths: int = 0
    rounds_won: int = 0
    rounds_played: int = 0
    kpr: float = 0.0            # kills per round
    dpr: float = 0.0            # deaths per round
    kd: float = 0.0             # kill/death ratio
    headshot_rate: float = 0.0
    opening_kills: int = 0      # 首杀数（对枪胜率代理）
    opening_deaths: int = 0
    opening_kill_rate: float = 0.0
    awp_kills: int = 0
    awp_share: float = 0.0      # AWP 击杀占比
    thrusmoke_kills: int = 0
    thrusmoke_rate: float = 0.0
    flash_assist_kills: int = 0  # 闪致击杀（assistedflash）——道具覆盖有效性
    flash_assist_rate: float = 0.0
    trade_kills: int = 0         # 交易击杀（队友刚阵亡 5s 内复仇）——抱团清点
    trade_rate: float = 0.0
    # 手枪局 + 经济
    pistol_wins: int = 0
    pistol_rounds: int = 0
    pistol_win_rate: float = 0.0
    post_pistol_wins: int = 0
    post_pistol_rounds: int = 0
    post_pistol_win_rate: float = 0.0
    eco_wins: int = 0
    eco_rounds: int = 0
    force_wins: int = 0
    force_rounds: int = 0
    full_wins: int = 0
    full_rounds: int = 0
    avg_equip_value: float = 0.0
    # 道具（每回合，按投掷数）
    flashes_per_round: float = 0.0
    smokes_per_round: float = 0.0
    mollies_per_round: float = 0.0
    he_per_round: float = 0.0


# --- 工具函数 ---

def _safe_rate(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def _classify_buy(equip: float) -> str:
    if equip <= ECO_MAX:
        return "eco"
    if equip <= FORCE_MAX:
        return "force"
    return "full"


def _steamid_to_group(player_info: pd.DataFrame) -> dict[str, int]:
    """steamid -> 队伍组号（player_info.team_number 的值 2/3，跨半场稳定）。"""
    return {str(r.steamid): int(r.team_number) for r in player_info.itertuples(index=False)}


def _group_roster(player_info: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    """队伍组号 -> 该队 5 人的 steamid 排序元组（跨 demo 稳定的队伍身份）。

    team_number 在同一张地图内稳定、但跨地图会翻转（主/客队摇号），
    所以队伍身份必须用 roster（5 个 steamid 的集合）而不是 team_number。
    """
    out: dict[int, tuple[str, ...]] = {}
    for g, sub in player_info.groupby("team_number"):
        out[int(g)] = tuple(sorted(str(s) for s in sub["steamid"]))
    return out


def _group_side_first_half(ticks_at_start: pd.DataFrame, sid2group: dict[str, int]) -> dict[int, str]:
    """队伍组号 -> 上半场的边（'CT'/'T'），用 team_name（"CT"/"TERRORIST"）。"""
    group_side: dict[int, str] = {}
    for r in ticks_at_start.itertuples(index=False):
        g = sid2group.get(str(r.steamid))
        if g is not None:
            group_side[g] = "CT" if str(r.team_name).upper().startswith("CT") else "T"
    return group_side


def _winner_group(winner_side: str, round_num: int, first_half_side: dict[int, str]) -> int | None:
    """把 round_end 的 winner('CT'/'T') 映射回队伍组号（考虑半场换边）。"""
    target = str(winner_side).upper()
    second_half = round_num > FIRST_HALF_ROUNDS
    for g, side in first_half_side.items():
        s = side.upper()
        if second_half:
            s = "T" if s == "CT" else "CT"
        if s == target:
            return g
    return None


def _grenade_family(gtype: str) -> str | None:
    """把 grenade_type 归到 flash/smoke/molly/he（忽略 decoy）。"""
    t = str(gtype).lower()
    if "flash" in t:
        return "flash"
    if "smoke" in t:
        return "smoke"
    if "molotov" in t or "incendiary" in t:
        return "molly"
    if "hegrenade" in t:
        return "he"
    return None


# --- 主入口 ---

def compute_map_features(dem_path: str | Path) -> dict:
    """解析一张地图的 .dem，返回两队各一条特征（含 map 名与比分）。"""
    dem_path = Path(dem_path)
    parser = DemoParser(str(dem_path))

    header = parser.parse_header()
    map_name = header.get("map_name", dem_path.stem)

    player_info = parser.parse_player_info()          # steamid, name, team_number
    round_end = parser.parse_event("round_end")       # round, tick, winner, reason
    kills = parser.parse_event("player_death")        # 击杀明细
    freeze = parser.parse_event("round_freeze_end")   # 每回合开始 tick
    grenades = parser.parse_grenades()                # 投掷物

    sid2group = _steamid_to_group(player_info)
    group_roster = _group_roster(player_info)
    groups = sorted(set(sid2group.values()))
    roster_keys = {g: ",".join(group_roster[g]) for g in groups}

    # 回合边界：freeze_end 的 tick 即每回合开始
    freeze_ticks = sorted(int(t) for t in freeze["tick"].tolist())
    start_ticks = [t + 1 for t in freeze_ticks]
    tick2round = {t: i + 1 for i, t in enumerate(start_ticks)}

    # 第 1 回合的边（用于把 CT/T 映射回队伍）
    ticks_start = parser.parse_ticks(["team_name"], ticks=[start_ticks[0]])
    first_half_side = _group_side_first_half(ticks_start, sid2group)

    feats: dict[int, TeamMapFeatures] = {g: TeamMapFeatures() for g in groups}

    # ---- 击杀效率（按队伍聚合 player_death）----
    if kills is not None and len(kills):
        k = kills.copy()
        k["attacker_group"] = k["attacker_steamid"].astype(str).map(sid2group)
        k["victim_group"] = k["user_steamid"].astype(str).map(sid2group)
        k = k.dropna(subset=["attacker_group", "victim_group"])
        k["attacker_group"] = k["attacker_group"].astype(int)
        k["victim_group"] = k["victim_group"].astype(int)

        # 交易击杀（补枪复仇）：A 击杀 V，而 V 刚在 5s 内杀了 A 的队友 => A 补枪复仇，
        # 是"抱团清点/少打多"的战术信号。CS2 64 tick/s。
        TRADE_TICKS = 5 * 64
        k_sorted = k.sort_values("tick").reset_index(drop=True)
        ticks_arr = k_sorted["tick"].to_numpy(dtype=float)
        att_sid = k_sorted["attacker_steamid"].astype(str).to_numpy()
        vic_sid = k_sorted["user_steamid"].astype(str).to_numpy()
        vg = k_sorted["victim_group"].to_numpy()
        ag = k_sorted["attacker_group"].to_numpy()
        trade_flag = np.zeros(len(k_sorted), dtype=bool)
        for i in range(len(k_sorted)):
            trade_flag[i] = ((att_sid == vic_sid[i]) & (vg == ag[i])
                             & (ticks_arr >= ticks_arr[i] - TRADE_TICKS)
                             & (ticks_arr < ticks_arr[i])).any()
        k_sorted["trade"] = trade_flag

        kk = k.groupby("attacker_group").size()
        dd = k.groupby("victim_group").size()
        for g in groups:
            f = feats[g]
            f.kills = int(kk.get(g, 0))
            f.deaths = int(dd.get(g, 0))
            f.headshot_rate = _safe_rate(int((k[k["attacker_group"] == g]["headshot"] == True).sum()), f.kills)
            f.awp_kills = int((k[k["attacker_group"] == g]["weapon"].astype(str).str.lower() == "awp").sum())
            f.awp_share = _safe_rate(f.awp_kills, f.kills)
            f.thrusmoke_kills = int((k[k["attacker_group"] == g]["thrusmoke"] == True).sum())
            f.thrusmoke_rate = _safe_rate(f.thrusmoke_kills, f.kills)
            f.flash_assist_kills = int((k[k["attacker_group"] == g]["assistedflash"] == True).sum())
            f.flash_assist_rate = _safe_rate(f.flash_assist_kills, f.kills)
            f.trade_kills = int(k_sorted[k_sorted["attacker_group"] == g]["trade"].sum())
            f.trade_rate = _safe_rate(f.trade_kills, f.kills)

        # 首杀（每回合第一个击杀）：player_death 无 round 字段，用 tick 归组
        def _round_of(tick: int) -> int:
            r = 1
            for i, t in enumerate(freeze_ticks):
                if tick >= t:
                    r = i + 1
                else:
                    break
            return r

        k["round"] = k["tick"].astype(int).map(_round_of)
        first_kill = k.sort_values("tick").groupby("round").head(1)
        for row in first_kill.itertuples(index=False):
            feats[int(row.attacker_group)].opening_kills += 1
            feats[int(row.victim_group)].opening_deaths += 1

    # ---- 回合 + 手枪局 + 经济 ----
    if round_end is not None and len(round_end):
        re_ = round_end[round_end["winner"].notna()].copy()
        re_["round"] = re_["round"].astype(int)

        # 每回合双方的经济档位（freeze 后第一 tick 的 equip value）
        equip_start = parser.parse_ticks(["current_equip_value"], ticks=start_ticks)
        equip_start["group"] = equip_start["steamid"].astype(str).map(sid2group)
        equip_start = equip_start.dropna(subset=["group"])
        equip_start["group"] = equip_start["group"].astype(int)
        equip_start["round"] = equip_start["tick"].map(tick2round)
        equip_by_round = equip_start.groupby(["round", "group"])["current_equip_value"].mean().reset_index()

        for r in sorted(re_["round"].unique()):
            w = re_[re_["round"] == r]["winner"].iloc[0]
            wg = _winner_group(str(w), r, first_half_side)
            rows = equip_by_round[equip_by_round["round"] == r]
            for g in groups:
                f = feats[g]
                f.rounds_played += 1
                if wg == g:
                    f.rounds_won += 1
                ev = rows[rows["group"] == g]["current_equip_value"]
                if len(ev):
                    v = float(ev.iloc[0])
                    f.avg_equip_value += v
                    # 手枪局本身不是 eco/force/full，单独统计
                    if r not in PISTOL_ROUNDS:
                        buy = _classify_buy(v)
                        if buy == "eco":
                            f.eco_rounds += 1
                            if wg == g:
                                f.eco_wins += 1
                        elif buy == "force":
                            f.force_rounds += 1
                            if wg == g:
                                f.force_wins += 1
                        else:
                            f.full_rounds += 1
                            if wg == g:
                                f.full_wins += 1
                # 手枪局 + 手枪局后小分区间
                if r in PISTOL_ROUNDS:
                    f.pistol_rounds += 1
                    if wg == g:
                        f.pistol_wins += 1
                if r in POST_PISTOL_ROUNDS:
                    f.post_pistol_rounds += 1
                    if wg == g:
                        f.post_pistol_wins += 1

    # ---- 道具（按投掷数：只数 Projectile 实体，避免 deployed 重复计）----
    if grenades is not None and len(grenades):
        gr = grenades[grenades["grenade_type"].astype(str).str.contains("Projectile", case=False)].copy()
        gr["group"] = gr["steamid"].astype(str).map(sid2group)
        gr = gr.dropna(subset=["group"])
        gr["group"] = gr["group"].astype(int)
        gr["family"] = gr["grenade_type"].astype(str).map(_grenade_family)
        throws = gr.dropna(subset=["family"]).groupby(["group", "family"])["grenade_entity_id"].nunique()
        for g in groups:
            f = feats[g]
            n = max(f.rounds_played, 1)
            f.flashes_per_round = _safe_rate(int(throws.get((g, "flash"), 0)), n)
            f.smokes_per_round = _safe_rate(int(throws.get((g, "smoke"), 0)), n)
            f.mollies_per_round = _safe_rate(int(throws.get((g, "molly"), 0)), n)
            f.he_per_round = _safe_rate(int(throws.get((g, "he"), 0)), n)

    # ---- 收尾：派生比率 ----
    for g in groups:
        f = feats[g]
        n = max(f.rounds_played, 1)
        f.kpr = _safe_rate(f.kills, n)
        f.dpr = _safe_rate(f.deaths, n)
        f.kd = _safe_rate(f.kills, f.deaths)
        f.opening_kill_rate = _safe_rate(f.opening_kills, n)
        f.pistol_win_rate = _safe_rate(f.pistol_wins, f.pistol_rounds)
        f.post_pistol_win_rate = _safe_rate(f.post_pistol_wins, f.post_pistol_rounds)
        f.avg_equip_value = round(f.avg_equip_value / n, 1)

    result: dict = {"map_name": map_name, "teams": {}}
    winner_group = max(groups, key=lambda g: feats[g].rounds_won)
    for g in groups:
        key = roster_keys[g]
        entry = asdict(feats[g])
        entry["team_number"] = g
        entry["side_first_half"] = first_half_side.get(g)
        entry["roster"] = list(group_roster[g])
        result["teams"][key] = entry
    result["winner_roster"] = roster_keys[winner_group]
    # 完整性：MR12 常规赛赢家至少 13 分；不足 13 说明 GOTV demo 被截断
    result["complete"] = feats[winner_group].rounds_won >= 13
    return result


if __name__ == "__main__":
    import sys

    res = compute_map_features(sys.argv[1])
    print(json.dumps(res, indent=2, ensure_ascii=False))
