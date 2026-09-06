"""team_style.py —— Phase 3：全量选手 opener 分类 + 队伍风格聚合。

读取全量 player_behavior/player_util/player_kills（240 选手 / 597 图），
做两件事：

  1. 选手级：按 steamid（全量身份干净）算 HLTV Opening/Entrying/Trading +
     首杀时机，输出 data/player_hlvt_all.csv/json，识别 opener。
  2. 队伍级：把 roster_key（排序 steamid）聚类成真实队伍（overlap>=3 处理换人），
     从 demo 文件名解析队名（多数投票），聚合成队伍战术身份（谁 opener / 谁 AWP /
     谁支援），输出 data/team_style_memory.json。

opener 判据：卷入率>=0.22 且首杀净正 且 时机中位<=0.45（default 阶段早抢对枪）。
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .player_style import entry_profile, load_behavior, load_kills, load_util

MIN_ROUNDS = 80       # 选手样本下限（opener 判定）
TEAM_MIN_ROUNDS = 100  # 队伍样本下限（风格聚合）
AWP_THRESHOLD = 0.15  # awp 击杀/回合 超过此值算主狙


# ---------------------------------------------------------------- 选手级

def group_by_steamid(beh: pd.DataFrame) -> dict[str, dict]:
    """steamid -> {name, steamids, roster_key, n_rounds}。"""
    out: dict[str, dict] = {}
    for sid, g in beh.groupby(beh["steamid"].astype(str)):
        out[sid] = {
            "name": str(g["name"].mode().iloc[0]),
            "steamids": [sid],
            "roster_key": str(g["roster_key"].mode().iloc[0]),
            "n_rounds": int(len(g)),
        }
    return out


def classify(hlvt: dict, n_rounds: int) -> str:
    """opener / support（entry 在干净 traded 死亡率下不再单独分档）。"""
    part = hlvt.get("opening_participation") or 0.0
    med = hlvt.get("first_kill_median_frac")
    net = (hlvt.get("opening_kills") or 0) - (hlvt.get("opening_deaths") or 0)
    if med is None or part < 0.22 or net <= 0:
        return "support"
    return "opener"


def build_all(behavior_path=None, util_path=None, kills_path=None) -> pd.DataFrame:
    beh = load_behavior(behavior_path)
    kills = load_kills(kills_path)
    players = group_by_steamid(beh)

    rows = []
    for sid, info in players.items():
        n = info["n_rounds"]
        if n < MIN_ROUNDS:
            continue
        hlvt = entry_profile(kills, info["steamids"], n)
        if not hlvt:
            continue
        hlvt["role"] = classify(hlvt, n)
        hlvt["n_rounds"] = n
        hlvt["steamid"] = sid
        hlvt["name"] = info["name"]
        hlvt["roster_key"] = info["roster_key"]
        rows.append(hlvt)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(
            ["opening_participation", "opening_kills"], ascending=False, na_position="last"
        ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- 队伍级

# CS2 竞技图池：从队名尾部剥掉「-m<num>-<map>[-p<num>]」或 BO1 的「-<map>[-p<num>]」尾巴
_MAP_TAIL = re.compile(
    r"-(?:m\d+-)?(?:ancient|anubis|dust2|inferno|mirage|nuke|overpass|vertigo|train)"
    r"(?:-p\d+)?$"
)


def parse_teams(demo_path: str) -> tuple[str | None, str | None]:
    """demo 文件名 -> (队A, 队B)。格式 <a>-vs-<b>[-m<num>]-<map>[-p<num>].dem，
    队名可含连字符/数字（the-mongolz / natus-vincere / 100-thieves），BO1 无 -m<num>。"""
    fn = demo_path.replace("\\", "/").split("/")[-1].replace(".dem", "")
    if "-vs-" not in fn:
        return None, None
    a, rest = fn.split("-vs-", 1)
    b = _MAP_TAIL.sub("", rest)
    return a, b


def cluster_rosters(rosters: list[str], min_overlap: int = 3) -> dict[str, list[str]]:
    """union-find：两个 roster 共享 >=min_overlap 个 steamid 算同一队。"""
    sets = {r: set(r.split(",")) for r in rosters}
    parent = {r: r for r in rosters}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rs = list(sets)
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            if len(sets[rs[i]] & sets[rs[j]]) >= min_overlap:
                parent[find(rs[i])] = find(rs[j])
    clusters: dict[str, list[str]] = defaultdict(list)
    for r in rs:
        clusters[find(r)].append(r)
    return clusters


def _name_counter(rs: list[str], roster_demos: dict[str, set[str]]) -> Counter:
    cnt: Counter[str] = Counter()
    for r in rs:
        for dp in roster_demos.get(r, []):
            a, b = parse_teams(dp)
            if a:
                cnt[a] += 1
            if b:
                cnt[b] += 1
    return cnt


def label_teams(clusters: dict[str, list[str]],
                roster_demos: dict[str, set[str]],
                demo_sides: dict[str, dict[str, str]]) -> dict[str, str]:
    """队名两步：先多数投票（自己出现全、对手分散），再用可靠大队伍反推对手名。

    只打一个对手的小队伍，自己的名字和对手名票数打平，多数投票会乱选；
    但那个对手（大队伍）名字可靠，从「另一边的名字」反推即可。
    """
    # roster -> cluster root
    r2c: dict[str, str] = {}
    for root, rs in clusters.items():
        for r in rs:
            r2c[r] = root

    label: dict[str, str] = {}
    confident: dict[str, bool] = {}
    for root, rs in clusters.items():
        cnt = _name_counter(rs, roster_demos)
        if not cnt:
            label[root] = "?"
            confident[root] = False
            continue
        label[root] = cnt.most_common(1)[0][0]
        top = cnt.most_common(1)[0][1]
        second = cnt.most_common(2)[1][1] if len(cnt) >= 2 else 0
        confident[root] = top >= 3 and top >= 2 * second

    # 传播：可靠队伍的名字 -> 对手（只打一个对手的小队）
    for dp, sides in demo_sides.items():
        a, b = parse_teams(dp)
        if not a:
            continue
        cs = [(side, r2c[r]) for side, r in sides.items() if r in r2c]
        if len(cs) != 2:
            continue
        (_, c1), (_, c2) = cs
        if confident.get(c1) and not confident.get(c2):
            label[c2] = b if label[c1] == a else (a if label[c1] == b else label[c2])
        elif confident.get(c2) and not confident.get(c1):
            label[c1] = b if label[c2] == a else (a if label[c2] == b else label[c1])
    return label


def build_team_style(behavior_path=None, util_path=None, kills_path=None) -> dict:
    beh = load_behavior(behavior_path)
    util = load_util(util_path)
    kills = load_kills(kills_path)

    # 选手级指标（含 entry_profile）
    players = group_by_steamid(beh)
    sid_metrics: dict[str, dict] = {}
    for sid, info in players.items():
        n = info["n_rounds"]
        b = beh[beh["steamid"].astype(str) == sid]
        u = util[util["steamid"].astype(str) == sid]
        m = {
            "name": info["name"],
            "n_rounds": n,
            "awp_per_round": float(b["awp_kills"].mean()) if n else 0.0,
            "first_contact": float(b["first_contact"].mean()) if n else 0.0,
            "throws_per_round": round(len(u) / max(n, 1), 2),
        }
        m.update(entry_profile(kills, [sid], n))
        sid_metrics[sid] = m

    # roster -> demos；demo -> {side: roster}
    roster_demos: dict[str, set[str]] = defaultdict(set)
    demo_sides: dict[str, dict[str, str]] = defaultdict(dict)
    for r, dp, side in beh[["roster_key", "demo_path", "side"]].drop_duplicates().itertuples(index=False):
        roster_demos[str(r)].add(dp)
        demo_sides[dp][side] = str(r)

    rosters = sorted(beh["roster_key"].astype(str).unique())
    clusters = cluster_rosters(rosters)
    labels = label_teams(clusters, roster_demos, demo_sides)

    # 同名队伍加后缀（falcons 换人拆成多簇、都叫 falcons）
    name_seen: Counter[str] = Counter()
    def unique_name(nm: str) -> str:
        if nm not in name_seen:
            name_seen[nm] = 1
            return nm
        name_seen[nm] += 1
        return f"{nm}#{name_seen[nm]}"

    teams: dict[str, dict] = {}
    for root, rs in clusters.items():
        # 队内 steamid 并集 + 每个选手的总回合
        sids: set[str] = set()
        for r in rs:
            sids |= set(r.split(","))
        members = []
        for sid in sids:
            mm = sid_metrics.get(sid)
            if mm is None:
                continue
            mm = dict(mm)
            mm["steamid"] = sid
            members.append(mm)
        if not members:
            continue
        n_team = int(sum(m["n_rounds"] for m in members))
        if n_team < TEAM_MIN_ROUNDS:
            continue

        members.sort(key=lambda m: -m["n_rounds"])
        # 主力 5 人（回合数最多）
        core = members[:5]
        # 角色判定
        opener = max(core, key=lambda m: (m.get("opening_kills") or 0), default=None)
        awp = max(core, key=lambda m: m["awp_per_round"], default=None)
        non_opener = [m for m in core if m is not opener]
        support = max(non_opener, key=lambda m: m["throws_per_round"], default=None)

        def role_of(m):
            is_awp = m["awp_per_round"] >= AWP_THRESHOLD
            is_op = m is opener
            if is_op and is_awp:
                return "opener+awp"
            if is_op:
                return "opener"
            if is_awp:
                return "awp"
            if m is support:
                return "support"
            return "rifler"

        teams[root] = {
            "team": unique_name(labels[root]),
            "n_rosters": len(rs),
            "n_rounds": n_team,
            "players": [
                {
                    "name": m["name"],
                    "role": role_of(m),
                    "n_rounds": m["n_rounds"],
                    "awp_per_round": round(m["awp_per_round"], 3),
                    "throws_per_round": m["throws_per_round"],
                    "first_contact": round(m["first_contact"], 3),
                    "opening": f"{m.get('opening_kills')}/{m.get('opening_deaths')}",
                    "first_kill_median_frac": m.get("first_kill_median_frac"),
                }
                for m in core
            ],
        }
    return {"version": 1, "built_at": datetime.now(timezone.utc).isoformat(),
            "n_teams": len(teams), "teams": teams}


def _print_teams(mem: dict) -> None:
    print(f"\n=== 队伍风格（{mem['n_teams']} 队）===")
    teams = mem["teams"]
    # 按 opener 首杀数排序
    def opener_kills(t):
        op = next((p for p in t["players"] if p["role"] == "opener"), None)
        return int(op["opening"].split("/")[0]) if op else 0
    ordered = sorted(teams.values(), key=opener_kills, reverse=True)
    for t in ordered[:40]:
        roles = ", ".join(f"{p['name']}[{p['role']}]" for p in t["players"])
        print(f"  {t['team']:<16} ({t['n_rounds']}回) {roles}")


def main() -> None:
    df = build_all()
    print(f"全量选手（回合≥{MIN_ROUNDS}）: {len(df)} 人")
    print(f"role 分布:\n{df['role'].value_counts().to_string()}\n")

    oe = df[df["role"] == "opener"].copy()
    oe["opening_net"] = oe["opening_kills"] - oe["opening_deaths"]
    oe = oe.sort_values("opening_net", ascending=False)
    cols = ["name", "opening_kills", "opening_deaths", "opening_duel_success",
            "opening_participation", "first_kill_median_frac", "traded_death_rate"]
    print("=== opener 排行（opening 净胜降序，前 30）===")
    print(oe[cols].head(30).to_string(index=False))

    csv_out = config.DATA_DIR / "player_hlvt_all.csv"
    json_out = config.DATA_DIR / "player_hlvt_all.json"
    df.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(
        {"version": 1, "built_at": datetime.now(timezone.utc).isoformat(),
         "players": df.to_dict(orient="records")},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n已写入 {csv_out}\n已写入 {json_out}")

    mem = build_team_style()
    _print_teams(mem)
    out = config.DATA_DIR / "team_style_memory.json"
    out.write_text(json.dumps(mem, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n队伍风格已写入 {out}")


if __name__ == "__main__":
    main()
