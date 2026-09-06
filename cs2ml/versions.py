"""CS2 游戏版本标注（coarse patch-era tagging）。

CS2 是持续更新的，跨版本间部分操作/打法不可复刻，因此每个样本需要标注其
所属的"玩法时代"。这里用少量会实质改变 meta 的重大更新作为分界线，把比赛
日期映射为一个粗粒度 era 标签（不是完整补丁日志）。

参考: https://esports.gg/news/counter-strike-2/all-cs2-updates/
"""
from __future__ import annotations

# (date_str, label)，按日期升序。date_str 为分界日：该日及之后归入该 label。
# 只收录会实质改变 meta 的更新（地图池 / 命中判定 / 动画 / 引擎），不含纯外观更新。
# 参考: https://esports.gg/news/counter-strike-2/all-cs2-updates/ 、HLTV、SteamDB 补丁日志。
CS2_MAJOR_PATCHES: list[tuple[str, str]] = [
    ("2024-06-25", "cs2-2024-mid"),         # 社区地图 / 光照
    ("2024-11-13", "cs2-2024-q4"),          # 伤害预测 / 动画 / 地图指引
    ("2025-01-29", "cs2-2025-train"),       # Vertigo→Train 地图池大改
    ("2025-07-29", "cs2-2025-animgraph"),   # AnimGraph2 / 地图重做 / 命中判定
    ("2025-10-15", "cs2-2025-q4"),          # Source2 引擎 / 拆弹延迟 / 穿透
    ("2026-01-21", "cs2-2026-s4"),          # Season4：Train→Anubis 地图池 / 跳跃机制 / MP7-MP5
    ("2026-04-20", "cs2-2026-animgraph2"),  # AnimGraph2 正式上线 / 后坐力镜头重做
    ("2026-07-08", "cs2-2026-s5"),          # Season5：Overpass→Cache 地图池 / C4 爆炸伤害机制重做(冲击波/不穿墙)
]

# 补充说明（不计入分界，仅作参考）:
# - 2026-04-21 后坐力镜头运动向 CS:GO 回归的微调 → 归入 animgraph2 时代（同批上线）
# - 2026-04-28 Cache 回归（仅竞技/休闲/死亡竞赛/重赛，未进 Active Duty）→ 不影响职业赛
# - 2026-07-09 / 07-20 C4 重做后续修补（移除 1 点最低伤害 / 烟雾交互）→ 归入 s5 时代

LEGACY_LABEL = "cs2-legacy"


def version_era(day_iso: str | None) -> str:
    """ISO 时间戳 → era 标签。早于首个分界日的归入 legacy。"""
    d = (day_iso or "")[:10]
    label = LEGACY_LABEL
    for date_str, lbl in CS2_MAJOR_PATCHES:
        if d >= date_str:
            label = lbl
        else:
            break
    return label
