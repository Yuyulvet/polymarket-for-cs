"""HLTV GOTV demo 解压 + 解析。

两段式流水线：
  1. 解压：HLTV demo 以 .rar 打包（一个 BO3 ≈ 一张或多张地图的 .dem），
     用 7z.exe 解出 .dem。rarfile 需要外部后端，这里直接用 subprocess 调 7z，
     更透明、可控。
  2. 解析：demoparser2（Rust）抽取回合/击杀/伤害/投掷物事件 → Polars DataFrame。

设计目标：为"学习引擎"提供比赛结果之外的**过程级**特征——回合经济、双方
击杀/ADR、投掷物、手枪局/翻盘等"打法"信号（这些正是跨版本不可复刻的部分）。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config

# 7z.exe 的常见安装位置（winget 装到 Program Files）
_7Z_CANDIDATES = (
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
)


def find_7z() -> str | None:
    """定位 7z.exe（先看 PATH，再看标准安装路径）。"""
    for name in ("7z.exe", "7za.exe", "7z"):
        p = shutil.which(name)
        if p:
            return p
    for loc in _7Z_CANDIDATES:
        if Path(loc).exists():
            return loc
    return None


def extract_rar(rar_path: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """解压 .rar，返回解出的 .dem 文件路径列表（已排序）。"""
    rar_path = Path(rar_path)
    out_dir = Path(out_dir) if out_dir else (rar_path.parent / rar_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    seven_zip = find_7z()
    if not seven_zip:
        raise RuntimeError(
            "未找到 7z.exe 解压后端。请安装 7-Zip（winget install 7zip.7zip）"
            " 或把 7z.exe 加入 PATH。"
        )
    subprocess.run(
        [seven_zip, "x", str(rar_path), f"-o{out_dir}", "-y"],
        check=True,
        capture_output=True,
    )
    demos = sorted(out_dir.rglob("*.dem"))
    if not demos:
        # 有些压缩包把 .dem 放在子目录，rglob 已覆盖；再放宽一次确认
        demos = sorted(out_dir.rglob("*"))
    return demos


# --- demo 解析（demoparser2）---

# 事件名在 CS2 demo 里是固定字符串，这里集中列出我们关心的类型。
KILL_EVENT = "player_death"
ROUND_END_EVENT = "round_end"


def parse_demo(dem_path: str | Path) -> dict[str, object]:
    """解析单个 .dem，返回事件 DataFrames（rounds / kills / damages / grenades）。

    返回 dict 键：
      - header: dict（地图、比分等头部信息）
      - rounds: Polars DataFrame（回合结束事件）
      - kills:  Polars DataFrame（击杀事件）
      - damages: Polars DataFrame（伤害事件）
      - grenades: Polars DataFrame（投掷物）
    """
    from demoparser2 import DemoParser  # 延迟导入，避免无依赖时 import 本模块就报错

    dem_path = Path(dem_path)
    parser = DemoParser(str(dem_path))

    out: dict[str, object] = {}
    try:
        out["header"] = parser.parse_header()
    except Exception:
        out["header"] = {}

    # 回合 / 击杀 / 伤害 走 parse_event；投掷物有专门接口 parse_grenades()。
    for key, ev in (
        ("rounds", ROUND_END_EVENT),
        ("kills", KILL_EVENT),
        ("damages", "player_hurt"),
    ):
        try:
            out[key] = parser.parse_event(ev)
        except Exception:
            out[key] = None
    try:
        out["grenades"] = parser.parse_grenades()
    except Exception:
        out["grenades"] = None
    try:
        out["player_info"] = parser.parse_player_info()
    except Exception:
        out["player_info"] = None
    return out


if __name__ == "__main__":
    import sys

    seven = find_7z()
    print("7z backend:", seven or "NOT FOUND")
    if len(sys.argv) > 1 and seven:
        demos = extract_rar(sys.argv[1])
        print("extracted demos:", demos)
