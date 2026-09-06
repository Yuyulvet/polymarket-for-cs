"""SQLite storage layer."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import config
from .versions import version_era

# 地图名规范化：小写 + 合并别名（如 "cache" -> "de_cache"）。
_MAP_ALIASES = {"cache": "de_cache", "dust2": "de_dust2", "dust 2": "de_dust2"}


def normalize_map(name: Any) -> str | None:
    n = (name or "").strip().lower()
    if n in ("", "unknown"):
        return None
    return _MAP_ALIASES.get(n, n)


def extract_maps(m: dict[str, Any]) -> list[str]:
    """从 match dict 的 games 字段提取规范化地图名列表（含顺序）。"""
    games = m.get("games")
    if not isinstance(games, list):
        return []
    out: list[str] = []
    for g in games:
        if isinstance(g, dict):
            mn = normalize_map(g.get("map_name"))
            if mn:
                out.append(mn)
    return out

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    slug TEXT,
    status TEXT,
    parsed_status TEXT,
    bo_type INTEGER,
    winner_team_id INTEGER,
    tier TEXT,
    start_date TEXT,
    end_date TEXT,
    team1_id INTEGER,
    team2_id INTEGER,
    team1_score INTEGER,
    team2_score INTEGER,
    stars INTEGER,
    tournament_id INTEGER,
    ai_pred_winner INTEGER,
    ai_pred_team1_score INTEGER,
    ai_pred_team2_score INTEGER,
    version_era TEXT,
    maps TEXT,
    raw_json TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_matches_start ON matches(start_date);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    slug TEXT,
    name TEXT,
    image_url TEXT,
    first_seen TEXT,
    last_seen TEXT
);

CREATE TABLE IF NOT EXISTS tournaments (
    id INTEGER PRIMARY KEY,
    slug TEXT,
    name TEXT,
    tier TEXT,
    prize REAL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS backfill_days (
    date TEXT PRIMARY KEY,
    status TEXT,            -- ok | empty | error
    match_count INTEGER,
    error TEXT,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS demos (
    demo_id TEXT PRIMARY KEY,        -- hltv download/demo/{id}
    hltv_match_id INTEGER,
    bo3gg_match_id INTEGER,
    tier TEXT,
    start_date TEXT,
    version_era TEXT,
    bo_type INTEGER,
    maps TEXT,                        -- json list（bo3gg 预期地图）
    team1 TEXT,                       -- bo3gg 队名
    team2 TEXT,                       -- bo3gg 队名
    hltv_team1 TEXT,                  -- hltv 队名（下载时定映射，与 team1 对齐）
    hltv_team2 TEXT,                  -- hltv 队名（与 team2 对齐）
    hltv_event TEXT,
    ambiguous INTEGER DEFAULT 0,      -- 模糊匹配有竞争候选
    rar_path TEXT,
    extracted_dir TEXT,
    map_files TEXT,                   -- json list（解出的 .dem 路径）
    map_check TEXT,                   -- ok | mismatch | pending
    status TEXT,                      -- downloaded | extracted | parsed | failed | skip
    error TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_demos_bo3gg ON demos(bo3gg_match_id);
CREATE INDEX IF NOT EXISTS idx_demos_status ON demos(status);

CREATE TABLE IF NOT EXISTS map_rosters (
    demo_path TEXT,                    -- .dem 文件路径
    roster_key TEXT,                   -- 5 个 steamid 排序元组（roster 身份）
    match_id TEXT,                     -- 比赛分组（父目录名）
    bo3gg_match_id INTEGER,
    start_date TEXT,
    map_name TEXT,
    team_number INTEGER,               -- demo 内 team_number（2/3，跨图翻转）
    steamids TEXT,                     -- json list[str] 5 名选手
    won_map INTEGER,                   -- 该 roster 是否赢下这张地图（0/1）
    complete INTEGER,                  -- demo 是否完整（胜方>=13 分）
    PRIMARY KEY (demo_path, roster_key)  -- 每张地图有两方 roster
);
CREATE INDEX IF NOT EXISTS idx_map_rosters_match ON map_rosters(match_id);
CREATE INDEX IF NOT EXISTS idx_map_rosters_bo3gg ON map_rosters(bo3gg_match_id);
"""


def connect(db_path: Any = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or config.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive schema migrations for DBs created before a column existed."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(matches)")}
    if "version_era" not in cols:
        conn.execute("ALTER TABLE matches ADD COLUMN version_era TEXT")
        conn.commit()
        _backfill_version_era(conn)
    if "maps" not in cols:
        conn.execute("ALTER TABLE matches ADD COLUMN maps TEXT")
        conn.commit()
        _backfill_maps(conn)


def _backfill_version_era(conn: sqlite3.Connection) -> None:
    """One-time: tag existing rows with the era derived from their start_date."""
    rows = conn.execute("SELECT id, start_date FROM matches WHERE version_era IS NULL").fetchall()
    if not rows:
        return
    with conn:
        for r in rows:
            conn.execute("UPDATE matches SET version_era=? WHERE id=?",
                         (version_era(r["start_date"]), r["id"]))


def _backfill_maps(conn: sqlite3.Connection) -> None:
    """One-time: parse games out of raw_json into the denormalized maps column."""
    rows = conn.execute("SELECT id, raw_json FROM matches WHERE maps IS NULL").fetchall()
    if not rows:
        return
    with conn:
        for r in rows:
            try:
                m = json.loads(r["raw_json"]) if r["raw_json"] else {}
            except (ValueError, TypeError):
                m = {}
            conn.execute("UPDATE matches SET maps=? WHERE id=?",
                         (json.dumps(extract_maps(m)), r["id"]))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_match(conn: sqlite3.Connection, m: dict[str, Any], fetched_at: str | None = None) -> None:
    ai = m.get("ai_predictions") or {}
    if isinstance(ai, str):
        try:
            ai = json.loads(ai)
        except (ValueError, TypeError):
            ai = {}
    fetched_at = fetched_at or _now()
    conn.execute(
        """
        INSERT OR REPLACE INTO matches (id, slug, status, parsed_status, bo_type, winner_team_id,
            tier, start_date, end_date, team1_id, team2_id, team1_score, team2_score, stars,
            tournament_id, ai_pred_winner, ai_pred_team1_score, ai_pred_team2_score, version_era,
            maps, raw_json, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            m.get("id"), m.get("slug"), m.get("status"), m.get("parsed_status"), m.get("bo_type"),
            m.get("winner_team_id"), m.get("tier"), m.get("start_date"), m.get("end_date"),
            m.get("team1_id"), m.get("team2_id"), m.get("team1_score"), m.get("team2_score"),
            m.get("stars"), m.get("tournament"),
            ai.get("prediction_winner_team_id") if isinstance(ai, dict) else None,
            ai.get("prediction_team1_score") if isinstance(ai, dict) else None,
            ai.get("prediction_team2_score") if isinstance(ai, dict) else None,
            version_era(m.get("start_date")),
            json.dumps(extract_maps(m)),
            json.dumps(m, ensure_ascii=False), fetched_at,
        ),
    )


def upsert_teams(conn: sqlite3.Connection, teams: dict[str, dict], fetched_at: str | None = None) -> None:
    fetched_at = fetched_at or _now()
    for tid, t in teams.items():
        conn.execute(
            """
            INSERT INTO teams (id, slug, name, image_url, first_seen, last_seen)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET slug=excluded.slug, name=excluded.name,
                image_url=COALESCE(excluded.image_url, teams.image_url), last_seen=excluded.last_seen
            """,
            (int(tid), t.get("slug"), t.get("name"), t.get("image_url"), fetched_at, fetched_at),
        )


def upsert_tournaments(conn: sqlite3.Connection, tournaments: dict[str, dict]) -> None:
    now = _now()
    for tid, t in tournaments.items():
        conn.execute(
            """
            INSERT INTO tournaments (id, slug, name, tier, prize, updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET slug=excluded.slug, name=excluded.name,
                tier=excluded.tier, prize=excluded.prize, updated_at=excluded.updated_at
            """,
            (int(tid), t.get("slug"), t.get("name"), t.get("tier"), t.get("prize"), now),
        )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))
