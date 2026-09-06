"""S 级赛事 demo 采集 + 下载时队名映射 manifest。

按用户指示（"样本优先从 S 级赛事"，"队名映射在下载时就定"）：

  1. 从 SQLite 取近 N 个月的 S 级 finished 比赛（bo3.gg 队名、日期、地图）。
  2. 分页抓取 HLTV /results，构建 (match_id, team1, team2, date, event, bo) 索引。
  3. 按"队名对 + 日期"把 bo3.gg 比赛映射到 HLTV match_id（下载时定映射）。
  4. 抓 match 页拿 demo_id -> 下载 .rar -> 7z 解出每张地图 .dem。
  5. 用 demo header 的 map_name 交叉验证地图集（防止错配）。
  6. manifest 落 SQLite `demos` 表；解出 .dem 后立即删除原始 .rar（回收磁盘）。

用法：
    python -m cs2ml.collect --dry-run                 # 只做映射，不下载（校验匹配质量）
    python -m cs2ml.collect --months 3 --limit 5      # 下载近 3 个月 S 级，最多 5 场
"""
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config, store
from .demo_parse import extract_rar
from .hltv import HLTVClient

# HLTV /results 每页 100 条（pagination 显示 "1 - 100 of ..."）
_RESULTS_PAGE_SIZE = 100

# 队名归一化：只留小写字母数字，去掉空格/符号/大小写差异。
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_name(name: str) -> str:
    return _NON_ALNUM.sub("", (name or "").lower())


def _sim(a: str, b: str) -> int:
    """队名相似度：2=归一后完全相同，1=子串包含（>=4 字符），0=不匹配。"""
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return 0
    if na == nb:
        return 2
    if len(na) >= 4 and len(nb) >= 4 and (na in nb or nb in na):
        return 1
    return 0


# --- HLTV results 页解析 ---

def parse_results_page(page_html: str) -> list[dict]:
    """把 /results 页 HTML 解析成 match 记录列表（含日期/队名/赛事/bo）。"""
    out: list[dict] = []
    for block in page_html.split('<div class="result-con'):
        m = re.search(r'data-zonedgrouping-entry-unix="(\d+)"', block)
        if not m:
            continue
        mm = re.search(r'href="/matches/(\d+)/', block)
        if not mm:
            continue
        teams = re.findall(r'<div class="team[^"]*">([^<]+)</div>', block)
        if len(teams) < 2:
            continue
        ev = re.search(r'<span class="event-name">([^<]+)</span>', block)
        botype = re.search(r'<div class="map map-text">([^<]+)</div>', block)
        out.append({
            "match_id": int(mm.group(1)),
            "date": datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc).date(),
            "team1": html.unescape(teams[0].strip()),
            "team2": html.unescape(teams[1].strip()),
            "event": html.unescape(ev.group(1).strip()) if ev else "",
            "bo": html.unescape(botype.group(1).strip()) if botype else "",
        })
    return out


def build_hltv_index(client: HLTVClient, since: datetime, max_pages: int = 60) -> list[dict]:
    """分页抓取 /results 直到最老一条早于 `since`，返回全部 match 记录。"""
    out: list[dict] = []
    for page in range(max_pages):
        html = client.results(page * _RESULTS_PAGE_SIZE)
        rows = parse_results_page(html)
        if not rows:
            break
        out.extend(rows)
        if min(r["date"] for r in rows) < since.date():
            break
    return out


# --- bo3.gg -> HLTV 匹配 ---

def match_to_hltv(bo3: dict, hltv_index: list[dict], day_window: int = 2) -> dict | None:
    """按队名对 + 日期把 bo3.gg 比赛映射到一条 HLTV 记录。

    返回 {hltv, hltv_team1, hltv_team2, score, ambiguous}（hltv 队名已按 bo3 顺序对齐），
    无候选返回 None。
    """
    t1, t2 = bo3["team1"], bo3["team2"]
    date = bo3["date"]  # date 对象

    cands: list[dict] = []
    for h in hltv_index:
        if abs((h["date"] - date).days) > day_window:
            continue
        direct_ok = _sim(t1, h["team1"]) > 0 and _sim(t2, h["team2"]) > 0
        swap_ok = _sim(t1, h["team2"]) > 0 and _sim(t2, h["team1"]) > 0
        if not (direct_ok or swap_ok):
            continue
        if direct_ok and swap_ok:
            # 两个方向都成立（如同名队对同名队），取分高的方向
            sd = _sim(t1, h["team1"]) + _sim(t2, h["team2"])
            ss = _sim(t1, h["team2"]) + _sim(t2, h["team1"])
            direct = sd >= ss
        else:
            direct = direct_ok
        if direct:
            score = _sim(t1, h["team1"]) + _sim(t2, h["team2"])
            a1, a2 = h["team1"], h["team2"]
        else:
            score = _sim(t1, h["team2"]) + _sim(t2, h["team1"])
            a1, a2 = h["team2"], h["team1"]
        cands.append({
            "hltv": h, "hltv_team1": a1, "hltv_team2": a2,
            "score": score,
            "daydiff": abs((h["date"] - date).days),
        })

    if not cands:
        return None
    # 排序：分数降序，日期差升序
    cands.sort(key=lambda c: (-c["score"], c["daydiff"]))
    best = cands[0]
    ambiguous = len(cands) > 1 and cands[1]["score"] == best["score"]
    return {
        "hltv": best["hltv"], "hltv_team1": best["hltv_team1"], "hltv_team2": best["hltv_team2"],
        "score": best["score"], "ambiguous": ambiguous,
    }


# --- manifest 持久化 ---

def upsert_demo(conn, d: dict) -> None:
    conn.execute(
        """
        INSERT INTO demos (demo_id, hltv_match_id, bo3gg_match_id, tier, start_date,
            version_era, bo_type, maps, team1, team2, hltv_team1, hltv_team2, hltv_event,
            ambiguous, rar_path, extracted_dir, map_files, map_check, status, error, fetched_at)
        VALUES (:demo_id, :hltv_match_id, :bo3gg_match_id, :tier, :start_date,
            :version_era, :bo_type, :maps, :team1, :team2, :hltv_team1, :hltv_team2, :hltv_event,
            :ambiguous, :rar_path, :extracted_dir, :map_files, :map_check, :status, :error, :fetched_at)
        ON CONFLICT(demo_id) DO UPDATE SET
            bo3gg_match_id=excluded.bo3gg_match_id, status=excluded.status,
            rar_path=excluded.rar_path, extracted_dir=excluded.extracted_dir,
            map_files=excluded.map_files, map_check=excluded.map_check,
            error=excluded.error, fetched_at=excluded.fetched_at
        """,
        d,
    )


def _iso_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


# --- 主入口 ---

def collect(months: int = 3, limit: int | None = None, dry_run: bool = False) -> dict:
    conn = store.connect()
    since = datetime.now(timezone.utc) - timedelta(days=months * 30)
    since_iso = since.strftime("%Y-%m-%d")

    rows = conn.execute(
        """
        SELECT m.id, m.start_date, m.maps, m.bo_type, m.version_era, m.tier,
               t1.name AS t1, t2.name AS t2
        FROM matches m
        JOIN teams t1 ON t1.id = m.team1_id
        JOIN teams t2 ON t2.id = m.team2_id
        WHERE m.tier = 's' AND m.status = 'finished' AND m.start_date >= ?
          AND m.id NOT IN (SELECT bo3gg_match_id FROM demos
                           WHERE bo3gg_match_id IS NOT NULL
                             AND status IN ('downloaded','extracted','parsed'))
        ORDER BY m.start_date DESC
        """,
        (since_iso,),
    ).fetchall()
    print(f"S 级 finished 比赛（近 {months} 个月）：{len(rows)} 场")

    # 已下载的 demo_id 集合，用于跳过
    done_ids = {r["demo_id"] for r in conn.execute("SELECT demo_id FROM demos")}
    done_matches = {r["bo3gg_match_id"] for r in conn.execute(
        "SELECT bo3gg_match_id FROM demos WHERE status IN ('downloaded','extracted','parsed')")}

    client = HLTVClient()
    hltv_index = build_hltv_index(client, since, max_pages=months * 15)
    print(f"HLTV 索引条数：{len(hltv_index)}")

    matched = unmatched = 0
    results = {"matched": 0, "unmatched": 0, "downloaded": 0, "failed": 0, "skipped": 0}

    for r in rows:
        if limit is not None and results["matched"] >= limit:
            break
        bo3 = {
            "id": r["id"], "team1": r["t1"], "team2": r["t2"],
            "date": datetime.fromisoformat(r["start_date"]).date(),
            "maps": json.loads(r["maps"]) if r["maps"] else [],
            "bo_type": r["bo_type"], "version_era": r["version_era"],
            "tier": r["tier"], "start_date": r["start_date"],
        }
        m = match_to_hltv(bo3, hltv_index)
        if m is None:
            unmatched += 1
            print(f"  [unmatched] {bo3['team1']} vs {bo3['team2']} ({bo3['date']})")
            continue
        matched += 1
        results["matched"] += 1
        tag = "AMBIGUOUS" if m["ambiguous"] else "ok"
        print(f"  [{tag}] {bo3['team1']} vs {bo3['team2']} ({bo3['date']}) "
              f"-> HLTV #{m['hltv']['match_id']} "
              f"({m['hltv_team1']} vs {m['hltv_team2']}, {m['hltv']['event']})")

        if r["id"] in done_matches:
            results["skipped"] += 1
            continue
        if dry_run:
            continue

        # 抓 demo_id + 下载 + 解压
        try:
            page = client.match_page(m["hltv"]["match_id"])
            demo_ids = client.parse_demo_ids(page)
        except Exception as e:  # noqa: BLE001
            results["failed"] += 1
            print(f"    match_page 失败：{e}")
            continue

        for did in demo_ids:
            if did in done_ids:
                continue
            base = config.DATA_DIR / "demos" / f"hltv-{m['hltv']['match_id']}-{did}"
            try:
                rar = client.download(did, base.parent)
                demos = extract_rar(rar, base)
                Path(rar).unlink(missing_ok=True)  # .dem 已解出，rar 纯冗余
            except Exception as e:  # noqa: BLE001
                results["failed"] += 1
                print(f"    下载/解压 {did} 失败：{e}")
                upsert_demo(conn, {
                    "demo_id": did, "hltv_match_id": m["hltv"]["match_id"],
                    "bo3gg_match_id": r["id"], "tier": bo3["tier"],
                    "start_date": bo3["start_date"], "version_era": bo3["version_era"],
                    "bo_type": bo3["bo_type"], "maps": json.dumps(bo3["maps"]),
                    "team1": bo3["team1"], "team2": bo3["team2"],
                    "hltv_team1": m["hltv_team1"], "hltv_team2": m["hltv_team2"],
                    "hltv_event": m["hltv"]["event"], "ambiguous": int(m["ambiguous"]),
                    "rar_path": "", "extracted_dir": str(base), "map_files": "[]",
                    "map_check": None, "status": "failed", "error": str(e)[:500],
                    "fetched_at": store._now(),
                })
                continue

            map_files = [str(p) for p in demos]
            map_check = _check_maps(demos, bo3["maps"])
            status = "extracted"
            upsert_demo(conn, {
                "demo_id": did, "hltv_match_id": m["hltv"]["match_id"],
                "bo3gg_match_id": r["id"], "tier": bo3["tier"],
                "start_date": bo3["start_date"], "version_era": bo3["version_era"],
                "bo_type": bo3["bo_type"], "maps": json.dumps(bo3["maps"]),
                "team1": bo3["team1"], "team2": bo3["team2"],
                "hltv_team1": m["hltv_team1"], "hltv_team2": m["hltv_team2"],
                "hltv_event": m["hltv"]["event"], "ambiguous": int(m["ambiguous"]),
                "rar_path": str(rar), "extracted_dir": str(base),
                "map_files": json.dumps(map_files), "map_check": map_check,
                "status": status, "error": None, "fetched_at": store._now(),
            })
            results["downloaded"] += 1
            print(f"    demo {did}: {len(demos)} 张地图 -> {base} (map_check={map_check})")
        conn.commit()

    conn.close()
    print(f"\n结果：matched={matched} unmatched={unmatched} "
          f"downloaded={results['downloaded']} failed={results['failed']} skipped={results['skipped']}")
    return results


def _check_maps(demos: list[Path], expected: list[str]) -> str | None:
    """用 demo header 的 map_name 交叉验证（actual ⊆ expected 视为 ok）。"""
    if not expected:
        return "pending"
    from demoparser2 import DemoParser

    actual: set[str] = set()
    for d in demos:
        try:
            hdr = DemoParser(str(d)).parse_header()
            mn = hdr.get("map_name")
            if mn:
                actual.add(str(mn))
        except Exception:
            continue
    if not actual:
        return "pending"
    exp = {store.normalize_map(m) for m in expected if store.normalize_map(m)}
    if actual <= exp:
        return "ok"
    return "mismatch"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    collect(args.months, args.limit, args.dry_run)
