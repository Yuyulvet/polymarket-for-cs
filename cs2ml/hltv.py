"""HLTV demo downloader (curl_cffi TLS impersonation).

HLTV is the canonical public source for pro CS2 GOTV demos. This module:
  - lists recent finished matches (results endpoint)
  - extracts GOTV demo download ids from a match page
  - downloads the demo file to disk (resumable: skips existing non-empty files)

Polite by design: single-threaded, rate-limited. Anti-bot: curl_cffi with a
Chrome TLS fingerprint (plain requests gets Cloudflare-challenged).

Usage:
    python -m cs2ml.hltv download <match_id> [--out data/demos]
    python -m cs2ml.hltv find <team1> <team2> [--days 7]
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from curl_cffi import requests as cffi_requests

from . import config

HLTV_BASE = "https://www.hltv.org"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
REQUEST_INTERVAL_SEC = 2.0  # polite delay between requests

_DEMO_RE = re.compile(r"/download/demo/(\d+)")
_MATCH_RE = re.compile(r"/matches/(\d+)/([a-z0-9-]+)")
_EXT_BY_CTYPE = {
    "application/x-rar-compressed": ".rar",
    "application/rar": ".rar",
    "application/octet-stream": ".dem",
}


class HLTVClient:
    def __init__(self, delay: float = REQUEST_INTERVAL_SEC):
        self.delay = delay
        self.session = cffi_requests.Session(impersonate="chrome")
        self.session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, url: str, timeout: float = 60.0, **kw: Any):
        self._throttle()
        return self.session.get(url, timeout=timeout, **kw)

    def results(self, offset: int = 0) -> str:
        r = self._get(f"{HLTV_BASE}/results", params={"offset": offset})
        r.raise_for_status()
        return r.text

    def match_page(self, match_id: int) -> str:
        r = self._get(f"{HLTV_BASE}/matches/{match_id}/x")
        r.raise_for_status()
        return r.text

    @staticmethod
    def parse_demo_ids(html: str) -> list[str]:
        return sorted(set(_DEMO_RE.findall(html)))

    def download(self, demo_id: str, out_dir: Path) -> Path | None:
        """Download a demo by id, returning the saved path (None if already there).

        HLTV demos are served as .rar archives (one per match, containing the
        per-map .dem files). Filename comes from the final redirected URL.

        Large (~540MB) and slow, so: single streaming GET with no hard timeout,
        written to a `.part` temp file and renamed only on completion.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._throttle()

        # curl_cffi's Response has no __enter__/__exit__, so no `with` here.
        resp = self.session.get(f"{HLTV_BASE}/download/demo/{demo_id}",
                                allow_redirects=True, stream=True,
                                timeout=None)
        resp.raise_for_status()
        cd = resp.headers.get("content-disposition", "")
        fname = None
        m = re.search(r'filename="?([^";]+)"?', cd)
        if m:
            fname = m.group(1)
        if not fname:
            fname = Path(resp.url.split("?")[0]).name
        if not Path(fname).suffix:
            ext = _EXT_BY_CTYPE.get(resp.headers.get("content-type", "").split(";")[0], ".dem")
            fname += ext
        total = int(resp.headers.get("content-length", 0) or 0)

        out = out_dir / fname
        if out.exists() and (total and out.stat().st_size >= total):
            return out  # already complete

        part = out_dir / (fname + ".part")
        got = 0
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)

        if total and got < total:
            part.unlink(missing_ok=True)  # incomplete — don't leave a broken .part
            raise IOError(f"download incomplete: {got}/{total} bytes")
        part.replace(out)
        return out

    # --- bo3.gg -> HLTV match mapping ---

    def find_match(self, team1: str, team2: str, since_days: int = 7) -> int | None:
        """Find the most recent HLTV match id between two team names (fuzzy substring).

        Scans recent results pages and, for candidate matches, checks whether the
        match page contains both team names. Bounded and rate-limited.
        """
        n1, n2 = team1.lower(), team2.lower()
        for offset in (0, 50, 100):
            html = self.results(offset)
            for mid, _ in _MATCH_RE.findall(html)[:30]:
                low = self.match_page(int(mid)).lower()
                if n1 in low and n2 in low:
                    return int(mid)
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("match_id", type=int)
    d.add_argument("--out", default=str(config.DATA_DIR / "demos"))
    f = sub.add_parser("find")
    f.add_argument("team1")
    f.add_argument("team2")
    f.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    c = HLTVClient()
    if args.cmd == "download":
        html = c.match_page(args.match_id)
        demo_ids = c.parse_demo_ids(html)
        print(f"match {args.match_id}: {len(demo_ids)} demo link(s): {demo_ids}")
        for did in demo_ids:
            p = c.download(did, Path(args.out))
            print(f"  downloaded -> {p} ({p.stat().st_size if p and p.exists() else 0} bytes)")
    elif args.cmd == "find":
        mid = c.find_match(args.team1, args.team2, args.days)
        print(f"find {args.team1} vs {args.team2}: {mid}")
