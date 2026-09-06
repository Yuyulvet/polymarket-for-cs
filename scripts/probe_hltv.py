"""Probe HLTV's live HTML to confirm demo-link structure before building the downloader."""
import re
from curl_cffi import requests as cffi

BASE = "https://www.hltv.org"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

s = cffi.Session(impersonate="chrome")
s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})

print("=== results page ===")
r = s.get(f"{BASE}/results?offset=0", timeout=30)
print("status", r.status_code, "len", len(r.text), "ctype", r.headers.get("content-type"))
match_ids = sorted(set(re.findall(r'/matches/(\d+)/', r.text)))
print("match ids found:", len(match_ids), match_ids[:5])

if match_ids:
    mid = match_ids[0]
    print(f"\n=== match page /matches/{mid} ===")
    r2 = s.get(f"{BASE}/matches/{mid}/x", timeout=30)
    print("status", r2.status_code, "len", len(r2.text))
    demos = sorted(set(re.findall(r'/download/demo/(\d+)', r2.text)))
    print("demo links:", demos)
    # also check for a 'demo' section / per-map demo anchors
    print("has 'GOTV' text:", "GOTV" in r2.text, "| has 'Demo' text:", "Demo" in r2.text)
