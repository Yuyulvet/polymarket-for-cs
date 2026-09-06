"""Dump recent CS2 update contents to a UTF-8 file for curation."""
import requests
from datetime import datetime, timezone

r = requests.get("https://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/",
                 params={"appid": 730, "count": 300, "maxlength": 0, "format": "json"}, timeout=30)
items = r.json()["appnews"]["newsitems"]

lines = []
for it in items:
    d = datetime.fromtimestamp(it["date"], tz=timezone.utc).strftime("%Y-%m-%d")
    if d <= "2026-04-20":
        continue
    title = (it.get("title") or "").strip()
    body = (it.get("contents") or "").strip()
    lines.append(f"### {d} | {title}")
    lines.append(body)
    lines.append("")

with open("reports/steam_news_dump.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("wrote", len(lines), "lines")
