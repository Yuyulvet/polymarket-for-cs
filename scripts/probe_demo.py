"""Probe one HLTV demo end-to-end: extract .rar -> parse .dem -> summarize.

Validates the full demo-learning pipeline on a single match before scaling up.

Usage:
    python scripts/probe_demo.py data/demos/<file>.rar
"""
from __future__ import annotations

import sys
from pathlib import Path

from cs2ml.demo_parse import extract_rar, parse_demo


def _shape(df) -> str:
    if df is None:
        return "N/A"
    try:
        return f"{df.height} rows x {df.width} cols"
    except Exception:
        return "?"


def main() -> None:
    rar_path = Path(sys.argv[1])
    demos = extract_rar(rar_path)
    print(f"extracted {len(demos)} demo(s):")
    for d in demos:
        print(f"  {d.name}  ({d.stat().st_size} bytes)")

    for d in demos:
        print(f"\n=== parse {d.name} ===")
        out = parse_demo(d)
        hdr = out.get("header") or {}
        if isinstance(hdr, dict):
            print("  header:", {k: hdr.get(k) for k in ("map_name", "server_name") if k in hdr})
        else:
            print("  header:", type(hdr).__name__)
        for key in ("rounds", "kills", "damages", "grenades", "player_info"):
            df = out.get(key)
            print(f"  {key:12s} {_shape(df)}")
            if df is not None and key in ("rounds", "kills"):
                try:
                    print("    cols:", list(df.columns)[:30])
                except Exception:
                    pass


if __name__ == "__main__":
    main()
