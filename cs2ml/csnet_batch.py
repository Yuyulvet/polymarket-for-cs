"""Batch cs-net spatial-only feature extraction over the v2 rebuild pool.

Reads data/map1/rebuild/full-20260916/map_labels.parquet (593 qualified maps),
parses each .dem with cs-net's demo_parser, runs the pretrained spatial-only
heads (GPU), and writes one parquet per demo under data/csnet_features/.
Resume-safe: existing outputs are skipped.

Paper-only research tooling (Phase 0 of the cs-net evaluation).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "extern" / "cs-net"))
sys.path.insert(0, str(ROOT / "extern" / "cs-net" / "scripts"))

from demo_parser.extract import parse_demo, save_demo_json  # noqa: E402
from cs2ml.csnet_features import (  # noqa: E402
    CSNET_ROOT, SpatialOnlyPredictor, extract_demo)

LABELS = ROOT / "data/map1/rebuild/full-20260916/map_labels.parquet"
JSON_DIR = ROOT / "data/csnet_json"
FEAT_DIR = ROOT / "data/csnet_features"


def main() -> int:
    labels = pd.read_parquet(LABELS)
    demos = sorted(set(labels["demo_path"]))
    print(f"pool: {len(demos)} demos", flush=True)
    predictor = SpatialOnlyPredictor(CSNET_ROOT / "checkpoints",
                                     device="cuda")
    maps_dir = CSNET_ROOT / "maps/optimized_obj_files"
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    done = fail = 0
    for i, demo in enumerate(demos):
        slug = demo.replace("/", "_").replace("\\", "_").replace(".dem", "")
        feat_path = FEAT_DIR / f"{slug}.parquet"
        if feat_path.exists():
            done += 1
            continue
        dem_path = ROOT / demo if not Path(demo).is_absolute() else Path(demo)
        if not dem_path.is_file():
            print(f"[{i}] MISSING {demo}", flush=True)
            fail += 1
            continue
        json_path = JSON_DIR / f"{slug}.json.gz"
        try:
            if not json_path.exists():
                t0 = time.time()
                data = parse_demo(str(dem_path), interval=0.25, verbose=False)
                save_demo_json(data, str(json_path), compact=True,
                               compress=True)
                parse_s = time.time() - t0
            else:
                parse_s = 0.0
            t0 = time.time()
            rows = extract_demo(json_path, predictor, maps_dir)
            infer_s = time.time() - t0
            pd.DataFrame(rows).to_parquet(feat_path, index=False)
            done += 1
            print(f"[{i}] OK {slug} rows={len(rows)} "
                  f"parse={parse_s:.0f}s infer={infer_s:.0f}s", flush=True)
        except Exception as exc:  # keep batch going; log honestly
            fail += 1
            print(f"[{i}] FAIL {slug}: {type(exc).__name__}: {exc}",
                  flush=True)
    print(json.dumps({"done": done, "fail": fail}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
