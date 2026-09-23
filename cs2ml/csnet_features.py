"""Extract per-player spatial-only features from CS2 demos using cs-net v4.

Phase 0 of the cs-net evaluation (docs/decision pending; see memory
cs2-csnet-evaluation): run the PRETRAINED spatial-only heads (winrate /
alive_end / future_kill) on rounds parsed by cs-net's own demo_parser, and
emit per-player per-round aggregates that can join our round_dataset for
prematch map-winner evaluation.

Paper-only research tooling: reads local .dem/json.gz files, writes parquet.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CSNET_ROOT = Path(__file__).resolve().parent.parent / "extern" / "cs-net"
for _sub in ("scripts",):
    _p = str(CSNET_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from create_training_data import _convert_inventory_indices  # noqa: E402
from replay_tool.filter import filter_data  # noqa: E402
from spatial_only_predictor import SpatialOnlyPredictor  # noqa: E402
from training_data.map_loader import get_map_geometry  # noqa: E402
from training_data.round_processor import process_round  # noqa: E402

SPATIAL_TASKS = ("winrate", "alive_end", "future_kill")


def _round_features(out: dict, meta: dict,
                    demo_path: str, round_idx: int) -> list[dict]:
    """Aggregate per-tick task curves (slot-parallel lists) into per-player
    per-round summaries over early (freeze+opening) and mid segments."""
    ticks = out["ticks"]
    players = meta.get("players") or []
    teams = meta.get("teams") or []
    winner = meta.get("winner")
    if not ticks or not players:
        return []
    n = len(ticks)

    def _slot_vals(task: str, idxs) -> list[float]:
        acc = [[] for _ in players]
        for i in idxs:
            series = ticks[i].get(task)
            if not series:
                continue
            for slot, v in enumerate(series):
                if v is not None:
                    acc[slot].append(float(v))
        return [float(np.mean(a)) if a else np.nan for a in acc]

    early_idx = range(0, max(1, n // 8))
    mid_idx = range(n // 4, max(n // 4 + 1, n // 2))
    rows = []
    for slot, pinfo in enumerate(players):
        row = {"demo_path": demo_path, "round_idx": round_idx,
               "slot": slot, "player": pinfo.get("name"),
               "steamid": str(pinfo.get("steamid", "")),
               "team": teams[slot] if slot < len(teams) else None,
               "winner": winner}
        for task in SPATIAL_TASKS:
            row[f"early_{task}"] = _slot_vals(task, early_idx)[slot]
            row[f"mid_{task}"] = _slot_vals(task, mid_idx)[slot]
        rows.append(row)
    return rows


def extract_demo(json_gz: Path, predictor: SpatialOnlyPredictor,
                 maps_dir: Path) -> list[dict]:
    with gzip.open(json_gz, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("format") != "cs2.demo.v2":
        raise ValueError(f"format:{data.get('format')}")
    map_name = data.get("map", "unknown")
    filter_data(data)
    try:
        map_geom = get_map_geometry(map_name, maps_dir)
    except FileNotFoundError:
        map_geom = None
    players_meta = data.get("players")
    rows: list[dict] = []
    for round_idx, round_data in enumerate(data.get("rounds", [])):
        _convert_inventory_indices(
            {"weapons": data.get("weapons", {}), "rounds": [round_data]})
        try:
            sample = process_round(
                round_data, map_geom=map_geom,
                source_file=json_gz.name, match_teams=None,
                players_meta=players_meta, tick_interval=0.25,
                compute_depth=map_geom is not None,
                places=data.get("places"))
        except Exception:
            continue  # skip malformed rounds, keep batch going
        if int(sample["meta"].get("T", 0)) < 8:
            continue
        out = predictor.predict_round_full(sample, chunk=32)
        rows.extend(_round_features(out, sample.get("meta", {}),
                                    str(json_gz), round_idx))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="cs-net demo JSON(.gz) files")
    parser.add_argument("--models-dir", type=Path,
                        default=CSNET_ROOT / "checkpoints")
    parser.add_argument("--maps-dir", type=Path,
                        default=CSNET_ROOT / "maps" / "optimized_obj_files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    predictor = SpatialOnlyPredictor(args.models_dir, device=args.device)
    all_rows: list[dict] = []
    for path in args.inputs:
        path = Path(path)
        try:
            rows = extract_demo(path, predictor, args.maps_dir)
        except Exception as exc:
            print(f"SKIP {path.name}: {type(exc).__name__}:{exc}",
                  file=sys.stderr)
            continue
        print(f"OK {path.name}: {len(rows)} player-round rows", flush=True)
        all_rows.extend(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows)
    frame.to_parquet(args.out, index=False)
    print(json.dumps({"rows": len(frame), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
