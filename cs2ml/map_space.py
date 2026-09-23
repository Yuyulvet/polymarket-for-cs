"""Versioned seven-map metadata and explicit radar/world affine calibration.

Geometry metadata is not a validated nav mesh, and XY is never silently given Z.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import requests

MAPS = tuple('de_'+m for m in ('cache', 'mirage', 'dust2', 'inferno', 'ancient', 'nuke', 'anubis'))


def validate_map(name):
    if name not in MAPS:
        raise ValueError('unsupported_map')
    return name


def build_catalog(manifest, map_data, version):
    version = str(version)
    if not re.fullmatch(r'[0-9]+', version) or str(manifest.get('client_version')) != version:
        raise ValueError('release_version_mismatch')
    result = {}
    for name in MAPS:
        if name not in manifest.get('maps', []) or name not in map_data:
            raise ValueError(f'missing_map: {name}')
        item = map_data[name]
        values = np.asarray([item.get(k) for k in ('pos_x', 'pos_y', 'scale')], dtype=float)
        if not np.isfinite(values).all() or values[2] <= 0:
            raise ValueError(f'invalid_transform: {name}')
        result[name] = {'overview': item, 'nav_status': 'not_downloaded_or_validated',
                        'geometry_status': 'not_downloaded_or_validated',
                        'broadcast_calibration_status': 'required', 'world_z_from_radar': None}
    return {'schema_version': 1, 'client_version': version, 'maps': result,
            'source': f'https://github.com/pnxenopoulos/awpy-data/releases/tag/{version}',
            'eligible_for_inference': False}


def fetch_catalog(version, output):
    if not re.fullmatch(r'[0-9]+', str(version)):
        raise ValueError('version_must_be_explicit_numeric')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base = f'https://github.com/pnxenopoulos/awpy-data/releases/download/{version}'
    fetched = {}
    for filename in ('manifest.json', 'map_data.json'):
        response = requests.get(f'{base}/{filename}', timeout=30)
        response.raise_for_status()
        if len(response.content) > 2_000_000:
            raise ValueError('unexpected_metadata_size')
        fetched[filename] = response.content
    catalog = build_catalog(json.loads(fetched['manifest.json']), json.loads(fetched['map_data.json']), version)
    catalog['retrieved_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    catalog['metadata_sha256'] = {k: hashlib.sha256(v).hexdigest() for k, v in fetched.items()}
    output.mkdir(parents=True, exist_ok=False)
    for filename, content in fetched.items():
        (output/filename).write_bytes(content)
    (output/'catalog.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    return catalog


def apply_affine(points, matrix):
    points, matrix = np.asarray(points, float), np.asarray(matrix, float)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 2):
        raise ValueError('invalid_affine_shape')
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError('nonfinite_coordinates')
    return np.column_stack([points, np.ones(len(points))]) @ matrix


def fit_calibration(spec):
    """User-supplied fixed map landmarks, not moving players. Held-out QA required."""
    validate_map(spec['map_name'])
    size = np.asarray(spec['frame_size'], float)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any() or (size != np.floor(size)).any():
        raise ValueError('invalid_frame_size')
    if not re.fullmatch(r'[0-9]+', str(spec['client_version'])):
        raise ValueError('invalid_client_version')
    max_error = float(spec['max_error_world_units'])
    if not np.isfinite(max_error) or max_error <= 0:
        raise ValueError('invalid_error_threshold')
    sets = []
    for name, minimum in (('anchors', 3), ('checkpoints', 2)):
        entries = spec[name]
        if len(entries) < minimum:
            raise ValueError(f'insufficient_{name}')
        pixel = np.asarray([r['pixel'] for r in entries], float)
        world = np.asarray([r['world_xy'] for r in entries], float)
        if pixel.shape != (len(entries), 2) or world.shape != pixel.shape:
            raise ValueError('invalid_point_shape')
        if not np.isfinite(pixel).all() or not np.isfinite(world).all():
            raise ValueError('nonfinite_coordinates')
        if (pixel < 0).any() or (pixel >= size).any():
            raise ValueError('anchor_outside_frame')
        if len(np.unique(pixel, axis=0)) != len(pixel):
            raise ValueError('duplicate_landmarks')
        sets.append((pixel, world))
    (p, w), (cp, cw) = sets
    if set(map(tuple, p)) & set(map(tuple, cp)):
        raise ValueError('checkpoints_overlap_training_anchors')
    design = np.column_stack([p, np.ones(len(p))])
    matrix, _, rank, _ = np.linalg.lstsq(design, w, rcond=None)
    if rank != 3 or np.linalg.matrix_rank(matrix[:2]) != 2:
        raise ValueError('degenerate_calibration')
    train_error = np.linalg.norm(apply_affine(p, matrix)-w, axis=1)
    check_error = np.linalg.norm(apply_affine(cp, matrix)-cw, axis=1)
    if max(train_error.max(), check_error.max()) > max_error:
        raise ValueError('calibration_error_exceeds_threshold')
    return {'schema_version': 1, 'map_name': spec['map_name'],
            'client_version': str(spec['client_version']), 'frame_size': size.astype(int).tolist(),
            'pixel_to_world_xy': matrix.tolist(), 'fit_max_error': float(train_error.max()),
            'heldout_max_error': float(check_error.max()), 'error_unit': 'game_world_unit',
            'calibration_status': 'landmark_checks_passed_not_live_validated',
            'world_z': None, 'eligible_for_inference': False,
            'landmark_spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}


def project_pixels(calibration, points, frame_size, map_name, client_version):
    if (list(frame_size) != calibration['frame_size'] or map_name != calibration['map_name']
            or str(client_version) != calibration['client_version']):
        raise ValueError('calibration_context_mismatch')
    points = np.asarray(points, float)
    if (points < 0).any() or (points >= np.asarray(frame_size)).any():
        raise ValueError('pixel_outside_frame')
    return apply_affine(points, calibration['pixel_to_world_xy'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('catalog')
    fetch.add_argument('--version', required=True)
    fetch.add_argument('--output', type=Path, required=True)
    fit = sub.add_parser('calibrate')
    fit.add_argument('--spec', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'catalog':
        result = fetch_catalog(args.version, args.output)
        print(f"Verified overview metadata for {len(result['maps'])} maps; geometry not yet validated.")
    else:
        result = fit_calibration(json.loads(args.spec.read_text(encoding='utf-8')))
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
