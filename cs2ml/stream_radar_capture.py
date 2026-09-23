"""Bounded, read-only radar evidence capture. Never produces player or trade signals."""
from __future__ import annotations
import argparse
import datetime as dt
import io
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

from PIL import Image

from .map_space import MAPS, validate_map
from .stream_score_watch import validate_binding


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def radar_box(size, roi):
    values = [float(v) for v in roi.split(',')]
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid_roi')
    x, y, w, h = values
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > 1 or y+h > 1:
        raise ValueError('roi_outside_frame')
    iw, ih = size
    box = (int(x*iw), int(y*ih), int((x+w)*iw), int((y+h)*ih))
    if box[2]-box[0] < 8 or box[3]-box[1] < 8:
        raise ValueError('roi_too_small')
    return box


def choose_format(formats, minimum_height=720, target_height=1080):
    candidates = [f for f in formats if f.get('url') and f.get('vcodec') not in (None, 'none')
                  and (f.get('height') or 0) >= minimum_height and not f.get('has_drm')]
    if not candidates:
        raise ValueError('no_sufficient_resolution_video')
    return min(candidates, key=lambda f: (abs(f['height']-target_height), -(f.get('tbr') or 0)))


def resolve_video(page_url):
    import yt_dlp
    parsed = urlsplit(page_url)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('public_https_page_without_credentials_or_query_required')
    with yt_dlp.YoutubeDL({'quiet': True, 'noplaylist': True, 'socket_timeout': 20,
                          'extractor_retries': 1}) as ydl:
        info = ydl.extract_info(page_url, download=False)
    selected = choose_format(info.get('formats', []))
    return selected['url'], {'page_url': page_url, 'width': selected.get('width'),
                             'height': selected['height'], 'fps': selected.get('fps')}


class EvidenceWriter:
    def __init__(self, output, binding, map_name, roi, source, *, max_bytes=500_000_000, full_every=50):
        validate_map(map_name)
        self.binding = validate_binding(**{k: binding[k] for k in ('event_id', 'team_a', 'team_b', 'map_number')})
        radar_box((10000, 10000), roi)
        if max_bytes <= 0 or full_every < 1:
            raise ValueError('invalid_capture_limits')
        self.output, self.roi = Path(output), roi
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/'frames').mkdir()
        self.count, self.bytes, self.full_every = 0, 0, full_every
        self.max_bytes = max_bytes
        self.frame_size = None
        meta = {'schema_version': 1, 'binding': self.binding, 'map_name': map_name,
                'binding_status': 'pending_manual_confirmation', 'roi': roi, 'source': source,
                'timing_basis': 'decoded_at_is_local_decode_time_not_network_receipt_or_game_time',
                'eligible_for_inference': False, 'created_at': now(),
                'max_image_bytes': max_bytes, 'full_frame_every_samples': full_every}
        (self.output/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    def log(self, record):
        record.setdefault('recorded_at', now())
        record['eligible_for_inference'] = False
        with (self.output/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')

    def save(self, image, *, pts_seconds, decoded_at, decoded_monotonic_ns):
        if self.frame_size is not None and image.size != self.frame_size:
            raise ValueError('frame_size_changed_recalibration_required')
        if pts_seconds is not None and not math.isfinite(pts_seconds):
            raise ValueError('nonfinite_pts')
        if (self.output/'STOP').exists():
            raise InterruptedError('stop_file')
        self.frame_size = image.size
        box = radar_box(image.size, self.roi)
        index = self.count+1
        blobs = {}
        crop = io.BytesIO()
        image.crop(box).save(crop, format='PNG')
        blobs[f'frames/{index:07d}_radar.png'] = crop.getvalue()
        if self.count % self.full_every == 0:
            full = io.BytesIO()
            image.convert('RGB').save(full, format='JPEG', quality=95)
            blobs[f'frames/{index:07d}_full.jpg'] = full.getvalue()
        required = sum(map(len, blobs.values()))
        if self.bytes+required > self.max_bytes:
            raise InterruptedError('image_byte_budget')
        for path, content in blobs.items():
            (self.output/path).write_bytes(content)
        self.bytes += required
        self.count = index
        record = {'type': 'radar_frame', 'sample': index, 'frame_size': image.size, 'crop_box': box,
                  'video_pts_seconds': pts_seconds, 'decoded_at': decoded_at,
                  'decoded_monotonic_ns': decoded_monotonic_ns, 'source_timestamp': None,
                  'network_received_at': None, 'files': list(blobs), 'image_bytes_total': self.bytes,
                  'scene_status': 'unverified_may_be_replay', 'player_observations': None}
        self.log(record)
        return record


def capture(source, writer, *, seconds=300, sample_hz=5, max_frames=1500):
    import av
    if not math.isfinite(seconds) or not math.isfinite(sample_hz) or seconds <= 0 or not 0 < sample_hz <= 30 or max_frames < 1:
        raise ValueError('invalid_capture_limits')
    deadline = time.monotonic()+seconds
    last_pts, last_saved = None, None
    writer.log({'type': 'capture_start', 'sample_hz': sample_hz, 'max_frames': max_frames, 'seconds': seconds})
    reason = 'end_of_stream'
    try:
        with av.open(str(source), timeout=(15, 15)) as container:
            for frame in container.decode(video=0):
                decoded_mono, decoded_at = time.monotonic_ns(), now()
                if time.monotonic() >= deadline:
                    reason = 'duration_limit'
                    break
                if (writer.output/'STOP').exists():
                    reason = 'stop_file'
                    break
                pts = float(frame.time) if frame.time is not None else None
                if pts is None or not math.isfinite(pts):
                    raise ValueError('missing_or_invalid_video_pts')
                if last_pts is not None and pts < last_pts:
                    raise ValueError('video_pts_regressed_new_session_required')
                last_pts = pts
                if last_saved is not None and pts-last_saved < 1/sample_hz-1e-6:
                    continue
                writer.save(frame.to_image(), pts_seconds=pts, decoded_at=decoded_at, decoded_monotonic_ns=decoded_mono)
                last_saved = pts
                if writer.count >= max_frames:
                    reason = 'frame_limit'
                    break
    except InterruptedError as exc:
        reason = str(exc)
    except Exception as exc:
        # Exception messages from decoders can contain signed stream URLs.
        writer.log({'type': 'capture_stop', 'reason': 'capture_error', 'error_type': type(exc).__name__, 'frames': writer.count})
        raise RuntimeError('capture_failed_see_sanitized_session_log') from None
    writer.log({'type': 'capture_stop', 'reason': reason, 'frames': writer.count})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--page-url')
    group.add_argument('--video', type=Path)
    group.add_argument('--image', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--event-id', required=True)
    p.add_argument('--team-a', required=True)
    p.add_argument('--team-b', required=True)
    p.add_argument('--map-number', type=int, required=True)
    p.add_argument('--map-name', choices=MAPS, required=True)
    p.add_argument('--roi', required=True, help='normalized x,y,width,height; must be checked on this HUD')
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--sample-hz', type=float, default=5)
    p.add_argument('--max-frames', type=int, default=1500)
    args = p.parse_args()
    binding = validate_binding(args.event_id, args.team_a, args.team_b, args.map_number)
    source, info = None, {'kind': 'offline_image' if args.image else 'offline_video'}
    if args.page_url:
        try:
            source, info = resolve_video(args.page_url)
        except Exception as exc:
            raise SystemExit(f'stream_resolution_failed: {type(exc).__name__}') from None
        info['kind'] = 'public_stream'
    else:
        source = args.image or args.video
        if not source.is_file():
            raise FileNotFoundError(source)
        info['path'] = str(source.resolve())
    writer = EvidenceWriter(args.output, binding, args.map_name, args.roi, info)
    if args.image:
        with Image.open(args.image) as image:
            writer.save(image, pts_seconds=None, decoded_at=now(), decoded_monotonic_ns=time.monotonic_ns())
        writer.log({'type': 'capture_stop', 'reason': 'offline_image', 'frames': writer.count})
    else:
        capture(source, writer, seconds=args.seconds, sample_hz=args.sample_hz, max_frames=args.max_frames)
    print(f'{writer.count} evidence frames saved to {args.output}; inference disabled.')


if __name__ == '__main__':
    main()
