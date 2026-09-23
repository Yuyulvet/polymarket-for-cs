"""stream_score_watch.py —— 已绑定比赛的直播画面候选变化记录器。

实验目的：测 Polymarket CS2 盘口改价相对「直播画面可见比分变化」的到达延迟。
与 realtime_record.py 的盘口记录共用本机时钟，对齐后得到 lag 分布：
  - 价格稳定跟随画面变化（滞后几秒）→ 改单者与我们看同一直播（人或慢 bot）
  - 价格稳定领先画面变化（超前十秒级以上）→ 场上有官方 feed 快 bot

日志语义（与 live_feed_store 一致）：
  - received_at / monotonic_ns 为本地接收/记录时间，不是游戏发生时间；
  - source_timestamp 恒为 null；eligible_for_inference 恒 false；
  - 保存的变化截图只作人工核验证据，OCR 不自动回填比分语义；
  - 原始像素变化叫 candidate_visual_change，绝不冒充比分事件；
  - event_id、两队和地图序号是每个 session 的强制绑定字段。

用法：
  python -m cs2ml.stream_score_watch --url https://kick.com/cct_cs --event-id 123 \
    --team-a Lavked --team-b Honved --map-number 2 --calibrate 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

OUT_ROOT = config.DATA_DIR / "stream_watch"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_stream(url: str) -> tuple[str, str]:
    """返回 (直接视频流 m3u8, 展示标题)。选 ~480p 以省解码资源。"""
    import yt_dlp
    opts = {"quiet": True, "noplaylist": True, "socket_timeout": 30,
            "extractor_retries": 1}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    fmts = [f for f in info.get("formats", []) if f.get("vcodec") not in (None, "none")]
    if not fmts:
        raise ValueError("no_video_format")
    fmts.sort(key=lambda f: abs((f.get("height") or 480) - 480))
    return fmts[0]["url"], str(info.get("title") or "")


def crop_box(size: tuple[int, int], spec: str) -> tuple[int, int, int, int]:
    """spec = fx,fy,fw,fh（0-1 比例）→ (x0,y0,x1,y1) 像素。"""
    w, h = size
    fx, fy, fw, fh = (float(v) for v in spec.split(","))
    return int(w * fx), int(h * fy), int(w * (fx + fw)), int(h * (fy + fh))


def validate_binding(event_id: str, team_a: str, team_b: str, map_number: int) -> dict:
    """Return a canonical session binding or reject an ambiguous identity."""
    event_id = str(event_id).strip()
    team_a, team_b = str(team_a).strip(), str(team_b).strip()
    if not event_id.isdigit():
        raise ValueError("event_id_must_be_numeric")
    if not team_a or not team_b:
        raise ValueError("both_team_names_are_required")
    if team_a.casefold() == team_b.casefold():
        raise ValueError("teams_must_be_distinct")
    if int(map_number) < 1:
        raise ValueError("map_number_must_be_positive")
    return {"event_id": event_id, "team_a": team_a, "team_b": team_b,
            "map_number": int(map_number),
            "market": f"Map {int(map_number)} Winner"}


class Watcher:
    def __init__(self, out_dir: Path, stream_url: str, page_url: str, title: str,
                 binding: dict, crop: str, cooldown: float, threshold: float, mask: str):
        self.out_dir = out_dir
        self.stream_url = stream_url
        self.page_url = page_url
        self.title = title
        self.binding = binding
        self.crop = crop
        self.cooldown = cooldown
        self.threshold = threshold
        # mask: 分号分隔的多个 "a,b" 横向比例区间，全部置零（保留区间外参与 diff）
        self.mask = [tuple(float(v) for v in part.split(","))
                     for part in mask.split(";")] if mask else []
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = out_dir / "events.jsonl"
        # Do not persist a transient signed m3u8 URL. The public page URL is sufficient
        # provenance and can be resolved again for a later session.
        self.meta = {"schema_version": 2, "page_url": page_url, "title": title,
                     "binding": binding, "binding_status": "pending_manual_confirmation",
                     "crop": crop, "mask": mask, "threshold": threshold,
                     "cooldown_seconds": cooldown,
                     "timing_basis": "local_receive_time_not_game_time",
                     "source_timestamp": None, "eligible_for_inference": False}
        (out_dir / "metadata.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        self.prev = None
        self.prev_img = None
        self.prev_received_at = None
        self.prev_frame = None
        self.last_event = 0.0
        self.n_frames = 0
        self.last_frame_mono = time.monotonic()

    def _log(self, record: dict):
        record.setdefault("received_at", _utc_now())
        record.setdefault("monotonic_ns", time.monotonic_ns())
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save(self, img: Image.Image, name: str) -> str:
        path = self.frames_dir / name
        img.save(path, quality=85)
        return str(path)

    def run(self, seconds: float, calibrate: int = 0, sample_sec: float = 1.0):
        import av
        deadline = time.monotonic() + seconds
        container = av.open(self.stream_url, timeout=30)
        stream = container.streams.video[0]
        self._log({"type": "watch_start", "title": self.title, **self.binding})
        print(f"[{_utc_now()}] watching: {self.title}", flush=True)
        last_sample = -1e18
        try:
            for frame in container.decode(stream):
                self.last_frame_mono = time.monotonic()
                ft = float(frame.time) if frame.time is not None else self.n_frames
                if ft - last_sample < sample_sec and not calibrate:
                    continue
                last_sample = ft
                img = frame.to_image()
                self.n_frames += 1
                sampled_at = _utc_now()
                if (self.out_dir / "quarantine.json").exists():
                    self._log({"type": "watch_stop", "reason": "identity_quarantined",
                               "frames": self.n_frames, "received_at": sampled_at})
                    return
                if calibrate:
                    full_path = self._save(img, f"calib_{self.n_frames:05d}_full.jpg")
                    crop_path = self._save(img.crop(crop_box(img.size, self.crop)),
                                           f"calib_{self.n_frames:05d}_crop.jpg")
                    self._log({"type": "calibration_frame", "frame": self.n_frames,
                               "received_at": sampled_at, "full_path": full_path,
                               "crop_path": crop_path})
                    print(f"[CALIB] {full_path}", flush=True)
                    if self.n_frames >= calibrate:
                        return
                    continue
                box = crop_box(img.size, self.crop)
                crop_img = img.crop(box)
                if self.n_frames == 1:
                    full_path = self._save(img, "identity_000001_full.jpg")
                    crop_path = self._save(crop_img, "identity_000001_crop.jpg")
                    self._log({"type": "identity_frame", "frame": self.n_frames,
                               "received_at": sampled_at, "full_path": full_path,
                               "crop_path": crop_path})
                cur = np.array(crop_img.convert("L"), dtype=np.int16, copy=True)
                if self.mask:  # 遮掉计时器/队名动画等高频变化区
                    w = cur.shape[1]
                    for a, b in self.mask:
                        cur[:, int(w * a):int(w * b)] = 0
                if self.n_frames % 60 == 1:  # 每分钟留一张参考帧
                    self._save(img.crop(box), f"ref_{self.n_frames:06d}.jpg")
                if self.prev is not None and cur.shape == self.prev.shape:
                    delta = np.abs(cur - self.prev)
                    frac = float((delta > 30).mean())
                    if frac > self.threshold and time.monotonic() - self.last_event > self.cooldown:
                        self.last_event = time.monotonic()
                        stem = f"candidate_{self.n_frames:06d}_f{frac:.3f}"
                        before_path = self._save(self.prev_img, f"{stem}_before.jpg")
                        after_path = self._save(crop_img, f"{stem}_after.jpg")
                        self._log({"type": "candidate_visual_change",
                                   "before_frame": self.prev_frame,
                                   "after_frame": self.n_frames,
                                   "before_received_at": self.prev_received_at,
                                   "received_at": sampled_at,
                                   "changed_fraction": round(frac, 4),
                                   "before_path": before_path, "after_path": after_path,
                                   **self.binding})
                        print(f"[{sampled_at}] candidate frac={frac:.3f} {after_path}", flush=True)
                self.prev = cur
                self.prev_img = crop_img.copy()
                self.prev_received_at = sampled_at
                self.prev_frame = self.n_frames
                if time.monotonic() > deadline:
                    self._log({"type": "watch_stop", "reason": "time_budget",
                               "frames": self.n_frames})
                    return
        finally:
            container.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--event-id", required=True, help="Polymarket event id")
    ap.add_argument("--team-a", required=True, help="与 Polymarket outcome 一致的 A 队名")
    ap.add_argument("--team-b", required=True, help="与 Polymarket outcome 一致的 B 队名")
    ap.add_argument("--map-number", required=True, type=int)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--calibrate", type=int, default=0, help="存 N 帧后退出（定标用）")
    ap.add_argument("--crop", default="0.30,0.0,0.40,0.09",
                    help="比分板区域 fx,fy,fw,fh（比例，默认顶部居中 40% 宽 9% 高）")
    ap.add_argument("--mask", default="0.40,0.60",
                    help="crop 内需遮罩的横向比例区间，分号分隔多段，如 \"0,0.22;0.36,0.76;0.94,1\""
                         "（默认遮中央 40%-60%，挡回合计时器）；空串=不遮")
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--threshold", type=float, default=0.10)
    args = ap.parse_args(argv)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    if out_dir.exists():
        sys.exit(f"output exists: {out_dir}")
    out_dir.mkdir(parents=True)

    binding = validate_binding(args.event_id, args.team_a, args.team_b, args.map_number)
    url, title = resolve_stream(args.url)
    watcher = Watcher(out_dir, url, args.url, title, binding, args.crop,
                      args.cooldown, args.threshold, args.mask)
    try:
        watcher.run(seconds=args.hours * 3600, calibrate=args.calibrate)
    except Exception as exc:  # 连接断开等：留证据退出，不自动重连不静默恢复
        watcher._log({"type": "watch_stop", "reason": f"{type(exc).__name__}: {exc}",
                      "frames": watcher.n_frames})
        print(f"[{_utc_now()}] stopped: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
    print(f"[{_utc_now()}] done, {watcher.n_frames} frames -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
