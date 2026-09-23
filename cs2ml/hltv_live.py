"""Bounded public Scorebot capture for source validation, never model execution.

Only the endpoint advertised by one public match page is used. No reconnect,
redirect, alternate endpoint, authentication, browser cookies or impersonation.
Engine.IO 3's WebSocket-only transport is intentional; unsupported protocols
stop the capture. This is a small recorder, not a general Socket.IO client.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from .hltv_scorebot_protocol import ProtocolError, ScorebotProtocol
from .live_feed_store import RawFeedRecorder, RecordingLimitError
from .map1_hltv import HLTVSourceError, MAX_HTML_BYTES, fetch_match_page, normalize_match_url


# Observed on the public match page on 2026-09-16, not a list to rotate through.
SCOREBOT_HOST = "scorebot-lb.hltv.org"
MAX_PACKET_BYTES = 1024 * 1024


class _ScoreboardParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes = []

    def handle_starttag(self, tag, attrs):
        if any(key == "id" and value == "scoreboardElement" for key, value in attrs):
            keys = [key for key, _ in attrs]
            if len(keys) != len(set(keys)):
                raise HLTVSourceError("Duplicate scoreboard attributes")
            self.nodes.append(dict(attrs))


def discover_scorebot(html: str, match_url: str) -> dict:
    """Extract a single visible scoreboard's metadata; fail closed on ambiguity."""
    match_url = normalize_match_url(match_url)
    if not isinstance(html, str) or len(html.encode("utf-8")) > MAX_HTML_BYTES:
        raise HLTVSourceError("Invalid or oversized match page")
    parser = _ScoreboardParser()
    parser.feed(html)
    if len(parser.nodes) != 1:
        raise HLTVSourceError("Expected exactly one public scoreboardElement")
    attrs = parser.nodes[0]
    endpoints = (attrs.get("data-scorebot-url") or "").split(",")
    endpoint = endpoints[-1].strip()
    # Exact allowlist, including authority/port/path; no arbitrary fetch target.
    if endpoint not in {f"https://{SCOREBOT_HOST}", f"https://{SCOREBOT_HOST}/"}:
        raise HLTVSourceError("Unrecognized public scorebot endpoint; review required")
    scorebot_id = attrs.get("data-scorebot-id") or ""
    if not re.fullmatch(r"[1-9][0-9]{0,11}", scorebot_id):
        raise HLTVSourceError("Invalid public scorebot ID")
    return {
        "match_url": match_url,
        "hltv_match_id": urlsplit(match_url).path.split("/")[2],
        "scorebot_id": scorebot_id,
        "endpoint": endpoint.rstrip("/"),
        "websocket_url": f"wss://{SCOREBOT_HOST}/socket.io/?EIO=3&transport=websocket",
        "page_attributes": {key: value for key, value in attrs.items()
                            if key.startswith("data-") and "logo" not in key},
    }


def _connect(url, timeout):
    import websocket

    # Validate text ourselves after preserving malformed bytes as evidence.
    ws = websocket.WebSocket(skip_utf8_validation=True)
    try:
        ws.connect(url, timeout=timeout, redirect_limit=0, http_no_proxy=["*"],
                   suppress_origin=True,
                   header={"User-Agent": "CS2Map1Research/1.0 (public feed validation)"})
        # websocket-client can return a redirect response when redirect_limit=0.
        if ws.getstatus() != 101:
            raise HLTVSourceError(f"Scorebot HTTP {ws.getstatus()}; no retry or bypass")
    except BaseException:
        try:
            ws.close(timeout=1)
        except Exception:
            pass  # Retain the actual handshake error/status.
        raise
    return ws


def _stream(ws, protocol, recorder, seconds, *, monotonic=time.monotonic, counts=None):
    """Save each complete message before decoding. No inferred game timestamps."""
    import websocket

    started = monotonic()
    deadline = started + seconds
    handshake_deadline = min(deadline, started + 8)
    next_ping = None
    pong_deadline = None
    counts = Counter() if counts is None else counts
    while True:
        now = monotonic()
        if now >= deadline:
            if not protocol.subscribed:
                raise HLTVSourceError("Capture ended before subscription")
            return dict(counts)
        if not protocol.subscribed and now >= handshake_deadline:
            raise HLTVSourceError("Scorebot handshake/subscription timeout")
        if pong_deadline is not None and now >= pong_deadline:
            raise HLTVSourceError("Scorebot heartbeat timeout")
        if next_ping is not None and now >= next_ping and pong_deadline is None:
            ws.send("2")
            recorder.append("control_sent", "2")
            pong_deadline = now + protocol.ping_timeout_seconds
            next_ping = None
        wake_at = min(value for value in (deadline, next_ping, pong_deadline,
                      None if protocol.subscribed else handshake_deadline) if value is not None)
        ws.settimeout(max(0.001, min(1.0, wake_at - monotonic())))
        try:
            opcode, data = ws.recv_data(control_frame=True)
        except websocket.WebSocketTimeoutException:
            continue
        if len(data) > MAX_PACKET_BYTES:
            recorder.append("oversized_packet", {"opcode": opcode, "size": len(data)})
            raise HLTVSourceError("Scorebot message exceeds packet budget")
        if opcode == websocket.ABNF.OPCODE_CLOSE:
            recorder.append("socket_close", {"hex": data.hex()})
            raise HLTVSourceError("Scorebot disconnected; no reconnect")
        if opcode in (websocket.ABNF.OPCODE_PING, websocket.ABNF.OPCODE_PONG):
            # WebSocket control frames are not Engine.IO's text heartbeats.
            recorder.append("websocket_control", {"opcode": opcode, "hex": data.hex()})
            continue
        if opcode != websocket.ABNF.OPCODE_TEXT:
            recorder.append("unsupported_packet", {"opcode": opcode,
                            "encoding": "base64", "data": base64.b64encode(data).decode("ascii")})
            raise ProtocolError("Unsupported binary Scorebot message")
        try:
            packet = data.decode("utf-8") if isinstance(data, bytes) else data
        except UnicodeDecodeError:
            recorder.append("unsupported_packet", {"opcode": opcode,
                            "encoding": "base64", "data": base64.b64encode(data).decode("ascii")})
            raise ProtocolError("Invalid UTF-8 Scorebot message") from None
        if len(packet.encode("utf-8")) > MAX_PACKET_BYTES:
            raise HLTVSourceError("Scorebot message exceeds packet budget")
        raw = recorder.append("socket_packet", packet)
        result = protocol.feed(packet)
        now = monotonic()
        if result["kind"] == "open":
            if protocol.ping_interval_seconds < 1:
                raise ProtocolError("Sub-second heartbeat interval is not supported by this probe")
            next_ping = now + protocol.ping_interval_seconds
        elif result["kind"] == "pong":
            if pong_deadline is None:
                raise ProtocolError("Unsolicited Engine.IO pong")
            pong_deadline = None
            next_ping = now + protocol.ping_interval_seconds
        elif result["kind"] == "event":
            name = result["event_name"]
            counts[name] += 1
            recorder.append("scorebot_event", {
                "name": name, "data": result["event_payload"],
                "raw_sequence": raw["sequence"], "packet_received_at": raw["received_at"],
                "packet_monotonic_ns": raw["monotonic_ns"],
                "possible_backfill": name == "fullLog",
                "eligible_for_inference": False,
            })
        for outgoing in result.get("outgoing", []):
            ws.send(outgoing)
            recorder.append("control_sent", outgoing)


def capture(match_url: str, directory: Path, seconds=30, *, max_records=10000,
            max_bytes=16 * 1024 * 1024, page_reader=fetch_match_page, connector=_connect):
    """One page read and one short connection, with a new evidence directory."""
    match_url = normalize_match_url(match_url)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 1 <= seconds <= 300:
        raise ValueError("seconds must be between 1 and 300")
    metadata = {"source": "hltv_scorebot_public", "match_url": match_url,
                "hltv_match_id": urlsplit(match_url).path.split("/")[2],
                "capture_seconds": seconds, "paper_only": True,
                "source_clock_verified": False, "model_input_enabled": False,
                "automatic_reconnect": False}
    summary = {"status": "stopped", "reason": None, "event_counts": {},
               "source_clock_verified": False, "model_input_enabled": False,
               "live_trading_enabled": False, "directory": str(Path(directory).resolve())}
    ws = None
    counts = Counter()
    with RawFeedRecorder(directory, metadata, max_records=max_records, max_bytes=max_bytes) as recorder:
        try:
            page = page_reader(match_url)
            recorder.append("match_page", page)
            source = discover_scorebot(page["html"], match_url)
            recorder.append("source_discovered", source)
            ws = connector(source["websocket_url"], min(8, seconds))
            recorder.append("connected", {"endpoint": source["endpoint"]})
            _stream(ws, ScorebotProtocol(source["scorebot_id"]), recorder, seconds, counts=counts)
            summary.update(status="captured" if counts else "no_events", event_counts=counts,
                           reason="duration_limit")
        except KeyboardInterrupt:
            summary["reason"] = "user_interrupted"
        except Exception as exc:
            # Third-party exception strings may contain response bodies; do not copy.
            safe = isinstance(exc, (HLTVSourceError, ProtocolError, RecordingLimitError))
            summary["reason"] = str(exc) if safe else type(exc).__name__
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int):
                summary["http_status"] = status_code
        finally:
            summary["event_counts"] = dict(counts)
            if ws is not None:
                try:
                    ws.close(timeout=1)
                except Exception as exc:
                    summary["close_error"] = type(exc).__name__
            if not recorder.closed:
                try:
                    recorder.append("capture_stopped", summary)
                except (RecordingLimitError, OSError):
                    pass
            summary["records_written"] = recorder.records_written
            summary["bytes_written"] = recorder.bytes_written
        # Separate tiny manifest remains available if the journal hit its cap.
        with (Path(directory) / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-url", required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args(argv)
    directory = args.output_dir or Path("data/hltv_live") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        result = capture(args.match_url, directory, args.seconds,
                         max_records=args.max_records, max_bytes=args.max_bytes)
    except (ValueError, OSError, RecordingLimitError) as exc:
        parser.exit(2, f"Capture could not start: {type(exc).__name__}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "captured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
