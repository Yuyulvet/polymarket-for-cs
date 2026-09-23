"""Record 5EPlay CS2 live state from the page's MQTT push stream.

The HTTP endpoint is used only to seed an explicitly ineligible initial
snapshot.  Decision-eligible updates come from the exact per-match MQTT topic
and are timestamped immediately on local receipt.  Raw match payloads and
source version markers are retained for later timing audits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import ssl
import threading
import time
from typing import Callable

from curl_cffi import requests
import paho.mqtt.client as mqtt

from .fivee_state import BASE, DETAIL_PATH, state_hash, summarize_payload, utc_now


BROKER = "post-cn-7mz2e5hc90i.mqtt.aliyuncs.com"
BROKER_PORT = 443
CREDENTIAL_URL = "https://www.5eplay.com/api/restrict/matchscore"
# The broker hard-caps connections at 60 minutes (observed 2026-09-19: two
# connections dropped at exactly ~60.2 min). Credentials from the token
# endpoint are single-use/short-lived: reconnecting with stale credentials is
# refused ("Not authorized") forever. So every connection uses freshly fetched
# credentials and is proactively rotated before the broker cap.
ROTATE_SECONDS = 3000.0


def topic_for(match_id: str) -> str:
    return f"csgo/product/detail/{match_id}"


def source_version_timestamp(value) -> float | None:
    """Decode the observed 5E version clock (Unix deciseconds, zero padded)."""
    try:
        timestamp = int(str(value)) / 10.0
    except (TypeError, ValueError):
        return None
    return timestamp if 1_000_000_000 <= timestamp <= 4_000_000_000 else None


def summarize_mqtt_message(payload: dict, expected_match_id: str) -> tuple[dict, dict, dict]:
    if not isinstance(payload, dict) or payload.get("event_name") != "csgo-detail":
        raise ValueError("mqtt_event_not_csgo_detail")
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("match"), dict):
        raise ValueError("mqtt_match_not_object")
    wrapped = {"success": True, "data": {"match": data["match"]}}
    summary, raw_match = summarize_payload(wrapped, expected_match_id)
    versions = {"from_ver": data.get("from_ver"), "this_ver": data.get("this_ver")}
    return summary, raw_match, versions


class FiveEMqttRecorder:
    def __init__(
        self, match_id: str, out_path: Path, *, duration_seconds: float,
        request_timeout: float = 20.0,
        request_get: Callable | None = None, request_post: Callable | None = None,
        wall_clock: Callable[[], str] = utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ):
        if not str(match_id).strip():
            raise ValueError("match_id_required")
        if duration_seconds <= 0 or request_timeout <= 0:
            raise ValueError("duration_and_timeout_must_be_positive")
        self.match_id = str(match_id)
        self.out_path = Path(out_path)
        self.duration_seconds = float(duration_seconds)
        self.request_timeout = float(request_timeout)
        self.request_get = request_get or requests.get
        self.request_post = request_post or requests.post
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self.topic = topic_for(self.match_id)
        self.detail_url = BASE + DETAIL_PATH.format(match_id=self.match_id)
        self._handle = None
        self._lock = threading.Lock()
        self._previous_hash = None
        self._connection_count = 0
        self._awaiting_recovery_snapshot = False
        self._message_count = 0
        self._snapshot_count = 0
        self._records_written = 0

    def _write(self, record_type: str, *, recv_utc: str | None = None,
               recv_mono: float | None = None, **fields) -> None:
        row = {
            "schema_version": 1, "record_type": record_type,
            "recv_utc": recv_utc or self.wall_clock(),
            "recv_mono": self.monotonic_clock() if recv_mono is None else recv_mono,
            **fields,
        }
        with self._lock:
            self._handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self._handle.flush()
            self._records_written += 1

    def _seed_http_snapshot(self) -> None:
        try:
            response = self.request_get(
                self.detail_url, impersonate="chrome", timeout=self.request_timeout)
            response.raise_for_status()
            summary, raw_match = summarize_payload(response.json(), self.match_id)
            digest = state_hash(summary)
            self._previous_hash = digest
            self._write(
                "state_snapshot", state_hash=digest, initial_snapshot=True,
                recovery_snapshot=False, decision_eligible=False,
                timing_quality="initial_http_snapshot", summary=summary,
                raw_match=raw_match, source_transport="http",
            )
            self._snapshot_count += 1
        except Exception as exc:
            self._write(
                "source_error", source_transport="http", phase="initial_snapshot",
                error_type=type(exc).__name__, error=str(exc)[:500])

    def _credentials(self) -> dict:
        response = self.request_post(
            CREDENTIAL_URL, json={"topic": self.topic}, impersonate="chrome",
            timeout=self.request_timeout)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not payload.get("success") or not isinstance(data, dict):
            raise ValueError(f"mqtt_credentials_failed:{payload.get('message')!r}")
        required = ("client_id", "username", "password")
        if any(not data.get(key) for key in required):
            raise ValueError("mqtt_credentials_incomplete")
        return data

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        success = not bool(getattr(reason_code, "is_failure", reason_code != 0))
        self._connack_ok = success
        self._connack.set()
        if success:
            self._connection_count += 1
            self._awaiting_recovery_snapshot = self._connection_count > 1
        self._write(
            "mqtt_connected", source_transport="mqtt", success=success,
            reason_code=str(reason_code), connection_count=self._connection_count,
            recovery_connection=self._awaiting_recovery_snapshot,
        )
        if success:
            result, message_id = client.subscribe(self.topic, qos=0)
            self._write(
                "mqtt_subscribe_requested", source_transport="mqtt",
                topic=self.topic, result_code=int(result), message_id=message_id)

    def _on_subscribe(self, client, userdata, message_id, reason_codes, properties) -> None:
        self._write(
            "mqtt_subscribed", source_transport="mqtt", topic=self.topic,
            message_id=message_id,
            reason_codes=[str(reason) for reason in reason_codes],
        )

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties) -> None:
        self._dropped = True
        self._write(
            "mqtt_disconnected", source_transport="mqtt",
            reason_code=str(reason_code),
            disconnect_flags=str(disconnect_flags),
        )

    def _on_message(self, client, userdata, message) -> None:
        recv_utc, recv_mono = self.wall_clock(), self.monotonic_clock()
        self._message_count += 1
        try:
            if message.topic != self.topic:
                raise ValueError(f"mqtt_topic_mismatch:{message.topic!r}")
            payload = json.loads(message.payload.decode("utf-8"))
            summary, raw_match, versions = summarize_mqtt_message(payload, self.match_id)
            digest = state_hash(summary)
            changed = digest != self._previous_hash
            recovering = self._awaiting_recovery_snapshot
            source_this_ts = source_version_timestamp(versions["this_ver"])
            recv_ts = datetime.fromisoformat(recv_utc.replace("Z", "+00:00")).timestamp()
            source_lag = None if source_this_ts is None else recv_ts - source_this_ts
            if changed:
                self._write(
                    "state_snapshot", recv_utc=recv_utc, recv_mono=recv_mono,
                    state_hash=digest, initial_snapshot=False,
                    recovery_snapshot=recovering,
                    decision_eligible=not recovering,
                    timing_quality=("mqtt_recovery_receive_time" if recovering
                                    else "mqtt_push_receive_time"),
                    summary=summary, raw_match=raw_match,
                    source_transport="mqtt", source_topic=message.topic,
                    source_from_ver=versions["from_ver"],
                    source_this_ver=versions["this_ver"],
                    source_this_ts=source_this_ts,
                    source_lag_seconds=source_lag,
                    mqtt_qos=int(message.qos), mqtt_retain=bool(message.retain),
                    mqtt_duplicate=bool(message.dup),
                )
                self._snapshot_count += 1
                self._previous_hash = digest
            self._write(
                "mqtt_message", recv_utc=recv_utc, recv_mono=recv_mono,
                source_transport="mqtt", source_topic=message.topic,
                state_hash=digest, state_changed=changed,
                source_from_ver=versions["from_ver"],
                source_this_ver=versions["this_ver"],
                source_this_ts=source_this_ts,
                source_lag_seconds=source_lag,
                payload_bytes=len(message.payload), recovery_message=recovering,
            )
            if recovering:
                self._awaiting_recovery_snapshot = False
        except Exception as exc:
            self._write(
                "mqtt_message_error", recv_utc=recv_utc, recv_mono=recv_mono,
                source_transport="mqtt", source_topic=getattr(message, "topic", None),
                error_type=type(exc).__name__, error=str(exc)[:500],
                payload_bytes=len(getattr(message, "payload", b"")),
            )

    def _build_client(self, credentials: dict):
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=credentials["client_id"], protocol=mqtt.MQTTv311,
            transport="websockets", reconnect_on_failure=False)
        client.username_pw_set(credentials["username"], credentials["password"])
        client.ws_set_options(path="/mqtt")
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        client.on_connect = self._on_connect
        client.on_subscribe = self._on_subscribe
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        return client

    def _wait(self, seconds: float, deadline: float) -> bool:
        """Sleep up to `seconds`, capped by `deadline`; True if budget remains."""
        remaining = deadline - self.monotonic_clock()
        if remaining <= 0:
            return False
        time.sleep(min(seconds, remaining))
        return self.monotonic_clock() < deadline

    def _connection_loop(self, deadline: float) -> tuple[str, int]:
        backoff = 1.0
        while self.monotonic_clock() < deadline:
            # Fresh credentials per connection: the broker refuses reused ones.
            try:
                credentials = self._credentials()
            except Exception as exc:
                self._write("source_error", source_transport="mqtt",
                            phase="credentials", error_type=type(exc).__name__,
                            error=str(exc)[:500])
                if not self._wait(backoff, deadline):
                    break
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            self._connack = threading.Event()
            self._connack_ok = False
            self._dropped = False
            client = self._build_client(credentials)
            try:
                client.connect(BROKER, BROKER_PORT, keepalive=30)
                client.loop_start()
                if not self._connack.wait(timeout=15.0):
                    raise TimeoutError("mqtt_connack_timeout")
                if not self._connack_ok:
                    raise RuntimeError("mqtt_connect_refused")
            except Exception as exc:
                client.loop_stop()
                try:
                    client.disconnect()
                except Exception:
                    pass
                self._write("source_error", source_transport="mqtt", phase="connect",
                            error_type=type(exc).__name__, error=str(exc)[:500])
                if not self._wait(backoff, deadline):
                    break
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            rotate_deadline = min(deadline,
                                  self.monotonic_clock() + ROTATE_SECONDS)
            while (self.monotonic_clock() < rotate_deadline
                   and not self._dropped):
                time.sleep(min(0.25, max(0.0, rotate_deadline - self.monotonic_clock())))
            reason = "rotated" if not self._dropped else "dropped"
            client.disconnect()
            client.loop_stop()
            self._write("mqtt_connection_closed", source_transport="mqtt",
                        reason=reason, messages_received=self._message_count,
                        snapshots_written=self._snapshot_count)
        return "deadline", 0

    def run(self) -> int:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as handle:
            self._handle = handle
            self._write(
                "session_start", match_id=self.match_id, source_transport="mqtt",
                topic=self.topic, broker=BROKER, broker_port=BROKER_PORT,
                detail_url=self.detail_url,
                connection_rotate_seconds=ROTATE_SECONDS)
            self._seed_http_snapshot()
            try:
                exit_reason, code = self._connection_loop(
                    self.monotonic_clock() + self.duration_seconds)
            except Exception as exc:
                self._write(
                    "source_error", source_transport="mqtt", phase="session",
                    error_type=type(exc).__name__, error=str(exc)[:500])
                exit_reason, code = "mqtt_error", 2
            self._write(
                "session_end", exit_reason=exit_reason,
                messages_received=self._message_count,
                snapshots_written=self._snapshot_count,
                connections=self._connection_count,
            )
            self._handle = None
        return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--out-dir", default="data/fivee")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_path = out_dir / f"mqtt_{args.match_id}_{stamp}.jsonl"
    meta_path = out_dir / f"mqtt_{args.match_id}_{stamp}.meta.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 1, "match_id": args.match_id,
        "topic": topic_for(args.match_id), "broker": BROKER,
        "initial_source": BASE + DETAIL_PATH.format(match_id=args.match_id),
        "timing_basis": "local_receive_time",
        "source_version_clock": (
            "this_ver observed as zero-padded Unix deciseconds; used for staleness audit"),
        "strict_backtest_policy": (
            "exclude initial HTTP and first post-reconnect MQTT snapshots"),
        "inventory_semantics": (
            "cash, HP, one displayed weapon, armour, kit and C4; not total equipment value"),
        "started_utc": utc_now(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder = FiveEMqttRecorder(
        args.match_id, out_path, duration_seconds=args.hours * 3600,
        request_timeout=args.request_timeout)
    code = recorder.run()
    print(f"{recorder._snapshot_count} snapshots -> {out_path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
