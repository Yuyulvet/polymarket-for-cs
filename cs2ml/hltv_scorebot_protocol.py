"""Pure Engine.IO 3 / Socket.IO wire-protocol revision 4 packet state machine.

This module performs no network requests, clock reads, gameplay interpretation,
or nested log decoding. It accepts individual text WebSocket packets, not HTTP
polling payloads. The transport owns EIO3 client ping scheduling and deadlines.

Protocol references:
https://github.com/socketio/engine.io-protocol/tree/v3
https://github.com/gigobyte/HLTV/blob/master/src/endpoints/connectToScorebot.ts
"""
from __future__ import annotations

import json
import math


MAX_PACKET_BYTES = 1024 * 1024
MAX_HEARTBEAT_MS = 300_000
MAX_IDENTIFIER_BYTES = 1024
MAX_JSON_DEPTH = 128


class ProtocolError(ValueError):
    """Malformed, unsupported or out-of-order scorebot packet; close transport."""


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ProtocolError(f"{name} must be a nonempty string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ProtocolError(f"{name} must be valid UTF-8") from exc
    if size > MAX_IDENTIFIER_BYTES:
        raise ProtocolError(f"{name} exceeds the identifier size limit")
    return value


def _reject_constant(value: str):
    raise ProtocolError("Non-finite JSON numeric constants are unsupported")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate JSON object keys are ambiguous")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError("JSON numeric value exceeds finite float range")
    return result


def _json(value: str):
    # Guard parser stack depth before decoding; brackets inside strings do not count.
    depth = 0
    quoted = escaped = False
    for char in value:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ProtocolError("JSON nesting exceeds the supported depth")
        elif char in "]}":
            depth -= 1
    try:
        return json.loads(value, parse_constant=_reject_constant, parse_float=_finite_float,
                          object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("Invalid or unsupported JSON packet payload") from exc


class ScorebotProtocol:
    """One connection only. Any protocol error poisons this instance.

    ``subscribed`` means the subscription frame was produced for the transport;
    it does not assert server acceptance or availability of a scoreboard feed.
    ``feed`` returns ``kind`` open/connect/pong/event; only connect has outgoing
    frames. Event payloads retain their delivered JSON type, including log strings.
    """

    def __init__(self, scorebot_id: str):
        self.scorebot_id = _identifier(scorebot_id, "scorebot_id")
        self.sid: str | None = None
        self.ping_interval_seconds: float | None = None
        self.ping_timeout_seconds: float | None = None
        self.opened = False
        self.subscribed = False
        self.failed = False

    def feed(self, packet: str) -> dict:
        """Consume one text WebSocket frame; no implicit retries or reconnection."""
        if self.failed:
            raise ProtocolError("Session already failed; create a new protocol instance")
        try:
            return self._feed(packet)
        except ProtocolError:
            self.failed = True
            self.opened = False
            self.subscribed = False
            raise

    def _feed(self, packet: str) -> dict:
        if not isinstance(packet, str):
            raise ProtocolError("Only text packets are supported; binary frames are rejected")
        if not packet:
            raise ProtocolError("Empty packet")
        if len(packet) > MAX_PACKET_BYTES:
            raise ProtocolError("Packet exceeds the UTF-8 byte size limit")
        try:
            byte_count = len(packet.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ProtocolError("Packet is not valid UTF-8") from exc
        if byte_count > MAX_PACKET_BYTES:
            raise ProtocolError("Packet exceeds the UTF-8 byte size limit")

        if packet[0] == "0":
            if self.opened:
                raise ProtocolError("Duplicate Engine.IO open packet")
            handshake = _json(packet[1:])
            if not isinstance(handshake, dict):
                raise ProtocolError("Engine.IO handshake must be a JSON object")
            sid = _identifier(handshake.get("sid"), "sid")
            values = []
            for name in ("pingInterval", "pingTimeout"):
                value = handshake.get(name)
                if type(value) is not int or not 1 <= value <= MAX_HEARTBEAT_MS:
                    raise ProtocolError(f"{name} must be an integer in 1..{MAX_HEARTBEAT_MS} milliseconds")
                values.append(value / 1000.)
            self.sid = sid
            self.ping_interval_seconds, self.ping_timeout_seconds = values
            self.opened = True
            return {"kind": "open"}

        if not self.opened:
            raise ProtocolError("Engine.IO open packet required before other packets")
        if packet[0] == "1":
            raise ProtocolError("Engine.IO connection closed")
        if packet[0] == "2":
            raise ProtocolError("Unexpected server ping: EIO3 requires client-originated pings")
        if packet == "3":
            return {"kind": "pong"}
        if packet.startswith("3"):
            raise ProtocolError("Pong payloads and transport probes are unsupported")
        if packet == "40":
            if self.subscribed:
                raise ProtocolError("Duplicate Socket.IO namespace connect packet")
            payload = json.dumps({"token": "", "listId": self.scorebot_id}, separators=(",", ":"), ensure_ascii=False)
            frame = "42" + json.dumps(["readyForMatch", payload], separators=(",", ":"), ensure_ascii=False)
            self.subscribed = True
            return {"kind": "connect", "outgoing": [frame]}
        if packet.startswith("41"):
            raise ProtocolError("Socket.IO namespace disconnected")
        if packet.startswith("44"):
            raise ProtocolError("Socket.IO error packet")
        if packet.startswith("42"):
            if not self.subscribed:
                raise ProtocolError("Socket.IO namespace connect required before events")
            if not packet[2:].startswith("["):
                raise ProtocolError("Socket.IO namespaces, acknowledgement IDs or non-array events are unsupported")
            event = _json(packet[2:])
            if not isinstance(event, list) or len(event) != 2 or not isinstance(event[0], str):
                raise ProtocolError("Socket.IO event must be exactly [event_name_string, payload]")
            return {"kind": "event", "event_name": event[0], "event_payload": event[1]}
        if packet.startswith(("45", "46", "b")):
            raise ProtocolError("Socket.IO/Engine.IO binary packets are unsupported")
        if packet.startswith("43"):
            raise ProtocolError("Socket.IO acknowledgement packets are unsupported")
        if packet.startswith("40"):
            raise ProtocolError("Only the exact default-namespace Socket.IO v4 connect packet is supported")
        raise ProtocolError("Unsupported Engine.IO/Socket.IO packet")
