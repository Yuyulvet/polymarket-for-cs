from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import websocket

from cs2ml import hltv_live
from cs2ml.map1_hltv import HLTVSourceError, MAX_HTML_BYTES


MATCH_URL = "https://www.hltv.org/matches/2390000/alpha-vs-beta"
ENDPOINT = "https://scorebot-lb.hltv.org"
OPEN = '0{"sid":"fixture","upgrades":[],"pingInterval":25000,"pingTimeout":5000}'
EVENT = '42["scoreboard",{"round":3,"ct":2,"t":1}]'


def page_html(endpoint=ENDPOINT, scorebot_id="456", extra=""):
    return (f'<div id="scoreboardElement" data-scorebot-url="{endpoint}" '
            f'data-scorebot-id="{scorebot_id}" {extra}></div>')


def text_packet(value):
    return websocket.ABNF.OPCODE_TEXT, value.encode("utf-8")


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FakeSocket:
    def __init__(self, packets, clock, close_error=None):
        self.packets = list(packets)
        self.clock = clock
        self.close_error = close_error
        self.sent = []
        self.close_calls = []
        self.timeouts = []
        self.timeout = 1.

    def send(self, value):
        self.sent.append(value)

    def settimeout(self, value):
        self.timeout = value
        self.timeouts.append(value)

    def recv_data(self, control_frame=False):
        if not control_frame:
            raise AssertionError("Expected explicit control-frame observation")
        if not self.packets:
            self.clock.now += self.timeout
            raise websocket.WebSocketTimeoutException("fixture timeout")
        self.clock.now += .001
        packet = self.packets.pop(0)
        if isinstance(packet, BaseException):
            raise packet
        return packet

    def close(self, timeout):
        self.close_calls.append(timeout)
        if self.close_error is not None:
            raise self.close_error


class ScorebotDiscoveryTests(unittest.TestCase):
    def test_public_metadata_ids_remain_distinct_and_endpoint_is_fixed(self):
        source = hltv_live.discover_scorebot(page_html(ENDPOINT + "/", extra='data-team-logo="ignored"'), MATCH_URL)
        self.assertEqual(source["hltv_match_id"], "2390000")
        self.assertEqual(source["scorebot_id"], "456")
        self.assertEqual(source["endpoint"], ENDPOINT)
        self.assertEqual(source["websocket_url"], "wss://scorebot-lb.hltv.org/socket.io/?EIO=3&transport=websocket")
        self.assertNotIn("data-team-logo", source["page_attributes"])

    def test_missing_or_multiple_scoreboards_are_rejected(self):
        for html in ("<html></html>", page_html() + page_html()):
            with self.subTest(html=html), self.assertRaisesRegex(HLTVSourceError, "exactly one"):
                hltv_live.discover_scorebot(html, MATCH_URL)

    def test_duplicate_attributes_are_rejected_even_when_values_agree(self):
        for extra in ('id="scoreboardElement"', 'data-scorebot-id="456"', 'class="a" class="a"'):
            with self.subTest(extra=extra), self.assertRaisesRegex(HLTVSourceError, "Duplicate"):
                hltv_live.discover_scorebot(page_html(extra=extra), MATCH_URL)

    def test_endpoint_cannot_redirect_to_other_host_port_path_or_credentials(self):
        invalid = ["http://scorebot-lb.hltv.org", "https://evil.example",
                   ENDPOINT + ":443", ENDPOINT + "/extra", ENDPOINT + "?x=1",
                   "https://user:pass@scorebot-lb.hltv.org", ENDPOINT + ".evil.example",
                   ENDPOINT + ",https://evil.example"]
        for endpoint in invalid:
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(HLTVSourceError, "endpoint"):
                hltv_live.discover_scorebot(page_html(endpoint), MATCH_URL)

    def test_scorebot_id_must_be_bounded_positive_decimal(self):
        for value in ("", "0", "-1", "1.5", " 123", "123 ", "1e3", "1234567890123"):
            with self.subTest(value=value), self.assertRaisesRegex(HLTVSourceError, "ID"):
                hltv_live.discover_scorebot(page_html(scorebot_id=value), MATCH_URL)

    def test_match_url_cannot_be_arbitrary_fetch_target(self):
        for url in ("http://www.hltv.org/matches/1/x", "https://evil.example/matches/1/x",
                    MATCH_URL + "?token=secret", MATCH_URL + "#fragment"):
            with self.subTest(url=url), self.assertRaises(HLTVSourceError):
                hltv_live.discover_scorebot(page_html(), url)

    def test_oversized_and_nontext_html_are_rejected(self):
        for html in (None, b"raw", "x" * (MAX_HTML_BYTES + 1)):
            with self.subTest(kind=type(html).__name__), self.assertRaises(HLTVSourceError):
                hltv_live.discover_scorebot(html, MATCH_URL)


class ScorebotConnectTests(unittest.TestCase):
    def test_connect_disables_redirect_proxy_origin_and_auth_and_rejects_http_403(self):
        for status in (101, 302, 403):
            with self.subTest(status=status):
                ws = Mock()
                ws.getstatus.return_value = status
                with patch("websocket.WebSocket", return_value=ws) as constructor:
                    if status == 101:
                        self.assertIs(hltv_live._connect("wss://scorebot-lb.hltv.org/socket.io/", 2), ws)
                        ws.close.assert_not_called()
                    else:
                        with self.assertRaisesRegex(HLTVSourceError, f"HTTP {status}"):
                            hltv_live._connect("wss://scorebot-lb.hltv.org/socket.io/", 2)
                        ws.close.assert_called_once_with(timeout=1)
                constructor.assert_called_once()
                ws.connect.assert_called_once()
                kwargs = ws.connect.call_args.kwargs
                self.assertEqual(kwargs["redirect_limit"], 0)
                self.assertEqual(kwargs["http_no_proxy"], ["*"])
                self.assertTrue(kwargs["suppress_origin"])
                self.assertNotIn("cookie", kwargs)
                self.assertNotIn("Authorization", kwargs["header"])
                self.assertNotIn("Cookie", kwargs["header"])

    def test_failed_connection_is_closed_without_retry(self):
        ws = Mock()
        ws.connect.side_effect = ConnectionError("fixture")
        with patch("websocket.WebSocket", return_value=ws):
            with self.assertRaises(ConnectionError):
                hltv_live._connect("wss://scorebot-lb.hltv.org/socket.io/", 2)
        ws.connect.assert_called_once()
        ws.close.assert_called_once_with(timeout=1)


class ScorebotCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.serial = 0

    def run_capture(self, packets=(), *, seconds=1, html=None, page_error=None,
                    connection_error=None, close_error=None, **kwargs):
        self.serial += 1
        directory = self.root / f"session-{self.serial}"
        clock = Clock()
        ws = FakeSocket(packets, clock, close_error=close_error)
        page = {"url": MATCH_URL, "html": page_html() if html is None else html,
                "received_at": "2026-09-16T00:00:00Z", "sha256": "fixture"}
        reader = Mock(return_value=page, side_effect=page_error)
        connector = Mock(return_value=ws, side_effect=connection_error)
        real_stream = hltv_live._stream

        def immediate_stream(*args, **stream_kwargs):
            return real_stream(*args, monotonic=clock, **stream_kwargs)

        with patch("cs2ml.hltv_live._stream", side_effect=immediate_stream):
            summary = hltv_live.capture(MATCH_URL, directory, seconds, page_reader=reader,
                                        connector=connector, **kwargs)
        rows = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        saved = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary, saved)
        reader.assert_called_once_with(MATCH_URL)
        return summary, rows, ws, connector

    def test_capture_preserves_duplicate_packets_raw_receipts_and_backfill_status(self):
        full_log = '42["fullLog","{\\"old\\":true}"]'
        packets = [text_packet(value) for value in (OPEN, "40", EVENT, EVENT, full_log)]
        summary, rows, ws, connector = self.run_capture(packets)
        self.assertEqual(summary["status"], "captured")
        self.assertEqual(summary["reason"], "duration_limit")
        self.assertEqual(summary["event_counts"], {"scoreboard": 2, "fullLog": 1})
        self.assertFalse(summary["source_clock_verified"])
        self.assertFalse(summary["model_input_enabled"])
        self.assertFalse(summary["live_trading_enabled"])
        connector.assert_called_once_with("wss://scorebot-lb.hltv.org/socket.io/?EIO=3&transport=websocket", 1)
        self.assertEqual(ws.close_calls, [1])
        by_sequence = {row["sequence"]: row for row in rows}
        parsed = [row for row in rows if row["event_type"] == "scorebot_event"]
        self.assertEqual(len(parsed), 3)
        for row in parsed:
            raw = by_sequence[row["payload"]["raw_sequence"]]
            self.assertEqual(raw["event_type"], "socket_packet")
            self.assertEqual(row["payload"]["packet_received_at"], raw["received_at"])
            self.assertEqual(row["payload"]["packet_monotonic_ns"], raw["monotonic_ns"])
            self.assertIsNone(raw["source_timestamp"])
            self.assertFalse(row["payload"]["eligible_for_inference"])
        self.assertEqual([row["payload"]["possible_backfill"] for row in parsed], [False, False, True])
        self.assertEqual(rows[-1]["event_type"], "capture_stopped")

    def test_disconnect_keeps_prior_event_counts_without_reconnecting(self):
        summary, rows, ws, connector = self.run_capture([
            text_packet(OPEN), text_packet("40"), text_packet(EVENT),
            (websocket.ABNF.OPCODE_CLOSE, b"\x03\xe8")])
        self.assertEqual(summary["status"], "stopped")
        self.assertIn("no reconnect", summary["reason"])
        self.assertEqual(summary["event_counts"], {"scoreboard": 1})
        connector.assert_called_once()
        self.assertEqual(ws.close_calls, [1])
        self.assertTrue(any(row["event_type"] == "socket_close" for row in rows))

    def test_explicit_user_interrupt_closes_and_records_stop(self):
        summary, rows, ws, connector = self.run_capture([KeyboardInterrupt()])
        self.assertEqual(summary["reason"], "user_interrupted")
        self.assertEqual(rows[-1]["event_type"], "capture_stopped")
        self.assertEqual(ws.close_calls, [1])
        connector.assert_called_once()

    def test_page_and_handshake_403_stop_once_and_unknown_exception_text_is_not_copied(self):
        summary, rows, _, connector = self.run_capture(page_error=HLTVSourceError("HLTV HTTP 403; no retry or bypass"))
        self.assertIn("HTTP 403", summary["reason"])
        connector.assert_not_called()
        self.assertEqual([row["event_type"] for row in rows], ["capture_stopped"])
        error = websocket.WebSocketBadStatusException("SECRET_RESPONSE_BODY", status_code=403)
        summary, rows, _, connector = self.run_capture(connection_error=error)
        connector.assert_called_once()
        self.assertEqual(summary["http_status"], 403)
        self.assertEqual(summary["reason"], "WebSocketBadStatusException")
        self.assertNotIn("SECRET_RESPONSE_BODY", json.dumps([summary, rows]))

    def test_malformed_protocol_packet_is_saved_before_rejection(self):
        packet = "42{broken"
        summary, rows, _, connector = self.run_capture([text_packet(OPEN), text_packet("40"), text_packet(packet)])
        self.assertEqual(summary["status"], "stopped")
        self.assertIn(packet, [row["payload"] for row in rows if row["event_type"] == "socket_packet"])
        self.assertFalse(any(row["event_type"] == "scorebot_event" for row in rows))
        connector.assert_called_once()

    def test_binary_and_invalid_utf8_are_saved_as_inert_base64_then_stop(self):
        for opcode, data in ((websocket.ABNF.OPCODE_BINARY, b"\x00\xff"),
                             (websocket.ABNF.OPCODE_TEXT, b"\xff\xfe")):
            with self.subTest(opcode=opcode):
                summary, rows, _, connector = self.run_capture([(opcode, data)])
                self.assertEqual(summary["status"], "stopped")
                raw = next(row["payload"] for row in rows if row["event_type"] == "unsupported_packet")
                self.assertEqual(raw["encoding"], "base64")
                self.assertEqual(base64.b64decode(raw["data"]), data)
                connector.assert_called_once()

    def test_record_and_byte_limits_stop_without_second_network_attempt(self):
        summary, rows, _, connector = self.run_capture(max_records=1)
        self.assertIn("max_records", summary["reason"])
        self.assertEqual(len(rows), 1)
        connector.assert_not_called()
        summary, rows, _, connector = self.run_capture(html=page_html() + "x" * 5000, max_bytes=2000)
        self.assertIn("max_bytes", summary["reason"])
        self.assertLessEqual(summary["bytes_written"], 2000)
        connector.assert_not_called()

    def test_invalid_duration_and_existing_directory_fail_before_network(self):
        reader, connector = Mock(), Mock()
        for seconds in (0, 301, True, float("nan"), float("inf"), "30"):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                hltv_live.capture(MATCH_URL, self.root / "invalid", seconds,
                                  page_reader=reader, connector=connector)
        with self.assertRaises(FileExistsError):
            hltv_live.capture(MATCH_URL, self.root, 1, page_reader=reader, connector=connector)
        reader.assert_not_called()
        connector.assert_not_called()

    def test_handshake_and_heartbeat_timeouts_are_bounded_without_sleep(self):
        summary, _, _, connector = self.run_capture([text_packet(OPEN)], seconds=10)
        self.assertIn("handshake/subscription timeout", summary["reason"])
        connector.assert_called_once()
        fast_open = '0{"sid":"fixture","pingInterval":1000,"pingTimeout":1000}'
        summary, rows, ws, connector = self.run_capture([text_packet(fast_open), text_packet("40")], seconds=10)
        self.assertIn("heartbeat timeout", summary["reason"])
        self.assertIn("2", ws.sent)
        self.assertTrue(all(0 < timeout <= 1 for timeout in ws.timeouts))
        connector.assert_called_once()

    def test_close_failure_does_not_erase_original_reason_or_summary(self):
        summary, rows, ws, _ = self.run_capture([KeyboardInterrupt()], close_error=OSError("SECRET_CLOSE_BODY"))
        self.assertEqual(summary["reason"], "user_interrupted")
        self.assertEqual(summary["close_error"], "OSError")
        self.assertEqual(rows[-1]["event_type"], "capture_stopped")
        self.assertNotIn("SECRET_CLOSE_BODY", json.dumps(summary))
        self.assertEqual(ws.close_calls, [1])

    def test_oversized_complete_message_records_size_without_saving_payload(self):
        with patch("cs2ml.hltv_live.MAX_PACKET_BYTES", 16):
            summary, rows, _, connector = self.run_capture([(websocket.ABNF.OPCODE_TEXT, b"x" * 17)])
        self.assertIn("packet budget", summary["reason"])
        oversize = next(row["payload"] for row in rows if row["event_type"] == "oversized_packet")
        self.assertEqual(oversize, {"opcode": websocket.ABNF.OPCODE_TEXT, "size": 17})
        self.assertFalse(any(row["event_type"] == "socket_packet" for row in rows))
        connector.assert_called_once()


if __name__ == "__main__":
    unittest.main()
