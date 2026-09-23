from __future__ import annotations

import json
import unittest

from cs2ml.hltv_scorebot_protocol import (MAX_HEARTBEAT_MS, MAX_PACKET_BYTES,
                                        ProtocolError, ScorebotProtocol)


def opening(**overrides):
    return "0" + json.dumps({"sid": "public-session", "pingInterval": 25000, "pingTimeout": 5000,
                              **overrides}, separators=(",", ":"))


def connected():
    protocol = ScorebotProtocol("000123")
    protocol.feed(opening())
    protocol.feed("40")
    return protocol


class HandshakeTests(unittest.TestCase):
    def test_open_exposes_heartbeat_seconds_without_outgoing_ping(self):
        protocol = ScorebotProtocol("000123")
        self.assertFalse(protocol.opened)
        self.assertFalse(protocol.subscribed)
        self.assertIsNone(protocol.ping_interval_seconds)
        self.assertIsNone(protocol.ping_timeout_seconds)
        result = protocol.feed(opening())
        self.assertEqual(result, {"kind": "open"})
        self.assertTrue(protocol.opened)
        self.assertFalse(protocol.subscribed)
        self.assertEqual(protocol.sid, "public-session")
        self.assertEqual(protocol.ping_interval_seconds, 25.)
        self.assertEqual(protocol.ping_timeout_seconds, 5.)

    def test_subscription_inner_payload_is_a_json_string_and_id_stays_string(self):
        protocol = ScorebotProtocol("000123")
        protocol.feed(opening())
        result = protocol.feed("40")
        self.assertEqual(result["kind"], "connect")
        self.assertEqual(len(result["outgoing"]), 1)
        frame = result["outgoing"][0]
        self.assertTrue(frame.startswith("42"))
        event, payload = json.loads(frame[2:])
        self.assertEqual(event, "readyForMatch")
        self.assertIsInstance(payload, str)
        self.assertEqual(json.loads(payload), {"token": "", "listId": "000123"})
        self.assertTrue(protocol.subscribed)

    def test_identifier_is_not_coerced_or_normalized(self):
        for ident in (123, True, None, "", "   ", "x"*1025, "\ud800"):
            with self.subTest(ident=repr(ident)):
                with self.assertRaises(ProtocolError):
                    ScorebotProtocol(ident)
        protocol = ScorebotProtocol('a"中')
        protocol.feed(opening())
        outer = json.loads(protocol.feed("40")["outgoing"][0][2:])
        self.assertEqual(json.loads(outer[1])["listId"], 'a"中')

    def test_invalid_heartbeat_values_fail_closed(self):
        for field in ("pingInterval", "pingTimeout"):
            for value in (None, True, False, "25000", 25000., 0, -1, MAX_HEARTBEAT_MS+1):
                with self.subTest(field=field, value=value):
                    protocol = ScorebotProtocol("1")
                    with self.assertRaises(ProtocolError):
                        protocol.feed(opening(**{field: value}))
                    self.assertFalse(protocol.opened)
                    self.assertTrue(protocol.failed)
                    self.assertIsNone(protocol.ping_interval_seconds)

    def test_heartbeat_integer_bounds_are_inclusive(self):
        protocol = ScorebotProtocol("1")
        protocol.feed(opening(pingInterval=1, pingTimeout=MAX_HEARTBEAT_MS))
        self.assertEqual(protocol.ping_interval_seconds, .001)
        self.assertEqual(protocol.ping_timeout_seconds, MAX_HEARTBEAT_MS/1000)

    def test_invalid_handshake_shapes_and_missing_fields(self):
        for packet in ("0", "0[]", "0null", "0true", "0{}", '0{"sid":"a"}',
                       opening(sid=""), opening(sid=123), '0{"sid":"a","sid":"b"}',
                       opening()+"junk"):
            with self.subTest(packet=packet):
                with self.assertRaises(ProtocolError):
                    ScorebotProtocol("1").feed(packet)

    def test_duplicate_open_and_connect_do_not_resubscribe(self):
        protocol = ScorebotProtocol("1")
        protocol.feed(opening())
        with self.assertRaisesRegex(ProtocolError, "Duplicate.*open"):
            protocol.feed(opening())
        protocol = connected()
        with self.assertRaisesRegex(ProtocolError, "Duplicate.*connect"):
            protocol.feed("40")
        self.assertFalse(protocol.subscribed)
        self.assertTrue(protocol.failed)


class EventTests(unittest.TestCase):
    def test_objects_arrays_scalars_and_unknown_event_names_are_preserved(self):
        protocol = connected()
        for payload in ({"CT": [{"steamId": "0005", "hp": 100}]}, [1, {"a": True}],
                        "text", None, True, 42, 1.5):
            with self.subTest(payload=payload):
                frame = "42" + json.dumps(["unknownPublicEvent", payload])
                self.assertEqual(protocol.feed(frame), {"kind": "event", "event_name": "unknownPublicEvent",
                                                        "event_payload": payload})

    def test_log_strings_are_never_decoded_twice(self):
        protocol = connected()
        for name in ("log", "fullLog"):
            for payload in ('{"log":[{"RoundStart":{}}]}', "not-json", "null"):
                result = protocol.feed("42"+json.dumps([name, payload]))
                self.assertEqual(result["event_payload"], payload)
                self.assertIsInstance(result["event_payload"], str)

    def test_exact_two_element_event_shape_is_required(self):
        for value in ([], ["log"], ["log", {}, {}], [7, {}], {"event": "log"}, None):
            with self.subTest(value=value):
                with self.assertRaises(ProtocolError):
                    connected().feed("42"+json.dumps(value))

    def test_malformed_or_nonfinite_json_is_rejected(self):
        for packet in ('42["log",', '42["log",{}]junk', '42["log",NaN]',
                       '42["log",Infinity]', '42["log",-Infinity]', '42["log",1e999]',
                       '42["log",{"a":1,"a":2}]'):
            with self.subTest(packet=packet):
                with self.assertRaises(ProtocolError):
                    connected().feed(packet)

    def test_deeply_nested_payload_is_a_protocol_error_not_recursion_error(self):
        packet = '42["log",' + "["*2000 + "0" + "]"*2000 + "]"
        with self.assertRaises(ProtocolError):
            connected().feed(packet)

    def test_size_limit_is_utf8_bytes_not_characters(self):
        packet = '42["log","' + "中"*(MAX_PACKET_BYTES//3) + '"]'
        self.assertLess(len(packet), MAX_PACKET_BYTES)
        self.assertGreater(len(packet.encode("utf-8")), MAX_PACKET_BYTES)
        with self.assertRaisesRegex(ProtocolError, "size limit"):
            connected().feed(packet)

    def test_maximum_size_packet_is_allowed(self):
        prefix, suffix = '42["log","', '"]'
        payload = "x"*(MAX_PACKET_BYTES-len(prefix)-len(suffix))
        packet = prefix+payload+suffix
        self.assertEqual(len(packet.encode("utf-8")), MAX_PACKET_BYTES)
        self.assertEqual(connected().feed(packet)["event_payload"], payload)
        with self.assertRaises(ProtocolError):
            connected().feed(prefix+payload+"x"+suffix)


class StateAndControlTests(unittest.TestCase):
    def test_non_open_packet_before_handshake_is_rejected(self):
        for packet in ("40", "3", "2", "1", '42["log",{}]', "41", "44error"):
            with self.subTest(packet=packet):
                with self.assertRaisesRegex(ProtocolError, "open packet required"):
                    ScorebotProtocol("1").feed(packet)

    def test_event_before_namespace_connect_is_rejected(self):
        protocol = ScorebotProtocol("1")
        protocol.feed(opening())
        with self.assertRaisesRegex(ProtocolError, "connect required"):
            protocol.feed('42["scoreboard",{}]')

    def test_pong_before_and_after_namespace_connect_has_no_outgoing(self):
        protocol = ScorebotProtocol("1")
        protocol.feed(opening())
        self.assertEqual(protocol.feed("3"), {"kind": "pong"})
        protocol.feed("40")
        self.assertEqual(protocol.feed("3"), {"kind": "pong"})

    def test_eio3_rejects_server_ping_and_transport_probe(self):
        for packet in ("2", "2probe", "3probe", "5", "6"):
            with self.subTest(packet=packet):
                with self.assertRaises(ProtocolError):
                    connected().feed(packet)

    def test_disconnect_and_socket_error_poison_connection(self):
        for packet in ("1", "41", '44{"message":"denied"}', '44"error"'):
            with self.subTest(packet=packet):
                protocol = connected()
                with self.assertRaises(ProtocolError):
                    protocol.feed(packet)
                self.assertFalse(protocol.opened)
                self.assertFalse(protocol.subscribed)
                with self.assertRaisesRegex(ProtocolError, "already failed"):
                    protocol.feed(opening())

    def test_unsupported_namespace_acks_and_binary_are_explicit_errors(self):
        for packet in ('40/live,', '40{"sid":"socket-v5"}', '42/live,["log",{}]',
                       '421["log",{}]', '431[{}]', '451-["log",{"_placeholder":true,"num":0}]',
                       '461-[{}]', 'b4AAAA', b'42["log",{}]'):
            with self.subTest(packet=packet):
                with self.assertRaises(ProtocolError):
                    connected().feed(packet)

    def test_empty_invalid_utf8_and_unknown_packets_fail(self):
        for packet in ("", "\ud800", "9", "4", "not-a-packet", ' 42["log",{}]', None):
            with self.subTest(packet=repr(packet)):
                with self.assertRaises(ProtocolError):
                    connected().feed(packet)


if __name__ == "__main__":
    unittest.main()
