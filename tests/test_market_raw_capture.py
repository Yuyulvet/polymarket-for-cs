import io
import json
from pathlib import Path
import tempfile
import unittest

import websocket

from cs2ml.market_raw_capture import market_binding, record_socket


class Socket:
    def __init__(self, messages, on_exhausted=None):
        self.messages = iter(messages)
        self.sent = []
        self.on_exhausted = on_exhausted

    def recv(self):
        try:
            return next(self.messages)
        except StopIteration:
            if self.on_exhausted is not None:
                self.on_exhausted()
            return ''

    def send(self, value):
        self.sent.append(value)


class DroppingSocket:
    def recv(self):
        raise websocket.WebSocketConnectionClosedException('dropped')

    def send(self, value):
        pass


class MarketCaptureTests(unittest.TestCase):
    def test_binding_and_rejection(self):
        event = {'id': '1', 'markets': [{'id': '2', 'outcomes': '["A", "B"]',
                 'clobTokenIds': '["101", "102"]', 'conditionId': 'condition'}]}
        result = market_binding(event, 1, 2)
        self.assertEqual(result['outcome_tokens'], {'A': '101', 'B': '102'})
        self.assertFalse(result['eligible_for_trading'])
        with self.assertRaises(ValueError):
            market_binding(event, 9, 2)
        with self.assertRaises(ValueError):
            market_binding(event, 1, 3)
        event['markets'][0]['clobTokenIds'] = '["101", "101"]'
        with self.assertRaises(ValueError):
            market_binding(event, 1, 2)

    def test_preserves_array_snapshot_delta_and_heartbeat(self):
        messages = ['[{"event_type":"book"}]', '{"event_type":"price_change"}', 'PING']
        ws, handle = Socket(messages), io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            result = record_socket(ws, handle, seconds=2, max_bytes=10000,
                                   stop_path=Path(directory) / 'STOP', clock=lambda: 1,
                                   wall_clock=lambda: 'test-receipt')
        rows = [json.loads(line) for line in handle.getvalue().splitlines()]
        self.assertEqual([r['raw'] for r in rows], messages)
        self.assertEqual(result['messages'], 3)
        self.assertEqual(result['exit_reason'], 'connection_closed')
        self.assertEqual(ws.sent, ['PONG'])
        self.assertTrue(all(not r['eligible_for_trading'] for r in rows))

    def test_stop_size_and_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / 'STOP'
            handle = io.StringIO()
            result = record_socket(Socket(['oversized']), handle, seconds=2,
                                   max_bytes=1, stop_path=stop, clock=lambda: 1)
            self.assertEqual(result['exit_reason'], 'byte_limit')
            self.assertEqual(handle.getvalue(), '')
            stop.touch()
            result = record_socket(Socket([]), handle, seconds=2,
                                   max_bytes=1000, stop_path=stop, clock=lambda: 1)
            self.assertEqual(result['exit_reason'], 'stop_file')
            ticks = iter([0, 0, 3])
            result = record_socket(Socket([]), handle, seconds=2,
                                   max_bytes=1000, stop_path=stop, clock=lambda: next(ticks))
            self.assertEqual(result['exit_reason'], 'deadline')

    def test_reconnects_on_drop_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / 'STOP'

            def connector():
                socket = Socket(['[{"event_type":"book"}]'], on_exhausted=stop.touch)
                return socket

            handle = io.StringIO()
            result = record_socket(DroppingSocket(), handle, seconds=5,
                                   max_bytes=10000, stop_path=stop,
                                   assets=['101'], connector=connector,
                                   clock=lambda: 1, wall_clock=lambda: 'test-receipt')
        self.assertEqual(result['reconnects'], 1)
        self.assertEqual(result['messages'], 1)
        rows = [json.loads(line) for line in handle.getvalue().splitlines()]
        self.assertEqual(rows[0]['raw'], '[{"event_type":"book"}]')

    def test_drop_without_assets_still_fails_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(websocket.WebSocketException):
                record_socket(DroppingSocket(), io.StringIO(), seconds=5,
                              max_bytes=10000,
                              stop_path=Path(directory) / 'STOP',
                              clock=lambda: 1)

    def test_empty_close_with_assets_reconnects(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / 'STOP'

            def connector():
                socket = Socket(['[{"event_type":"book"}]'], on_exhausted=stop.touch)
                return socket

            handle = io.StringIO()
            result = record_socket(Socket([]), handle, seconds=5,
                                   max_bytes=10000, stop_path=stop,
                                   assets=['101'], connector=connector,
                                   clock=lambda: 1, wall_clock=lambda: 'test-receipt')
        self.assertEqual(result['reconnects'], 1)
        self.assertEqual(result['messages'], 1)
        rows = [json.loads(line) for line in handle.getvalue().splitlines()]
        self.assertEqual(rows[0]['raw'], '[{"event_type":"book"}]')


if __name__ == '__main__':
    unittest.main()
