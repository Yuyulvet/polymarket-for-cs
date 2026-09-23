"""Bounded, read-only public market evidence capture; never submits orders."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from curl_cffi import requests
import websocket

WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/market'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def market_binding(event, event_id, market_id):
    if str(event.get('id')) != str(event_id):
        raise ValueError('event_id_mismatch')
    matches = [m for m in event.get('markets', []) if str(m.get('id')) == str(market_id)]
    if len(matches) != 1:
        raise ValueError('market_not_unique')
    market = matches[0]
    def array(key):
        value = market.get(key, [])
        return json.loads(value) if isinstance(value, str) else value
    outcomes, tokens = array('outcomes'), array('clobTokenIds')
    if (not isinstance(outcomes, list) or not isinstance(tokens, list)
            or len(outcomes) != 2 or len(tokens) != 2
            or len(set(outcomes)) != 2 or len(set(tokens)) != 2
            or not all(isinstance(x, str) and x for x in outcomes + tokens)):
        raise ValueError('invalid_binary_market_binding')
    return {'event_id': str(event_id), 'market_id': str(market_id),
            'condition_id': market.get('conditionId'),
            'market_name': market.get('groupItemTitle') or market.get('question'),
            'outcome_tokens': dict(zip(outcomes, tokens)),
            'event': event, 'eligible_for_trading': False}


def record_socket(ws, handle, *, seconds, max_bytes, stop_path, assets=None,
                  connector=None, clock=time.monotonic, wall_clock=utc_now):
    """Keep raw array snapshots and deltas alike, with local receipt timestamps.

    If `assets` is given, a dropped connection is re-established (the
    resubscribe yields a fresh full-depth book snapshot) and capture continues
    until the deadline; reconnects are counted in the result. Without `assets`
    a drop propagates, preserving legacy fail-fast behaviour.
    """
    if not math.isfinite(seconds) or seconds <= 0 or max_bytes <= 0:
        raise ValueError('invalid_capture_limits')
    if connector is None:
        connector = lambda: _subscribe(assets)
    deadline = clock() + seconds
    next_ping = clock() + 10
    rows, byte_count, reason, reconnects = 0, 0, 'deadline', 0
    while clock() < deadline:
        if stop_path.exists():
            reason = 'stop_file'
            break
        if clock() >= next_ping:
            try:
                ws.send('PING')
            except websocket.WebSocketException:
                pass  # recv will surface the broken connection
            next_ping = clock() + 10
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except websocket.WebSocketException:
            if assets is None:
                raise
            ws = _reconnect(connector, deadline, stop_path, clock)
            if ws is None:
                reason = 'reconnect_exhausted'
                break
            reconnects += 1
            next_ping = clock() + 10
            continue
        received_at, received_mono = wall_clock(), clock()
        if raw is None or raw == '':
            # Server closed the socket. With `assets` we re-establish the
            # connection (fresh full-depth snapshot on resubscribe); without,
            # preserve the legacy fail-fast behaviour.
            if assets is None:
                reason = 'connection_closed'
                break
            ws = _reconnect(connector, deadline, stop_path, clock)
            if ws is None:
                reason = 'reconnect_exhausted'
                break
            reconnects += 1
            next_ping = clock() + 10
            continue
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        if raw == 'PING':
            ws.send('PONG')
        line = json.dumps({'type': 'market_raw', 'received_at': received_at,
                           'received_monotonic_seconds': received_mono,
                           'raw': raw, 'eligible_for_trading': False}) + '\n'
        size = len(line.encode('utf-8'))
        if byte_count + size > max_bytes:
            reason = 'byte_limit'
            break
        handle.write(line)
        handle.flush()
        byte_count += size
        rows += 1
    return {'messages': rows, 'bytes': byte_count, 'exit_reason': reason,
            'reconnects': reconnects}


def _subscribe(assets):
    ws = websocket.create_connection(WS_URL, timeout=10)
    ws.send(json.dumps({'assets_ids': list(assets), 'type': 'market'}))
    ws.settimeout(1)
    return ws


def _reconnect(connector, deadline, stop_path, clock):
    """Retry until deadline; None if the budget expired first."""
    while clock() < deadline and not stop_path.exists():
        try:
            return connector()
        except websocket.WebSocketException:
            time.sleep(1)
        except OSError:
            time.sleep(1)
    return None


def _array(value):
    return json.loads(value) if isinstance(value, str) else (value or [])


def select_market_bindings(event, event_id, market_ids) -> list[dict]:
    """Validate and bind an ordered list of binary markets from an event."""
    ordered = []
    for market_id in market_ids:
        binding = market_binding(event, event_id, market_id)
        outcomes = binding['outcome_tokens']
        ordered.append({
            'event_id': binding['event_id'],
            'market_id': binding['market_id'], 'condition_id': binding['condition_id'],
            'market_name': binding['market_name'],
            'outcomes': list(outcomes), 'tokens': dict(outcomes),
        })
    return ordered


def capture_event_markets(event, event_id, market_ids, output, seconds=120,
                          max_bytes=100_000_000):
    """One websocket, all selected markets' tokens, full depth raw capture."""
    if not market_ids:
        raise ValueError('market_ids_required')
    bindings = select_market_bindings(event, event_id, list(market_ids))
    assets = [token for binding in bindings for token in binding['tokens'].values()]
    if len(set(assets)) != len(assets):
        raise ValueError('duplicate_token_across_markets')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    resuming = metadata_path.exists()
    if resuming and (output / "summary.json").exists():
        raise ValueError("market_capture_already_exists")
    if not resuming:
        metadata = {
            'started_at': utc_now(), 'seconds': seconds, 'max_bytes': max_bytes,
            'mode': 'read_only_raw_evidence_multi',
            'binding': {k: bindings[0][k] for k in
                        ('event_id', 'market_id', 'condition_id', 'market_name',
                         'tokens')},
            'markets': bindings,
        }
        metadata_path.write_text(json.dumps(
            metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    ws = None
    result = {'messages': None, 'exit_reason': 'error'}
    try:
        ws = websocket.create_connection(WS_URL, timeout=10)
        ws.send(json.dumps({'assets_ids': assets, 'type': 'market'}))
        ws.settimeout(1)
        mode = "a" if resuming else "x"
        with (output / 'events.jsonl').open(mode, encoding='utf-8') as handle:
            result = record_socket(ws, handle, seconds=seconds,
                                   max_bytes=max_bytes, stop_path=output / 'STOP',
                                   assets=assets)
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        raise
    finally:
        if ws is not None:
            ws.close()
        (output / 'summary.json').write_text(json.dumps(
            {**result, 'ended_at': utc_now()}, indent=2), encoding='utf-8')
    return result


def capture(event_id, market_id, output, seconds=120, max_bytes=100_000_000):
    if not math.isfinite(seconds) or seconds <= 0 or max_bytes <= 0:
        raise ValueError('invalid_capture_limits')
    response = requests.get(f'https://gamma-api.polymarket.com/events/{int(event_id)}',
                            timeout=20, impersonate='chrome')
    response.raise_for_status()
    binding = market_binding(response.json(), event_id, market_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'metadata.json').write_text(json.dumps(
        {'started_at': utc_now(), 'binding': binding, 'seconds': seconds,
         'max_bytes': max_bytes, 'mode': 'read_only_raw_evidence'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    ws = None
    result = {'messages': None, 'exit_reason': 'error'}
    try:
        ws = websocket.create_connection(WS_URL, timeout=10)
        ws.send(json.dumps({'assets_ids': list(binding['outcome_tokens'].values()),
                            'type': 'market'}))
        ws.settimeout(1)
        with (output / 'events.jsonl').open('x', encoding='utf-8') as handle:
            result = record_socket(ws, handle, seconds=seconds,
                                   max_bytes=max_bytes, stop_path=output / 'STOP',
                                   assets=list(binding['outcome_tokens'].values()))
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        raise
    finally:
        if ws is not None:
            ws.close()
        (output / 'summary.json').write_text(json.dumps(
            {**result, 'ended_at': utc_now()}, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-id', type=int, required=True)
    parser.add_argument('--market-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=120)
    args = parser.parse_args()
    print(json.dumps(capture(args.event_id, args.market_id, args.output, args.seconds)))


if __name__ == '__main__':
    main()
