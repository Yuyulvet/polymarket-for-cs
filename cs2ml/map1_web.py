"""Loopback-only paper workbench: python -m cs2ml.map1_web --port 8765."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from .map1 import DEFAULT_DIR
from .map1_desk import PaperDesk
from .map1_store import encode

ASSETS = Path(__file__).with_name("map1_web_assets")


def make_server(desk, port=8765):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_):
            pass  # No form contents or source evidence in access logs.

        def allowed_hosts(self):
            p = self.server.server_port
            return {f"127.0.0.1:{p}", f"localhost:{p}"}

        def send(self, status, payload, kind="application/json; charset=utf-8", filename=None):
            if isinstance(payload, (dict, list)):
                payload = encode(payload).encode("utf-8")
            elif isinstance(payload, str):
                payload = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def valid_host(self):
            if self.headers.get("Host") not in self.allowed_hosts():
                self.deny("loopback_host_required")
                return False
            return True

        def deny(self, reason):
            # Drain only a bounded rejected body before closing. Closing a TCP
            # socket with unread bytes can reset it (notably on Windows) and
            # discard the 403 response. Never parse or execute rejected input.
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if 0 < length <= 32768:
                try:
                    self.rfile.read(length)
                except OSError:
                    pass
            self.close_connection = True
            self.send(403, {"error": reason})

        def do_GET(self):
            if not self.valid_host():
                return
            path = urlsplit(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            try:
                if path in assets:
                    name, kind = assets[path]
                    self.send(200, (ASSETS / name).read_bytes(), kind)
                elif path == "/api/state":
                    self.send(200, {**desk.state(), "csrf_token": token})
                elif path == "/api/export/report":
                    self.send(200, desk.report(), filename="map1-paper-report.json")
                elif path in {"/api/export/snapshots", "/api/export/resolutions", "/api/export/hltv_evidence"}:
                    kind = path.rsplit("/", 1)[1]
                    rows = getattr(desk.store, kind)()
                    self.send(200, "".join(encode(row) + "\n" for row in rows),
                              "application/x-ndjson; charset=utf-8", f"map1-{kind}.jsonl")
                else:
                    self.send(404, {"error": "not_found"})
            except Exception as exc:
                self.send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            if not self.valid_host():
                return
            origin = self.headers.get("Origin")
            if (origin not in {f"http://{host}" for host in self.allowed_hosts()}
                    or not secrets.compare_digest(self.headers.get("X-CSRF-Token", ""), token)
                    or self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
                self.deny("same_origin_and_csrf_required")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("invalid_request_size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request_must_be_object")
                path = urlsplit(self.path).path
                if path == "/api/start":
                    if desk.data_error:
                        raise ValueError(desk.data_error)
                    desk.running = True
                    self.send(200, {"running": True, "mode": "paper_only"})
                elif path == "/api/stop":
                    desk.running = False
                    self.send(200, {"running": False, "active_round_may_finish": True})
                elif path in {"/api/discover", "/api/capture", "/api/settle", "/api/confirm",
                              "/api/hltv_import", "/api/hltv_identities", "/api/hltv_confirm"}:
                    self.send(202, desk.submit(path.rsplit("/", 1)[1], payload))
                else:
                    self.send(404, {"error": "not_found"})
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--db", type=Path, help="Separate database for a new frozen research protocol")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024..65535")
    desk = PaperDesk(args.data, args.db)
    try:
        server = make_server(desk, args.port)
        print(f"Paper-only workbench: http://127.0.0.1:{server.server_port} (collector PAUSED)", flush=True)
        print(f"Database: {desk.store.path.resolve()}", flush=True)
        try:
            server.serve_forever(poll_interval=.25)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        desk.close()


if __name__ == "__main__":
    main()
