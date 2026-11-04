"""Serve the local dashboard using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import logging
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from offline_rift.inference.transport import TRANSPORTS
from offline_rift.ui.service import LiveDashboard

ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, dashboard: LiveDashboard, port: int = 8765):
        self.dashboard = dashboard
        self.assets = {
            route: (files("offline_rift.ui").joinpath("static", name).read_bytes(), content_type)
            for route, (name, content_type) in ASSETS.items()
        }
        super().__init__(("127.0.0.1", port), DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def do_GET(self):
        # Bind to loopback and reject cross-origin/DNS-rebinding requests.
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update({"127.0.0.1", "localhost"})
        if self.headers.get("Host") not in hosts:
            self.send_error(403)
            return
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            self.send_error(403)
            return
        route = urlsplit(self.path).path
        if route == "/api/state":
            payload = json.dumps(self.server.dashboard.state(), allow_nan=False).encode()
            self._send(payload, "application/json; charset=utf-8")
        elif route in self.server.assets:
            self._send(*self.server.assets[route])
        else:
            self.send_error(404)

    def _send(self, payload: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A tab closed while a game poll was in flight.

    def log_message(self, format, *args):
        logging.getLogger(__name__).debug(format, *args)


def add_ui_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--team", type=int, choices=(100, 200), default=100)
    parser.add_argument(
        "--interval", type=float, default=10, help="Poll interval, minimum 2 seconds"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Local dashboard port (default: 8765)"
    )
    parser.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default="auto",
        help="auto uses Windows curl.exe in WSL when available, otherwise local HTTP",
    )
    parser.add_argument("--ca-bundle", help="Optional Riot root certificate for TLS verification")


def run_ui(
    model_dir: str | Path, *, team=100, interval=10, port=8765, transport="auto", ca_bundle=None
):
    if not 1 <= port <= 65535:
        raise ValueError("UI port must be between 1 and 65535")
    if not math.isfinite(interval) or interval < 2:
        raise ValueError("UI polling interval must be at least 2 seconds")
    import torch

    from offline_rift.inference.live_client import LiveClientInference

    # Small CPU inference batches do not need PyTorch's default all-core thread pool.
    torch.set_num_threads(1)
    client = LiveClientInference(
        model_dir,
        team=team,
        interval_seconds=interval,
        transport=transport,
        verify_tls=ca_bundle or False,
    )
    try:
        dashboard = LiveDashboard(client)
        with DashboardServer(dashboard, port) as server:
            print(f"Offline Rift UI: http://127.0.0.1:{server.server_port}", flush=True)
            print(f"Game transport: {client.source.transport}. Press Ctrl+C to stop.", flush=True)
            server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Completed policy bundle directory"
    )
    add_ui_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run_ui(
            args.model,
            team=args.team,
            interval=args.interval,
            port=args.port,
            transport=args.transport,
            ca_bundle=args.ca_bundle,
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
