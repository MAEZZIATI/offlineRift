"""Serve the Macro HUD and launch its native desktop overlay."""

from __future__ import annotations

import argparse
import json
import logging
import math
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from offline_rift.inference.transport import TRANSPORTS
from offline_rift.ui.service import LiveHudService


class HudServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, dashboard: LiveHudService, port: int = 8765):
        self.dashboard = dashboard
        static = files("offline_rift.ui").joinpath("static")
        self.assets = {
            "/": (static.joinpath("index.html").read_bytes(), "text/html; charset=utf-8")
        }
        for asset in static.joinpath("assets").iterdir():
            if asset.is_file() and asset.name.endswith((".js", ".css")):
                mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                self.assets[f"/assets/{asset.name}"] = (asset.read_bytes(), mime)
        super().__init__(("127.0.0.1", port), HudHandler)


class HudHandler(BaseHTTPRequestHandler):
    server: HudServer

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def _trusted(self):
        # Bind to loopback and reject cross-origin/DNS-rebinding requests.
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update({"127.0.0.1", "localhost"})
        if self.headers.get("Host") not in hosts:
            self.send_error(403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            self.send_error(403)
            return False
        return True

    def do_GET(self):
        if not self._trusted():
            return
        route = urlsplit(self.path).path
        if route == "/api/hud":
            self._send(
                json.dumps(self.server.dashboard.frame(), allow_nan=False).encode(),
                "application/json",
            )
        elif route == "/api/state":
            query = parse_qs(urlsplit(self.path).query)
            if set(query) - {"team", "actor", "role"} or any(
                len(values) != 1 for values in query.values()
            ):
                self.send_error(400)
                return
            try:
                team = int(query["team"][0]) if "team" in query else None
                actor = query.get("actor", [None])[0]
                role = query.get("role", [None])[0]
                payload = json.dumps(
                    self.server.dashboard.state(team=team, actor=actor, role=role), allow_nan=False
                ).encode()
            except (ValueError, KeyError):
                self.send_error(400)
                return
            self._send(payload, "application/json; charset=utf-8")
        elif route in self.server.assets:
            self._send(*self.server.assets[route])
        else:
            self.send_error(404)

    def do_POST(self):
        if not self._trusted():
            return
        if self.path != "/api/context":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected bounded JSON")
            payload = self.server.dashboard.configure(json.loads(self.rfile.read(length)))
            self._send(json.dumps(payload, allow_nan=False).encode(), "application/json")
        except (ValueError, TypeError, KeyError):
            self.send_error(400)

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
    from rift_common.actions import ROLES

    parser.add_argument(
        "--serve-only", action="store_true", help="Serve the HUD without launching a desktop host"
    )
    parser.add_argument("--desktop", type=Path, help="Override the native HUD executable")
    parser.add_argument("--position", choices=("top-right", "top-left"), default="top-right")
    parser.add_argument("--team", type=int, choices=(100, 200), default=100)
    parser.add_argument("--actor", default="active")
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument(
        "--interval",
        type=float,
        default=10,
        help="Poll interval in seconds (default: 10, minimum: 2)",
    )
    parser.add_argument("--port", type=int, default=8765, help="Local HUD port (default: 8765)")
    parser.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default="auto",
        help="auto uses Windows curl.exe in WSL when available, otherwise local HTTP",
    )
    parser.add_argument("--ca-bundle", help="Optional Riot root certificate for TLS verification")
    parser.add_argument(
        "--capture-session",
        type=Path,
        help="Opt in to recording measured inputs and served HUD advice",
    )


def run_hud(
    model_dir: str | Path,
    *,
    team=100,
    actor="active",
    role=None,
    interval=10,
    port=8765,
    transport="auto",
    ca_bundle=None,
    capture_session=None,
    serve_only=False,
    desktop=None,
    position="top-right",
):
    if not 1 <= port <= 65535:
        raise ValueError("UI port must be between 1 and 65535")
    if not math.isfinite(interval) or interval < 2:
        raise ValueError("UI polling interval must be at least 2 seconds")
    import torch

    from offline_rift.inference.coach import MacroCoach
    from offline_rift.inference.live_client import LiveClientInference
    from offline_rift.ui.desktop import launch_desktop

    # Small CPU inference batches do not need PyTorch's default all-core thread pool.
    torch.set_num_threads(1)
    client = LiveClientInference(
        model_dir,
        team=team,
        actor=actor,
        role=role,
        interval_seconds=interval,
        transport=transport,
        verify_tls=ca_bundle or False,
        capture_session=capture_session,
    )
    try:
        if not isinstance(client.policy, MacroCoach):
            raise TypeError("The Macro HUD requires a completed six-policy coach bundle")
        dashboard = LiveHudService(client, model_dir=model_dir)
        with HudServer(dashboard, port) as server:
            print(f"Offline Rift HUD: http://127.0.0.1:{server.server_port}", flush=True)
            print(f"Game transport: {client.source.transport}. Press Ctrl+C to stop.", flush=True)
            if capture_session is not None:
                print(f"Session recording: {Path(capture_session).resolve()}", flush=True)
            print(f"Selected coach: {Path(model_dir).resolve()}", flush=True)
            if serve_only:
                server.serve_forever(poll_interval=0.5)
            else:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                process = None
                try:
                    process = launch_desktop(
                        server.server_port, executable=desktop, position=position
                    )
                    if process.wait() != 0:
                        raise RuntimeError("Native HUD exited with an error")
                finally:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                    server.shutdown()
                    thread.join(timeout=10)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        help="Completed six-policy coach directory (defaults to workflow selection)",
    )
    add_ui_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        from offline_rift.workflow import Workflow

        model = args.model or Workflow(Path("workflow.json")).state().get("model", {}).get("path")
        if not model:
            raise ValueError("No selected coach; train and select a six-policy bundle first")
        run_hud(
            model,
            team=args.team,
            actor=args.actor,
            role=args.role,
            interval=args.interval,
            port=args.port,
            transport=args.transport,
            ca_bundle=args.ca_bundle,
            capture_session=args.capture_session,
            serve_only=args.serve_only,
            desktop=args.desktop,
            position=args.position,
        )
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
