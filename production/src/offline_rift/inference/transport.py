"""Bounded local game API access, including Windows loopback from WSL."""

from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
from urllib.parse import urlparse

import requests

DEFAULT_ENDPOINT = "https://127.0.0.1:2999/liveclientdata/allgamedata"
TRANSPORTS = ("auto", "local", "windows")


class LiveClientUnavailable(RuntimeError):
    pass


class LiveDataSource:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout: float = 3,
        verify_tls: bool | str = False,
        transport: str = "auto",
        session: requests.Session | None = None,
    ):
        parsed = urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Live endpoint must be local HTTP(S)")
        if not math.isfinite(timeout) or timeout <= 0 or transport not in TRANSPORTS:
            raise ValueError("Invalid live timeout or transport")
        self.endpoint, self.timeout, self.verify_tls = endpoint, timeout, verify_tls
        self.windows_curl = shutil.which("curl.exe")
        self.is_wsl = "microsoft" in platform.release().lower()
        self.transport = transport
        if transport == "auto":
            self.transport = (
                "windows" if session is None and self.is_wsl and self.windows_curl else "local"
            )
        if self.transport == "windows" and not self.windows_curl:
            raise ValueError("Windows transport requires curl.exe on PATH and Windows interop")
        self.session = session or requests.Session()
        self.session.trust_env = False

    def fetch(self) -> dict:
        try:
            if self.transport == "windows":
                data = self._windows_fetch()
            else:
                response = self.session.get(
                    self.endpoint,
                    timeout=self.timeout,
                    verify=self.verify_tls,
                    allow_redirects=False,
                )
                response.raise_for_status()
                if not 200 <= response.status_code < 300:
                    raise ValueError("Live endpoint returned a redirect")
                data = response.json()
            if not isinstance(data, dict):
                raise TypeError("Expected JSON object")
            return data
        except (
            requests.RequestException,
            OSError,
            subprocess.SubprocessError,
            ValueError,
            TypeError,
        ):
            raise LiveClientUnavailable(
                f"Live Client unavailable via {self.transport} transport. "
                "Start a 5v5 Summoner's Rift match and wait for the map to load. "
                "For Windows + WSL, check curl.exe and Windows interop; see docs/EXECUTION.md."
            ) from None

    def _windows_fetch(self):
        command = [
            self.windows_curl,
            "--disable",  # Ignore curlrc: no redirects, proxies, or alternate destinations.
            "--silent",
            "--show-error",
            "--fail",
            "--noproxy",
            "*",
            "--max-time",
            str(self.timeout),
            "--max-filesize",
            "4194304",
            "--write-out",
            "\n%{http_code}",
        ]
        if self.verify_tls is False:
            command.append("--insecure")
        elif isinstance(self.verify_tls, str):
            certificate = self.verify_tls
            if self.is_wsl:
                certificate = subprocess.run(
                    ["wslpath", "-w", certificate],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                ).stdout.strip()
            command.extend(["--cacert", certificate])
        command.extend(["--url", self.endpoint])
        response = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=self.timeout + 1
        )
        body, status = response.stdout.rsplit("\n", 1)
        if not 200 <= int(status) < 300:
            raise ValueError("Unexpected live HTTP status")
        return json.loads(body)

    def close(self):
        self.session.close()
