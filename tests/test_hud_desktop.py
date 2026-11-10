"""Native HUD staging contracts and opt-in Windows smoke check with a local fixture."""

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from offline_rift.ui import desktop
from offline_rift.ui.server import HudServer
from rift_common.actions import ROLES


@pytest.fixture
def windows_package(tmp_path, monkeypatch):
    source = tmp_path / "wsl package"
    source.mkdir()
    executable = source / "OfflineRiftHUD.exe"
    executable.write_bytes(b"exe")
    (source / "resources").mkdir()
    (source / "resources/app.asar").write_bytes(b"application")
    (source / "runtime.dll").write_bytes(b"runtime")
    local = tmp_path / "Windows local app data"

    def run(args, **kwargs):
        if args[0] == "powershell.exe":
            assert args[-1] == "[Console]::Write($env:LOCALAPPDATA)"
            return SimpleNamespace(stdout="C:\\Users\\fixture\\AppData\\Local")
        assert args[:2] in (["wslpath", "-u"], ["wslpath", "-w"])
        return SimpleNamespace(
            stdout=str(local) if args[1] == "-u" else "\\\\wsl.localhost\\Fixture\\hud.exe"
        )

    monkeypatch.setattr(desktop.subprocess, "run", run)
    return executable, local


def test_wsl_package_is_staged_completely_and_reused_until_contents_change(windows_package):
    executable, local = windows_package
    staged = desktop._windows_local_executable(executable, {})
    assert staged.is_relative_to(local) and staged.read_bytes() == b"exe"
    assert (staged.parent / "runtime.dll").read_bytes() == b"runtime"
    assert (staged.parent / "resources/app.asar").read_bytes() == b"application"
    copied_at = staged.stat().st_mtime_ns
    assert desktop._windows_local_executable(executable, {}) == staged
    assert staged.stat().st_mtime_ns == copied_at
    (executable.parent / "resources/app.asar").write_bytes(b"updated")
    updated = desktop._windows_local_executable(executable, {})
    assert updated != staged and staged.exists()
    assert (updated.parent / "resources/app.asar").read_bytes() == b"updated"


def test_warm_launch_does_not_rehash_the_electron_runtime(windows_package, monkeypatch):
    executable, _ = windows_package
    staged = desktop._windows_local_executable(executable, {})
    digest = Mock(side_effect=AssertionError("Unchanged runtime was hashed again"))
    monkeypatch.setattr(desktop, "file_digest", digest)
    assert desktop._windows_local_executable(executable, {}) == staged
    digest.assert_not_called()


def test_ctrl_c_notifies_native_host_even_when_windows_proxy_has_exited(monkeypatch):
    from unittest.mock import MagicMock

    from offline_rift.inference.coach import MacroCoach
    from offline_rift.ui import server as ui

    client = Mock(policy=Mock(spec=MacroCoach))
    host = MagicMock()
    host.__enter__.return_value = host
    process = Mock()
    process.wait.side_effect = [KeyboardInterrupt, 0]
    process.poll.return_value = 130
    monkeypatch.setattr(
        "offline_rift.inference.live_client.LiveClientInference", Mock(return_value=client)
    )
    monkeypatch.setattr(ui, "LiveHudService", Mock())
    monkeypatch.setattr(ui, "HudServer", Mock(return_value=host))
    monkeypatch.setattr(desktop, "launch_desktop", Mock(return_value=process))
    ui.run_hud("unused-model")
    host.stop_desktop.assert_called_once()
    host.shutdown.assert_called_once()
    client.close.assert_called_once()
    process.terminate.assert_not_called()


def test_interrupted_copy_is_not_marked_complete_and_can_be_retried(windows_package, monkeypatch):
    executable, local = windows_package
    copy = desktop.shutil.copytree
    with monkeypatch.context() as patch:
        patch.setattr(desktop.shutil, "copytree", Mock(side_effect=OSError("copy interrupted")))
        with pytest.raises(OSError, match="interrupted"):
            desktop._windows_local_executable(executable, {})
    assert not list(local.rglob(".complete"))
    assert desktop.shutil.copytree is copy
    assert desktop._windows_local_executable(executable, {}).is_file()


def test_wsl_windows_drive_executable_does_not_need_copying(tmp_path, monkeypatch):
    executable = tmp_path / "hud.exe"
    run = Mock(return_value=SimpleNamespace(stdout="C:\\Apps\\hud.exe"))
    monkeypatch.setattr(desktop.subprocess, "run", run)
    assert desktop._windows_local_executable(executable, {}) == executable
    run.assert_called_once()


def test_launcher_uses_local_working_directory_and_keeps_secrets_out(windows_package, monkeypatch):
    executable, _ = windows_package
    monkeypatch.setattr(desktop.platform, "release", lambda: "microsoft-standard-WSL2")
    monkeypatch.setenv("RIOT_API_KEY", "test-key")
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    popen = Mock()
    monkeypatch.setattr(desktop.subprocess, "Popen", popen)
    desktop.launch_desktop(8765, executable=executable, position="top-left")
    args, kwargs = popen.call_args
    assert Path(args[0][0]).parent == kwargs["cwd"] != executable.parent
    assert args[0][1:] == [
        "--backend=http://127.0.0.1:8765",
        "--position=top-left",
        "--enable-logging=stderr",
    ]
    assert "RIOT_API_KEY" not in kwargs["env"]
    assert "ELECTRON_RUN_AS_NODE" not in kwargs["env"]
    assert not any("no-sandbox" in argument for argument in args[0])


@pytest.mark.skipif(
    not os.environ.get("RIFT_TEST_WINDOWS_HUD"),
    reason="Set RIFT_TEST_WINDOWS_HUD to a packaged Windows executable for a native smoke check",
)
def test_windows_native_hud_starts_and_closes_after_ctrl_c(tmp_path):
    executable = Path(os.environ["RIFT_TEST_WINDOWS_HUD"]).resolve()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"RIOT_API_KEY", "ELECTRON_RUN_AS_NODE"}
    }
    executable = desktop._windows_local_executable(executable, environment)
    report = executable.parent / f"smoke-{tmp_path.parent.name}.json"
    windows_report = subprocess.run(
        ["wslpath", "-w", str(report)], capture_output=True, text=True, check=True
    ).stdout.strip()
    frame = {
        "schema": "macro-hud-frame-v1",
        "session_id": "fixture",
        "context_version": 0,
        "status": "waiting_for_game",
        "message": "Waiting for League of Legends",
        "detail": "Synthetic fixture; no game API is contacted.",
        "hud": None,
        "valid_for_ms": 5000,
        "context": {"team": 100, "actor": "active", "role": None},
        "players": [],
        "model": {
            "display_name": "native-smoke-fixture",
            "fingerprint": "fixture",
            "policy_count": 6,
            "policies": {
                scope: {"algorithm": "bc", "candidate": "fixture"} for scope in ("TEAM", *ROLES)
            },
        },
    }
    with HudServer(SimpleNamespace(frame=lambda: frame), 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            process = subprocess.Popen(
                [
                    str(executable),
                    f"--backend=http://127.0.0.1:{server.server_port}",
                    f"--smoke-report={windows_report}",
                    "--smoke-keep-open",
                    "--enable-logging=stderr",
                ],
                cwd=executable.parent,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 20
            while not report.is_file() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.05)
            assert report.is_file(), process.communicate(timeout=5)
            result = json.loads(report.read_text())
            assert result["platform"] == "win32"
            assert result["alwaysOnTop"] and result["visible"]
            assert not result["interactive"] and not result["focusable"]
            assert result["status"] == "waiting_for_game"
            assert result["renderer"]["hasHud"] and result["renderer"]["hasBridge"]
            assert not result["renderer"]["nodeExposed"]
            assert result["renderer"]["title"] == frame["message"]
            assert result["frameRequests"] <= 2
            print("Native HUD requests in 6.5 seconds:", result["frameRequests"])
            # Reproduce terminal interruption of the WSL proxy. Shutdown must reach
            # the native process independently of that proxy's return code.
            process.send_signal(signal.SIGINT)
            server.stop_desktop()
            process.wait(timeout=5)
            native_id = int(result["pid"])
            stopped = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    (
                        f"Wait-Process -Id {native_id} -Timeout 5 -ErrorAction SilentlyContinue; "
                        f"if (Get-Process -Id {native_id} -ErrorAction SilentlyContinue) {{exit 1}}"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert stopped.returncode == 0, stopped.stderr
        finally:
            server.stop_desktop()
            server.shutdown()
            thread.join(timeout=5)
