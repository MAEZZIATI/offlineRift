from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock

import pytest
import requests
from offline_rift.inference.transport import LiveClientUnavailable, LiveDataSource


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(
        "offline_rift.inference.transport.platform.release", lambda: "WSL2-Microsoft"
    )
    monkeypatch.setattr(
        "offline_rift.inference.transport.shutil.which", lambda _: "/windows/curl.exe"
    )
    run = Mock(return_value=Mock(stdout=json.dumps({"gameData": {"gameTime": 120}}) + "\n200"))
    monkeypatch.setattr("offline_rift.inference.transport.subprocess.run", run)
    return run


def test_wsl_reads_windows_loopback_with_bounded_proxy_free_command(windows):
    source = LiveDataSource()
    assert source.transport == "windows"
    assert source.fetch()["gameData"]["gameTime"] == 120
    command = windows.call_args.args[0]
    assert command[:2] == ["/windows/curl.exe", "--disable"]
    assert command[command.index("--noproxy") + 1] == "*"
    assert command[-1] == "https://127.0.0.1:2999/liveclientdata/allgamedata"
    assert "--insecure" in command
    assert "--location" not in command
    assert windows.call_args.kwargs["timeout"] == 4
    assert "shell" not in windows.call_args.kwargs
    source.close()


@pytest.mark.parametrize(
    "failure",
    [
        OSError("interop disabled"),
        subprocess.TimeoutExpired("curl.exe", 4),
        subprocess.CalledProcessError(7, "curl.exe"),
    ],
)
def test_windows_errors_are_recoverable_offline_states(windows, failure):
    windows.side_effect = [failure, Mock(stdout='{"gameData": {}}\n200')]
    source = LiveDataSource()
    with pytest.raises(LiveClientUnavailable, match="Windows interop"):
        source.fetch()
    assert source.fetch() == {"gameData": {}}


@pytest.mark.parametrize("output", ["{}\n302", "[]\n200", "not json\n200", "{}\n500"])
def test_windows_rejects_redirects_and_invalid_responses(windows, output):
    windows.return_value.stdout = output
    with pytest.raises(LiveClientUnavailable):
        LiveDataSource().fetch()


def test_explicit_local_and_injected_sessions_do_not_use_windows(windows):
    session = Mock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.return_value = {"gameData": {}}
    source = LiveDataSource(session=session)
    assert source.transport == "local"
    assert source.fetch() == {"gameData": {}}
    assert session.trust_env is False
    assert session.get.call_args.kwargs["allow_redirects"] is False
    session.get.side_effect = requests.ConnectionError()
    with pytest.raises(LiveClientUnavailable):
        source.fetch()
    windows.assert_not_called()
    assert LiveDataSource(transport="local").transport == "local"


def test_missing_windows_interop_and_native_auto(monkeypatch):
    monkeypatch.setattr("offline_rift.inference.transport.shutil.which", lambda _: None)
    monkeypatch.setattr("offline_rift.inference.transport.platform.release", lambda: "Linux")
    assert LiveDataSource().transport == "local"
    with pytest.raises(ValueError, match="interop"):
        LiveDataSource(transport="windows")


@pytest.mark.parametrize(
    "endpoint", ["https://example.com", "https://user@localhost", "file:///tmp/x"]
)
def test_transport_cannot_target_remote_urls(windows, endpoint):
    with pytest.raises(ValueError, match="local"):
        LiveDataSource(endpoint)
    windows.assert_not_called()


def test_windows_certificate_path_is_translated(windows):
    windows.side_effect = [Mock(stdout="C:\\certs\\riot.pem\n"), Mock(stdout="{}\n200")]
    assert LiveDataSource(verify_tls="/mnt/c/certs/riot.pem").fetch() == {}
    assert windows.call_args_list[0].args[0] == ["wslpath", "-w", "/mnt/c/certs/riot.pem"]
    command = windows.call_args.args[0]
    assert command[command.index("--cacert") + 1] == "C:\\certs\\riot.pem"
    assert "--insecure" not in command
