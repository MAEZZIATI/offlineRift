from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from offline_rift.inference.live_client import LiveClientUnavailable
from offline_rift.ui.server import DashboardServer, run_ui
from offline_rift.ui.service import LiveDashboard


@pytest.fixture
def dashboard(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "offline_rift.ui.service.time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]),
    )
    client = Mock(interval_seconds=10, team=100)
    client.policy.metadata = {
        "training_report": {
            "validation_prediction_counts": [0, 0, 0, 0, 20],
            "action_counts": [1, 1, 1, 1, 96],
            "validation_heuristic_agreement": 0.85,
            "validation_majority_baseline": 0.85,
        }
    }
    client.source.transport = "windows"
    client.recommend.side_effect = lambda snapshot, team: {
        "action_id": 4,
        "action": "Defensive Reset & Setup Vision",
        "team": team,
        "game_time": 120,
        "q_values": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    return LiveDashboard(client), clock


def test_tabs_and_teams_share_one_snapshot_and_model_inference(dashboard):
    service, clock = dashboard
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(lambda _: service.state(), range(8)))
    service.client.fetch_live_data.assert_called_once()
    service.client.adapter.snapshot.assert_called_once()
    assert service.client.recommend.call_count == 2
    assert states[0]["recommendations"]["100"]["team"] == 100
    assert states[0]["recommendations"]["200"]["team"] == 200
    assert states[0]["model"]["collapsed"] is True
    assert states[0]["model"]["action_coverage"] == 5
    assert states[0]["valid_for_ms"] == 10000
    states[0]["recommendations"].clear()
    assert service.state()["recommendations"]  # A caller cannot corrupt the cache.
    clock[0] = 9
    assert service.state()["valid_for_ms"] == 1000
    assert service.client.fetch_live_data.call_count == 1


@pytest.mark.parametrize(
    "error,status",
    [
        (LiveClientUnavailable("no game"), "offline"),
        (ValueError("bad map"), "invalid_state"),
        (RuntimeError("failed model"), "error"),
    ],
)
def test_expired_success_is_replaced_by_failure_then_recovers(dashboard, error, status):
    service, clock = dashboard
    assert service.state()["status"] == "live"
    clock[0] = 11
    service.client.fetch_live_data.side_effect = error
    failed = service.state()
    assert failed["status"] == status
    assert failed["recommendations"] == {}
    assert "action_id" not in failed
    clock[0] = 22
    service.client.fetch_live_data.side_effect = None
    assert service.state()["status"] == "live"
    service.close()
    service.client.close.assert_called_once()


def test_browser_assets_and_read_only_api_are_local(dashboard):
    service, _ = dashboard
    with DashboardServer(service, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        opener = build_opener(ProxyHandler({}))
        try:
            for path, content_type in [
                ("/", "text/html"),
                ("/app.js", "text/javascript"),
                ("/app.css", "text/css"),
            ]:
                with opener.open(origin + path) as response:
                    assert response.status == 200
                    assert content_type in response.headers["Content-Type"]
                    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
                    assert response.read()
            service.client.fetch_live_data.assert_not_called()  # Serving assets does no inference.
            with opener.open(origin + "/api/state") as response:
                state = json.load(response)
                assert state["status"] == "live"
                assert response.headers["Cache-Control"] == "no-store"
                assert "Access-Control-Allow-Origin" not in response.headers
            for request, code in [
                (Request(origin + "/api/state", headers={"Host": "foreign.example"}), 403),
                (
                    Request(origin + "/api/state", headers={"Origin": "https://foreign.example"}),
                    403,
                ),
                (Request(origin + "/api/state", method="POST", data=b"{}"), 501),
                (Request(origin + "/../workflow.json"), 404),
            ]:
                with pytest.raises(HTTPError) as error:
                    opener.open(request)
                assert error.value.code == code
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("interval", [float("nan"), float("inf"), 0, 1])
def test_invalid_intervals_fail_before_model_loading(interval):
    with pytest.raises(ValueError, match="at least 2"):
        run_ui("unused-model", interval=interval)


def test_server_startup_failure_closes_client(dashboard, monkeypatch):
    pytest.importorskip("torch")
    service, _ = dashboard
    monkeypatch.setattr(
        "offline_rift.inference.live_client.LiveClientInference", Mock(return_value=service.client)
    )
    monkeypatch.setattr(
        "offline_rift.ui.server.DashboardServer", Mock(side_effect=OSError("port in use"))
    )
    with pytest.raises(OSError, match="port in use"):
        run_ui("unused-model")
    service.client.close.assert_called_once()
