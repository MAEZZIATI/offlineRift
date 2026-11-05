import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from offline_rift.inference.live_client import LiveClientUnavailable, LiveStateAdapter
from offline_rift.ui.server import DashboardServer
from offline_rift.ui.service import LiveDashboard
from test_live import live_data


@pytest.fixture
def dashboard(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "offline_rift.ui.service.time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]),
    )
    client = Mock(team=100, actor="active", role=None, interval_seconds=10)
    client.source.transport = "local"
    client.adapter = LiveStateAdapter()
    client.fetch_live_data.return_value = live_data()
    client.policy.name = "test-policy"
    client.policy.metadata = {
        "algorithm": "cql",
        "training_report": {
            "action_counts": [10] * 15,
            "validation": {
                "prediction_counts": [10] * 15,
                "behavior_agreement": 0.2,
                "balanced_accuracy": 0.2,
                "macro_f1": 0.2,
                "collapsed": False,
            },
            "majority_baseline": 0.3,
            "per_group": {},
        },
    }
    client.recommend.side_effect = lambda snapshot, team, actor, role: {
        "status": "recommendation",
        "action_id": 0,
        "team": team,
        "actor": actor,
        "role": role,
        "game_time": snapshot.time,
    }
    return LiveDashboard(client), clock


def test_subject_changes_share_snapshot_but_not_cached_advice(dashboard):
    service, clock = dashboard
    first = service.state(team=100, actor="team")
    second = service.state(team=100, actor="Player1#TEST", role="TOP")
    assert first["recommendation"]["actor"] == "team"
    assert second["recommendation"]["role"] == "TOP"
    service.client.fetch_live_data.assert_called_once()
    assert service.client.recommend.call_count == 2
    second["recommendation"].clear()
    assert service.state(team=100, actor="Player1#TEST", role="TOP")["recommendation"]
    clock[0] = 11
    service.client.fetch_live_data.side_effect = LiveClientUnavailable("no game")
    failure = service.state()
    assert failure["status"] == "offline" and "recommendation" not in failure
    clock[0] = 22
    service.client.fetch_live_data.side_effect = None
    assert service.state()["status"] == "live"


def test_local_read_only_api_validates_context_and_keeps_assets_local(dashboard):
    service, _ = dashboard
    with DashboardServer(service, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = build_opener(ProxyHandler({}))
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            for asset in ("/", "/app.js", "/app.css"):
                with opener.open(origin + asset) as response:
                    assert response.status == 200 and response.read()
                    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            with opener.open(origin + "/api/state?team=100&actor=team") as response:
                result = json.load(response)
                assert result["recommendation"]["actor"] == "team"
            for request, code in (
                (Request(origin + "/api/state?team=100&role=invalid"), 400),
                (Request(origin + "/api/state?team=100&team=200"), 400),
                (
                    Request(origin + "/api/state", headers={"Origin": "https://foreign.example"}),
                    403,
                ),
                (Request(origin + "/../workflow.json"), 404),
            ):
                with pytest.raises(HTTPError) as error:
                    opener.open(request)
                assert error.value.code == code
        finally:
            server.shutdown()
            thread.join(timeout=5)
