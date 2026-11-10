import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from offline_rift.inference.live_client import LiveClientUnavailable, LiveStateAdapter
from offline_rift.ui.server import HudServer
from offline_rift.ui.service import LiveHudService
from rift_common.actions import GROUPS, ROLES
from rift_common.metrics import behavior_metrics
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
    client.observe_once.side_effect = lambda: client.adapter.snapshot(client.fetch_live_data())
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
    return LiveHudService(client), clock


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


@pytest.mark.parametrize("role", ROLES)
def test_scoped_diagnostics_use_the_selected_role(dashboard, role):
    service, _ = dashboard
    metadata = service.client.policy.metadata
    metadata["advisory_groups"] = [role]
    report = metadata["training_report"]
    report["per_group"][role] = behavior_metrics([3, 3, 3, 12], [3, 3, -1, 12], 15)
    report["counts_by_group"] = [[1] * 15 for _ in range(7)]
    report["counts_by_group"][GROUPS.index(role)][3] = 100
    scoped = LiveHudService(service.client)
    assert scoped.model["behavior_agreement"] == 0.75
    assert scoped.model["majority_baseline"] == 0.75
    assert scoped.model["advisory_groups"] == [role]


def test_local_read_only_api_validates_context_and_keeps_assets_local(dashboard):
    service, _ = dashboard
    with HudServer(service, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = build_opener(ProxyHandler({}))
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            for asset in server.assets:
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


@pytest.mark.parametrize("role", ROLES)
def test_hud_pairs_actual_macro_and_selected_role_without_calibrated_metrics(dashboard, role):
    service, clock = dashboard
    frame = service.configure({"role": role})
    assert frame["status"] == "live"
    assert frame["hud"]["role"]["player_role"] == role
    assert frame["hud"]["macro"]["title"]
    assert frame["hud"]["telemetry"]["ai_confidence"] is None
    assert "winrate_impact" not in frame["hud"]["macro"]
    before = service.client.recommend.call_count
    clock[0] = 4
    assert service.frame()["hud"]["valid_for_ms"] == 6000
    assert service.client.recommend.call_count == before
    assert frame["context_version"] == 1


def test_conflict_withdraws_role_but_preserves_team_direction(dashboard):
    service, _ = dashboard
    macro = {"status": "recommendation", "action_id": 8, "action": "Prepare dragon"}
    service.client.recommend.side_effect = lambda *args: {
        "status": "coordination_conflict",
        "role": "TOP",
        "team": 100,
        "macro_recommendation": macro,
        "coordination": {"status": "withheld_conflict"},
        "action": "must never appear",
        "instruction": "must never appear",
    }
    payload = service.frame()["hud"]
    assert payload["macro"]["title"] == "Prepare dragon"
    assert payload["coordination"] == "withheld_conflict"
    assert payload["role"]["status"] == "withheld_conflict"
    assert payload["role"]["action"] == payload["role"]["details"] == ""


def test_missing_role_uses_team_policy_without_guessing(dashboard):
    service, _ = dashboard
    service.client.fetch_live_data.return_value = live_data(missing_roles=True)
    service.client.recommend.side_effect = lambda observation, team, actor, role: {
        "status": "recommendation" if actor == "team" else "needs_role",
        "team": team,
        "action_id": 6,
        "role": "" if actor != "team" else "TEAM",
    }
    frame = service.frame()
    assert frame["hud"]["role"] is None
    assert frame["hud"]["macro"]["title"]
    assert "Choose your role" in frame["hud"]["notice"]


def test_team_abstention_does_not_produce_an_empty_live_hud(dashboard):
    service, _ = dashboard
    service.client.recommend.side_effect = lambda *args: {"status": "insufficient_support"}
    frame = service.configure({"actor": "team"})
    assert frame["status"] == "insufficient_support"
    assert frame["hud"] is None


def test_restart_changes_session_and_offline_withdraws_hud(dashboard):
    service, clock = dashboard
    first = service.frame()
    clock[0] = 11
    service.client.fetch_live_data.return_value = live_data(100)
    restarted = service.frame()
    assert first["session_id"] != restarted["session_id"]
    clock[0] = 22
    service.client.fetch_live_data.side_effect = LiveClientUnavailable("game ended")
    frame = service.frame()
    assert frame["hud"] is None and frame["status"] == "waiting_for_game"


def test_observed_player_selects_their_actual_team_and_explicit_role(dashboard):
    service, _ = dashboard
    service.frame()
    frame = service.configure({"actor": "Player6#TEST", "role": "UTILITY"})
    assert frame["context"] == {"actor": "Player6#TEST", "team": 200, "role": "UTILITY"}
    for invalid in ({"role": "ADC"}, {"team": True}, {"actor": "guessed"}, {"model": "other"}):
        with pytest.raises(ValueError):
            service.configure(invalid)
