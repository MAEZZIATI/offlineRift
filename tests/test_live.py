from unittest.mock import Mock

import numpy as np
import pytest
from offline_rift.inference.live_client import LiveClientInference, LiveStateAdapter
from rift_common.actions import ROLES
from rift_common.artifacts import read_json
from rift_common.state import StateEncoder
from rift_common.timeline import read_match


def live_data(t=120, missing_roles=False):
    return {
        "gameData": {"gameTime": t, "mapNumber": 11},
        "activePlayer": {
            "riotId": "Player1#TEST",
            "championStats": {"currentHealth": 800, "maxHealth": 1000},
        },
        "allPlayers": [
            {
                "riotId": f"Player{i + 1}#TEST",
                "championName": f"Champion{i}",
                "team": "ORDER" if i < 5 else "CHAOS",
                "position": "" if missing_roles else ROLES[i % 5],
                "level": 1 + t // 120,
                "items": [{"itemID": 1001}] if i == 0 and t >= 15 else [],
                "isDead": False,
                "scores": {"creepScore": t // 10, "kills": 0, "deaths": 0, "assists": 0},
            }
            for i in range(10)
        ],
        "events": {"Events": [{"EventID": 0, "EventName": "GameStart", "EventTime": 0}]},
    }


def test_training_live_feature_parity_and_role_identity(match_payloads, catalog):
    offline = read_match(*match_payloads())
    adapter = LiveStateAdapter()
    adapter.snapshot(live_data(60))
    current = adapter.snapshot(live_data(120))
    assert current.active_player == "Player1#TEST"
    assert current.player(current.active_player).role == "TOP"
    assert all(p.position is None for p in current.players)
    encoder = StateEncoder(catalog)
    np.testing.assert_array_equal(
        encoder.encode(current, 100, current.active_player, previous=adapter.previous),
        encoder.encode(offline[2], 100, "1", previous=offline[1]),
    )
    # Unreported roles stay unavailable; no champion-name or future-position inference.
    current = adapter.snapshot(live_data(180, missing_roles=True))
    assert current.player(current.active_player).role == ""
    assert (
        encoder.encode(current, 100, current.active_player)[encoder.names.index("role_known")] == 0
    )


def test_history_rollbacks_duplicate_clock_and_incomplete_events():
    adapter = LiveStateAdapter()
    adapter.snapshot(live_data(60))
    adapter.snapshot(live_data(120))
    adapter.snapshot(live_data(120))
    assert len(adapter.history) == 2 and adapter.previous.time == 60
    adapter.snapshot(live_data(0))
    assert len(adapter.history) == 1 and adapter.previous is None
    data = live_data(600)
    data["events"] = {"Events": []}
    current = adapter.snapshot(data)
    assert not current.history_complete
    assert not current.objectives["dragon"]["enabled"]


def test_future_history_marker_cannot_enable_objectives_and_game_end_clears_advice():
    data = live_data(600)
    data["events"]["Events"] = [{"EventName": "GameStart", "EventTime": 700}]
    current = LiveStateAdapter().snapshot(data)
    assert not current.history_complete and not current.objectives["dragon"]["enabled"]
    data["events"]["Events"] = [{"EventName": "GameEnd", "EventTime": 599}]
    with pytest.raises(ValueError, match="Game has ended"):
        LiveStateAdapter().snapshot(data)


def test_active_player_is_not_silently_replaced_with_team():
    client = LiveClientInference(policy=Mock(), session=Mock())
    data = live_data()
    data["activePlayer"] = {}
    snapshot = client.adapter.snapshot(data)
    assert client.recommend(snapshot, 100)["status"] == "needs_player"
    client.policy.recommend.assert_not_called()
    client.close()


def test_opt_in_availability_capture_masks_history_and_excludes_identity(tmp_path, catalog):
    policy = Mock(encoder=StateEncoder(catalog))
    client = LiveClientInference(
        policy=policy, session=Mock(), capture_availability=tmp_path / "capture"
    )
    first = client.adapter.snapshot(live_data(120))
    client.recommend(first, 100)
    client.recommend(first, 100)
    later = client.adapter.snapshot(live_data(180))
    client.recommend(later, 100, role="TOP")
    report = read_json(tmp_path / "capture/report.json")
    assert report["polls"] == 2 and report["duplicate_polls"] == 1
    assert report["known_feature_counts"]["self_cs_rate_known"] == 1
    assert report["observation_sessions"] == 1
    assert len(report["cells"]) == 2
    stored = (tmp_path / "capture/report.json").read_text()
    assert "Player1" not in stored and "Champion" not in stored
    assert "recommendation" not in report and "observations" not in report
    reset = client.adapter.snapshot(live_data(0))
    client.recommend(reset, 100)
    client.close(capture_status="interrupted")
    report = read_json(tmp_path / "capture/report.json")
    assert report["status"] == "interrupted" and report["observation_sessions"] == 2
    with pytest.raises(FileExistsError):
        LiveClientInference(
            policy=policy, session=Mock(), capture_availability=tmp_path / "capture"
        )


def test_capture_is_disabled_by_default_and_interrupt_retains_progress(
    tmp_path, catalog, monkeypatch
):
    client = LiveClientInference(policy=Mock(), session=Mock())
    assert client.capture is None
    client.close()
    client = LiveClientInference(
        policy=Mock(encoder=StateEncoder(catalog)),
        session=Mock(),
        capture_availability=tmp_path / "capture",
    )

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "fetch_live_data", interrupt)
    client.run(max_polls=1)
    report = read_json(tmp_path / "capture/report.json")
    assert report["status"] == "interrupted" and report["polls"] == 0


def test_session_capture_explicit_champion_adc_and_name_redaction(tmp_path, catalog):
    policy = Mock(encoder=StateEncoder(catalog))
    policy.name = "fixture"
    policy.metadata = {
        "algorithm": "cql",
        "training_report": {
            "action_counts": [10] * 15,
            "majority_baseline": 0.3,
            "validation": {
                "prediction_counts": [10] * 15,
                "behavior_agreement": 0.5,
                "balanced_accuracy": 0.5,
                "macro_f1": 0.5,
                "collapsed": False,
            },
        },
    }
    policy.recommend.return_value = {
        "status": "recommendation",
        "actor": "Player1#TEST",
        "team": 100,
        "role": "BOTTOM",
        "action_id": 3,
        "scores": [None] * 15,
    }
    data = live_data(120, missing_roles=True)
    data["allPlayers"][0]["championName"] = "Ezreal"
    data["activePlayer"]["currentGold"] = 1200
    data["events"]["Events"].append(
        {
            "EventName": "ChampionKill",
            "EventTime": 110,
            "KillerName": "Player1#TEST",
            "VictimName": "Player6#TEST",
        }
    )
    session = Mock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.return_value = data
    client = LiveClientInference(
        policy=policy,
        session=session,
        team=100,
        actor="champion:Ezreal",
        role="BOTTOM",
        capture_session=tmp_path / "session",
    )
    from offline_rift.ui.service import LiveHudService

    assert LiveHudService(client).state()["recommendation"]["role"] == "BOTTOM"
    saved = read_json(tmp_path / "session/samples/000001.json")
    assert saved["active_current_gold"] == 1200
    assert saved["active_slot"] == 0
    assert saved["features"]["role_BOTTOM"] == 1
    assert saved["events"][1]["KillerNameSlot"] == 0
    assert "Player1" not in (tmp_path / "session/samples/000001.json").read_text()
    # User-selected champion must exist on the requested side.
    data["allPlayers"][0]["championName"] = "Other"
    with pytest.raises(ValueError, match="Waiting for selected champion"):
        client.infer_once()
    assert read_json(tmp_path / "session/report.json")["samples"] == 1
    data["allPlayers"][0]["championName"] = "Ezreal"
    data["events"]["Events"].append({"EventName": "GameEnd", "EventTime": 120, "Result": "Win"})
    with pytest.raises(ValueError, match="Game has ended"):
        client.infer_once()
    client.close()
    assert read_json(tmp_path / "session/report.json")["game_end_observed"]
    assert read_json(tmp_path / "session/samples/000002.json")["events"][-1]["Result"] == "Win"
    with pytest.raises(FileExistsError):
        LiveClientInference(policy=policy, session=Mock(), capture_session=tmp_path / "session")
    from rift_research.session_audit import audit_session

    audited = audit_session(tmp_path / "session", tmp_path / "audit", reported_outcome="win")
    assert audited["samples"] == 2
    assert audited["game_end_observed"]
    assert audited["self_reported_outcome"] == "win"
    assert audited["policy_winrate"] is None
    assert len(audited["sample_sha256"]) == 2
    with pytest.raises(FileExistsError):
        audit_session(tmp_path / "session", tmp_path / "audit")


def test_swiftplay_capture_uses_ranked_policy_for_summoners_rift(tmp_path, catalog):
    policy = Mock(encoder=StateEncoder(catalog))
    data = live_data()
    data["gameData"]["gameMode"] = "SWIFTPLAY"
    session = Mock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.return_value = data
    client = LiveClientInference(
        policy=policy, session=session, capture_session=tmp_path / "session"
    )
    client.infer_once()
    policy.recommend.assert_called_once()
    client.close()
    assert read_json(tmp_path / "session/report.json")["samples"] == 1
    assert (
        read_json(tmp_path / "session/samples/000001.json")["decision"]["status"] == "invalid_state"
    )
    from rift_research.session_audit import audit_session

    audit = audit_session(tmp_path / "session", tmp_path / "audit")
    assert audit["mode_mismatch"] and not audit["game_end_observed"]
    assert audit["self_reported_outcome"] is None
    assert audit["voluntary_dominant_fraction"] == 0
