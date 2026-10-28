from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import requests

from data_pipeline import TimelineProcessor
from features import ROLES, FeatureConfig, FeatureEncoder, ItemCatalog
from live_client import LiveClientInference, LiveClientUnavailable, LiveStateAdapter


def live_payload(timestamp=0):
    return {
        "gameData": {"mapNumber": 11, "gameTime": timestamp},
        "allPlayers": [
            {
                "riotId": f"player{i}#EUW",
                "summonerName": f"player{i}",
                "team": "ORDER" if i < 5 else "CHAOS",
                "position": ROLES[i % 5],
                "level": 1,
                "items": [],
            }
            for i in range(10)
        ],
        "events": {"Events": []},
    }


def fake_policy(profile="live"):
    policy = Mock()
    policy.config = FeatureConfig(profile=profile)
    policy.encoder = FeatureEncoder(policy.config, ItemCatalog())
    policy.recommend.return_value = {"action_id": 3, "action": "Group & Push Mid", "q_value": 0.2}
    return policy


def test_offline_live_feature_parity(match_payloads):
    config = FeatureConfig()
    match, timeline = match_payloads()
    offline = TimelineProcessor(config).parse(match, timeline)[0]
    live = LiveStateAdapter(config).snapshot(live_payload())
    encoder = FeatureEncoder(config, ItemCatalog())
    for team in (100, 200):
        np.testing.assert_array_equal(encoder.encode(offline, team), encoder.encode(live, team))


def test_live_events_idempotence_and_new_game():
    data = live_payload(1400)
    data["events"]["Events"] = [
        {"EventName": "BaronKill", "EventTime": 1300, "KillerName": "player0#EUW"}
    ]
    adapter = LiveStateAdapter(FeatureConfig())
    for _ in range(2):
        state = adapter.snapshot(data)
        assert state.objectives["baron"]["eta"] == 260
        assert state.objectives["baron_buff"]["team"] == 100
    assert adapter.snapshot(live_payload(0)).objectives["baron"]["eta"] == 1200


def test_offline_endpoint_and_recovery_without_stale_recommendations():
    session, policy = Mock(), fake_policy()
    response = Mock(status_code=200)
    response.json.return_value = live_payload()
    session.get.side_effect = [requests.ConnectionError(), response]
    client = LiveClientInference(policy=policy, session=session, interval_seconds=0.001)
    results = []
    client.run(results.append, max_polls=2)
    assert results[0]["status"] == "offline"
    assert "action_id" not in results[0]
    assert results[1]["status"] == "partial_observation"
    assert results[1]["q_value"] == 0.2
    assert policy.recommend.call_count == 1
    session.close.assert_called_once()


def test_local_endpoint_only_and_full_policy_rejected():
    with pytest.raises(ValueError, match="local"):
        LiveClientInference(policy=fake_policy(), endpoint="https://example.com/")
    with pytest.raises(ValueError, match="profile"):
        LiveClientInference(policy=fake_policy("full"))
    # The requested HTTP endpoint remains supported as an explicit override.
    client = LiveClientInference(
        policy=fake_policy(),
        session=Mock(),
        endpoint="http://127.0.0.1:2999/liveclientdata/allgamedata",
    )
    assert client.endpoint.startswith("http:")


def test_malformed_live_json():
    session = Mock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.side_effect = ValueError()
    with pytest.raises(LiveClientUnavailable):
        LiveClientInference(policy=fake_policy(), session=session).fetch_live_data()
