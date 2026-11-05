from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import requests
from offline_rift.inference.live_client import (
    LiveClientInference,
    LiveClientUnavailable,
    LiveStateAdapter,
)
from rift_common.pipeline.data_pipeline import TimelineProcessor
from rift_common.pipeline.features import ROLES, FeatureConfig, FeatureEncoder, ItemCatalog


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
    policy.recommend.return_value = {"action_id": 5, "action": "Group mid", "q_value": 0.2}
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


def test_corrected_offline_and_live_dragon_phase_match(match_payloads):
    import copy

    config = FeatureConfig.corrected_2026()
    match, timeline = match_payloads()
    match["info"].update(gameVersion="16.22.1", queueId=420, gameDuration=2100)
    match["info"].pop("gameEndTimestamp", None)
    match["info"].pop("gameStartTimestamp", None)
    template = copy.deepcopy(timeline["info"]["frames"][0])
    frames = []
    for t in range(0, 2101, 60):
        frame = copy.deepcopy(template)
        frame.update(timestamp=t * 1000, events=[])
        frames.append(frame)
    timeline["info"]["frames"] = frames
    live = live_payload(2000)
    for t, pid in ((310, 1), (620, 6), (930, 1), (1240, 6), (1550, 1), (1860, 1)):
        event = {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": t * 1000,
            "monsterType": "DRAGON",
            "monsterSubType": "FIRE_DRAGON",
            "killerId": pid,
        }
        frames[(t + 59) // 60]["events"].append(event)
        live["events"]["Events"].append(
            {
                "EventName": "DragonKill",
                "EventTime": t,
                "DragonType": "Fire",
                "KillerName": f"player{pid - 1}#EUW",
            }
        )
    offline = next(s for s in TimelineProcessor(config).parse(match, timeline) if s.time == 2000)
    observed = LiveStateAdapter(config).snapshot(live)
    assert offline.objectives == observed.objectives
    assert observed.objectives["dragon"]["eta"] == 220
    encoder = FeatureEncoder(config, ItemCatalog())
    np.testing.assert_array_equal(encoder.encode(offline, 100), encoder.encode(observed, 100))


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


def test_failed_policy_load_closes_transport(monkeypatch):
    session = Mock()
    monkeypatch.setattr(
        "offline_rift.inference.live_client.MacroPolicy",
        Mock(side_effect=RuntimeError("invalid policy")),
    )
    with pytest.raises(RuntimeError, match="invalid policy"):
        LiveClientInference("broken-bundle", session=session)
    session.close.assert_called_once()
