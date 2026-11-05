from unittest.mock import Mock

import numpy as np
from offline_rift.inference.live_client import LiveClientInference, LiveStateAdapter
from rift_common.actions import ROLES
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


def test_active_player_is_not_silently_replaced_with_team():
    client = LiveClientInference(policy=Mock(), session=Mock())
    data = live_data()
    data["activePlayer"] = {}
    snapshot = client.adapter.snapshot(data)
    assert client.recommend(snapshot, 100)["status"] == "needs_player"
    client.policy.recommend.assert_not_called()
    client.close()
