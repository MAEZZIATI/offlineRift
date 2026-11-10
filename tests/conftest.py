"""Deterministic real-format fixtures; never used as research training data."""

import pytest
from rift_common.actions import ROLES
from rift_common.items import ItemCatalog


@pytest.fixture
def catalog():
    return ItemCatalog({"1001": {"gold": {"total": 300}}, "2001": {"gold": {"total": 1200}}})


@pytest.fixture
def match_payloads():
    def make(match_id="KR_test", win=True):
        participants = [
            {
                "participantId": i + 1,
                "teamId": 100 if i < 5 else 200,
                "teamPosition": ROLES[i % 5],
                "championName": f"Champion{i}",
                "puuid": f"player-{i + 1}",
                "win": win if i < 5 else not win,
            }
            for i in range(10)
        ]
        positions = [(1400, 9000), (4500, 8500), (7000, 7000), (9500, 1500), (10500, 1500)]
        frames = []
        for t in range(0, 481, 60):
            players = {}
            for i in range(10):
                x, y = positions[i % 5]
                players[str(i + 1)] = {
                    "participantId": i + 1,
                    "position": {"x": x, "y": y},
                    "level": 1 + t // 120,
                    "minionsKilled": t // 10 if i % 5 != 1 else 0,
                    "jungleMinionsKilled": t // 10 if i % 5 == 1 else 0,
                    "currentGold": 300,
                    "totalGold": 500 + t * 2,
                    "championStats": {
                        "health": 0 if i == 0 and t == 300 else 800,
                        "healthMax": 1000,
                    },
                }
            events = []
            if t == 60:
                events.append(
                    {
                        "type": "ITEM_PURCHASED",
                        "timestamp": 15000,
                        "participantId": 1,
                        "itemId": 1001,
                    }
                )
            if t == 300:
                events.append(
                    {
                        "type": "CHAMPION_KILL",
                        "timestamp": 270000,
                        "killerId": 6,
                        "victimId": 1,
                        "assistingParticipantIds": [7],
                    }
                )
            if t == 480:
                events.append({"type": "GAME_END", "timestamp": 480000})
            frames.append({"timestamp": t * 1000, "participantFrames": players, "events": events})
        return {
            "metadata": {"matchId": match_id},
            "info": {
                "participants": participants,
                "queueId": 420,
                "mapId": 11,
                "gameVersion": "16.22.1",
                "gameDuration": 480,
            },
        }, {"metadata": {"matchId": match_id}, "info": {"frames": frames}}

    return make
