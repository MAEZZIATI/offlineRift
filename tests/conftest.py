from __future__ import annotations

import pytest

from features import ROLES


@pytest.fixture
def match_payloads():
    def make(match_id="EUW1_1", win=True):
        participants = [
            {
                "participantId": i + 1,
                "teamId": 100 if i < 5 else 200,
                "teamPosition": ROLES[i % 5],
                "win": win if i < 5 else not win,
            }
            for i in range(10)
        ]
        match = {
            "metadata": {"matchId": match_id},
            "info": {"participants": participants, "mapId": 11, "gameDuration": 125},
        }
        frames = []
        for timestamp in (0, 60000, 120000, 125000):
            frames.append(
                {
                    "timestamp": timestamp,
                    "participantFrames": {
                        str(i + 1): {
                            "totalGold": 500 + timestamp / 100,
                            "level": 1 + timestamp // 60000,
                            "position": {"x": 7000 + i * 10, "y": 7000 + i * 10},
                        }
                        for i in range(10)
                    },
                    "events": [],
                }
            )
        # Events often appear in a containing frame with a later timestamp.
        frames[1]["events"] = [
            {"type": "ITEM_PURCHASED", "timestamp": 15000, "participantId": 1, "itemId": 1001},
            {
                "type": "ITEM_UNDO",
                "timestamp": 25000,
                "participantId": 1,
                "beforeId": 1001,
                "afterId": 0,
            },
            {
                "type": "ELITE_MONSTER_KILL",
                "timestamp": 35000,
                "monsterType": "DRAGON",
                "killerId": 1,
            },
        ]
        frames[-1]["events"] = [{"type": "GAME_END", "timestamp": 125000}]
        timeline = {"info": {"frames": frames, "frameInterval": 60000}}
        return match, timeline

    return make
