from datetime import datetime, timezone

import pytest
from rift_common.artifacts import atomic_json
from rift_research.adherence import review_adherence


def session_fixture(tmp_path, *, action=12, stop=240, champions=None):
    session = tmp_path / "session"
    count = 0
    for moment in range(120, stop + 1, 5):
        count += 1
        players = champions or [{"slot": 0, "championName": "Example", "team": "ORDER"}]
        atomic_json(
            session / "samples" / f"{count:06d}.json",
            {
                "observed_utc": datetime.fromtimestamp(
                    1_800_000_000 + moment, timezone.utc
                ).isoformat(),
                "game": {"gameTime": moment},
                "active_slot": 0,
                "players": [
                    {**p, "isDead": False, "scores": {"creepScore": moment // 5}} for p in players
                ],
                "decision": {
                    "status": "recommendation",
                    "action_id": action,
                    "role": "TOP",
                    "subject_slot": 0,
                    "issued_game_time": moment,
                },
                "events": [
                    {
                        "EventName": "ChampionKill",
                        "EventTime": 150,
                        "KillerNameSlot": 0,
                        "VictimNameSlot": 5,
                    }
                ],
            },
        )
    atomic_json(session / "report.json", {"schema": "live-audit-session-v1", "samples": count})
    return session


def test_live_combat_is_consistency_not_proof_and_farming_lane_is_unverifiable(tmp_path):
    session = session_fixture(tmp_path)
    report = review_adherence(session, tmp_path / "review")
    assert report["counts"]["consistent_evidence"] == 1
    assert len(report["windows"]) == 3
    assert report["verified_following_rate"] is None
    with pytest.raises(FileExistsError):
        review_adherence(session, tmp_path / "review")
    other = tmp_path / "other"
    session = session_fixture(other, action=1)
    report = review_adherence(session, other / "review")
    assert all(row["status"] == "unverifiable" for row in report["windows"])
    assert report["windows"][0]["reason"] == "farming_evidence_present; lane_or_jungle_unverifiable"


def test_postgame_roster_clock_binding_and_measured_activity(tmp_path, match_payloads):
    match, timeline = match_payloads()
    match["info"]["gameStartTimestamp"] = 1_800_000_000_000
    players = [
        {
            "slot": p["participantId"] - 1,
            "championName": p["championName"],
            "team": "ORDER" if p["teamId"] == 100 else "CHAOS",
        }
        for p in match["info"]["participants"]
    ]
    players[0]["rawChampionName"] = "game_character_displayname_Champion0"
    players[0]["championName"] = "Localized display name"
    session = session_fixture(tmp_path, action=1, champions=players)
    raw = tmp_path / "match.json"
    atomic_json(raw, {"match": match, "timeline": timeline})
    report = review_adherence(session, tmp_path / "review", match=raw)
    assert report["windows"][0]["status"] == "supported"
    assert report["windows"][0]["match_interval"] == [120, 180]
    match["info"]["gameStartTimestamp"] -= 999_000
    atomic_json(raw, {"match": match, "timeline": timeline})
    with pytest.raises(ValueError, match="roster and game clock"):
        review_adherence(session, tmp_path / "wrong", match=raw)
