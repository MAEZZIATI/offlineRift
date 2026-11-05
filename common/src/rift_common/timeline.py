"""Causal reconstruction at real frame timestamps, preserving episode boundaries."""

from collections import Counter
from itertools import pairwise

from rift_common.objectives import ObjectiveRules, ObjectiveTracker
from rift_common.state import Observation, PlayerState


def read_match(match, timeline):
    info = match["info"]
    ObjectiveRules.validate_patch(info.get("gameVersion", ""))
    participants = info["participants"]
    if info.get("mapId") != 11 or info.get("queueId") != 420 or len(participants) != 10:
        raise ValueError("Only completed ranked Summoner's Rift 5v5 matches are supported")
    if len({p["participantId"] for p in participants}) != 10 or any(
        sum(p["teamId"] == t for p in participants) != 5 for t in (100, 200)
    ):
        raise ValueError("Invalid participant roster")
    if {bool(p["win"]) for p in participants if p["teamId"] == 100} not in ({True}, {False}):
        raise ValueError("Inconsistent match result")
    if any(
        len({bool(p["win"]) for p in participants if p["teamId"] == team}) != 1
        for team in (100, 200)
    ) or bool(participants[0]["win"]) == bool(
        next(p for p in participants if p["teamId"] != participants[0]["teamId"])["win"]
    ):
        raise ValueError("Inconsistent team outcome")
    match_id = match["metadata"]["matchId"]
    if timeline.get("metadata", {}).get("matchId", match_id) != match_id:
        raise ValueError("Timeline/match ID mismatch")
    frames = timeline["info"]["frames"]
    times = [f["timestamp"] / 1000 for f in frames]
    if len(frames) < 3 or times[0] != 0 or any(b <= a for a, b in pairwise(times)):
        raise ValueError("Need complete, strictly ordered real frames starting at zero")
    events = sorted((e for f in frames for e in f.get("events", [])), key=lambda e: e["timestamp"])
    end_events = [e["timestamp"] / 1000 for e in events if e["type"] == "GAME_END"]
    end = min(end_events) if end_events else float(info["gameDuration"])
    if abs(times[-1] - end) > 1:
        raise ValueError("Incomplete terminal frame; no extrapolated action labels")
    by_id = {p["participantId"]: p for p in participants}
    inventories = {pid: Counter() for pid in by_id}
    known = dict.fromkeys(by_id, True)
    scores = {pid: Counter(kills=0, deaths=0, assists=0) for pid in by_id}
    tracker = ObjectiveTracker(ObjectiveRules())
    normalized = []
    observations = []
    event_index = 0
    for frame, timestamp in zip(frames, times):
        while event_index < len(events) and events[event_index]["timestamp"] / 1000 <= timestamp:
            e = events[event_index]
            event_index += 1
            kind = e["type"]
            pid = e.get("participantId")
            moment = e["timestamp"] / 1000
            if pid in inventories:
                inv = inventories[pid]
                if kind == "ITEM_PURCHASED":
                    inv[e["itemId"]] += 1
                elif kind in {"ITEM_SOLD", "ITEM_DESTROYED"}:
                    item = e["itemId"]
                    if inv[item] <= 0:
                        known[pid] = False
                    inv[item] = max(0, inv[item] - 1)
                elif kind == "ITEM_UNDO":
                    before, after = e.get("beforeId", 0), e.get("afterId", 0)
                    if before:
                        if inv[before] <= 0:
                            known[pid] = False
                        inv[before] = max(0, inv[before] - 1)
                    if after:
                        inv[after] += 1
            if kind == "CHAMPION_KILL":
                killer, victim = e.get("killerId"), e.get("victimId")
                if killer in scores:
                    scores[killer]["kills"] += 1
                if victim in scores:
                    scores[victim]["deaths"] += 1
                for aid in e.get("assistingParticipantIds", []):
                    if aid in scores:
                        scores[aid]["assists"] += 1
                normalized.append(
                    {
                        "kind": "kill",
                        "time": moment,
                        "actor": str(killer),
                        "victim": str(victim),
                        "assists": [str(x) for x in e.get("assistingParticipantIds", [])],
                        "victim_team": by_id.get(victim, {}).get("teamId"),
                    }
                )
            elif kind == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
                normalized.append({"kind": "tower", "time": moment, "team": e.get("teamId")})
            elif kind == "ELITE_MONSTER_KILL":
                monster = {
                    "RIFTHERALD": "herald",
                    "RIFT_HERALD": "herald",
                    "BARON_NASHOR": "baron",
                    "DRAGON": "dragon",
                }.get(e.get("monsterType"), e.get("monsterType", "").lower())
                team = e.get("killerTeamId", by_id.get(e.get("killerId"), {}).get("teamId"))
                subtype = e.get("monsterSubType")
                tracker.consume(
                    monster,
                    moment,
                    team,
                    is_elder=None
                    if monster == "dragon" and subtype is None
                    else subtype == "ELDER_DRAGON",
                )
            elif kind == "ITEM_PURCHASED":
                normalized.append({"kind": "purchase", "time": moment, "actor": str(pid)})
        players = []
        for p in participants:
            pid = p["participantId"]
            pf = frame["participantFrames"].get(str(pid), {})
            stats = pf.get("championStats", {})
            pos = pf.get("position")
            hp, maximum = stats.get("health"), stats.get("healthMax")
            lane, jungle = pf.get("minionsKilled"), pf.get("jungleMinionsKilled")
            players.append(
                PlayerState(
                    id=str(pid),
                    team=p["teamId"],
                    champion=p.get("championName", ""),
                    role=p.get("teamPosition", ""),
                    level=pf.get("level"),
                    items=tuple(inventories[pid].elements()),
                    inventory_known=known[pid],
                    cs=lane + jungle if lane is not None and jungle is not None else None,
                    lane_cs=lane,
                    jungle_cs=jungle,
                    kills=scores[pid]["kills"],
                    deaths=scores[pid]["deaths"],
                    assists=scores[pid]["assists"],
                    health=hp / maximum if hp is not None and maximum else None,
                    is_dead=hp <= 0 if hp is not None else None,
                    position=(pos["x"], pos["y"]) if pos else None,
                )
            )
        observations.append(
            Observation(timestamp, tuple(players), tracker.state(timestamp), tuple(normalized))
        )
    return observations
