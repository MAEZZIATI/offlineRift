"""Weak, interval-based executed-option reconstruction. No claim of inferred intent."""

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from rift_common.actions import Action
from rift_common.objectives import PITS


@dataclass(frozen=True)
class Behavior:
    action: int
    evidence: str


@dataclass(frozen=True)
class ActivityEvidence:
    representative: Behavior
    candidates: tuple[int, ...]
    identified: bool
    death_participation: bool = False
    combat_credit: bool = False


def player_activity(start, end, player_id):
    """Preserve overlapping observations; death participation never identifies fight intent."""
    representative = player_behavior(start, end, player_id)
    if representative is None:
        return None
    if representative.action != Action.SKIRMISH:
        return ActivityEvidence(representative, (int(representative.action),), True)
    events = [
        e for e in end.events if e.get("kind") == "kill" and start.time < e["time"] <= end.time
    ]
    died = any(e.get("victim") == str(player_id) for e in events)
    credited = any(
        e.get("actor") == str(player_id) or str(player_id) in e.get("assists", []) for e in events
    )
    other = player_behavior(
        start,
        replace(end, events=tuple(e for e in end.events if e.get("kind") != "kill")),
        player_id,
    )
    candidates = {int(Action.SKIRMISH)}
    # Removing a death event must never invent a voluntary return to base.
    if (
        other is not None
        and other.action != Action.CONTINUE
        and not (died and other.action == Action.RESET)
    ):
        candidates.add(int(other.action))
    return ActivityEvidence(
        representative,
        tuple(sorted(candidates)),
        len(candidates) == 1 and not (died and not credited),
        died,
        credited,
    )


def team_activity(start, end, team):
    representative = team_behavior(start, end, team)
    players = [player_activity(start, end, p.id) for p in start.players if p.team == team]
    if representative is None or any(p is None for p in players):
        return None
    # Only coherent executed macro options identify a target. No consensus is
    # unresolved, not a positive "continue" strategy. Combat does not identify intent.
    identified = (
        representative.action not in {Action.CONTINUE, Action.SKIRMISH}
        and sum(p.identified and p.representative.action == representative.action for p in players)
        >= 3
    )
    return ActivityEvidence(
        representative,
        (int(representative.action),),
        identified,
        any(p.death_participation for p in players),
        any(p.combat_credit for p in players),
    )


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - b))


def zone(position, team):
    if position is None:
        return None
    x, y = position
    if not (np.isfinite([x, y]).all() and 0 <= x <= 15000 and 0 <= y <= 15000):
        return None
    base = (0, 0) if team == 100 else (15000, 15000)
    if distance(position, base) < 2500:
        return "base"
    if abs(x - y) <= 1600:
        return "mid"
    if y - x > 1600 and (x < 2300 or y > 12700):
        return "top"
    if x - y > 1600 and (y < 2300 or x > 12700):
        return "bottom"
    return "jungle"


def player_behavior(start, end, player_id):
    """Future frames identify the recorded action only; they never enter start-state features.

    Combat includes kills, assists and deaths, not just successful fights. Moving to
    another lane is a rotation, not a certified gank. Wave manipulation is unobserved.
    """
    player = start.player(player_id)
    after = end.player(player_id)
    if player.is_dead is True:
        return Behavior(Action.WAIT_RESPAWN, "dead_at_start")
    events = [e for e in end.events if start.time < e["time"] <= end.time]
    if any(
        e.get("kind") == "kill"
        and str(player_id) in [e.get("actor"), e.get("victim"), *e.get("assists", [])]
        for e in events
    ):
        return Behavior(Action.SKIRMISH, "combat_participation_including_deaths")
    source, target = zone(player.position, player.team), zone(after.position, player.team)
    if source is None or target is None:
        return None
    moved = distance(player.position, after.position)
    if target == "base" and source != "base":
        return Behavior(Action.RESET, "base_arrival_without_observed_death")
    for action, kind in (
        (Action.PREPARE_DRAGON, "dragon"),
        (Action.PREPARE_BARON, "baron"),
        (Action.PREPARE_HERALD, "herald"),
    ):
        objective = start.objectives.get(kind, {})
        if (
            objective.get("enabled")
            and objective.get("eta", 9999) <= 120
            and distance(after.position, PITS[kind]) <= 2200
            and (
                distance(player.position, PITS[kind]) - distance(after.position, PITS[kind]) >= 1000
                or distance(player.position, PITS[kind]) <= 2200
            )
        ):
            return Behavior(action, "approach_or_maintain_objective_area_over_interval")
    if source == target:
        if (
            source in {"top", "mid", "bottom"}
            and player.lane_cs is not None
            and after.lane_cs is not None
            and after.lane_cs > player.lane_cs
        ):
            action = {"top": Action.FARM_TOP, "mid": Action.FARM_MID, "bottom": Action.FARM_BOTTOM}[
                source
            ]
            return Behavior(action, "same_lane_band_and_lane_cs_increase")
        if (
            source == "jungle"
            and player.jungle_cs is not None
            and after.jungle_cs is not None
            and after.jungle_cs > player.jungle_cs
        ):
            return Behavior(Action.CLEAR_JUNGLE, "jungle_area_and_jungle_cs_increase")
        return Behavior(Action.CONTINUE, "same_coarse_area_over_interval")
    if moved >= 1500:
        action = {
            "top": Action.ROTATE_TOP,
            "mid": Action.ROTATE_MID,
            "bottom": Action.ROTATE_BOTTOM,
            "jungle": Action.ROTATE_JUNGLE,
        }.get(target)
        if action is not None:
            return Behavior(action, "observed_movement_to_new_area")
    return Behavior(Action.CONTINUE, "no_large_resolved_area_transition")


def team_behavior(start, end, team):
    observed = [player_behavior(start, end, p.id) for p in start.players if p.team == team]
    if any(value is None for value in observed):
        return None
    counts = Counter(value.action for value in observed)
    for action in (
        Action.SKIRMISH,
        Action.PREPARE_DRAGON,
        Action.PREPARE_BARON,
        Action.PREPARE_HERALD,
        Action.RESET,
        Action.ROTATE_TOP,
        Action.ROTATE_MID,
        Action.ROTATE_BOTTOM,
        Action.ROTATE_JUNGLE,
    ):
        if counts[action] >= 3:
            return Behavior(action, "at_least_three_matching_player_options")
    return Behavior(Action.CONTINUE, "no_resolved_majority_team_transition")
