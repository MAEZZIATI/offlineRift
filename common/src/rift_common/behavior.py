"""Weak, interval-based executed-option reconstruction. No claim of inferred intent."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from rift_common.actions import Action
from rift_common.objectives import PITS


@dataclass(frozen=True)
class Behavior:
    action: int
    evidence: str


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
