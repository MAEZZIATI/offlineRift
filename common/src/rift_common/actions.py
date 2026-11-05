"""Versioned next-behavior options, not instantaneous formation categories."""

from enum import IntEnum

import numpy as np

ACTION_SCHEMA = "next-action-v1"
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
GROUPS = (*ROLES, "UNKNOWN", "TEAM")


class Action(IntEnum):
    CONTINUE = 0
    FARM_TOP = 1
    FARM_MID = 2
    FARM_BOTTOM = 3
    CLEAR_JUNGLE = 4
    ROTATE_TOP = 5
    ROTATE_MID = 6
    ROTATE_BOTTOM = 7
    PREPARE_DRAGON = 8
    PREPARE_BARON = 9
    PREPARE_HERALD = 10
    RESET = 11
    SKIRMISH = 12
    WAIT_RESPAWN = 13
    ROTATE_JUNGLE = 14


ACTION_NAMES = dict(
    enumerate(
        (
            "Continue current activity",
            "Farm top lane",
            "Farm mid lane",
            "Farm bottom lane",
            "Clear jungle camps",
            "Rotate toward top",
            "Rotate toward mid",
            "Rotate toward bottom",
            "Prepare the dragon area",
            "Prepare the Baron area",
            "Prepare the Herald area",
            "Return to base and regroup",
            "Participate in a nearby fight",
            "Wait to respawn",
            "Rotate through jungle",
        )
    )
)


def group_index(team_scope, role):
    return GROUPS.index("TEAM" if team_scope else role if role in ROLES else "UNKNOWN")


def feasible_actions(observations, names):
    """Necessary conditions only; does not establish route safety or tactical value."""
    observed = np.atleast_2d(observations)
    allowed = np.ones((len(observed), len(Action)), dtype=bool)
    teams = observed[:, names.index("scope_team")] > 0
    allowed[teams, 1:5] = False
    allowed[:, Action.WAIT_RESPAWN] = False
    dead = (
        (observed[:, names.index("self_dead")] > 0)
        & (observed[:, names.index("self_dead_known")] > 0)
        & ~teams
    )
    allowed[dead] = False
    allowed[dead, Action.WAIT_RESPAWN] = True
    for action, kind in (
        (Action.PREPARE_DRAGON, "dragon"),
        (Action.PREPARE_BARON, "baron"),
        (Action.PREPARE_HERALD, "herald"),
    ):
        known = observed[:, names.index(f"{kind}_enabled")] > 0
        soon = observed[:, names.index(f"{kind}_eta")] <= 120 / 1200 + 1e-6
        allowed[:, action] &= known & soon
    return allowed


ACTION_DETAILS = {
    Action.CONTINUE: (
        "Continue your current task and reassess after a relevant event.",
        "Keep checking the map, waves and your team's plans.",
    ),
    Action.FARM_TOP: (
        "Stay with top-lane income: collect reachable minions without forcing a rotation.",
        "Check the wave, visible threats and whether an objective needs you.",
    ),
    Action.FARM_MID: (
        "Stay with mid-lane income: collect reachable minions without forcing a rotation.",
        "Check the wave, visible threats and whether an objective needs you.",
    ),
    Action.FARM_BOTTOM: (
        "Stay with bottom-lane income: collect reachable minions without forcing a rotation.",
        "Check the wave, visible threats and whether an objective needs you.",
    ),
    Action.CLEAR_JUNGLE: (
        "Work through available jungle camps while keeping a route to your next decision.",
        "Camp availability, lane pressure and route safety are not observed by this policy.",
    ),
    Action.ROTATE_TOP: (
        "Move toward top lane to reinforce or reposition; this is not a guaranteed gank opportunity.",
        "Check the top wave, visible opponents and the cost of leaving your current task.",
    ),
    Action.ROTATE_MID: (
        "Move toward mid lane to reinforce or reposition.",
        "Check the mid wave, visible opponents and the cost of leaving your current task.",
    ),
    Action.ROTATE_BOTTOM: (
        "Move toward bottom lane to reinforce or reposition; this is not a guaranteed gank opportunity.",
        "Check the bottom wave, visible opponents and the cost of leaving your current task.",
    ),
    Action.PREPARE_DRAGON: (
        "Approach or maintain presence around the dragon area for the upcoming window.",
        "Verify the objective, vision, teammates and a safe route before committing.",
    ),
    Action.PREPARE_BARON: (
        "Approach or maintain presence around the Baron area for the upcoming window.",
        "Verify the objective, vision, teammates and a safe route before committing.",
    ),
    Action.PREPARE_HERALD: (
        "Approach or maintain presence around the Herald area for the upcoming window.",
        "Verify the objective, vision, teammates and a safe route before committing.",
    ),
    Action.RESET: (
        "Return toward your base to buy, recover or regroup; the model does not select purchases.",
        "Check whether the wave and nearby objectives permit leaving now.",
    ),
    Action.SKIRMISH: (
        "Consider joining or responding to a nearby fight where you can contribute.",
        "A fight being safe, reachable or favorable is not established by the model.",
    ),
    Action.WAIT_RESPAWN: (
        "Wait until you respawn, then reassess your next destination.",
        "The policy should not recommend a voluntary move while you are dead.",
    ),
    Action.ROTATE_JUNGLE: (
        "Reposition through jungle toward your next task, without assuming camps or paths are safe.",
        "Check vision, visible opponents and lane priorities yourself.",
    ),
}


def action_catalog():
    return {
        int(action): {
            "name": ACTION_NAMES[action],
            "instruction": ACTION_DETAILS[action][0],
            "check": ACTION_DETAILS[action][1],
            "nominal_seconds": 60,
        }
        for action in Action
    }
