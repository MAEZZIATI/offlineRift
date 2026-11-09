"""Shared six-policy scopes and conservative symbolic direction comparison."""

from rift_common.actions import ROLES, Action

POLICY_SCOPES = ("TEAM", *ROLES)
DESTINATIONS = frozenset(
    int(a)
    for a in (
        Action.ROTATE_TOP,
        Action.ROTATE_MID,
        Action.ROTATE_BOTTOM,
        Action.PREPARE_DRAGON,
        Action.PREPARE_BARON,
        Action.PREPARE_HERALD,
    )
)


def alignment(macro, specialist):
    if macro.get("status") != "recommendation" or specialist.get("status") != "recommendation":
        return "unresolved"
    first, second = macro.get("action_id"), specialist.get("action_id")
    if first == second:
        return "aligned"
    if first in DESTINATIONS and second in DESTINATIONS:
        return "conflict"
    return "independent"
