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


def withhold_conflicting_role(result):
    """Keep the macro proposal visible; do not issue a contradictory destination.

    This is an explicit abstention rule, not learned coordination or an alternative
    action recommendation. Apply after persistence as well as initial routing.
    """
    macro = result.get("macro_recommendation")
    if not macro or alignment(macro, result) != "conflict":
        return result
    withheld = {**result}
    for key in ("action_id", "action", "instruction", "check", "scores", "score_meaning"):
        withheld.pop(key, None)
    withheld.update(
        status="coordination_conflict",
        message="Role direction withheld because it conflicts with the team proposal. Review the team direction against your map.",
        coordination={
            "status": "withheld_conflict",
            "withheld_action_id": result["action_id"],
            "meaning": "Serving constraint; no compatible alternative is inferred",
        },
    )
    return withheld
