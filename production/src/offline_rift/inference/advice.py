"""Persist eligible advice briefly; withdraw on lost support, abstention or death."""

from copy import deepcopy

from rift_common.coaching import alignment


class AdviceController:
    def __init__(self, *, minimum_hold=20, confirmations=2):
        self.minimum_hold, self.confirmations = minimum_hold, confirmations
        self.contexts = {}

    def select(self, result):
        result = deepcopy(result)
        if "game_time" not in result:
            return result
        key = (result.get("team"), result.get("actor"), result.get("role"))
        now = result["game_time"]
        old = self.contexts.get(key)
        voluntary = result.get("status") == "recommendation"
        if not voluntary or old is None or now < old["seen"] or now >= old["expires"]:
            return self._replace(key, result)
        action = result.get("action_id")
        supported = result.get("scores", [])
        previous = old["result"]["action_id"]
        if previous >= len(supported) or supported[previous] is None:
            return self._replace(key, result)
        if action == previous:
            old["pending"], old["count"] = None, 0
        elif now > old["seen"]:
            old["count"] = old["count"] + 1 if old["pending"] == action else 1
            old["pending"] = action
            if now - old["issued"] >= self.minimum_hold and old["count"] >= self.confirmations:
                return self._replace(key, result)
        old["seen"] = now
        held = deepcopy(old["result"])
        held.update(
            game_time=now,
            scores=result.get("scores"),
            advice_issued_game_time=old["issued"],
            advice_expires_game_time=old["expires"],
            retained_advice=action != previous,
        )
        if "macro_recommendation" in result:
            held["macro_recommendation"] = result["macro_recommendation"]
            held["coordination"] = {
                **result.get("coordination", {}),
                "status": alignment(result["macro_recommendation"], held),
            }
        return held

    def _replace(self, key, result):
        now = result["game_time"]
        result.update(
            advice_issued_game_time=now,
            advice_expires_game_time=now + result.get("horizon_seconds", 60),
            retained_advice=False,
        )
        if result.get("status") == "recommendation":
            self.contexts[key] = {
                "result": deepcopy(result),
                "issued": now,
                "seen": now,
                "expires": result["advice_expires_game_time"],
                "pending": None,
                "count": 0,
            }
        else:
            self.contexts.pop(key, None)
        return result
