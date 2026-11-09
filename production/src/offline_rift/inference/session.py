"""Explicit opt-in measured live samples, with player names omitted."""

import math
from datetime import datetime, timezone
from pathlib import Path

from rift_common.artifacts import atomic_json, file_digest


def numbers(values):
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }


class SessionCapture:
    def __init__(self, output, *, model, team, actor, role):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.ended = False
        self.report = {
            "schema": "live-audit-session-v1",
            "status": "waiting",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(model),
            "team": team,
            "actor_selection": actor
            if actor in {"active", "team"} or actor.startswith("champion:")
            else "explicit_player",
            "role": role,
            "samples": 0,
            "poll_attempts": 0,
            "last_game_time": None,
            "limitations": [
                "Measured live polls, not Match-V5 training transitions",
                "No unseen positions, vision, waves, casts or cooldowns inferred",
                "Recorded recommendations do not establish adherence or policy outcomes",
            ],
        }
        if model is not None:
            self.report["bundle_sha256"] = file_digest(
                Path(model)
                / ("coach.json" if (Path(model) / "coach.json").exists() else "bundle.json")
            )
        self.save()

    def save(self):
        atomic_json(self.output / "report.json", self.report)

    def record(self, data):
        players = data.get("allPlayers", [])
        aliases = {}
        for index, player in enumerate(players):
            for key in ("riotId", "summonerName", "riotIdGameName"):
                if player.get(key):
                    aliases.setdefault(player[key], set()).add(index)

        def slot(name):
            matches = aliases.get(name, set())
            return next(iter(matches)) if len(matches) == 1 else None

        active = data.get("activePlayer", {})
        self.aliases = aliases
        active_slots = {slot(active.get(k)) for k in ("riotId", "summonerName", "riotIdGameName")}
        active_slots.discard(None)
        events = []
        for event in data.get("events", {}).get("Events", []):
            cleaned = {
                k: event[k]
                for k in (
                    "EventID",
                    "EventName",
                    "EventTime",
                    "DragonType",
                    "TurretKilled",
                    "InhibKilled",
                    "WinningTeam",
                    "Result",
                )
                if k in event and isinstance(event[k], (str, int, float, bool))
            }
            for key in ("KillerName", "VictimName", "Recipient"):
                if key in event:
                    cleaned[key + "Slot"] = slot(event[key])
            if "Assisters" in event:
                cleaned["AssisterSlots"] = [slot(name) for name in event["Assisters"]]
            events.append(cleaned)
        self.ended = any(
            event.get("EventName") == "GameEnd"
            and event.get("EventTime", float("inf")) <= data.get("gameData", {}).get("gameTime", -1)
            for event in events
        )
        self.report["samples"] += 1
        self.report.update(
            status="recording", last_game_time=data.get("gameData", {}).get("gameTime")
        )
        self.path = self.output / "samples" / f"{self.report['samples']:06d}.json"
        self.sample = {
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "game": {
                k: data["gameData"][k]
                for k in (
                    "gameTime",
                    "mapNumber",
                    "gameMode",
                    "gameVersion",
                )
                if k in data.get("gameData", {})
            },
            "active_slot": next(iter(active_slots)) if len(active_slots) == 1 else None,
            "active_stats": numbers(active.get("championStats", {})),
            "active_ability_levels": {
                key: value["abilityLevel"]
                for key, value in active.get("abilities", {}).items()
                if isinstance(value, dict) and isinstance(value.get("abilityLevel"), int)
            },
            "active_current_gold": active.get("currentGold"),
            "players": [
                {
                    "slot": index,
                    **{
                        k: p[k]
                        for k in (
                            "championName",
                            "rawChampionName",
                            "team",
                            "position",
                            "level",
                            "isDead",
                            "respawnTimer",
                        )
                        if k in p
                    },
                    "scores": numbers(p.get("scores", {})),
                    "items": [
                        {k: item[k] for k in ("itemID", "count", "slot") if k in item}
                        for item in p["items"]
                    ]
                    if p.get("items") is not None
                    else None,
                }
                for index, p in enumerate(players)
            ],
            "events": events,
        }
        atomic_json(self.path, self.sample)
        self.save()

    def decision(self, result, *, observation=None, feature_names=None):
        # Context identifiers are intentionally excluded from persisted advice.
        self.sample["decision"] = {
            k: result[k]
            for k in (
                "status",
                "action_id",
                "action",
                "scores",
                "score_meaning",
                "method",
                "critic_target",
                "role",
                "horizon_seconds",
                "advice_issued_game_time",
                "advice_expires_game_time",
                "retained_advice",
            )
            if k in result
        }
        if "macro_recommendation" in result:
            self.sample["macro_decision"] = {
                key: value
                for key, value in result["macro_recommendation"].items()
                if key
                in {
                    "status",
                    "action_id",
                    "action",
                    "scores",
                    "critic_target",
                    "advice_issued_game_time",
                    "advice_expires_game_time",
                }
            }
            self.sample["coordination"] = result.get("coordination", {})
        matches = self.aliases.get(result.get("actor"), set())
        self.sample["decision"]["subject_slot"] = next(iter(matches)) if len(matches) == 1 else None
        self.sample["decision"]["issued_game_time"] = result.get(
            "advice_issued_game_time", self.sample["game"].get("gameTime")
        )
        self.sample["decision"]["delivery"] = "served_or_printed; user attention not verified"
        if observation is not None:
            self.sample["features"] = dict(zip(feature_names, observation.tolist()))
        atomic_json(self.path, self.sample)

    def close(self, status):
        self.report.update(status=status, game_end_observed=self.ended)
        self.save()

    def heartbeat(self, status):
        self.report["poll_attempts"] += 1
        self.report.update(
            last_poll_utc=datetime.now(timezone.utc).isoformat(), last_poll_status=status
        )
        self.save()
