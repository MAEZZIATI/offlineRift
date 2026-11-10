"""Causal public-observation state shared by dataset construction and live inference."""

from dataclasses import dataclass, field

import numpy as np

from rift_common.actions import ROLES
from rift_common.items import ItemCatalog

STATE_SCHEMA = "decision-state-v1"


@dataclass(frozen=True)
class PlayerState:
    id: str
    team: int
    champion: str = ""
    role: str = ""
    level: float | None = None
    items: tuple[int, ...] = ()
    inventory_known: bool = False
    cs: float | None = None
    lane_cs: float | None = None
    jungle_cs: float | None = None
    kills: float | None = None
    deaths: float | None = None
    assists: float | None = None
    is_dead: bool | None = None
    health: float | None = None
    position: tuple[float, float] | None = None


@dataclass(frozen=True)
class Observation:
    time: float
    players: tuple[PlayerState, ...]
    objectives: dict = field(default_factory=dict)
    events: tuple[dict, ...] = ()
    active_player: str | None = None
    history_complete: bool = True

    def player(self, player_id):
        matches = [p for p in self.players if p.id == str(player_id)]
        if len(matches) != 1:
            raise ValueError("Select a player present in this observation")
        return matches[0]


def _mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


class StateEncoder:
    """No x/y, final match statistics or hidden team gold enter the policy state."""

    def __init__(self, catalog: ItemCatalog):
        self.catalog = catalog

    @property
    def names(self):
        names = ["time", "scope_team", "team_red", "time_under_14m", "time_over_25m"]
        names += [f"role_{role}" for role in ROLES] + ["role_known"]
        for name in (
            "self_level",
            "self_items",
            "self_cs",
            "self_kills",
            "self_deaths",
            "self_assists",
            "self_health",
            "self_dead",
            "self_cs_rate",
            "self_level_rate",
        ):
            names += [name, name + "_known"]
        for name in (
            "ally_level",
            "enemy_level",
            "ally_items",
            "enemy_items",
            "ally_cs",
            "enemy_cs",
            "ally_kills",
            "enemy_kills",
            "ally_deaths_60",
            "enemy_deaths_60",
            "own_towers_lost",
            "enemy_towers_lost",
        ):
            names += [name, name + "_known"]
        for kind in ("dragon", "baron", "herald"):
            names += [kind + "_eta", kind + "_available", kind + "_enabled"]
        return names

    def encode(self, current: Observation, team: int, player_id=None, *, role=None, previous=None):
        if team not in (100, 200) or any(
            sum(p.team == side for p in current.players) != 5 for side in (100, 200)
        ):
            raise ValueError("A valid 5v5 team perspective is required")
        focal = current.player(player_id) if player_id is not None else None
        if focal is not None and focal.team != team:
            raise ValueError("Selected player does not belong to this team")
        role = role or (focal.role if focal else "")
        values = [
            current.time / 3600,
            float(focal is None),
            float(team == 200),
            float(current.time < 840),
            float(current.time >= 1500),
        ]
        values += [float(focal is not None and role == r) for r in ROLES] + [
            float(focal is not None and role in ROLES)
        ]

        def add(value, scale=1):
            values.extend(
                [float(value) / scale if value is not None else 0, float(value is not None)]
            )

        def power(player):
            value, _, known = self.catalog.power(list(player.items), 2500)
            return value if known and player.inventory_known else None

        before = None
        if focal is not None and previous is not None and 30 <= current.time - previous.time <= 120:
            try:
                before = previous.player(focal.id)
            except ValueError:
                pass
        for value, scale in (
            (focal.level if focal else None, 18),
            (power(focal) if focal else None, 20000),
            (focal.cs if focal else None, 300),
            (focal.kills if focal else None, 20),
            (focal.deaths if focal else None, 20),
            (focal.assists if focal else None, 30),
            (focal.health if focal else None, 1),
            (focal.is_dead if focal else None, 1),
        ):
            add(value, scale)
        elapsed = (current.time - previous.time) / 60 if before else 1
        add(
            max(0, focal.cs - before.cs) / elapsed
            if before and focal.cs is not None and before.cs is not None
            else None,
            15,
        )
        add(
            max(0, focal.level - before.level) / elapsed
            if before and focal.level is not None and before.level is not None
            else None,
            3,
        )
        groups = [[p for p in current.players if p.team == side] for side in (team, 300 - team)]
        for getter, scale in (
            (lambda p: p.level, 18),
            (power, 20000),
            (lambda p: p.cs, 300),
            (lambda p: p.kills, 20),
        ):
            for players in groups:
                add(_mean([getter(p) for p in players]), scale)
        for side in (team, 300 - team):
            add(
                sum(
                    e.get("kind") == "kill"
                    and e.get("victim_team") == side
                    and current.time - 60 < e["time"] <= current.time
                    for e in current.events
                )
                if current.history_complete
                and not any(
                    e.get("kind") == "kill"
                    and e.get("victim_team") not in (100, 200)
                    and current.time - 60 < e["time"] <= current.time
                    for e in current.events
                )
                else None,
                5,
            )
        for side in (team, 300 - team):
            add(
                sum(e.get("kind") == "tower" and e.get("team") == side for e in current.events)
                if current.history_complete
                and not any(
                    e.get("kind") == "tower" and e.get("team") not in (100, 200)
                    for e in current.events
                )
                else None,
                11,
            )
        for kind in ("dragon", "baron", "herald"):
            obj = current.objectives.get(kind, {})
            known = bool(obj.get("enabled"))
            values += [
                obj.get("eta", 0) / 1200 if known else 0,
                float(bool(obj.get("available")) and known),
                float(known),
            ]
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (len(self.names),) or not np.isfinite(result).all() or current.time < 0:
            raise ValueError("Invalid decision state")
        return result
