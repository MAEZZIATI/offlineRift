"""Versioned, team-relative observation contract shared by offline and live paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
PITS = {"dragon": (9866, 4414), "baron": (5007, 10471), "herald": (5007, 10471)}
MAP_SIZE = 15000.0
SCHEMA_VERSION = "rift-v1"


@dataclass
class FeatureConfig:
    """Illustrative classic timers; override from JSON for the dataset's patch.

    Timers are estimates, not authoritative spawn detection. Ability cooldowns
    are deliberately unknown: neither source exposes enough cast/haste data.
    """

    interval_seconds: int = 10
    grid_size: int = 4
    profile: str = "live"
    first_spawns: dict[str, float] = field(
        default_factory=lambda: {"dragon": 300, "baron": 1200, "herald": 840}
    )
    respawns: dict[str, float] = field(
        default_factory=lambda: {"dragon": 300, "baron": 360, "herald": 360}
    )
    herald_despawn: float = 1170
    epic_buff_durations: dict[str, float] = field(
        default_factory=lambda: {"baron": 180, "elder": 150}
    )
    item_spike_gold: int = 2500

    def __post_init__(self):
        if self.profile not in {"live", "full"}:
            raise ValueError("profile must be live or full")
        if self.interval_seconds != 10 or not 1 <= self.grid_size <= 16:
            raise ValueError("Use 10-second intervals and grid_size in [1,16]")
        for key in PITS:
            if key not in self.first_spawns or key not in self.respawns:
                raise ValueError(f"Missing timer for {key}")
        if any(v < 0 for v in self.first_spawns.values()) or any(
            v <= 0 for v in self.respawns.values()
        ):
            raise ValueError("Invalid objective timers")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Snapshot:
    time: float
    players: list[dict[str, Any]]
    objectives: dict[str, dict[str, Any]]
    frame_age: float = 0.0


class ItemCatalog:
    """Data Dragon item JSON. Values are inventory investment proxies, not DPS."""

    def __init__(self, data: dict | None = None):
        self.data = (data or {}).get("data", data or {})

    def value(self, item_id: int) -> float:
        return float(self.data.get(str(item_id), {}).get("gold", {}).get("total", 0))

    def power(self, items: list[int], threshold: int) -> tuple[float, int, bool]:
        known = all(str(i) in self.data for i in items)
        return (
            sum(self.value(i) for i in items),
            sum(self.value(i) >= threshold for i in items),
            known,
        )


class ObjectiveTracker:
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.kills: dict[str, tuple[float, int | None]] = {}
        self.buff_kills: dict[str, tuple[float, int | None]] = {}

    def consume(self, kind: str, timestamp: float, team: int | None = None):
        if kind in PITS:
            self.kills[kind] = (timestamp, team)
        if kind in self.config.epic_buff_durations:
            self.buff_kills[kind] = (timestamp, team)

    def state(self, timestamp: float) -> dict[str, dict]:
        result = {}
        for kind in PITS:
            killed = self.kills.get(kind)
            # Herald modeled as one spawn. Patch-specific respawns require a custom tracker.
            unavailable = kind == "herald" and (
                timestamp >= self.config.herald_despawn or killed is not None
            )
            spawn = (
                killed[0] + self.config.respawns[kind] if killed else self.config.first_spawns[kind]
            )
            result[kind] = {
                "eta": max(0.0, spawn - timestamp),
                "available": not unavailable and timestamp >= spawn,
                "enabled": not unavailable,
            }
        for kind, duration in self.config.epic_buff_durations.items():
            kill = self.buff_kills.get(kind)
            result[f"{kind}_buff"] = {
                "remaining": max(0, kill[0] + duration - timestamp) if kill else 0,
                "team": kill[1] if kill else None,
            }
        return result


class FeatureEncoder:
    def __init__(self, config: FeatureConfig, catalog: ItemCatalog):
        self.config, self.catalog = config, catalog

    @property
    def names(self) -> list[str]:
        names = ["time", "frame_age", "gold_difference", "gold_known"]
        for role in ROLES:
            names += [f"{role}_level_gap", f"{role}_level_known"]
        for side in ("ally", "enemy"):
            for i in range(5):
                names += [
                    f"{side}_{i}_{name}"
                    for name in (
                        "level",
                        "level_known",
                        "item_gold",
                        "item_spikes",
                        "items_known",
                        "x",
                        "y",
                        "position_known",
                        "cooldown_remaining",
                        "cooldown_known",
                    )
                ]
            names += [f"{side}_grid_{i}" for i in range(self.config.grid_size**2)]
        for objective in PITS:
            names += [f"{objective}_eta", f"{objective}_available", f"{objective}_enabled"]
        for buff in self.config.epic_buff_durations:
            names += [f"{buff}_buff_ally", f"{buff}_buff_enemy"]
        return names

    @staticmethod
    def team_players(snapshot: Snapshot, team: int) -> list[dict]:
        return sorted(
            (p for p in snapshot.players if p["team"] == team),
            key=lambda p: (p.get("champion", "").lower(), p["id"]),
        )

    def encode(self, snapshot: Snapshot, team: int) -> np.ndarray:
        sides = [self.team_players(snapshot, team), self.team_players(snapshot, 300 - team)]
        if team not in (100, 200) or any(len(side) != 5 for side in sides):
            raise ValueError("Requires a 5v5 Summoner's Rift snapshot")
        full = self.config.profile == "full"
        gold_known = full and all(p.get("gold") is not None for s in sides for p in s)
        gold = (
            sum(p["gold"] for p in sides[0]) - sum(p["gold"] for p in sides[1]) if gold_known else 0
        )
        values = [
            snapshot.time / 3600,
            snapshot.frame_age / 60 if full else 0,
            gold / 10000,
            float(gold_known),
        ]
        for role in ROLES:
            pair = [next((p for p in side if p.get("role") == role), None) for side in sides]
            # Live 'position' roles may be absent or differ from final match roles.
            known = full and all(p is not None and p.get("level") is not None for p in pair)
            values += [(pair[0]["level"] - pair[1]["level"]) / 18 if known else 0, float(known)]
        for side in sides:
            grid = np.zeros((self.config.grid_size, self.config.grid_size))
            for p in side:
                value, spikes, item_known = self.catalog.power(
                    p.get("items", []), self.config.item_spike_gold
                )
                item_known = item_known and p.get("inventory_known", True)
                pos = p.get("position") if full else None
                cooldown = p.get("cooldown_remaining") if full else None
                values += [
                    p["level"] / 18 if p.get("level") is not None else 0,
                    float(p.get("level") is not None),
                    value / 20000 if item_known else 0,
                    spikes / 6 if item_known else 0,
                    float(item_known),
                    pos[0] / MAP_SIZE if pos else 0,
                    pos[1] / MAP_SIZE if pos else 0,
                    float(pos is not None),
                    cooldown / 300 if cooldown is not None else 0,
                    float(cooldown is not None),
                ]
                if pos:
                    x, y = np.clip(
                        np.array(pos) / MAP_SIZE * self.config.grid_size,
                        0,
                        self.config.grid_size - 1,
                    ).astype(int)
                    grid[x, y] += 0.2
            values += grid.ravel().tolist()
        for kind in PITS:
            obj = snapshot.objectives[kind]
            values += [
                obj["eta"] / 600 if obj["enabled"] else 0,
                float(obj["available"]),
                float(obj["enabled"]),
            ]
        for buff in self.config.epic_buff_durations:
            obj = snapshot.objectives.get(f"{buff}_buff", {})
            remaining = obj.get("remaining", 0) / 180
            values += [
                remaining if obj.get("team") == team else 0,
                remaining if obj.get("team") == 300 - team else 0,
            ]
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (len(self.names),) or not np.isfinite(array).all():
            raise ValueError("Invalid encoded observation")
        return array
