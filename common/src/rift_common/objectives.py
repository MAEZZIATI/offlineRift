"""Pinned patch-scoped objective estimates; no live-coordinate or buff-retention claims."""

import json
from dataclasses import dataclass, field
from pathlib import Path

PITS = {"dragon": (9866, 4414), "baron": (5007, 10471), "herald": (5007, 10471)}
SPEC = json.loads(Path(__file__).with_name("objective_rules.json").read_text())


@dataclass(frozen=True)
class ObjectiveRules:
    first_spawns: dict = field(default_factory=lambda: SPEC["parameters"]["first_spawns"].copy())
    respawns: dict = field(default_factory=lambda: SPEC["parameters"]["respawns"].copy())
    herald_despawn: float = 1200
    herald_uncertainty_start: float = 1170
    elder_respawn: float = 360
    epic_buff_durations: dict = field(default_factory=lambda: {"baron": 180, "elder": 150})

    @staticmethod
    def validate_patch(version):
        if ".".join(version.split(".")[:2]) not in SPEC["supported_patches"]:
            raise ValueError(f"No objective specification for patch {version}")


class ObjectiveTracker:
    def __init__(self, config: ObjectiveRules):
        self.config = config
        self.kills: dict[str, tuple[float, int | None]] = {}
        self.buff_kills: dict[str, tuple[float, int | None]] = {}
        self.elemental_kills = {100: 0, 200: 0}
        self.unattributed_elementals = 0
        self.elder_phase = False
        self.dragon_phase_unknown = False

    def consume(
        self, kind: str, timestamp: float, team: int | None = None, *, is_elder: bool | None = False
    ):
        if kind == "elder" or (kind == "dragon" and is_elder is True):
            self.elder_phase = True
            self.dragon_phase_unknown = False
            self.kills["dragon"] = (timestamp, team)
        elif kind == "dragon" and not self.elder_phase:
            if is_elder is None:
                self.dragon_phase_unknown = True
            elif team in self.elemental_kills:
                self.elemental_kills[team] += 1
            else:
                self.unattributed_elementals += 1
            maximum = max(self.elemental_kills.values())
            if maximum >= 4:
                self.elder_phase = True
                self.dragon_phase_unknown = False
            elif maximum + self.unattributed_elementals >= 4:
                self.dragon_phase_unknown = True
        if kind in PITS:
            self.kills[kind] = (timestamp, team)
        if kind in self.config.epic_buff_durations:
            self.buff_kills[kind] = (timestamp, team)
        if kind == "dragon" and is_elder is True:
            self.buff_kills["elder"] = (timestamp, team)

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
            uncertain = (
                kind == "herald"
                and killed is None
                and self.config.herald_uncertainty_start <= timestamp < self.config.herald_despawn
            ) or (kind == "dragon" and self.dragon_phase_unknown)
            if kind == "dragon" and self.elder_phase and killed:
                spawn = killed[0] + self.config.elder_respawn
                result[kind].update(eta=max(0.0, spawn - timestamp), available=timestamp >= spawn)
            result[kind]["availability_known"] = not uncertain
            if uncertain:
                # Enabled is the shared evidence mask: never encode unknown availability as true.
                result[kind].update(eta=0.0, available=False, enabled=False)
        for kind, duration in self.config.epic_buff_durations.items():
            kill = self.buff_kills.get(kind)
            result[f"{kind}_buff"] = {
                "remaining": max(0, kill[0] + duration - timestamp) if kill else 0,
                "team": kill[1] if kill else None,
            }
        return result
