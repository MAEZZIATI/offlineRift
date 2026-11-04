"""Weak macro labels and explicit, validated offline MDP transition storage."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from offline_rift.pipeline.actions import ACTION_NAMES, ACTION_SCHEMA, action_names
from offline_rift.pipeline.features import (
    MAP_SIZE,
    PITS,
    SCHEMA_VERSION,
    FeatureConfig,
    FeatureEncoder,
    ItemCatalog,
    Snapshot,
)


class MacroActionLabeler:
    """Geometric weak supervision; neither ground-truth intent nor observed vision.

    Priority: base > available objective area > cross-map > split > defense >
    lane group > dispersed coverage > unclassified. No future state is used.
    """

    def __init__(
        self, pit_radius: float = 2000, mid_width: float = 1800, max_frame_age: float = 90
    ):
        self.pit_radius, self.mid_width, self.max_frame_age = pit_radius, mid_width, max_frame_age

    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(np.asarray(a) - b))

    def classify(self, snapshot: Snapshot, team: int) -> tuple[int, float]:
        if team not in (100, 200):
            raise ValueError("team must be 100 or 200")
        sides = [
            [p.get("position") for p in snapshot.players if p["team"] == t]
            for t in (team, 300 - team)
        ]
        if any(len(s) != 5 or any(p is None for p in s) for s in sides):
            return 11, 0.1
        if snapshot.frame_age > self.max_frame_age:
            return 11, 0.1
        allies, enemies = sides
        confidence = max(0.25, 0.9 - snapshot.frame_age / 120)
        base = (0, 0) if team == 100 else (MAP_SIZE, MAP_SIZE)
        active = [p for p in allies if self._distance(p, base) > 2200]
        if len(active) <= 2:
            return 9, confidence  # Presence at base is observable; a recall is not.
        for action, kind in enumerate(PITS):
            obj = snapshot.objectives[kind]
            if (
                obj["enabled"]
                and obj["eta"] <= 60
                and sum(self._distance(p, PITS[kind]) <= self.pit_radius for p in active) >= 3
            ):
                return action, confidence

        def forward(p):
            return sum(p) > MAP_SIZE if team == 100 else sum(p) < MAP_SIZE

        side_pressure = [p for p in active if abs(p[0] - p[1]) > self.mid_width and forward(p)]
        for kind, pit in PITS.items():
            obj = snapshot.objectives[kind]
            if (
                obj["enabled"]
                and obj["eta"] <= 60
                and sum(self._distance(p, pit) <= self.pit_radius for p in enemies) >= 3
                and any(self._distance(p, pit) > 6500 for p in side_pressure)
            ):
                return 3, confidence
        # Three or four together plus isolated side pressure supports both 1-4 and 1-3-1.
        mid = [p for p in active if abs(p[0] - p[1]) <= self.mid_width]
        if len(mid) >= 3 and 1 <= len(side_pressure) <= 2:
            return 4, confidence
        own_half = [p for p in active if not forward(p)]
        if len(own_half) >= 3 and sum(not forward(p) for p in enemies) >= 3:
            return 8, confidence
        if len(mid) >= 3:
            return 5, confidence
        if sum(p[1] - p[0] > self.mid_width for p in active) >= 3:
            return 6, confidence
        if sum(p[0] - p[1] > self.mid_width for p in active) >= 3:
            return 7, confidence
        # Distinct occupied bands support coverage, but not farming/rotation intent.
        if (
            len(
                {
                    0 if abs(p[0] - p[1]) <= self.mid_width else (1 if p[1] > p[0] else 2)
                    for p in active
                }
            )
            == 3
        ):
            return 10, min(confidence, 0.6)
        return 11, min(confidence, 0.3)

    def label(self, snapshot: Snapshot, team: int) -> int:
        return self.classify(snapshot, team)[0]

    def label_window(self, window: list[Snapshot], team: int) -> int:
        if not window:
            raise ValueError("Empty window")
        labels = [self.label(snapshot, team) for snapshot in window]
        return int(np.bincount(labels).argmax())


class MDPDatasetBuilder:
    def __init__(
        self,
        config: FeatureConfig,
        catalog: ItemCatalog,
        *,
        reward_mode: str = "sparse",
        gold_scale: float = 10000,
    ):
        if reward_mode not in {"sparse", "gold"} or gold_scale <= 0:
            raise ValueError("Invalid reward configuration")
        self.config, self.catalog = config, catalog
        self.encoder = FeatureEncoder(config, catalog)
        self.labeler = MacroActionLabeler()
        self.action_schema = ACTION_SCHEMA
        self.action_names = ACTION_NAMES
        self.reward_mode, self.gold_scale = reward_mode, gold_scale
        self.rows: list[dict] = []
        self.match_ids: set[str] = set()

    def add_match(
        self, match: dict, snapshots: list[Snapshot], *, catalog: ItemCatalog | None = None
    ):
        encoder = FeatureEncoder(self.config, catalog) if catalog is not None else self.encoder
        match_id = match["metadata"]["matchId"]
        if match_id in self.match_ids:
            raise ValueError(f"Duplicate match: {match_id}")
        if len(snapshots) < 2 or any(b.time <= a.time for a, b in pairwise(snapshots)):
            raise ValueError("Need at least two increasing snapshots")
        participants = match["info"]["participants"]
        pending = []
        for team in (100, 200):
            outcomes = {bool(p["win"]) for p in participants if p["teamId"] == team}
            if len(outcomes) != 1:
                raise ValueError("Inconsistent team outcome")
            outcome = 1.0 if outcomes.pop() else -1.0
            for i, (state, next_state) in enumerate(pairwise(snapshots)):
                terminal = i == len(snapshots) - 2
                action, confidence = self.labeler.classify(state, team)
                reward = outcome if terminal else 0.0
                if self.reward_mode == "gold":

                    def gold_difference(s, perspective=team):
                        if any(p.get("gold") is None for p in s.players):
                            raise ValueError("Gold shaping requires known historical gold")
                        return sum(
                            p["gold"] * (1 if p["team"] == perspective else -1) for p in s.players
                        )

                    reward += (
                        gold_difference(next_state) - gold_difference(state)
                    ) / self.gold_scale
                pending.append(
                    {
                        "observations": encoder.encode(state, team),
                        "next_observations": encoder.encode(next_state, team),
                        "actions": action,
                        "rewards": reward,
                        "terminals": terminal,
                        "episode_ids": f"{match_id}:{team}",
                        "match_ids": match_id,
                        "teams": team,
                        "timestamps": state.time,
                        "next_timestamps": next_state.time,
                        "label_confidence": confidence,
                    }
                )
        self.rows.extend(pending)
        self.match_ids.add(match_id)

    def save(self, path: str | Path, *, provenance: dict | None = None):
        if not self.rows:
            raise ValueError("Cannot save an empty dataset")
        arrays = {key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}
        for key in ("observations", "next_observations", "rewards", "label_confidence"):
            arrays[key] = arrays[key].astype(np.float32)
        arrays["actions"] = arrays["actions"].astype(np.int64)
        arrays["metadata"] = np.asarray(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "feature_names": self.encoder.names,
                    "feature_config": self.config.to_dict(),
                    "action_names": self.action_names,
                    "action_schema": self.action_schema,
                    "reward_mode": self.reward_mode,
                    "gold_scale": self.gold_scale,
                    "item_catalog": self.catalog.data,
                    "provenance": provenance or {},
                }
            )
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def load_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "terminals",
            "episode_ids",
            "match_ids",
            "metadata",
            "timestamps",
            "next_timestamps",
            "label_confidence",
        }
        if not required.issubset(archive.files):
            raise ValueError("Dataset is missing required arrays")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
    encoder = FeatureEncoder(
        FeatureConfig(**metadata["feature_config"]), ItemCatalog(metadata["item_catalog"])
    )
    if metadata["schema_version"] != SCHEMA_VERSION or metadata["feature_names"] != encoder.names:
        raise ValueError("Incompatible feature schema")
    n = len(arrays["actions"])
    names = action_names(metadata)
    if n == 0 or any(len(value) != n for value in arrays.values()):
        raise ValueError("Empty or misaligned transition arrays")
    if arrays["observations"].shape != (n, len(encoder.names)) or (
        arrays["next_observations"].shape != arrays["observations"].shape
    ):
        raise ValueError("Invalid observation shape")
    for key in (
        "actions",
        "rewards",
        "terminals",
        "episode_ids",
        "match_ids",
        "timestamps",
        "next_timestamps",
    ):
        if arrays[key].shape != (n,):
            raise ValueError(f"Invalid {key} shape")
    for key in ("observations", "next_observations", "rewards", "timestamps", "next_timestamps"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Non-finite {key}")
    confidence = arrays["label_confidence"]
    if (
        confidence.shape != (n,)
        or not np.isfinite(confidence).all()
        or np.any((confidence < 0) | (confidence > 1))
    ):
        raise ValueError("Invalid label confidence")
    if not np.issubdtype(arrays["actions"].dtype, np.integer) or np.any(
        (arrays["actions"] < 0) | (arrays["actions"] >= len(names))
    ):
        raise ValueError("Actions must be integers within the action contract")
    if not np.isin(arrays["terminals"], [0, 1]).all():
        raise ValueError("Invalid terminal flags")
    seen = set()
    start = 0
    for end in range(1, n + 1):
        if end != n and arrays["episode_ids"][end] == arrays["episode_ids"][start]:
            continue
        episode_id = str(arrays["episode_ids"][start])
        if episode_id in seen:
            raise ValueError("Episodes must be contiguous")
        seen.add(episode_id)
        if not arrays["terminals"][end - 1] or arrays["terminals"][start : end - 1].any():
            raise ValueError("Each complete episode must have one final terminal")
        if len(set(arrays["match_ids"][start:end])) != 1:
            raise ValueError("Episode crosses match boundaries")
        if np.any(arrays["next_timestamps"][start:end] <= arrays["timestamps"][start:end]):
            raise ValueError("Invalid time ordering")
        if end - start > 1 and (
            not np.array_equal(
                arrays["next_observations"][start : end - 1],
                arrays["observations"][start + 1 : end],
            )
            or not np.array_equal(
                arrays["next_timestamps"][start : end - 1], arrays["timestamps"][start + 1 : end]
            )
        ):
            raise ValueError("Broken next-state continuity")
        start = end
    return arrays, metadata
