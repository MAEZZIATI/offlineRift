"""Immutable explicit option transitions, validated without stitching missing actions."""

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from rift_common.actions import ACTION_NAMES, ACTION_SCHEMA, ROLES, feasible_actions, group_index
from rift_common.behavior import player_activity, team_activity
from rift_common.state import STATE_SCHEMA, STATE_SCHEMAS, StateEncoder
from rift_common.timeline import read_match

DATASET_SCHEMA = "option-transitions-v3"


def episode_returns(arrays, gamma):
    """Real terminal rewards, discounted over complete stored actor episodes; targets only."""
    result = np.empty(len(arrays["rewards"]), dtype=np.float32)
    continuation = 0.0
    for i in range(len(result) - 1, -1, -1):
        if arrays["terminals"][i]:
            continuation = 0.0
        continuation = (
            float(arrays["rewards"][i])
            + gamma ** (float(arrays["durations"][i]) / 60) * continuation
        )
        result[i] = continuation
    return result


class DatasetBuilder:
    def __init__(self, *, minimum_time=90):
        self.minimum_time = minimum_time
        self.rows = []
        self.matches = set()
        self.excluded = Counter()
        self.evidence = Counter()

    def add_match(self, match, timeline, catalog):
        match_id = match["metadata"]["matchId"]
        if match_id in self.matches:
            raise ValueError("Duplicate match ID")
        observations = read_match(match, timeline)
        if any(b.time - a.time > 90 for a, b in pairwise(observations)):
            raise ValueError("Coarse interval exceeds 90s; no invented intermediate decisions")
        encoder = StateEncoder(catalog)
        participants = match["info"]["participants"]
        pending = []
        for team in (100, 200):
            outcome = 1.0 if next(p for p in participants if p["teamId"] == team)["win"] else -1.0
            actors = [
                None,
                *[p.id for p in observations[0].players if p.team == team and p.role in ROLES],
            ]
            self.excluded["unknown_role_player_episodes"] += 5 - (len(actors) - 1)
            for actor in actors:
                episode = []
                evidence = []
                role = "TEAM" if actor is None else observations[0].player(actor).role
                group = group_index(actor is None, role)
                for index, (current, following) in enumerate(pairwise(observations)):
                    if current.time < self.minimum_time:
                        continue
                    activity = (
                        team_activity(current, following, team)
                        if actor is None
                        else player_activity(current, following, actor)
                    )
                    if activity is None:
                        self.excluded["incomplete_behavior_episodes"] += 1
                        episode = []
                        break
                    behavior = activity.representative
                    state = encoder.encode(
                        current, team, actor, previous=observations[index - 1] if index else None
                    )
                    next_state = encoder.encode(following, team, actor, previous=current)
                    mask = feasible_actions(state, encoder.names)[0]
                    if not mask[behavior.action]:
                        self.excluded["infeasible_behavior_episodes"] += 1
                        episode = []
                        break
                    terminal = index == len(observations) - 2
                    candidates = np.zeros(len(ACTION_NAMES), dtype=bool)
                    candidates[list(activity.candidates)] = True
                    episode.append(
                        {
                            "observations": state,
                            "next_observations": next_state,
                            "actions": int(behavior.action),
                            "activity_candidates": candidates,
                            "action_identified": activity.identified,
                            "death_participation": activity.death_participation,
                            "combat_credit": activity.combat_credit,
                            "rewards": outcome if terminal else 0.0,
                            "terminals": terminal,
                            "durations": following.time - current.time,
                            "timestamps": current.time,
                            "next_timestamps": following.time,
                            "match_ids": match_id,
                            "episode_ids": f"{match_id}:{team}:{actor or 'team'}",
                            "teams": team,
                            "actors": actor or "team",
                            "groups": group,
                            "action_masks": mask,
                            "next_action_masks": feasible_actions(next_state, encoder.names)[0],
                        }
                    )
                    evidence.append(behavior.evidence)
                if episode and episode[-1]["terminals"]:
                    pending.extend(episode)
                    self.evidence.update(evidence)
        if not pending:
            raise ValueError("No complete actor episodes with supported behavior evidence")
        self.rows.extend(pending)
        self.matches.add(match_id)

    def save(self, path, *, catalog, provenance):
        if not self.rows:
            raise ValueError("No transitions to save")
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
        for key in ("observations", "next_observations", "rewards", "durations"):
            arrays[key] = arrays[key].astype(np.float32)
        for key in ("actions", "groups", "teams"):
            arrays[key] = arrays[key].astype(np.int64)
        metadata = {
            "dataset_schema": DATASET_SCHEMA,
            "state_schema": STATE_SCHEMA,
            "action_schema": ACTION_SCHEMA,
            "feature_names": StateEncoder(catalog).names,
            "action_names": ACTION_NAMES,
            "item_catalog": catalog.data,
            "reward": "terminal_team_win_plus1_loss_minus1",
            "time_unit_seconds": 60,
            "minimum_time": self.minimum_time,
            "provenance": provenance,
            "excluded": dict(self.excluded),
            "behavior_evidence": dict(self.evidence),
            "target_schema": "executed-evidence-v3",
            "target_limitations": "Representative actions remain descriptive; ambiguous intervals are retained but excluded from action-specific fitting",
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata)))
        return metadata


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
    from rift_common.items import ItemCatalog

    names = StateEncoder(
        ItemCatalog(metadata["item_catalog"]), schema=metadata["state_schema"]
    ).names
    if (
        metadata.get("dataset_schema")
        not in {DATASET_SCHEMA, "option-transitions-v2", "option-transitions-v1"}
        or metadata.get("state_schema") not in STATE_SCHEMAS
        or metadata.get("action_schema") != ACTION_SCHEMA
        or metadata["feature_names"] != names
        or {int(k): v for k, v in metadata["action_names"].items()} != ACTION_NAMES
    ):
        raise ValueError("Incompatible next-action dataset; legacy formation data cannot be loaded")
    required = {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "terminals",
        "durations",
        "timestamps",
        "next_timestamps",
        "match_ids",
        "episode_ids",
        "teams",
        "actors",
        "groups",
        "action_masks",
        "next_action_masks",
    }
    if metadata["dataset_schema"] in {DATASET_SCHEMA, "option-transitions-v2"}:
        required |= {
            "activity_candidates",
            "action_identified",
            "death_participation",
            "combat_credit",
        }
        expected_target = (
            "executed-evidence-v3"
            if metadata["dataset_schema"] == DATASET_SCHEMA
            else "executed-evidence-v2"
        )
        if metadata.get("target_schema") != expected_target:
            raise ValueError("Invalid activity target schema")
    if set(arrays) != required:
        raise ValueError("Missing or unexpected transition arrays")
    n = len(arrays["actions"])
    if not n or any(len(value) != n for value in arrays.values()):
        raise ValueError("Empty or misaligned transitions")
    if arrays["observations"].shape != (n, len(names)) or arrays["next_observations"].shape != (
        n,
        len(names),
    ):
        raise ValueError("Invalid state dimensions")
    for key in (
        "observations",
        "next_observations",
        "rewards",
        "durations",
        "timestamps",
        "next_timestamps",
    ):
        if not np.isfinite(arrays[key]).all():
            raise ValueError("Non-finite transition")
    if arrays["actions"].dtype.kind not in "iu" or np.any(
        (arrays["actions"] < 0) | (arrays["actions"] >= len(ACTION_NAMES))
    ):
        raise ValueError("Invalid action ID")
    if arrays["groups"].dtype.kind not in "iu" or np.any(
        (arrays["groups"] < 0) | (arrays["groups"] > 6)
    ):
        raise ValueError("Invalid actor group")
    if (
        arrays["terminals"].dtype != bool
        or np.any(arrays["durations"] <= 0)
        or not np.allclose(arrays["durations"], arrays["next_timestamps"] - arrays["timestamps"])
    ):
        raise ValueError("Invalid terminal or elapsed duration")
    for key in ("action_masks", "next_action_masks"):
        if (
            arrays[key].shape != (n, len(ACTION_NAMES))
            or arrays[key].dtype != bool
            or not arrays[key].any(axis=1).all()
        ):
            raise ValueError("Invalid feasibility masks")
    if not arrays["action_masks"][np.arange(n), arrays["actions"]].all():
        raise ValueError("Recorded action is infeasible")
    if "activity_candidates" in arrays:
        candidates = arrays["activity_candidates"]
        if (
            candidates.dtype != bool
            or candidates.shape != (n, len(ACTION_NAMES))
            or not candidates[np.arange(n), arrays["actions"]].all()
        ):
            raise ValueError("Invalid activity evidence candidates")
        for key in ("action_identified", "death_participation", "combat_credit"):
            if arrays[key].dtype != bool or arrays[key].shape != (n,):
                raise ValueError("Invalid activity evidence flags")
        if np.any(arrays["action_identified"] & (candidates.sum(1) != 1)):
            raise ValueError("Ambiguous activity marked identified")
    if not np.array_equal(
        arrays["action_masks"], feasible_actions(arrays["observations"], names)
    ) or not np.array_equal(
        arrays["next_action_masks"], feasible_actions(arrays["next_observations"], names)
    ):
        raise ValueError("Feasibility masks differ from shared contract")
    starts = np.r_[0, np.flatnonzero(arrays["episode_ids"][1:] != arrays["episode_ids"][:-1]) + 1]
    ends = np.r_[starts[1:], n]
    if len(np.unique(arrays["episode_ids"][starts])) != len(starts):
        raise ValueError("Non-contiguous episode")
    for start, end in zip(starts, ends):
        if (
            len(set(arrays["match_ids"][start:end])) != 1
            or len(set(arrays["actors"][start:end])) != 1
            or len(set(arrays["teams"][start:end])) != 1
            or len(set(arrays["groups"][start:end])) != 1
        ):
            raise ValueError("Actor/match boundary mismatch")
        if not arrays["terminals"][end - 1] or arrays["terminals"][start : end - 1].any():
            raise ValueError("Invalid episode terminal boundary")
        if not np.array_equal(
            arrays["next_observations"][start : end - 1], arrays["observations"][start + 1 : end]
        ) or not np.array_equal(
            arrays["next_timestamps"][start : end - 1], arrays["timestamps"][start + 1 : end]
        ):
            raise ValueError("Broken state/next-state continuity")
    if (
        np.any(arrays["rewards"][~arrays["terminals"]] != 0)
        or not np.isin(arrays["rewards"][arrays["terminals"]], [-1, 1]).all()
    ):
        raise ValueError("Invalid terminal-win reward")
    return arrays, metadata
