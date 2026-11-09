"""Inference-only composition of one macro policy and five role specialists."""

from pathlib import Path

import numpy as np
from rift_common.actions import GROUPS, ROLES, Action, feasible_actions
from rift_common.artifacts import file_digest, read_json
from rift_common.coaching import POLICY_SCOPES, alignment

from offline_rift.inference.policy import NextActionPolicy


class MacroCoach:
    """Route to separate networks; expose coordination without inventing plan conditioning."""

    def __init__(self, directory, *, require_complete=True):
        directory = Path(directory)
        manifest = read_json(directory / "coach.json")
        if require_complete:
            run = read_json(directory / "run.json")
            if (
                run.get("status") != "complete"
                or not run.get("serving_reload_verified")
                or file_digest(directory / "run.json") != manifest.get("run_sha256")
            ):
                raise ValueError("Coach is incomplete or its completion record is corrupted")
        if manifest.get("schema") != "six-policy-coach-v1" or set(manifest["policies"]) != set(
            POLICY_SCOPES
        ):
            raise ValueError("Incomplete six-policy coach")
        self.policies = {}
        for scope, entry in manifest["policies"].items():
            path = (directory / entry["path"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Coach policy path escapes bundle")
            if file_digest(path / "bundle.json") != entry["sha256"]:
                raise ValueError("Coach child integrity mismatch")
            policy = NextActionPolicy(path)
            if tuple(policy.advisory_groups) != (scope,) or tuple(
                policy.config.get("training_groups", ())
            ) != (scope,):
                raise ValueError("Coach child has incorrect serving scope")
            self.policies[scope] = policy
        if file_digest(directory / "metadata.json") != manifest["metadata_sha256"]:
            raise ValueError("Coach metadata integrity mismatch")
        self.metadata = read_json(directory / "metadata.json")
        reference = self.policies["TEAM"].metadata
        for policy in self.policies.values():
            for key in (
                "feature_names",
                "state_schema",
                "action_schema",
                "dataset_sha256",
                "provenance",
                "item_catalog",
                "target_schema",
            ):
                if policy.metadata[key] != reference[key]:
                    raise ValueError("Coach children have incompatible contracts")
            for key in ("training_matches", "validation_matches", "holdout_matches"):
                if policy.metadata["training_report"].get(key) != reference["training_report"].get(
                    key
                ):
                    raise ValueError("Coach children have incompatible match partitions")
        if self.metadata["dataset_sha256"] != reference["dataset_sha256"]:
            raise ValueError("Coach aggregate dataset differs")
        self.encoder = self.policies["TEAM"].encoder
        self.name, self.algorithm = directory.name, "cql"
        self.advisory_groups = POLICY_SCOPES
        self.counts = np.zeros((len(GROUPS), len(Action)), dtype=np.int64)
        for scope, policy in self.policies.items():
            self.counts[GROUPS.index(scope)] = policy.counts[GROUPS.index(scope)]

    def predict_batch(self, observations, groups, masks):
        observed, groups, masks = (
            np.asarray(observations, dtype=np.float32),
            np.asarray(groups),
            np.asarray(masks),
        )
        if (
            observed.ndim != 2
            or observed.shape[1] != len(self.encoder.names)
            or not np.isfinite(observed).all()
            or groups.shape != (len(observed),)
            or groups.dtype.kind not in "iu"
            or np.any((groups < 0) | (groups >= len(GROUPS)))
            or masks.shape != (len(observed), len(Action))
            or masks.dtype != bool
            or not np.array_equal(masks, feasible_actions(observed, self.encoder.names))
        ):
            raise ValueError("Invalid coach batch")
        predicted = np.full(len(observed), -1, dtype=np.int64)
        scores = np.full((len(observed), len(Action)), -np.inf, dtype=np.float32)
        for scope, policy in self.policies.items():
            selected = groups == GROUPS.index(scope)
            if selected.any():
                predicted[selected], scores[selected] = policy.predict_batch(
                    observed[selected], groups[selected], masks[selected]
                )
        return predicted, scores

    def recommend(self, current, team, actor=None, *, role=None, previous=None):
        macro = self.policies["TEAM"].recommend(current, team, previous=previous)
        if actor is None:
            return {**macro, "policy_scope": "TEAM"}
        role = role or current.player(actor).role
        if role not in ROLES:
            return {
                "status": "needs_role",
                "message": "Select the player's role.",
                "team": team,
                "actor": actor,
                "role": role,
                "game_time": current.time,
            }
        result = self.policies[role].recommend(current, team, actor, role=role, previous=previous)
        return {
            **result,
            "policy_scope": role,
            "macro_recommendation": macro,
            "coordination": {
                "status": alignment(macro, result),
                "meaning": "Symbolic direction compatibility; not learned plan conditioning or proof of tactical safety",
            },
        }


def load_policy(directory):
    directory = Path(directory)
    return (
        MacroCoach(directory)
        if (directory / "coach.json").is_file()
        else NextActionPolicy(directory)
    )
