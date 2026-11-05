"""Inference-only next-action policy; formation bundles are intentionally incompatible."""

from pathlib import Path

import numpy as np
import torch
from rift_common.actions import (
    ACTION_NAMES,
    ACTION_SCHEMA,
    ROLES,
    Action,
    action_catalog,
    feasible_actions,
    group_index,
)
from rift_common.artifacts import read_json, verify_bundle_manifest
from rift_common.items import ItemCatalog
from rift_common.network import OptionNetwork, rank_options
from rift_common.state import STATE_SCHEMA, StateEncoder


class NextActionPolicy:
    def __init__(self, directory, *, device="cpu"):
        directory = Path(directory)
        self.metadata = read_json(directory / "metadata.json")
        if (
            self.metadata.get("bundle_schema") != "next-action-policy-v1"
            or self.metadata.get("action_schema") != ACTION_SCHEMA
            or self.metadata.get("state_schema") != STATE_SCHEMA
        ):
            raise ValueError(
                "Legacy formation model rejected; train a next-action model with rift train"
            )
        verify_bundle_manifest(directory)
        self.name = directory.name
        self.encoder = StateEncoder(ItemCatalog(self.metadata["item_catalog"]))
        if (
            self.encoder.names != self.metadata["feature_names"]
            or {int(k): v for k, v in self.metadata["action_names"].items()} != ACTION_NAMES
        ):
            raise ValueError("Model contract mismatch")
        self.device = device
        self.algorithm = self.metadata["algorithm"]
        architecture = self.metadata.get("network_architecture", "shared_v1")
        bounded = self.metadata["train_config"].get("bounded_values", False)
        expected = (
            "independent_actor_v1"
            if self.algorithm == "iql"
            else "bounded_shared_v1"
            if bounded
            else "shared_v1"
        )
        if bounded and self.algorithm != "cql":
            raise ValueError("Bounded architecture requires CQL")
        if self.algorithm not in {"cql", "bc", "iql"} or architecture != expected:
            raise ValueError("Invalid model architecture contract")
        self.network = OptionNetwork(
            len(self.encoder.names), separate_actor=self.algorithm == "iql", bounded_values=bounded
        ).to(device)
        self.network.load_state_dict(
            torch.load(directory / "policy.pt", map_location=device, weights_only=True)
        )
        self.network.eval()
        self.config = self.metadata["train_config"]
        self.counts = np.asarray(self.metadata["training_report"]["counts_by_group"])
        if self.counts.shape != (7, len(Action)) or np.any(self.counts < 0):
            raise ValueError("Invalid model support contract")

    def predict_batch(self, observations, groups, masks):
        observed = np.asarray(observations, dtype=np.float32)
        if (
            observed.ndim != 2
            or observed.shape[1] != len(self.encoder.names)
            or not np.isfinite(observed).all()
        ):
            raise ValueError("Invalid policy observation")
        predicted = []
        scores = []
        with torch.inference_mode():
            for start in range(0, len(observed), 1024):
                values, logits = self.network(
                    torch.as_tensor(observed[start : start + 1024], device=self.device)
                )
                p, s, _ = rank_options(
                    values.cpu().numpy(),
                    logits.cpu().numpy(),
                    self.counts,
                    groups[start : start + 1024],
                    masks[start : start + 1024],
                    algorithm=self.algorithm,
                    minimum_support=self.config["minimum_support"],
                    behavior_floor=self.config["behavior_floor"],
                )
                predicted.append(p)
                scores.append(s)
        return np.concatenate(predicted), np.concatenate(scores)

    def recommend(self, current, team, actor=None, *, role=None, previous=None):
        player = current.player(actor) if actor is not None else None
        if player and player.team != team:
            raise ValueError("Player does not belong to selected team")
        role = (role or player.role) if player else "TEAM"
        context = {
            "team": team,
            "actor": actor or "team",
            "role": role,
            "game_time": current.time,
            "subject": player.champion if player else "Team",
            "horizon_seconds": 60,
        }
        if player and role not in ROLES:
            return {
                **context,
                "status": "needs_role",
                "message": "Select your role so the model can condition its recommendation.",
            }
        if current.time < 90:
            return {
                **context,
                "status": "opening",
                "message": "Follow your opening plan. The learned decision window starts after 90 seconds.",
            }
        state = self.encoder.encode(current, team, actor, role=role, previous=previous)
        masks = feasible_actions(state, self.encoder.names)
        group = group_index(actor is None, role)
        predicted, scores = self.predict_batch(state[None, :], np.asarray([group]), masks)
        if predicted[0] < 0:
            return {
                **context,
                "status": "insufficient_support",
                "message": "No feasible action has enough learned support for this observation.",
            }
        action = Action(int(predicted[0]))
        spec = action_catalog()[action]
        return {
            **context,
            "status": "respawning" if action == Action.WAIT_RESPAWN else "recommendation",
            "action_id": int(action),
            "action": spec["name"],
            "instruction": spec["instruction"],
            "check": spec["check"],
            "scores": [float(v) if np.isfinite(v) else None for v in scores[0]],
            "score_meaning": "discounted return estimate; not a win probability"
            if self.algorithm == "cql"
            else "advantage-weighted policy logit; not a win probability"
            if self.algorithm == "iql"
            else "behavior-support logit; not a win probability",
            "method": self.algorithm,
            "partial_observation": True,
        }
