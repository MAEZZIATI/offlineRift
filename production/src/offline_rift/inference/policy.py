"""Portable inference: classifier bundles need only Torch; legacy CQL loads lazily."""

from pathlib import Path
from typing import Protocol

import numpy as np
from rift_common.artifacts import read_json, verify_bundle_manifest
from rift_common.pipeline.actions import action_names
from rift_common.pipeline.features import SCHEMA_VERSION, FeatureConfig, FeatureEncoder, ItemCatalog


class Policy(Protocol):
    config: FeatureConfig
    encoder: FeatureEncoder
    metadata: dict

    def recommend(self, observation: np.ndarray) -> dict: ...


class MacroPolicy:
    def __init__(self, bundle_dir: str | Path, *, device: str = "cpu"):
        bundle_dir = Path(bundle_dir)
        verify_bundle_manifest(bundle_dir)
        self.metadata = read_json(bundle_dir / "metadata.json")
        bundle_type = self.metadata.get("bundle_type", "legacy-cql")
        if bundle_type not in {"legacy-cql", "imitation-v1"}:
            raise ValueError("Unsupported model bundle type")
        if bundle_type == "imitation-v1":
            verify_bundle_manifest(bundle_dir, required=True)
        self.config = FeatureConfig(**self.metadata["feature_config"])
        self.encoder = FeatureEncoder(self.config, ItemCatalog(self.metadata["item_catalog"]))
        if (
            self.metadata["schema_version"] != SCHEMA_VERSION
            or self.metadata["feature_names"] != self.encoder.names
        ):
            raise ValueError("Policy feature schema mismatch")
        self.action_names = action_names(self.metadata)
        self.device = device
        report = self.metadata["training_report"]
        self.prediction_mode = report["prediction_mode"]
        if self.prediction_mode not in {"cql", "balanced_bc", "hybrid"}:
            raise ValueError("Unsupported prediction mode")
        self.model = None
        if bundle_type == "legacy-cql":
            import d3rlpy
            from rift_common.models import cql  # noqa: F401

            self.model = d3rlpy.load_learnable(str(bundle_dir / "policy.d3"), device=device)
            if self.model.action_size != len(self.action_names) or tuple(
                self.model.observation_shape
            ) != (len(self.encoder.names),):
                raise ValueError("Policy dimensions mismatch")
        elif self.prediction_mode != "balanced_bc":
            raise ValueError("Classifier bundle requires balanced_bc prediction mode")
        self.predictor = None
        if self.prediction_mode != "cql":
            import torch
            from rift_common.models.predictor import create_predictor

            self.predictor = create_predictor(
                len(self.encoder.names), self.config.grid_size, report["encoder_type"]
            ).to(device)
            self.predictor.load_state_dict(
                torch.load(
                    bundle_dir / "macro_predictor.pt", map_location=device, weights_only=True
                )
            )
            self.predictor.eval()

    def _scores(self, observations):
        size = len(self.action_names)
        q_values = None
        if self.model is not None:
            q_values = self.model.predict_value(
                np.repeat(observations, size, axis=0),
                np.tile(np.arange(size), len(observations)),
            ).reshape(len(observations), size)
            if not np.isfinite(q_values).all():
                raise ValueError("Non-finite value prediction")
        if self.predictor is None:
            return q_values, q_values
        from rift_common.models.predictor import hybrid_scores, predictor_logits

        logits = predictor_logits(self.predictor, observations, self.device)
        values = q_values if self.prediction_mode == "hybrid" else np.zeros_like(logits)
        scores = hybrid_scores(
            logits,
            values,
            self.metadata["training_report"]["action_counts"],
            observations,
            self.encoder.names,
        )
        return scores, q_values

    def _validate(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        if (
            observations.ndim != 2
            or observations.shape[1] != len(self.encoder.names)
            or not np.isfinite(observations).all()
        ):
            raise ValueError("Invalid policy input")
        return observations

    def predict_batch(self, observations: np.ndarray, *, batch_size=1024) -> np.ndarray:
        observations = self._validate(observations)
        if batch_size < 1:
            raise ValueError("Prediction batch size must be positive")
        return (
            np.concatenate(
                [
                    self._scores(observations[start : start + batch_size])[0].argmax(axis=1)
                    for start in range(0, len(observations), batch_size)
                ]
            )
            if len(observations)
            else np.empty(0, dtype=np.int64)
        )

    def recommend(self, observations: np.ndarray) -> dict:
        observation = np.asarray(observations, dtype=np.float32)
        if observation.shape != (len(self.encoder.names),):
            raise ValueError("Invalid policy input")
        scores, values = self._scores(self._validate(observation[None, :]))
        action = int(scores[0].argmax())
        pattern = self.predictor is not None
        return {
            "action_id": action,
            "action": self.action_names[action],
            "action_schema": self.metadata["action_schema"],
            "prediction_method": self.prediction_mode,
            "macro_scores": [float(v) if np.isfinite(v) else None for v in scores[0]]
            if pattern
            else None,
            "score_meaning": "weak-label evidence; not probabilities"
            if pattern
            else "discounted-return estimates; not probabilities",
            "q_value": float(values[0, action]) if values is not None else None,
            "q_values": values[0].tolist() if values is not None else None,
            "evaluation": "weak formation labels; strategic utility unestablished",
        }
