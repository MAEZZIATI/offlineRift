"""d3rlpy 2.8 discrete CQL training and portable policy bundle loading."""

from __future__ import annotations

import json
from math import gcd
from pathlib import Path

import numpy as np

from offline_rift.pipeline.action_labeler import load_dataset
from offline_rift.pipeline.actions import ACTION_NAMES, action_names
from offline_rift.pipeline.features import (
    SCHEMA_VERSION,
    FeatureConfig,
    FeatureEncoder,
    ItemCatalog,
)


class OfflineMacroTrainer:
    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 64,
        alpha: float = 1.0,
        gamma: float = 0.99,
        seed: int = 42,
        encoder_type: str = "team_set",
        prediction_mode: str = "hybrid",
    ):
        import d3rlpy
        from d3rlpy.algos import DiscreteCQLConfig
        from d3rlpy.models.encoders import VectorEncoderFactory

        if (
            batch_size <= 0
            or alpha < 0
            or not 0 < gamma <= 1
            or encoder_type not in {"mlp", "team_set"}
            or prediction_mode not in {"cql", "balanced_bc", "hybrid"}
        ):
            raise ValueError("Invalid training hyperparameters")
        d3rlpy.seed(seed)
        self.seed = seed
        self.encoder_type = encoder_type
        self.prediction_mode = prediction_mode
        self.device = device
        self.model_config = DiscreteCQLConfig(
            batch_size=batch_size,
            alpha=alpha,
            gamma=gamma,
            encoder_factory=VectorEncoderFactory(hidden_units=[128, 128]),
        )
        self.model = self.model_config.create(device=device)

    @staticmethod
    def to_mdp_dataset(arrays: dict, action_size: int = len(ACTION_NAMES)):
        from d3rlpy.constants import ActionSpace
        from d3rlpy.dataset import MDPDataset

        # Complete terminal transitions do not bootstrap. d3rlpy supplies a zero
        # next observation there; all nonterminal next states were validated.
        return MDPDataset(
            observations=arrays["observations"],
            actions=arrays["actions"],
            rewards=arrays["rewards"],
            terminals=arrays["terminals"].astype(np.float32),
            action_space=ActionSpace.DISCRETE,
            action_size=action_size,
        )

    def train(
        self,
        dataset_path: str | Path,
        output_dir: str | Path,
        *,
        n_steps: int = 10000,
        n_steps_per_epoch: int = 1000,
        validation_fraction: float = 0.2,
    ) -> dict:
        if n_steps <= 0 or n_steps_per_epoch <= 0 or not 0 <= validation_fraction < 1:
            raise ValueError("Invalid fit settings")
        arrays, metadata = load_dataset(dataset_path)
        names = action_names(metadata)
        action_size = len(names)
        if self.encoder_type == "team_set":
            from offline_rift.training.encoders import TeamSetEncoderFactory

            self.model_config.encoder_factory = TeamSetEncoderFactory(
                grid_size=metadata["feature_config"]["grid_size"]
            )
        self.model = self.model_config.create(device=self.device)
        matches = np.unique(arrays["match_ids"])
        rng = np.random.default_rng(self.seed)
        rng.shuffle(matches)
        count = (
            max(1, int(len(matches) * validation_fraction))
            if (len(matches) > 1 and validation_fraction > 0)
            else 0
        )
        validation_ids = matches[:count]
        # Split by match, keeping both team perspectives together.
        validation_mask = np.isin(arrays["match_ids"], validation_ids)
        train_arrays = {key: value[~validation_mask] for key, value in arrays.items()}
        dataset = self.to_mdp_dataset(train_arrays, action_size)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from d3rlpy.logging import FileAdapterFactory

        self.model.fit(
            dataset,
            n_steps=n_steps,
            # d3rlpy rounds incomplete epochs down; choose a divisor so all requested
            # updates run, even when n_steps is not a multiple of the epoch setting.
            n_steps_per_epoch=gcd(n_steps, n_steps_per_epoch),
            experiment_name="macro_cql",
            show_progress=False,
            logger_adapter=FileAdapterFactory(root_dir=str(output / "logs")),
        )
        self.model.save(str(output / "policy.d3"))
        report = {
            "seed": self.seed,
            "training_matches": matches[count:].tolist(),
            "validation_matches": validation_ids.tolist(),
            "train_transitions": int((~validation_mask).sum()),
            "validation_transitions": int(validation_mask.sum()),
            "action_counts": np.bincount(train_arrays["actions"], minlength=action_size).tolist(),
            "encoder_type": self.encoder_type,
            "action_schema": metadata["action_schema"],
            "n_steps": n_steps,
            "optimizer_steps": int(self.model.grad_step),
        }
        from offline_rift.training.evaluation import classification_metrics
        from offline_rift.training.macro_predictor import (
            fit_predictor,
            hybrid_scores,
            predictor_logits,
        )

        report["prediction_mode"] = self.prediction_mode
        report["training_action_coverage"] = int(np.count_nonzero(report["action_counts"]))
        report["unsupported_training_actions"] = [
            names[i] for i, n in enumerate(report["action_counts"]) if not n
        ]
        predictor = None
        if self.prediction_mode != "cql":
            predictor, predictor_report = fit_predictor(
                train_arrays,
                output,
                grid_size=metadata["feature_config"]["grid_size"],
                encoder_type=self.encoder_type,
                n_steps=n_steps,
                batch_size=self.model_config.batch_size,
                seed=self.seed,
                device=self.device,
            )
            report["supervised_predictor"] = predictor_report
        if validation_mask.any():
            observations = arrays["observations"][validation_mask]
            truth = arrays["actions"][validation_mask]
            predictions = {"cql": self.model.predict(observations)}
            if predictor is not None:
                logits = predictor_logits(predictor, observations, self.device)
                q_values = np.column_stack(
                    [
                        self.model.predict_value(observations, np.full(len(observations), i))
                        for i in range(action_size)
                    ]
                )
                predictions["balanced_bc"] = hybrid_scores(
                    logits,
                    np.zeros_like(q_values),
                    report["action_counts"],
                    observations,
                    metadata["feature_names"],
                ).argmax(axis=1)
                predictions["hybrid"] = hybrid_scores(
                    logits,
                    q_values,
                    report["action_counts"],
                    observations,
                    metadata["feature_names"],
                ).argmax(axis=1)
            report["prediction_comparison"] = {
                method: classification_metrics(truth, predicted, action_size)
                for method, predicted in predictions.items()
            }
            selected = report["prediction_comparison"][self.prediction_mode]
            report.update({f"validation_{k}": v for k, v in selected.items()})
            report["collapsed_predictions"] = selected["collapsed_predictions"]
            majority = int(np.argmax(report["action_counts"]))
            report["validation_majority_baseline"] = float(np.mean(truth == majority))
            report["majority_comparison"] = classification_metrics(
                truth, np.full(len(truth), majority), action_size
            )
        metadata["training_report"] = report
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return report


class MacroPolicy:
    def __init__(self, bundle_dir: str | Path, *, device: str = "cpu"):
        import d3rlpy

        # Register custom factories before loading portable d3rlpy configuration.
        from offline_rift.training import encoders  # noqa: F401

        bundle_dir = Path(bundle_dir)
        self.metadata = json.loads((bundle_dir / "metadata.json").read_text())
        self.config = FeatureConfig(**self.metadata["feature_config"])
        self.encoder = FeatureEncoder(self.config, ItemCatalog(self.metadata["item_catalog"]))
        if (
            self.metadata["schema_version"] != SCHEMA_VERSION
            or self.metadata["feature_names"] != self.encoder.names
        ):
            raise ValueError("Policy feature schema mismatch")
        self.model = d3rlpy.load_learnable(str(bundle_dir / "policy.d3"), device=device)
        self.action_names = action_names(self.metadata)
        if self.model.action_size != len(self.action_names) or tuple(
            self.model.observation_shape
        ) != (len(self.encoder.names),):
            raise ValueError("Policy dimensions mismatch")

        self.device = device
        self.prediction_mode = self.metadata["training_report"]["prediction_mode"]
        self.predictor = None
        if self.prediction_mode != "cql":
            import torch

            from offline_rift.training.macro_predictor import create_predictor

            self.predictor = create_predictor(
                len(self.encoder.names),
                self.config.grid_size,
                self.metadata["training_report"]["encoder_type"],
            ).to(device)
            self.predictor.load_state_dict(
                torch.load(
                    bundle_dir / "macro_predictor.pt", map_location=device, weights_only=True
                )
            )
            self.predictor.eval()

    def recommend(self, observations: np.ndarray) -> dict:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape != (len(self.encoder.names),) or not np.isfinite(observations).all():
            raise ValueError("Invalid policy input")
        batch = observations.reshape(1, -1)
        action = int(self.model.predict(batch)[0])
        size = len(self.action_names)
        q_values = self.model.predict_value(np.repeat(batch, size, axis=0), np.arange(size))
        scores = None
        if self.predictor is not None:
            from offline_rift.training.macro_predictor import hybrid_scores, predictor_logits

            logits = predictor_logits(self.predictor, batch, self.device)
            values = (
                q_values.reshape(1, -1) if self.prediction_mode == "hybrid" else np.zeros((1, size))
            )
            scores = hybrid_scores(
                logits,
                values,
                self.metadata["training_report"]["action_counts"],
                batch,
                self.encoder.names,
            )[0]
            action = int(scores.argmax())
        if not np.isfinite(q_values).all() or action not in self.action_names:
            raise ValueError("Invalid policy output")
        return {
            "action_id": action,
            "action": self.action_names[action],
            "action_schema": self.metadata["action_schema"],
            "prediction_method": self.prediction_mode,
            "macro_scores": [float(v) if np.isfinite(v) else None for v in scores]
            if scores is not None
            else None,
            "score_meaning": "balanced weak-label evidence with bounded Q tie-breaker; not probabilities",
            "q_value": float(q_values[action]),
            "q_values": q_values.tolist(),
            "evaluation": "discounted_return; not a calibrated win probability",
        }
