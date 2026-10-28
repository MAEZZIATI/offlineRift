"""d3rlpy 2.8 discrete CQL training and portable policy bundle loading."""

from __future__ import annotations

import json
from math import gcd
from pathlib import Path

import numpy as np

from action_labeler import ACTION_NAMES, load_dataset
from features import SCHEMA_VERSION, FeatureConfig, FeatureEncoder, ItemCatalog


class OfflineMacroTrainer:
    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 64,
        alpha: float = 1.0,
        gamma: float = 0.99,
        seed: int = 42,
    ):
        import d3rlpy
        from d3rlpy.algos import DiscreteCQLConfig
        from d3rlpy.models.encoders import VectorEncoderFactory

        if batch_size <= 0 or alpha < 0 or not 0 < gamma <= 1:
            raise ValueError("Invalid training hyperparameters")
        d3rlpy.seed(seed)
        self.seed = seed
        self.model = DiscreteCQLConfig(
            batch_size=batch_size,
            alpha=alpha,
            gamma=gamma,
            encoder_factory=VectorEncoderFactory(hidden_units=[128, 128]),
        ).create(device=device)

    @staticmethod
    def to_mdp_dataset(arrays: dict):
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
            action_size=len(ACTION_NAMES),
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
        dataset = self.to_mdp_dataset(train_arrays)
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
            "action_counts": np.bincount(train_arrays["actions"], minlength=5).tolist(),
            "n_steps": n_steps,
            "optimizer_steps": int(self.model.grad_step),
        }
        if validation_mask.any():
            predicted = self.model.predict(arrays["observations"][validation_mask])
            report["validation_heuristic_agreement"] = float(
                np.mean(predicted == arrays["actions"][validation_mask])
            )
            majority = int(np.bincount(train_arrays["actions"], minlength=5).argmax())
            report["validation_majority_baseline"] = float(
                np.mean(arrays["actions"][validation_mask] == majority)
            )
            report["validation_prediction_counts"] = np.bincount(predicted, minlength=5).tolist()
        metadata["training_report"] = report
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return report


class MacroPolicy:
    def __init__(self, bundle_dir: str | Path, *, device: str = "cpu"):
        import d3rlpy

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
        if self.model.action_size != 5 or tuple(self.model.observation_shape) != (
            len(self.encoder.names),
        ):
            raise ValueError("Policy dimensions mismatch")

    def recommend(self, observations: np.ndarray) -> dict:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape != (len(self.encoder.names),) or not np.isfinite(observations).all():
            raise ValueError("Invalid policy input")
        batch = observations.reshape(1, -1)
        action = int(self.model.predict(batch)[0])
        q_values = self.model.predict_value(np.repeat(batch, 5, axis=0), np.arange(5))
        if not np.isfinite(q_values).all() or action not in ACTION_NAMES:
            raise ValueError("Invalid policy output")
        return {
            "action_id": action,
            "action": ACTION_NAMES[action],
            "q_value": float(q_values[action]),
            "q_values": q_values.tolist(),
            "evaluation": "discounted_return; not a calibrated win probability",
        }
