"""Maintained classifier training with bounded accelerator memory and portable bundles."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rift_common.artifacts import (
    atomic_json,
    file_digest,
    read_json,
    write_bundle_manifest,
)
from rift_common.models import encoders, predictor, splits
from rift_common.models.evaluation import classification_metrics
from rift_common.models.predictor import hybrid_scores, predictor_logits
from rift_common.models.splits import MatchSplit
from rift_common.pipeline.action_labeler import load_dataset
from rift_common.pipeline.actions import action_names

from offline_rift.training.macro_predictor import fit_predictor


@dataclass(frozen=True)
class ImitationTrainer:
    encoder_type: str = "mlp"
    batch_size: int = 128
    device: str = "cpu"
    seed: int = 42
    sampling: str = "sqrt_frequency"

    def train(
        self,
        dataset_path,
        output_dir,
        *,
        n_steps=10000,
        validation_fraction=0.2,
        split_from=None,
    ):
        if n_steps < 1 or self.batch_size < 1:
            raise ValueError("Training steps and batch size must be positive")
        arrays, metadata = load_dataset(dataset_path)
        names = action_names(metadata)
        if split_from:
            reference = read_json(Path(split_from) / "metadata.json")
            for key in ("feature_names", "feature_config", "action_schema"):
                if reference[key] != metadata[key]:
                    raise ValueError(f"Split reference uses a different {key}")
            split = MatchSplit.from_report(reference["training_report"], arrays["match_ids"])
        else:
            split = MatchSplit.create(
                arrays["match_ids"], seed=self.seed, validation_fraction=validation_fraction
            )
        validation = split.validation_mask(arrays["match_ids"])
        training = {key: value[~validation] for key, value in arrays.items()}
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "running",
            "algorithm": "imitation",
            "settings": {**asdict(self), "n_steps": n_steps},
            "dataset_path": str(Path(dataset_path).resolve()),
            "dataset_sha256": file_digest(dataset_path),
            "split_reference": str(Path(split_from).resolve()) if split_from else None,
            "review_status": "not_assessed",
            "promoted": False,
            "source_sha256": {
                "trainer": file_digest(__file__),
                "optimizer": file_digest(Path(__file__).with_name("macro_predictor.py")),
                **{
                    module.__name__: file_digest(module.__file__)
                    for module in (encoders, predictor, splits)
                },
            },
        }
        atomic_json(output / "run.json", manifest)
        try:
            model, optimization = fit_predictor(
                training,
                output,
                grid_size=metadata["feature_config"]["grid_size"],
                encoder_type=self.encoder_type,
                n_steps=n_steps,
                batch_size=self.batch_size,
                seed=self.seed,
                device=self.device,
                sampling=self.sampling,
            )
            counts = np.bincount(training["actions"], minlength=len(names)).tolist()
            report = {
                **split.to_dict(),
                "seed": self.seed,
                "encoder_type": self.encoder_type,
                "prediction_mode": "balanced_bc",
                "n_steps": n_steps,
                "optimizer_steps": n_steps,
                "supervised_predictor": optimization,
                "action_counts": counts,
                "training_action_coverage": int(np.count_nonzero(counts)),
                "unsupported_training_actions": [
                    names[i] for i, count in enumerate(counts) if not count
                ],
                "train_transitions": int((~validation).sum()),
                "validation_transitions": int(validation.sum()),
            }
            if validation.any():
                observed, truth = arrays["observations"][validation], arrays["actions"][validation]
                logits = predictor_logits(model, observed, self.device)
                predicted = hybrid_scores(
                    logits, np.zeros_like(logits), counts, observed, metadata["feature_names"]
                ).argmax(axis=1)
                metrics = classification_metrics(truth, predicted, len(names))
                report.update({f"validation_{key}": value for key, value in metrics.items()})
                report["prediction_comparison"] = {"balanced_bc": metrics}
                report["collapsed_predictions"] = metrics["collapsed_predictions"]
                report["validation_majority_baseline"] = float(np.mean(truth == np.argmax(counts)))
                report["majority_comparison"] = classification_metrics(
                    truth, np.full(len(truth), np.argmax(counts)), len(names)
                )
            metadata.update(bundle_type="imitation-v1", training_report=report)
            atomic_json(output / "metadata.json", metadata)
            write_bundle_manifest(output, ("metadata.json", "macro_predictor.pt"))
            manifest["status"] = "complete"
            return report
        except BaseException as error:
            manifest.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
            )
            raise
        finally:
            atomic_json(output / "run.json", manifest)
