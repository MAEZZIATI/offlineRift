"""Compare frozen serving bundles on identical matches without fitting or promotion."""

import time
from pathlib import Path

import numpy as np
import torch
from offline_rift.inference.policy import MacroPolicy
from rift_common.artifacts import atomic_json, file_digest
from rift_common.models.evaluation import classification_metrics
from rift_common.pipeline.action_labeler import load_dataset

from rift_research.evaluation import match_bootstrap, validate_split


def compare_bundles(dataset, bundles, output, *, samples=1000, unseen=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    arrays, metadata = load_dataset(dataset)
    predictions, metrics, artifacts, latency = {}, {}, {}, {}
    reference_mask = reference_counts = None
    for index, bundle in enumerate(bundles):
        bundle = Path(bundle)
        policy = MacroPolicy(bundle)
        mask = validate_split(arrays, metadata, policy.metadata, unseen=unseen)
        counts = policy.metadata["training_report"]["action_counts"]
        if reference_mask is None:
            reference_mask, reference_counts = mask, counts
        elif not np.array_equal(reference_mask, mask) or counts != reference_counts:
            raise ValueError("Comparison requires identical training counts and evaluation matches")
        observed, truth = arrays["observations"][mask], arrays["actions"][mask]
        name = f"bundle-{index + 1}"
        predictions[name] = policy.predict_batch(observed)
        metrics[name] = classification_metrics(truth, predictions[name], len(counts))
        example = observed[0]
        for _ in range(5):
            policy.recommend(example)
        durations = []
        for _ in range(50):
            started = time.perf_counter()
            policy.recommend(example)
            durations.append((time.perf_counter() - started) * 1000)
        latency[name] = {
            "p50_ms": float(np.median(durations)),
            "p95_ms": float(np.quantile(durations, 0.95)),
            "samples": len(durations),
            "cpu_threads": 1,
            "includes_live_transport": False,
        }
        artifacts[name] = {
            "path": str(bundle.resolve()),
            "sha256": {
                path.name: file_digest(path)
                for path in (
                    bundle / name for name in ("metadata.json", "policy.d3", "macro_predictor.pt")
                )
                if path.exists()
            },
            "weight_bytes": sum(
                (bundle / filename).stat().st_size
                for filename in ("policy.d3", "macro_predictor.pt")
                if (bundle / filename).exists()
            ),
        }
    if not predictions:
        raise ValueError("At least one model bundle is required")
    truth = arrays["actions"][reference_mask]
    predictions["majority"] = np.full(len(truth), np.argmax(reference_counts))
    metrics["majority"] = classification_metrics(
        truth, predictions["majority"], len(reference_counts)
    )
    report = {
        "dataset_sha256": file_digest(dataset),
        "bundles": artifacts,
        "metrics": metrics,
        "latency": latency,
        "evaluation_matches": np.unique(arrays["match_ids"][reference_mask]).tolist(),
        "evaluation_transitions": len(truth),
        "unseen_from_saved_splits": unseen,
        "bootstrap": match_bootstrap(
            truth,
            predictions,
            arrays["match_ids"][reference_mask],
            len(reference_counts),
            samples=samples,
            comparisons=[(name, "bundle-1") for name in predictions if name != "bundle-1"],
        ),
        "limitations": [
            "Agreement with unreviewed heuristic labels is not strategic utility",
            "Development comparisons do not establish untouched-test generalization",
            "Latency is a local CPU microbenchmark, excluding transport and feature encoding",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "predictions.npz",
        truth=truth,
        match_ids=arrays["match_ids"][reference_mask],
        **predictions,
    )
    atomic_json(output / "report.json", report)
    return report
