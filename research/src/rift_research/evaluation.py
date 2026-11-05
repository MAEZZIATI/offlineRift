"""Frozen next-action diagnostics, with explicit separation from policy-win-rate claims."""

from pathlib import Path

import numpy as np
from rift_common.actions import ACTION_NAMES, GROUPS
from rift_common.artifacts import atomic_json, file_digest
from rift_common.dataset import load_dataset
from rift_common.metrics import behavior_metrics


def describe_dataset(path):
    arrays, metadata = load_dataset(path)
    counts = np.zeros((len(GROUPS), len(ACTION_NAMES)), dtype=int)
    np.add.at(counts, (arrays["groups"], arrays["actions"]), 1)
    return {
        "matches": len(set(arrays["match_ids"])),
        "episodes": len(set(arrays["episode_ids"])),
        "transitions": len(arrays["actions"]),
        "duration_seconds": {
            "min": float(arrays["durations"].min()),
            "median": float(np.median(arrays["durations"])),
            "max": float(arrays["durations"].max()),
        },
        "action_names": ACTION_NAMES,
        "counts_by_group": {group: counts[i].tolist() for i, group in enumerate(GROUPS)},
        "behavior_evidence": metadata["behavior_evidence"],
        "excluded": metadata["excluded"],
        "state_schema": metadata["state_schema"],
        "action_schema": metadata["action_schema"],
    }


def evaluate_policy(dataset, bundle, output, *, samples=500):
    from offline_rift.inference.policy import NextActionPolicy

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    arrays, metadata = load_dataset(dataset)
    policy = NextActionPolicy(bundle)
    report = policy.metadata["training_report"]
    training, validation = set(report["training_matches"]), set(report["validation_matches"])
    if (
        not training
        or not validation
        or training & validation
        or training | validation != set(arrays["match_ids"])
    ):
        raise ValueError("Invalid or incompatible model match split")
    if metadata["provenance"]["fingerprint"] != policy.metadata["provenance"]["fingerprint"]:
        raise ValueError("Dataset differs from the frozen model corpus")
    mask = np.isin(arrays["match_ids"], list(validation))
    predicted, scores = policy.predict_batch(
        arrays["observations"][mask], arrays["groups"][mask], arrays["action_masks"][mask]
    )
    truth = arrays["actions"][mask]
    metrics = behavior_metrics(truth, predicted, len(ACTION_NAMES))
    if metrics != report["validation"]:
        raise ValueError("Serving predictions do not reproduce training evaluation")
    matches, groups = np.unique(arrays["match_ids"][mask], return_inverse=True)
    matrices = np.zeros((len(matches), len(ACTION_NAMES), len(ACTION_NAMES) + 1), dtype=int)
    np.add.at(matrices, (groups, truth, np.where(predicted < 0, len(ACTION_NAMES), predicted)), 1)
    if samples < 1:
        raise ValueError("Positive bootstrap samples required")
    rng = np.random.default_rng(20261105)
    estimates = []
    for _ in range(samples):
        matrix = matrices[rng.integers(len(matches), size=len(matches))].sum(0)
        supported = matrix.sum(1)
        proposed = matrix[:, : len(ACTION_NAMES)].sum(0)
        diagonal = matrix[:, : len(ACTION_NAMES)].diagonal()
        f1 = np.divide(
            2 * diagonal,
            supported + proposed,
            out=np.zeros(len(ACTION_NAMES)),
            where=supported + proposed > 0,
        )
        estimates.append(f1[(supported + proposed) > 0].mean())
    result = {
        "dataset_sha256": file_digest(dataset),
        "bundle": str(Path(bundle).resolve()),
        "algorithm": policy.algorithm,
        "validation_matches": sorted(validation),
        "metrics": metrics,
        "behavior_macro_f1_match_bootstrap_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
        "policy_winrate": None,
        "policy_improvement": None,
        "limitations": [
            "Behavior agreement and bootstrap intervals are diagnostics of inferred actions, not strategic utility",
            "These are previously consulted retrospective matches, not a new prospective test",
            "No simulator or authenticated live-game outcome study has been performed",
            "Shared team rewards and partial observations do not identify individual causal credit",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "predictions.npz",
        truth=truth,
        predicted=predicted,
        match_ids=arrays["match_ids"][mask],
        scores=scores,
    )
    atomic_json(output / "report.json", result)
    return result
