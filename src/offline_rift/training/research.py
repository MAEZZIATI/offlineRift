"""Read-only bundle ablations, match bootstrap, and blinded weak-label audit sampling."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from offline_rift.pipeline.action_labeler import load_dataset
from offline_rift.training.evaluation import classification_metrics

METRICS = ("heuristic_agreement", "balanced_accuracy", "macro_f1")


def match_bootstrap(
    truth, predictions, match_ids, size, *, samples=2000, seed=20261104, comparisons=None
):
    """Paired percentile intervals for pooled transition metrics, clustering both teams.

    Conditions on fitted weights and the existing development split. It measures neither
    optimizer-seed uncertainty nor human-label validity. Matches are the resampling unit.
    """
    if samples < 1 or len(truth) == 0:
        raise ValueError("Bootstrap needs positive samples and nonempty observations")
    matches, groups = np.unique(match_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(matches), size=(samples, len(matches)))
    multiplicity = np.zeros((samples, len(matches)), dtype=np.int64)
    np.add.at(multiplicity, (np.arange(samples)[:, None], draws), 1)
    estimates = {}
    for method, predicted in predictions.items():
        matrices = np.zeros((len(matches), size, size), dtype=np.int64)
        np.add.at(matrices, (groups, truth, predicted), 1)
        # Avoid a samples × matches × classes² temporary for larger held-out cohorts.
        confusion = (multiplicity @ matrices.reshape(len(matches), -1)).reshape(samples, size, size)
        support, proposed = confusion.sum(axis=2), confusion.sum(axis=1)
        diagonal = confusion.diagonal(axis1=1, axis2=2)
        recall = np.divide(diagonal, support, out=np.zeros_like(diagonal, float), where=support > 0)
        f1 = np.divide(
            2 * diagonal,
            support + proposed,
            out=np.zeros_like(diagonal, float),
            where=support + proposed > 0,
        )
        estimates[method] = np.column_stack(
            (
                diagonal.sum(axis=1) / support.sum(axis=1),
                recall.sum(axis=1) / (support > 0).sum(axis=1),
                f1.sum(axis=1) / (support + proposed > 0).sum(axis=1),
            )
        )

    def intervals(values):
        bounds = np.quantile(values, [0.025, 0.975], axis=0)
        return {name: bounds[:, i].tolist() for i, name in enumerate(METRICS)}

    comparisons = (
        comparisons
        if comparisons is not None
        else (
            ("cql_filtered", "cql_raw"),
            ("bc_filtered", "bc_raw"),
            ("hybrid_filtered", "hybrid_raw"),
            ("hybrid_filtered", "bc_filtered"),
            ("bc_filtered", "cql_filtered"),
        )
    )
    return {
        "samples": samples,
        "seed": seed,
        "unit": "match (both teams together)",
        "confidence": 0.95,
        "method": "paired percentile bootstrap; exploratory, unadjusted",
        "intervals": {name: intervals(values) for name, values in estimates.items()},
        "paired_differences": {
            f"{left}_minus_{right}": intervals(estimates[left] - estimates[right])
            for left, right in comparisons
            if left in estimates and right in estimates
        },
    }


def validate_split(arrays, metadata, bundle, *, unseen=False):
    """Refuse an evaluation on another dataset or on overlapping training matches."""
    for key in ("schema_version", "feature_names", "feature_config", "action_schema", "provenance"):
        if unseen and key == "provenance":
            continue  # A new corpus must share the feature/action contract, not its fingerprint.
        if metadata[key] != bundle[key]:
            raise ValueError(f"Dataset/bundle mismatch: {key}")
    report = bundle["training_report"]
    train, validation = report["training_matches"], report["validation_matches"]
    if unseen:
        if not train or not validation or set(train) & set(validation):
            raise ValueError("Invalid saved match split")
        mask = ~np.isin(arrays["match_ids"], train + validation)
        if not mask.any():
            raise ValueError("No unseen matches outside saved training/development splits")
        return mask
    if (
        not train
        or not validation
        or len(set(train)) != len(train)
        or len(set(validation)) != len(validation)
        or set(train) & set(validation)
        or set(train) | set(validation) != set(arrays["match_ids"])
    ):
        raise ValueError("Invalid or incomplete saved match split")
    mask = np.isin(arrays["match_ids"], validation)
    if int(mask.sum()) != report["validation_transitions"]:
        raise ValueError("Validation transition count differs from saved report")
    counts = np.bincount(arrays["actions"][~mask], minlength=len(report["action_counts"]))
    if counts.tolist() != report["action_counts"]:
        raise ValueError("Training label counts differ from saved report")
    return mask


def audit_rows(arrays, training_matches, patches, *, per_cell=3, seed=20261104):
    """Equal patch/label strata, uniform matches then uniform states within each match.

    Returns blinded reviewer rows and a separate key. No annotation is inferred.
    Windows may overlap across strata; IDs retain their match for clustered analysis.
    """
    if per_cell < 1:
        raise ValueError("Audit per-cell count must be positive")
    rng = np.random.default_rng(seed)
    eligible = np.isin(arrays["match_ids"], training_matches)
    rows, keys = [], []
    for patch in sorted(set(patches.values())):
        for label in range(12):
            indices = np.flatnonzero(
                eligible
                & (arrays["actions"] == label)
                & np.isin(arrays["match_ids"], [m for m, p in patches.items() if p == patch])
            )
            matches = np.unique(arrays["match_ids"][indices])
            for match in rng.permutation(matches)[:per_cell]:
                index = int(rng.choice(indices[arrays["match_ids"][indices] == match]))
                timestamp = int(arrays["timestamps"][index])
                sample = f"window-{len(rows) + 1:04d}"
                rows.append(
                    {
                        "sample_id": sample,
                        "match_id": str(match),
                        "patch": patch,
                        "team": int(arrays["teams"][index]),
                        "timestamp_seconds": timestamp,
                        "review_start_seconds": max(0, timestamp - 60),
                        "review_end_seconds": timestamp + 30,
                        "reviewer_1_label": "",
                        "reviewer_2_label": "",
                        "reviewer_1_evidence_sufficient": "",
                        "reviewer_2_evidence_sufficient": "",
                        "intent_observable": "",
                        "adjudicated_label": "",
                        "notes": "",
                    }
                )
                keys.append(
                    {
                        "sample_id": sample,
                        "weak_label": label,
                        "label_confidence": float(arrays["label_confidence"][index]),
                    }
                )
    # Hide patch/label ordering as well as label values from the reviewer sheet.
    order = rng.permutation(len(rows))
    rows, keys = [rows[i] for i in order], [keys[i] for i in order]
    for i, (row, key) in enumerate(zip(rows, keys), 1):
        row["sample_id"] = key["sample_id"] = f"window-{i:04d}"
    return rows, keys


def evaluate_bundle(
    dataset, bundle, output, *, samples=2000, seed=20261104, raw_dir=None, per_cell=3, unseen=False
):
    """Evaluate frozen weights without training or publishing a latest-model reference."""
    import torch

    from offline_rift.training.macro_predictor import hybrid_scores, predictor_logits
    from offline_rift.training.rl_agent import MacroPolicy

    torch.set_num_threads(1)
    dataset, bundle, output = Path(dataset), Path(bundle), Path(output)
    arrays, metadata = load_dataset(dataset)
    bundle_metadata = json.loads((bundle / "metadata.json").read_text())
    mask = validate_split(arrays, metadata, bundle_metadata, unseen=unseen)
    policy = MacroPolicy(bundle)
    if policy.predictor is None:
        raise ValueError("A companion predictor is required for the controlled ablations")
    observations, truth = arrays["observations"][mask], arrays["actions"][mask]
    training = bundle_metadata["training_report"]
    size = len(training["action_counts"])
    logits = predictor_logits(policy.predictor, observations, "cpu")
    q_values = np.column_stack(
        [
            policy.model.predict_value(observations, np.full(len(truth), action))
            for action in range(size)
        ]
    )
    if not np.isfinite(q_values).all() or not np.isfinite(logits).all():
        raise ValueError("Non-finite model scores")
    advantage = q_values - q_values.mean(axis=1, keepdims=True)
    hybrid = logits + 0.25 * np.clip(
        advantage / np.maximum(q_values.std(axis=1, keepdims=True), 1e-6), -2, 2
    )
    predictions = {"majority": np.full(len(truth), np.argmax(training["action_counts"]))}
    for name, scores in (("cql", q_values), ("bc", logits), ("hybrid", hybrid)):
        predictions[f"{name}_raw"] = scores.argmax(axis=1)
        # Reuse the serving constraints identically for all three score functions.
        predictions[f"{name}_filtered"] = hybrid_scores(
            scores.copy(),
            np.zeros_like(scores),
            training["action_counts"],
            observations,
            metadata["feature_names"],
        ).argmax(axis=1)
    metrics = {
        name: classification_metrics(truth, pred, size) for name, pred in predictions.items()
    }
    # Verify exact reproduction of the previous published counts before any inference.
    saved_comparisons = (
        ()
        if unseen
        else (
            ("cql", "cql_raw"),
            ("balanced_bc", "bc_filtered"),
            ("hybrid", "hybrid_filtered"),
        )
    )
    for saved, current in saved_comparisons:
        if (
            metrics[current]["confusion_matrix"]
            != training["prediction_comparison"][saved]["confusion_matrix"]
        ):
            raise ValueError(f"Saved predictions failed reproduction: {saved}")
    bootstrap = match_bootstrap(
        truth, predictions, arrays["match_ids"][mask], size, samples=samples, seed=seed
    )
    audit, key = [], []
    if raw_dir is not None:
        patches = {}
        for match in training["training_matches"]:
            payload = json.loads((Path(raw_dir) / f"{match}.json").read_text())
            patches[match] = ".".join(payload["match"]["info"]["gameVersion"].split(".")[:2])
        audit, key = audit_rows(
            arrays, training["training_matches"], patches, per_cell=per_cell, seed=seed
        )
    report = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "software": {package: version(package) for package in ("numpy", "torch", "d3rlpy")},
        "purpose": "frozen-model unseen-match evaluation"
        if unseen
        else "exploratory frozen-model development-set ablation",
        "dataset": str(dataset.resolve()),
        "bundle": str(bundle.resolve()),
        "schema": metadata["schema_version"],
        "profile": metadata["feature_config"]["profile"],
        "provenance": metadata["provenance"],
        "artifact_sha256": {},
        "training_matches": training["training_matches"],
        "reference_validation_matches": training["validation_matches"],
        "validation_matches": np.unique(arrays["match_ids"][mask]).tolist(),
        "excluded_previously_seen_matches": len(np.unique(arrays["match_ids"][~mask])),
        "validation_transitions": len(truth),
        "filter_changed_predictions": {
            method: int(
                np.count_nonzero(predictions[f"{method}_raw"] != predictions[f"{method}_filtered"])
            )
            for method in ("cql", "bc", "hybrid")
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "audit": {
            "sampled_windows": len(audit),
            "annotations_completed": 0,
            "split": "training only",
            "per_patch_label_maximum": per_cell,
            "seed": seed,
            "sampling": "uniform match then state within patch/label",
        },
    }
    # Stream hashes; Python 3.10-compatible, without loading model files into memory.
    for path in (
        dataset,
        bundle / "metadata.json",
        bundle / "policy.d3",
        bundle / "macro_predictor.pt",
        Path(__file__),
    ):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        report["artifact_sha256"][str(path.resolve())] = digest.hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "validation_predictions.npz",
        truth=truth,
        match_ids=arrays["match_ids"][mask],
        teams=arrays["teams"][mask],
        timestamps=arrays["timestamps"][mask],
        **predictions,
    )
    if audit:
        for filename, rows in (("audit_blinded.csv", audit), ("audit_key.csv", key)):
            with (output / filename).open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    temporary = output / "report.pending.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output / "report.json")
    return report


def add_research_arguments(parser):
    parser.add_argument("--dataset", type=Path, required=True, help="Existing transitions.npz")
    parser.add_argument("--bundle", type=Path, required=True, help="Completed hybrid/BC bundle")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; never overwritten"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--seed", type=int, default=20261104, help="Resampling seed, not training seed"
    )
    parser.add_argument(
        "--raw-dir", type=Path, help="Cached raw payloads for patch-stratified audit"
    )
    parser.add_argument("--audit-per-cell", type=int, default=3)
    parser.add_argument(
        "--unseen",
        action="store_true",
        help="Evaluate only matches absent from both saved training and development splits",
    )


def run_research(args):
    report = evaluate_bundle(
        args.dataset,
        args.bundle,
        args.output,
        samples=args.bootstrap_samples,
        seed=args.seed,
        raw_dir=args.raw_dir,
        per_cell=args.audit_per_cell,
        unseen=args.unseen,
    )
    print(f"Completed frozen-model ablations: {args.output / 'report.json'}")
    for name, values in report["metrics"].items():
        print(
            f"{name}: agreement={values['heuristic_agreement']:.4f}, "
            f"balanced_accuracy={values['balanced_accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}, "
            f"categories={values['prediction_coverage']}/12"
        )
