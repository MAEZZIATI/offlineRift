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
        "identified_transitions": int(
            arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool)).sum()
        ),
        "excluded": metadata["excluded"],
        "state_schema": metadata["state_schema"],
        "action_schema": metadata["action_schema"],
    }


def evaluate_policy(dataset, bundle, output, *, samples=500, new_matches=False, stress=False):
    from offline_rift.inference.coach import load_policy

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    arrays, metadata = load_dataset(dataset)
    policy = load_policy(bundle)
    report = policy.metadata["training_report"]
    training, validation = set(report["training_matches"]), set(report["validation_matches"])
    if not training or not validation or training & validation:
        raise ValueError("Invalid or incompatible model match split")
    corpus = set(arrays["match_ids"])
    if new_matches:
        evaluated_matches = corpus - (training | validation)
        if not evaluated_matches:
            raise ValueError("No new matches outside the frozen training/development corpus")
    else:
        if training | validation | set(report.get("holdout_matches", [])) != corpus:
            raise ValueError("Invalid or incompatible model match split")
        if metadata["provenance"]["fingerprint"] != policy.metadata["provenance"]["fingerprint"]:
            raise ValueError("Dataset differs from the frozen model corpus")
        expected_digest = policy.metadata.get("dataset_sha256")
        if expected_digest is not None and file_digest(dataset) != expected_digest:
            raise ValueError("Frozen dataset integrity mismatch")
        evaluated_matches = validation
    if any(
        metadata[key] != policy.metadata[key]
        for key in ("feature_names", "reward", "time_unit_seconds")
    ):
        raise ValueError("Evaluation dataset contract differs from frozen policy")
    mask = np.isin(arrays["match_ids"], list(evaluated_matches))
    predicted, scores = policy.predict_batch(
        arrays["observations"][mask], arrays["groups"][mask], arrays["action_masks"][mask]
    )
    truth = arrays["actions"][mask]
    metrics = behavior_metrics(truth, predicted, len(ACTION_NAMES))
    if not new_matches and metrics != report["validation"]:
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
        "bundle_sha256": file_digest(
            Path(bundle)
            / ("coach.json" if (Path(bundle) / "coach.json").exists() else "bundle.json")
        ),
        "algorithm": policy.algorithm,
        "evaluation_mode": "new_matches" if new_matches else "frozen_validation",
        "validation_matches": sorted(validation),
        "evaluated_matches": sorted(evaluated_matches),
        "excluded_previously_seen_matches": sorted(corpus & (training | validation))
        if new_matches
        else [],
        "evaluated_transitions": int(mask.sum()),
        "majority_baseline": float(np.mean(truth == policy.counts.sum(0).argmax())),
        "retention_coverage": float(np.mean(predicted >= 0)),
        "metrics": metrics,
        "identified_metrics": behavior_metrics(
            truth[
                arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))[mask]
            ],
            predicted[
                arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))[mask]
            ],
            len(ACTION_NAMES),
        )
        if arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))[mask].any()
        else None,
        "per_group": {
            group: {
                **behavior_metrics(
                    truth[arrays["groups"][mask] == i],
                    predicted[arrays["groups"][mask] == i],
                    len(ACTION_NAMES),
                ),
                "training_action_support": policy.counts[i].tolist(),
                "identified": behavior_metrics(
                    truth[
                        (arrays["groups"][mask] == i)
                        & arrays.get(
                            "action_identified", np.ones(len(arrays["actions"]), dtype=bool)
                        )[mask]
                    ],
                    predicted[
                        (arrays["groups"][mask] == i)
                        & arrays.get(
                            "action_identified", np.ones(len(arrays["actions"]), dtype=bool)
                        )[mask]
                    ],
                    len(ACTION_NAMES),
                )
                if np.any(
                    (arrays["groups"][mask] == i)
                    & arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))[
                        mask
                    ]
                )
                else None,
                "retention_coverage": float(np.mean(predicted[arrays["groups"][mask] == i] >= 0)),
                "majority_baseline": float(
                    np.mean(truth[arrays["groups"][mask] == i] == policy.counts[i].argmax())
                ),
                "largest_prediction_fraction": float(
                    np.bincount(
                        predicted[(arrays["groups"][mask] == i) & (predicted >= 0)],
                        minlength=len(ACTION_NAMES),
                    ).max()
                    / max(1, np.sum((arrays["groups"][mask] == i) & (predicted >= 0)))
                ),
            }
            for i, group in enumerate(GROUPS)
            if np.any(arrays["groups"][mask] == i)
        },
        "score_quantiles": np.quantile(
            scores[np.isfinite(scores)], [0, 0.01, 0.5, 0.99, 1]
        ).tolist()
        if np.isfinite(scores).any()
        else [],
        "score_meaning": "Q estimate" if policy.algorithm == "cql" else "policy logit",
        "behavior_macro_f1_match_bootstrap_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
        "policy_winrate": None,
        "policy_improvement": None,
        "limitations": [
            "Behavior agreement and bootstrap intervals are diagnostics of inferred actions, not strategic utility",
            "New match IDs exclude this bundle's training/development corpus; prior human or other-model consultation is not identifiable"
            if new_matches
            else "These are previously consulted retrospective matches, not a new prospective test",
            "No simulator or authenticated live-game outcome study has been performed",
            "Shared team rewards and partial observations do not identify individual causal credit",
        ],
    }
    if stress:
        from rift_common.actions import feasible_actions
        from rift_common.observation_masks import COLD_START_FIELDS, HEALTH_FIELDS

        result["availability_stress"] = {}
        for scenario, fields in {
            "unknown_health": HEALTH_FIELDS,
            "cold_history": COLD_START_FIELDS,
        }.items():
            observed = arrays["observations"][mask].copy()
            observed[:, [metadata["feature_names"].index(field) for field in fields]] = 0
            omitted_masks = feasible_actions(observed, metadata["feature_names"])
            still_feasible = (predicted >= 0) & omitted_masks[
                np.arange(len(predicted)), np.maximum(predicted, 0)
            ]
            changed, _ = policy.predict_batch(
                observed,
                arrays["groups"][mask],
                omitted_masks,
            )
            result["availability_stress"][scenario] = {
                group: {
                    "changed_fraction": float(
                        np.mean(
                            changed[arrays["groups"][mask] == i]
                            != predicted[arrays["groups"][mask] == i]
                        )
                    ),
                    "retention": float(np.mean(changed[arrays["groups"][mask] == i] >= 0)),
                    "original_choice_made_infeasible_fraction": float(
                        np.mean(((predicted >= 0) & ~still_feasible)[arrays["groups"][mask] == i])
                    ),
                    "changed_when_original_still_feasible": float(
                        np.mean(
                            changed[(arrays["groups"][mask] == i) & still_feasible]
                            != predicted[(arrays["groups"][mask] == i) & still_feasible]
                        )
                    )
                    if np.any((arrays["groups"][mask] == i) & still_feasible)
                    else None,
                }
                for i, group in enumerate(GROUPS)
                if np.any(arrays["groups"][mask] == i)
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
