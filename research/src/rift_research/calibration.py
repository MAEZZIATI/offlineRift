"""Match-separated calibration of retained coverage, never action quality or win rate."""

from pathlib import Path

import numpy as np
import torch
from offline_rift.inference.policy import NextActionPolicy
from rift_common import observation_masks
from rift_common.actions import ACTION_NAMES, GROUPS, feasible_actions
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.dataset import load_dataset
from rift_common.metrics import behavior_metrics
from rift_common.observation_masks import (
    COLD_START_FIELDS,
    EVIDENCE_FIELDS,
    HEALTH_FIELDS,
    evidence_patterns,
)


def fit_evidence_thresholds(
    confidence, groups, patterns, matches, rows, *, target, minimum_rows=200, minimum_matches=20
):
    """Calibration inputs only; repeated source row/signature pairs count once."""
    _, unique = np.unique(
        np.asarray([f"{row}:{pattern}" for row, pattern in zip(rows, patterns)]), return_index=True
    )
    keys = np.asarray([f"{GROUPS[group]}:{pattern}" for group, pattern in zip(groups, patterns)])
    thresholds = {}
    for key in np.unique(keys):
        selected = unique[keys[unique] == key]
        match_count = len(np.unique(matches[selected]))
        thresholds[str(key)] = {
            "rows": len(selected),
            "matches": match_count,
            "threshold": max(
                float(np.quantile(confidence[selected], 1 - target)), float(np.nextafter(0.0, 1.0))
            )
            if len(selected) >= minimum_rows and match_count >= minimum_matches
            else None,
        }
    return thresholds


def conditional_coverage(policy, states, groups, truth, identifiers, fitted, names, targets):
    scenarios = {}
    for scenario, fields in {
        "observed": (),
        "unobserved_health": HEALTH_FIELDS,
        "cold_start_history": COLD_START_FIELDS,
        "cold_start_unobserved_health": (*COLD_START_FIELDS, *HEALTH_FIELDS),
    }.items():
        observed = states.copy()
        observed[:, [names.index(name) for name in fields]] = 0
        masks = feasible_actions(observed, names)
        predicted, confidence = _confidence(policy, observed, groups, masks)
        scenarios[scenario] = (predicted, confidence, evidence_patterns(observed, names), masks)
    # Each original row contributes once per distinct evidence signature, even if two
    # scenarios produce identical missingness (e.g. health in a team observation).
    calibration = {
        "confidence": np.concatenate([value[1][fitted] for value in scenarios.values()]),
        "patterns": np.concatenate([value[2][fitted] for value in scenarios.values()]),
        "groups": np.tile(groups[fitted], len(scenarios)),
        "matches": np.tile(identifiers[fitted], len(scenarios)),
        "rows": np.tile(np.flatnonzero(fitted), len(scenarios)),
    }
    result = {
        "evidence_fields": list(EVIDENCE_FIELDS),
        "minimum_rows": 200,
        "minimum_matches": 20,
        "scenarios": {},
        "thresholds": [],
    }
    for target in targets:
        thresholds = fit_evidence_thresholds(**calibration, target=target)
        result["thresholds"].append({"target_coverage": target, "cells": thresholds})
        for scenario, (predicted, confidence, patterns, masks) in scenarios.items():
            retained = predicted.copy()
            keys = np.asarray([f"{GROUPS[g]}:{p}" for g, p in zip(groups, patterns)])
            cells = {}
            for key in np.unique(keys[~fitted]):
                selected = (keys == key) & ~fitted
                threshold = thresholds.get(str(key), {}).get("threshold")
                rejected = selected if threshold is None else selected & (confidence < threshold)
                retained[rejected] = -1
                coverage = float(np.mean(retained[selected] >= 0))
                enough = selected.sum() >= 200 and len(np.unique(identifiers[selected])) >= 20
                cells[str(key)] = {
                    "rows": int(selected.sum()),
                    "matches": len(np.unique(identifiers[selected])),
                    "coverage": coverage,
                    "calibration_supported": threshold is not None,
                    "evaluation_supported": bool(enough),
                    "coverage_screen_pass": bool(
                        threshold is not None and abs(coverage - target) <= 0.05
                    )
                    if enough
                    else None,
                }
            by_group = {}
            retention_checks = []
            for i, group in enumerate(GROUPS):
                selected = (groups == i) & ~fitted
                if not selected.any():
                    continue
                actions = {}
                for action in np.unique(truth[selected]):
                    action_rows = selected & (truth == action)
                    enough = (
                        action_rows.sum() >= 100 and len(np.unique(identifiers[action_rows])) >= 20
                    )
                    retention = float(np.mean(retained[action_rows] >= 0))
                    actions[ACTION_NAMES[int(action)]] = {
                        "rows": int(action_rows.sum()),
                        "matches": len(np.unique(identifiers[action_rows])),
                        "state_retention": retention,
                        "target_infeasible_after_omission": int(
                            np.sum(~masks[action_rows, action])
                        ),
                        "retention_screen_pass": bool(retention >= 0.5) if enough else None,
                    }
                    if enough:
                        retention_checks.append(retention >= 0.5)
                by_group[group] = {
                    **_summary(truth[selected], retained[selected]),
                    "by_action": actions,
                }
            coverage_checks = [
                c["coverage_screen_pass"]
                for c in cells.values()
                if c["coverage_screen_pass"] is not None
            ]
            result["scenarios"].setdefault(scenario, []).append(
                {
                    "target_coverage": target,
                    "evaluation": _summary(truth[~fitted], retained[~fitted]),
                    "evaluation_cells": cells,
                    "evaluation_by_group": by_group,
                    "coverage_screen_pass": bool(coverage_checks and all(coverage_checks)),
                    "retention_screen_pass": bool(retention_checks and all(retention_checks)),
                }
            )
    return result


def _confidence(policy, states, groups, masks):
    predicted, _ = policy.predict_batch(states, groups, masks)
    confidence = np.zeros(len(states))
    with torch.inference_mode():
        for start in range(0, len(states), 1024):
            end = min(start + 1024, len(states))
            _, logits = policy.network(torch.as_tensor(states[start:end], device=policy.device))
            probability = (
                logits.masked_fill(~torch.as_tensor(masks[start:end], device=policy.device), -1e9)
                .softmax(1)
                .cpu()
                .numpy()
            )
            choices = predicted[start:end]
            selected = choices >= 0
            confidence[start:end][selected] = probability[
                np.arange(end - start)[selected], choices[selected]
            ]
    return predicted, confidence


def _summary(truth, predicted):
    metrics = behavior_metrics(truth, predicted, len(ACTION_NAMES))
    selected = predicted >= 0
    count = int(selected.sum())
    return {
        "transitions": len(truth),
        "coverage": float(selected.mean()),
        "majority_behavior_baseline": float(max(metrics["action_support"]) / len(truth)),
        "retained_behavior_agreement": float(np.mean(truth[selected] == predicted[selected]))
        if count
        else None,
        "largest_prediction_fraction": max(metrics["prediction_counts"]) / count if count else None,
        "metrics": metrics,
    }


def calibrate_coverage(study, output, *, targets=(1.0, 0.9, 0.75, 0.5), conditional_evidence=False):
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not targets or any(not 0 < target <= 1 for target in targets):
        raise ValueError("Coverage targets must be in (0,1]")
    source = read_json(study / "study.json")
    if source["status"] != "complete":
        raise ValueError("Study must be complete before calibration")
    dataset = source["dataset"]["path"]
    if file_digest(dataset) != source["dataset_sha256"]:
        raise ValueError("Study dataset integrity mismatch")
    arrays, metadata = load_dataset(dataset)
    if not source["runs"]:
        raise ValueError("Study has no completed runs")
    first = NextActionPolicy(source["runs"][0]["bundle"])
    matches = np.array(sorted(first.metadata["training_report"]["validation_matches"]))
    if len(matches) < 2:
        raise ValueError(
            "Need at least two development matches for separate calibration/evaluation"
        )
    np.random.default_rng(20261105).shuffle(matches)
    count = len(matches) // 2
    calibration, evaluation = matches[:count], matches[count:]
    selected = np.isin(arrays["match_ids"], matches)
    states, groups = arrays["observations"][selected], arrays["groups"][selected]
    truth, identifiers = arrays["actions"][selected], arrays["match_ids"][selected]
    fitted = np.isin(identifiers, calibration)
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "source_study_sha256": file_digest(study / "study.json"),
        "source_sha256": file_digest(__file__),
        "observation_masks_sha256": file_digest(observation_masks.__file__),
        "protocol_sha256": file_digest(Path(__file__).parents[2] / "docs/NEXT_STUDY_PROTOCOL.md")
        if conditional_evidence
        else None,
        "dataset_sha256": source["dataset_sha256"],
        "calibration_matches": calibration.tolist(),
        "evaluation_matches": evaluation.tolist(),
        "target_coverages": list(targets),
        "split_seed": 20261105,
        "conditional_evidence": conditional_evidence,
        "runs": [],
        "policy_winrate": None,
        "limitations": [
            "Thresholds use calibration confidence ranks only; no outcome or behavior labels select thresholds",
            "Masked classifier/actor mass is not calibrated action correctness or win probability",
            "Coverage calibration and later evaluation matches are disjoint; both were previously consulted development data",
            "Abstention can concentrate predictions and discard rare actions or roles; inspect per-group diagnostics",
            "No finite-sample risk guarantee or live-distribution validity is claimed",
            "Threshold ties are retained together, so achieved coverage can exceed its target",
            "Coverage targets apply to all group rows, including existing support abstentions",
        ],
    }
    atomic_json(output / "report.json", report)
    try:
        for run in source["runs"]:
            print(f"Calibrating coverage for {run['name']}", flush=True)
            policy = NextActionPolicy(run["bundle"])
            split = policy.metadata["training_report"]
            training, validation = set(split["training_matches"]), set(split["validation_matches"])
            if (
                validation != set(matches)
                or not training
                or training & validation
                or training | validation != set(arrays["match_ids"])
                or policy.metadata["provenance"]["fingerprint"]
                != metadata["provenance"]["fingerprint"]
            ):
                raise ValueError("Calibration model splits differ")
            result = {
                "name": run["name"],
                "bundle_sha256": file_digest(Path(run["bundle"]) / "bundle.json"),
                "scenarios": {},
            }
            observed_thresholds = {}
            for scenario in ("observed", "cold_start_history", "cold_start_observed_thresholds"):
                observed = states.copy()
                if scenario != "observed":
                    observed[
                        :, [metadata["feature_names"].index(name) for name in COLD_START_FIELDS]
                    ] = 0
                masks = feasible_actions(observed, metadata["feature_names"])
                predicted, confidence = _confidence(policy, observed, groups, masks)
                curves = []
                for target in targets:
                    retained = predicted.copy()
                    thresholds = {}
                    for i, group in enumerate(GROUPS):
                        calibration_group = fitted & (groups == i)
                        if not calibration_group.any():
                            thresholds[group] = None
                            retained[groups == i] = -1
                            continue
                        threshold = (
                            observed_thresholds[target][group]
                            if scenario == "cold_start_observed_thresholds"
                            else max(
                                float(np.quantile(confidence[calibration_group], 1 - target)),
                                float(np.nextafter(0.0, 1.0)),
                            )
                        )
                        thresholds[group] = threshold
                        retained[(groups == i) & (confidence < threshold)] = -1
                    if scenario == "observed":
                        observed_thresholds[target] = thresholds
                    curves.append(
                        {
                            "target_coverage": target,
                            "thresholds_by_group": thresholds,
                            "threshold_source_scenario": "cold_start_history"
                            if scenario == "cold_start_history"
                            else "observed",
                            "calibration": _summary(truth[fitted], retained[fitted]),
                            "evaluation": _summary(truth[~fitted], retained[~fitted]),
                            "evaluation_by_group": {
                                group: _summary(
                                    truth[(groups == i) & ~fitted],
                                    retained[(groups == i) & ~fitted],
                                )
                                for i, group in enumerate(GROUPS)
                                if np.any((groups == i) & ~fitted)
                            },
                        }
                    )
                result["scenarios"][scenario] = curves
            if conditional_evidence:
                result["conditional_evidence"] = conditional_coverage(
                    policy,
                    states,
                    groups,
                    truth,
                    identifiers,
                    fitted,
                    metadata["feature_names"],
                    targets,
                )
            report["runs"].append(result)
            atomic_json(output / "report.json", report)
        report["status"] = "complete"
        return report
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        atomic_json(output / "report.json", report)
