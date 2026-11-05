"""Experimental support-intersection minimax decision regret under evidence removal."""

from pathlib import Path

import numpy as np
import torch
from offline_rift.inference.policy import NextActionPolicy
from rift_common.actions import GROUPS, feasible_actions
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.dataset import load_dataset
from rift_common.network import rank_options
from rift_common.observation_masks import COLD_START_FIELDS, HEALTH_FIELDS

from rift_research.calibration import _confidence, _summary


def match_coverage(baseline, confidence, groups, fitted, retained):
    """Fit confidence-only coverage to the candidate's calibration coverage per group."""
    predicted = baseline.copy()
    thresholds = {}
    for i, group in enumerate(GROUPS):
        calibration = fitted & (groups == i)
        if not calibration.any():
            predicted[groups == i] = -1
            thresholds[group] = None
            continue
        target = float(np.mean(retained[calibration] >= 0))
        if target == 0:
            threshold = float("inf")
        else:
            threshold = max(
                float(np.quantile(confidence[calibration], 1 - target)),
                float(np.nextafter(0.0, 1.0)),
            )
        predicted[(groups == i) & (confidence < threshold)] = -1
        thresholds[group] = {
            "target_coverage": target,
            "threshold": threshold if np.isfinite(threshold) else None,
            "abstain_all": target == 0,
        }
    return predicted, thresholds


def minimax_decision(values, allowed):
    """Arrays have (model/view, row, action) axes; no labels/outcomes are used."""
    values, allowed = np.asarray(values), np.asarray(allowed, dtype=bool)
    if values.shape != allowed.shape or values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Invalid decision views")
    common = allowed.all(axis=0)
    # An unsupported view makes the intersection empty. A finite placeholder keeps
    # subtraction defined; those rows are explicitly rejected below.
    best = np.where(allowed, values, -np.inf).max(axis=2, keepdims=True)
    best[~np.isfinite(best)] = 0
    regret = np.where(common, (best - values).max(axis=0), np.inf)
    predicted = regret.argmin(axis=1)
    worst = regret[np.arange(len(predicted)), predicted]
    predicted[~common.any(axis=1)] = -1
    return predicted, worst, common


def study_decision_regret(study, output):
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = read_json(study / "study.json")
    if source["status"] != "complete" or len(source["optimizer_seeds"]) < 2:
        raise ValueError("A completed study with at least two optimizer seeds is required")
    dataset = source["dataset"]["path"]
    if file_digest(dataset) != source["dataset_sha256"]:
        raise ValueError("Study dataset integrity mismatch")
    arrays, metadata = load_dataset(dataset)
    names = metadata["feature_names"]
    # Objective flags are action prerequisites, not removable optional statistics.
    history = tuple(
        name for name in COLD_START_FIELDS if not name.startswith(("dragon_", "baron_", "herald_"))
    )
    views = {
        "observed": (),
        "health": HEALTH_FIELDS,
        "recent_history": history,
        "health_and_history": (*HEALTH_FIELDS, *history),
    }
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "dataset_sha256": source["dataset_sha256"],
        "source_study_sha256": file_digest(study / "study.json"),
        "source_sha256": file_digest(__file__),
        "protocol_sha256": file_digest(Path(__file__).parents[2] / "docs/NEXT_STUDY_PROTOCOL.md"),
        "views": views,
        "tolerances": [0, 0.05, 0.1, 0.25],
        "variants": [],
        "policy_winrate": None,
        "policy_improvement": None,
        "limitations": [
            "Regret is a learned value gap, not causal or realized regret",
            "Optimizer seeds are not a calibrated uncertainty distribution",
            "Evidence removal can legitimately change the preferred action",
            "Previously consulted development matches; no policy quality estimate",
        ],
    }
    atomic_json(output / "report.json", report)
    torch.set_num_threads(2)
    try:
        for variant in source["variants"]:
            runs = [r for r in source["runs"] if r["name"].rsplit("-seed-", 1)[0] == variant]
            policies = [NextActionPolicy(run["bundle"]) for run in runs]
            if not policies or policies[0].algorithm != "cql":
                continue
            print(f"Decision-regret study: {variant}", flush=True)
            split = policies[0].metadata["training_report"]
            validation = set(split["validation_matches"])
            for policy in policies:
                split = policy.metadata["training_report"]
                training = set(split["training_matches"])
                if (
                    set(split["validation_matches"]) != validation
                    or training & validation
                    or training | validation != set(arrays["match_ids"])
                    or policy.metadata["provenance"]["fingerprint"]
                    != metadata["provenance"]["fingerprint"]
                ):
                    raise ValueError("Incompatible model corpus or splits")
            selected = np.isin(arrays["match_ids"], list(validation))
            states, groups = arrays["observations"][selected], arrays["groups"][selected]
            truth = arrays["actions"][selected]
            calibration_matches = np.array(sorted(validation))
            np.random.default_rng(20261105).shuffle(calibration_matches)
            fitted = np.isin(
                arrays["match_ids"][selected], calibration_matches[: len(calibration_matches) // 2]
            )
            if not fitted.any() or fitted.all():
                raise ValueError("Need at least two development matches for matched coverage")
            _, confidence = _confidence(
                policies[0], states, groups, arrays["action_masks"][selected]
            )
            values, support = [], []
            for policy in policies:
                for fields in views.values():
                    observed = states.copy()
                    observed[:, [names.index(name) for name in fields]] = 0
                    masks = feasible_actions(observed, names)
                    qs, allowed = [], []
                    with torch.inference_mode():
                        for start in range(0, len(states), 1024):
                            end = start + 1024
                            q, logits = policy.network(
                                torch.as_tensor(observed[start:end], device=policy.device)
                            )
                            q, logits = q.cpu().numpy(), logits.cpu().numpy()
                            _, _, a = rank_options(
                                q,
                                logits,
                                policy.counts,
                                groups[start:end],
                                masks[start:end],
                                algorithm="cql",
                                minimum_support=policy.config["minimum_support"],
                                behavior_floor=policy.config["behavior_floor"],
                            )
                            qs.append(q)
                            allowed.append(a)
                    values.append(np.concatenate(qs))
                    support.append(np.concatenate(allowed))
            values, support = np.stack(values), np.stack(support)
            baseline = np.where(support[0], values[0], -np.inf).argmax(axis=1)
            baseline[~support[0].any(axis=1)] = -1
            result = {
                "variant": variant,
                "bundles": {
                    run["name"]: file_digest(Path(run["bundle"]) / "bundle.json") for run in runs
                },
                "validation_matches": sorted(validation),
                "coverage_calibration_matches": calibration_matches[
                    : len(calibration_matches) // 2
                ].tolist(),
                "coverage_evaluation_matches": calibration_matches[
                    len(calibration_matches) // 2 :
                ].tolist(),
                "baseline": _summary(truth, baseline),
                "methods": {},
            }
            for method, indices in {
                "single_observed": [0],
                "seed_only": list(range(0, len(values), len(views))),
                "omission_only": list(range(len(views))),
                "seed_and_omission": list(range(len(values))),
            }.items():
                predicted, regret, common = minimax_decision(values[indices], support[indices])
                curves = []
                for tolerance in (None, *report["tolerances"]):
                    retained = predicted.copy()
                    if tolerance is not None:
                        retained[regret > tolerance] = -1
                    matched, thresholds = match_coverage(
                        baseline, confidence, groups, fitted, retained
                    )
                    curves.append(
                        {
                            "tolerance": tolerance,
                            "evaluation": _summary(truth, retained),
                            "per_group": {
                                group: _summary(truth[groups == i], retained[groups == i])
                                for i, group in enumerate(GROUPS)
                                if np.any(groups == i)
                            },
                            "matched_coverage": {
                                "thresholds_by_group": thresholds,
                                "candidate_evaluation": _summary(truth[~fitted], retained[~fitted]),
                                "confidence_only_evaluation": _summary(
                                    truth[~fitted], matched[~fitted]
                                ),
                            },
                        }
                    )
                result["methods"][method] = {
                    "selected_regret_quantiles": np.quantile(
                        regret[np.isfinite(regret)], [0, 0.5, 0.9, 1]
                    ).tolist()
                    if np.isfinite(regret).any()
                    else [],
                    "empty_support_intersection": int(np.sum(~common.any(axis=1))),
                    "disagreement_with_baseline": float(np.mean(predicted != baseline)),
                    "candidate_counts": np.bincount(
                        common.sum(axis=1), minlength=values.shape[-1] + 1
                    ).tolist(),
                    "curves": curves,
                }
            report["variants"].append(result)
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
