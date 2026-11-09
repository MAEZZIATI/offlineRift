"""Policy seed agreement, critic range and missing-observation stress diagnostics."""

from pathlib import Path

import numpy as np
import torch
from offline_rift.inference.policy import NextActionPolicy
from rift_common.actions import ACTION_NAMES, GROUPS, feasible_actions
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.dataset import load_dataset
from rift_common.metrics import behavior_metrics
from rift_common.observation_masks import COLD_START_FIELDS, HEALTH_FIELDS


def analyze_study(study, output):
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = read_json(study / "study.json")
    if source["status"] != "complete":
        raise ValueError("Study must be complete before sensitivity analysis")
    dataset = source["dataset"]["path"]
    if file_digest(dataset) != source["dataset_sha256"]:
        raise ValueError("Study dataset integrity mismatch")
    arrays, metadata = load_dataset(dataset)
    names = metadata["feature_names"]
    scenarios = {
        "unobserved_health": list(HEALTH_FIELDS),
        "cold_start_history": list(COLD_START_FIELDS),
    }
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "source_study_sha256": file_digest(study / "study.json"),
        "source_sha256": file_digest(__file__),
        "scenarios": scenarios,
        "runs": [],
        "seed_agreement": {},
        "limitations": [
            "Mask stress tests are controlled omissions, not observed live performance",
            "Seed agreement does not establish correctness or policy utility",
            "Critic ranges outside [-1,1] indicate unbounded estimates, not reward probabilities",
            "No calibrated win-rate or counterfactual policy improvement is estimated",
        ],
    }
    atomic_json(output / "report.json", report)
    predictions = {}
    try:
        for run in source["runs"]:
            print(f"Stress testing {run['name']}", flush=True)
            policy = NextActionPolicy(run["bundle"])
            selected = np.isin(
                arrays["match_ids"], policy.metadata["training_report"]["validation_matches"]
            )
            states, groups = arrays["observations"][selected], arrays["groups"][selected]
            truth, masks = arrays["actions"][selected], arrays["action_masks"][selected]
            baseline, _ = policy.predict_batch(states, groups, masks)
            predictions[run["name"]] = (baseline, groups)
            result = {
                "name": run["name"],
                "bundle_sha256": file_digest(Path(run["bundle"]) / "bundle.json"),
                "scenarios": {},
            }
            if policy.algorithm != "bc":
                q = []
                with torch.inference_mode():
                    for start in range(0, len(states), 1024):
                        values, _ = policy.network(torch.as_tensor(states[start : start + 1024]))
                        q.append(
                            values.numpy()[np.arange(len(values)), truth[start : start + 1024]]
                        )
                recorded = np.concatenate(q)
                result["recorded_action_q_quantiles"] = np.quantile(
                    recorded, [0, 0.01, 0.5, 0.99, 1]
                ).tolist()
                result["recorded_action_q_outside_return_bounds"] = float(
                    np.mean(np.abs(recorded) > 1)
                )
            for scenario, fields in scenarios.items():
                observed = states.copy()
                observed[:, [names.index(field) for field in fields]] = 0
                omitted_masks = feasible_actions(observed, names)
                predicted, _ = policy.predict_batch(observed, groups, omitted_masks)
                infeasible_targets = ~omitted_masks[np.arange(len(truth)), truth]
                result["scenarios"][scenario] = {
                    "disagreement": float(np.mean(baseline != predicted)),
                    "metrics": behavior_metrics(truth, predicted, len(ACTION_NAMES)),
                    "per_group_disagreement": {
                        group: float(np.mean(baseline[groups == i] != predicted[groups == i]))
                        for i, group in enumerate(GROUPS)
                        if np.any(groups == i)
                    },
                    "per_group": {
                        group: {
                            **behavior_metrics(
                                truth[groups == i], predicted[groups == i], len(ACTION_NAMES)
                            ),
                            "targets_made_infeasible": np.bincount(
                                truth[(groups == i) & infeasible_targets],
                                minlength=len(ACTION_NAMES),
                            ).tolist(),
                        }
                        for i, group in enumerate(GROUPS)
                        if np.any(groups == i)
                    },
                }
            report["runs"].append(result)
            atomic_json(output / "report.json", report)
        for variant in source["variants"]:
            seeds = source["optimizer_seeds"]
            first, groups = predictions[f"{variant}-seed-{seeds[0]}"]
            report["seed_agreement"][variant] = {
                str(seed): {
                    "agreement": float(np.mean(first == predictions[f"{variant}-seed-{seed}"][0])),
                    "per_group": {
                        group: float(
                            np.mean(
                                first[groups == i]
                                == predictions[f"{variant}-seed-{seed}"][0][groups == i]
                            )
                        )
                        for i, group in enumerate(GROUPS)
                        if np.any(groups == i)
                    },
                }
                for seed in seeds[1:]
            }
        if source.get("suite") == "readiness":
            originals = {run["name"]: run for run in source["runs"]}
            stressed = {run["name"]: run for run in report["runs"]}
            report["development_screen"] = {}
            for variant in source["variants"]:
                if not variant.startswith("cql-bounded-joint"):
                    continue
                failures = []
                for seed in source["optimizer_seeds"]:
                    name = f"{variant}-seed-{seed}"
                    original = originals[name]
                    if stressed[name]["scenarios"]["cold_start_history"]["disagreement"] > 0.2:
                        failures.append(f"Seed {seed}: history-removal disagreement exceeds 20%")
                    for group, metrics in original["per_group"].items():
                        if metrics["largest_prediction_fraction"] > 0.8:
                            failures.append(f"Seed {seed}: {group} concentration exceeds 80%")
                    reference = originals[f"cql-alpha-0.1-seed-{seed}"]
                    if (
                        original["metrics"]["prediction_coverage"]
                        < reference["metrics"]["prediction_coverage"]
                    ):
                        failures.append(f"Seed {seed}: action diversity below ordinary CQL")
                report["development_screen"][variant] = {
                    "passed": not failures,
                    "failures": failures,
                    "production_strategy_validated": False,
                }
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
