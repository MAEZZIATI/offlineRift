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
                predicted, _ = policy.predict_batch(
                    observed, groups, feasible_actions(observed, names)
                )
                result["scenarios"][scenario] = {
                    "disagreement": float(np.mean(baseline != predicted)),
                    "metrics": behavior_metrics(truth, predicted, len(ACTION_NAMES)),
                    "per_group_disagreement": {
                        group: float(np.mean(baseline[groups == i] != predicted[groups == i]))
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
