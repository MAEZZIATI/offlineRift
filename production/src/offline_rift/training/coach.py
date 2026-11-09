"""Six independently trained policies using the maintained explicit-transition trainer."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from shutil import copy2, copytree

import numpy as np
import rift_common
from rift_common.actions import GROUPS, Action
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.coaching import POLICY_SCOPES
from rift_common.dataset import load_dataset
from rift_common.metrics import behavior_metrics

from offline_rift.inference.coach import MacroCoach
from offline_rift.inference.policy import NextActionPolicy
from offline_rift.training.trainer import PolicyTrainer, TrainConfig


def scope_diagnostics(policy, arrays, scope):
    selected = (arrays["groups"] == GROUPS.index(scope)) & np.isin(
        arrays["match_ids"], policy.metadata["training_report"]["validation_matches"]
    )
    predictions, _ = policy.predict_batch(
        arrays["observations"][selected],
        arrays["groups"][selected],
        arrays["action_masks"][selected],
    )
    truth = arrays["actions"][selected]
    identified = arrays["action_identified"][selected]
    counts = np.bincount(predictions[predictions >= 0], minlength=len(Action))
    counts[Action.WAIT_RESPAWN] = 0
    return {
        "descriptive": behavior_metrics(truth, predictions, len(Action)),
        "identified": behavior_metrics(truth[identified], predictions[identified], len(Action))
        if identified.any()
        else None,
        "retention": float(np.mean(predictions >= 0)),
        "voluntary_concentration": float(counts.max() / max(1, counts.sum())),
        "identified_validation_rows": int(identified.sum()),
        "majority_baseline": float(np.mean(truth == policy.counts[GROUPS.index(scope)].argmax())),
    }


def train_coach(dataset, output, *, steps=10000, holdout_matches=(), seed=42):
    """Retain all comparisons; choose on development only, never publish a latest pointer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays, metadata = load_dataset(dataset)
    if metadata.get("target_schema") != "executed-evidence-v3":
        raise ValueError("Six-policy training requires the audited macro evidence v3 contract")
    report = {
        "status": "running",
        "dataset_path": str(Path(dataset).resolve()),
        "dataset_sha256": file_digest(dataset),
        "candidates": {},
        "selected": {},
        "policy_winrate": None,
        "selection": "Noncollapsed CQL candidates first, then highest within-scope identified development F1; BC is diagnostic only",
        "coverage": {},
        "source_sha256": {},
    }
    for root in (Path(__file__).parents[1], Path(rift_common.__file__).parent):
        for source in root.rglob("*.py"):
            name = str(source.relative_to(root.parent))
            destination = output / "sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source, destination)
            report["source_sha256"][name] = file_digest(source)
    atomic_json(output / "study.json", report)
    base = TrainConfig(
        steps=steps,
        seed=seed,
        split_seed=42,
        bounded_values=True,
        joint_history_dropout=0.2,
        holdout_matches=tuple(holdout_matches),
    )
    try:
        for scope in POLICY_SCOPES:
            rows = (arrays["groups"] == GROUPS.index(scope)) & arrays["action_identified"]
            report["coverage"][scope] = {
                "identified_rows": int(rows.sum()),
                "matches": len(set(arrays["match_ids"][rows])),
                "action_counts": np.bincount(
                    arrays["actions"][rows], minlength=len(Action)
                ).tolist(),
            }
            settings = replace(base, training_groups=(scope,), advisory_groups=(scope,))
            candidates = {
                "cql-td" if scope != "TEAM" else "cql-return-001": {
                    "alpha": 0.01,
                    "critic_target": "td" if scope != "TEAM" else "episode_return",
                },
                "cql-return-003": {"alpha": 0.03, "critic_target": "episode_return"},
                "bc": {"algorithm": "bc", "bounded_values": False},
            }
            report["candidates"][scope] = []
            for name, overrides in candidates.items():
                print(f"Training {scope}/{name}: {steps} updates", flush=True)
                directory = output / "candidates" / scope / name
                config = replace(settings, **overrides)
                PolicyTrainer(config).train(dataset, directory)
                diagnostic = scope_diagnostics(NextActionPolicy(directory), arrays, scope)
                report["candidates"][scope].append(
                    {
                        "name": name,
                        "bundle": str(directory.resolve()),
                        "settings": asdict(config),
                        **diagnostic,
                    }
                )
                atomic_json(output / "study.json", report)
            choices = [
                r for r in report["candidates"][scope] if r["settings"]["algorithm"] == "cql"
            ]
            selected = max(
                choices,
                key=lambda r: (
                    not r["descriptive"]["collapsed"],
                    (r["identified"] or {}).get("macro_f1", -1),
                    -r["voluntary_concentration"],
                ),
            )
            report["selected"][scope] = selected
            atomic_json(output / "study.json", report)
        directory = assemble_coach(dataset, report["selected"], output / "model", arrays=arrays)
        report.update(status="complete", bundle=str(directory.resolve()))
        return report
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        atomic_json(output / "study.json", report)


def assemble_coach(dataset, selections, directory, *, arrays=None):
    """Assemble verified immutable components; no optimizer or selection is rerun."""
    if arrays is None:
        arrays, _ = load_dataset(dataset)
    directory = Path(directory)
    directory.mkdir()
    entries = {}
    for scope, selected in selections.items():
        destination = directory / "policies" / scope
        copytree(selected["bundle"], destination)
        entries[scope] = {
            "path": str(destination.relative_to(directory)),
            "sha256": file_digest(destination / "bundle.json"),
        }
    first = read_json(directory / "policies/TEAM/metadata.json")
    aggregate = deepcopy(first)
    training = aggregate["training_report"]
    component_reports = [
        read_json(Path(item["bundle"]) / "metadata.json")["training_report"]
        for item in selections.values()
    ]
    for key in (
        "training_transitions",
        "retained_training_transitions",
        "excluded_ambiguous_training_updates",
        "identified_training_terminal_rows",
        "optimizer_steps",
    ):
        training[key] = sum(component[key] for component in component_reports)
    training["observed_action_q_mean"] = None
    training["critic_target"] = "per-policy; see selection"
    aggregate["train_config"] = {
        "per_policy": {scope: item["settings"] for scope, item in selections.items()}
    }
    counts = np.zeros((len(GROUPS), len(Action)), dtype=np.int64)
    for scope, selected in selections.items():
        child = read_json(Path(selected["bundle"]) / "metadata.json")
        counts[GROUPS.index(scope)] = child["training_report"]["counts_by_group"][
            GROUPS.index(scope)
        ]
    validation = np.isin(arrays["match_ids"], training["validation_matches"])
    predictions = np.full(int(validation.sum()), -1, dtype=np.int64)
    groups = arrays["groups"][validation]
    for scope, selected in selections.items():
        selected_rows = groups == GROUPS.index(scope)
        predictions[selected_rows], _ = NextActionPolicy(selected["bundle"]).predict_batch(
            arrays["observations"][validation][selected_rows],
            groups[selected_rows],
            arrays["action_masks"][validation][selected_rows],
        )
    truth = arrays["actions"][validation]
    training.update(
        counts_by_group=counts.tolist(),
        action_counts=counts.sum(0).tolist(),
        validation=behavior_metrics(truth, predictions, len(Action)),
        majority_baseline=float(np.mean(truth == counts.sum(0).argmax())),
        per_group={scope: selections[scope]["descriptive"] for scope in POLICY_SCOPES},
    )
    aggregate.update(
        bundle_schema="six-policy-coach-v1",
        advisory_groups=list(POLICY_SCOPES),
        training_report=training,
        selection=selections,
        architecture="six independent scoped networks; symbolic coordination only",
    )
    atomic_json(directory / "metadata.json", aggregate)
    atomic_json(
        directory / "coach.json",
        {
            "schema": "six-policy-coach-v1",
            "policies": entries,
            "metadata_sha256": file_digest(directory / "metadata.json"),
        },
    )
    restored, _ = MacroCoach(directory, require_complete=False).predict_batch(
        arrays["observations"][validation], groups, arrays["action_masks"][validation]
    )
    if not np.array_equal(restored, predictions):
        raise RuntimeError("Coach routing failed saved serving parity")
    atomic_json(
        directory / "run.json",
        {
            "status": "complete",
            "dataset_path": str(Path(dataset).resolve()),
            "dataset_sha256": file_digest(dataset),
            "report": training,
            "serving_reload_verified": True,
        },
    )
    manifest = read_json(directory / "coach.json")
    manifest["run_sha256"] = file_digest(directory / "run.json")
    atomic_json(directory / "coach.json", manifest)
    return directory
