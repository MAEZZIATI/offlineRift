"""Matched-split experiments using the maintained builder, trainer and serving loader."""

from dataclasses import asdict, replace
from pathlib import Path
from shutil import copy2

import numpy as np
from offline_rift.inference import policy
from offline_rift.training import augmentation, iql, trainer
from offline_rift.training.trainer import PolicyTrainer, TrainConfig
from offline_rift.workflow import Workflow
from rift_common import network, observation_masks
from rift_common.artifacts import atomic_json, file_digest

from rift_research import evaluation
from rift_research.evaluation import describe_dataset, evaluate_policy


def run_study(
    config,
    output,
    *,
    steps=10000,
    seeds=(42, 43),
    samples=500,
    suite="algorithms",
    holdout_matches=(),
    corpus_matches=None,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct optimizer seeds")
    if samples < 1:
        raise ValueError("Positive bootstrap samples required")
    base = TrainConfig(steps=steps, split_seed=42, holdout_matches=tuple(holdout_matches))
    variants = {
        "bc": {"algorithm": "bc"},
        "cql-alpha-0.03": {"alpha": 0.03},
        "cql-alpha-0.1": {"alpha": 0.1},
        "cql-alpha-0.3": {"alpha": 0.3},
        "iql-expectile-0.7": {"algorithm": "iql", "expectile": 0.7},
        "iql-expectile-0.9": {"algorithm": "iql", "expectile": 0.9},
    }
    if suite == "robustness":
        variants = {
            "cql-alpha-0.1": {},
            "cql-joint-history": {"joint_history_dropout": 0.2},
            "cql-bounded": {"bounded_values": True},
            "cql-bounded-joint": {"bounded_values": True, "joint_history_dropout": 0.2},
        }
    elif suite == "readiness":
        variants = {
            "cql-alpha-0.1": {},
            **{
                f"cql-bounded-joint-alpha-{alpha}": {
                    "alpha": alpha,
                    "bounded_values": True,
                    "joint_history_dropout": 0.2,
                }
                for alpha in (0.03, 0.1, 0.3)
            },
            "bc-joint-history": {"algorithm": "bc", "joint_history_dropout": 0.2},
        }
    elif suite in {"evidence", "evidence-low-alpha"}:
        base = replace(base, joint_history_dropout=0.2, bounded_values=True, minimum_score_gap=0.02)
        variants = {
            "cql-representative": {"target_mode": "representative"},
            "cql-identified": {"target_mode": "identified"},
            "bc-identified": {"algorithm": "bc", "bounded_values": False, "minimum_score_gap": 0},
        }
        if suite == "evidence-low-alpha":
            base = replace(base, minimum_score_gap=0)
            variants = {
                f"cql-identified-alpha-{alpha}": {"alpha": alpha, "target_mode": "identified"}
                for alpha in (0.01, 0.03)
            }
    elif suite != "algorithms":
        raise ValueError("Unknown experiment suite")
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        **{
            module.__name__: Path(module.__file__)
            for module in (
                trainer,
                augmentation,
                iql,
                network,
                observation_masks,
                policy,
                evaluation,
            )
        },
        "rift_research.experiments": Path(__file__),
    }
    for name, path in sources.items():
        destination = output / "sources" / (name.replace(".", "/") + ".py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(path, destination)
    report = {
        "status": "building",
        "split_seed": 42,
        "suite": suite,
        "optimizer_seeds": list(seeds),
        "variants": variants,
        "base_settings": asdict(base),
        "source_sha256": {name: file_digest(path) for name, path in sources.items()},
        "runs": [],
        "policy_winrate": None,
        "policy_improvement": None,
        "limitations": [
            "Previously consulted retrospective cache; development comparisons only",
            "All eligible matches used across grouped train/development splits",
            "Behavior diagnostics and concentration do not establish strategic utility",
            "No automatic selection or publication of a model as the latest policy",
        ],
    }
    atomic_json(output / "study.json", report)
    try:
        data = (
            Workflow(config).build(match_ids=set(corpus_matches))
            if corpus_matches is not None
            else Workflow(config).build()
        )
        report.update(
            status="running",
            dataset=data,
            dataset_sha256=file_digest(data["path"]),
            audit=describe_dataset(data["path"]),
        )
        atomic_json(output / "study.json", report)
        reference_split = None
        for variant, settings in variants.items():
            for seed in seeds:
                name = f"{variant}-seed-{seed}"
                print(f"Training {name}: {steps} steps", flush=True)
                bundle = output / name / "model"
                result = PolicyTrainer(replace(base, seed=seed, **settings)).train(
                    data["path"], bundle
                )
                split = result["validation_matches"]
                if reference_split is not None and split != reference_split:
                    raise ValueError("Experiment match splits differ")
                reference_split = split
                evaluated = evaluate_policy(
                    data["path"], bundle, output / name / "evaluation", samples=samples
                )
                report["runs"].append(
                    {
                        "name": name,
                        "bundle": str(bundle.resolve()),
                        "settings": asdict(replace(base, seed=seed, **settings)),
                        "metrics": evaluated["metrics"],
                        "identified_metrics": evaluated.get("identified_metrics"),
                        "retention_coverage": evaluated["retention_coverage"],
                        "per_group": evaluated["per_group"],
                        "score_quantiles": evaluated["score_quantiles"],
                        "last_optimizer": result["report"]["last_optimizer"],
                        "behavior_macro_f1_match_bootstrap_95": evaluated[
                            "behavior_macro_f1_match_bootstrap_95"
                        ],
                    }
                )
                atomic_json(output / "study.json", report)
                print(
                    f"Completed {name}: macro F1={evaluated['metrics']['macro_f1']:.4f}, "
                    f"coverage={evaluated['metrics']['prediction_coverage']}",
                    flush=True,
                )
        # Paired prediction disagreements use identical rows and held-out matches.
        reference = "cql-alpha-0.1" if "cql-alpha-0.1" in variants else next(iter(variants))
        baseline = np.load(output / f"{reference}-seed-{seeds[0]}/evaluation/predictions.npz")
        try:
            for run in report["runs"]:
                with np.load(output / run["name"] / "evaluation/predictions.npz") as archive:
                    if not np.array_equal(archive["match_ids"], baseline["match_ids"]):
                        raise ValueError("Paired evaluation rows differ")
                    run["disagreement_with_reference_cql"] = float(
                        np.mean(archive["predicted"] != baseline["predicted"])
                    )
        finally:
            baseline.close()
        report["status"] = "complete"
        return report
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        atomic_json(output / "study.json", report)
