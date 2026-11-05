"""New-objective research: inspect decision evidence and evaluate frozen next-action policies."""

import argparse
import json
from pathlib import Path

from rift_research.evaluation import describe_dataset, evaluate_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser(
        "dataset", help="Audit actor/role option coverage and measured durations"
    )
    describe.add_argument("--dataset", type=Path, required=True)
    evaluate = sub.add_parser(
        "evaluate", help="Reproduce held-out behavior/support diagnostics; not win rate"
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=500)
    study = sub.add_parser("study", help="Run matched-split CQL/IQL/imitation comparisons")
    study.add_argument("--config", type=Path, default=Path("workflow.json"))
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--steps", type=int, default=10000)
    study.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    study.add_argument("--bootstrap-samples", type=int, default=500)
    sensitivity = sub.add_parser(
        "sensitivity", help="Check optimizer-seed and missing-state stability"
    )
    sensitivity.add_argument("--study", type=Path, required=True)
    sensitivity.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sensitivity":
        from rift_research.sensitivity import analyze_study

        report = analyze_study(args.study, args.output)
        print(json.dumps({"status": report["status"], "runs": len(report["runs"])}))
        return
    if args.command == "study":
        from rift_research.experiments import run_study

        report = run_study(
            args.config,
            args.output,
            steps=args.steps,
            seeds=tuple(args.seeds),
            samples=args.bootstrap_samples,
        )
        print(json.dumps({"status": report["status"], "runs": len(report["runs"])}))
        return
    report = (
        describe_dataset(args.dataset)
        if args.command == "dataset"
        else evaluate_policy(args.dataset, args.bundle, args.output, samples=args.bootstrap_samples)
    )
    print(json.dumps(report, indent=2))
