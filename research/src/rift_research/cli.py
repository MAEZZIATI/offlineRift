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
    study.add_argument("--suite", choices=("algorithms", "robustness"), default="algorithms")
    sensitivity = sub.add_parser(
        "sensitivity", help="Check optimizer-seed and missing-state stability"
    )
    sensitivity.add_argument("--study", type=Path, required=True)
    sensitivity.add_argument("--output", type=Path, required=True)
    combat = sub.add_parser(
        "audit-combat", help="Audit combat event/CS/movement overlap in cached targets"
    )
    combat.add_argument("--dataset", type=Path, required=True)
    combat.add_argument("--raw", type=Path, required=True)
    combat.add_argument("--output", type=Path, required=True)
    calibration = sub.add_parser(
        "calibrate-coverage", help="Fit abstention coverage on disjoint development matches"
    )
    calibration.add_argument("--study", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument(
        "--conditional-evidence",
        action="store_true",
        help="Also fit exact availability-pattern cells with sparse-cell abstention",
    )
    regret = sub.add_parser(
        "decision-regret",
        help="Study supported minimax decisions across seeds and evidence omissions",
    )
    regret.add_argument("--study", type=Path, required=True)
    regret.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "decision-regret":
        from rift_research.decision_regret import study_decision_regret

        report = study_decision_regret(args.study, args.output)
        print(json.dumps({"status": report["status"], "variants": len(report["variants"])}))
        return
    if args.command == "calibrate-coverage":
        from rift_research.calibration import calibrate_coverage

        report = calibrate_coverage(
            args.study, args.output, conditional_evidence=args.conditional_evidence
        )
        print(json.dumps({"status": report["status"], "runs": len(report["runs"])}))
        return
    if args.command == "audit-combat":
        from rift_research.combat import audit_combat

        report = audit_combat(args.dataset, args.raw, args.output)
        print(json.dumps({"status": report["status"], "counts": report["counts"]}))
        return
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
            suite=args.suite,
        )
        print(json.dumps({"status": report["status"], "runs": len(report["runs"])}))
        return
    report = (
        describe_dataset(args.dataset)
        if args.command == "dataset"
        else evaluate_policy(args.dataset, args.bundle, args.output, samples=args.bootstrap_samples)
    )
    print(json.dumps(report, indent=2))
