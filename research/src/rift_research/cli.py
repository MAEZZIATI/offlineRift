"""Optional research commands, separate from the production workflow."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("workflow.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare", help="Compare frozen serving bundles on identical matches")
    compare.add_argument("--dataset", type=Path, required=True)
    compare.add_argument("--bundles", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=1000)
    compare.add_argument("--unseen", action="store_true")
    from rift_research.evaluation import add_research_arguments

    research = sub.add_parser(
        "evaluate", help="Frozen-model ablations, match bootstrap, and label audit sheet"
    )
    add_research_arguments(research)
    from rift_research.cohort import add_cohort_arguments

    cohort = sub.add_parser("cohort", help="Prepare/resume a role-stratified collection pilot")
    add_cohort_arguments(cohort)
    memory = sub.add_parser("memory", help="Fixed-split causal GRU vs matched-capacity snapshot BC")
    memory.add_argument("--dataset", type=Path, required=True)
    memory.add_argument("--output", type=Path, required=True)
    memory.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    memory.add_argument("--steps", type=int, default=5000)
    memory.add_argument("--batch-size", type=int, default=64)
    memory.add_argument("--split-seed", type=int, default=20261105)
    memory.add_argument("--bootstrap-samples", type=int, default=1000)
    memory.add_argument(
        "--evaluate-from",
        type=Path,
        help="Completed memory experiment; evaluate frozen models on unseen matches",
    )
    memory.add_argument(
        "--reference-bundle",
        type=Path,
        help="Optional plugged bundle for comparison on exactly the same unseen matches",
    )
    memory.add_argument(
        "--methods",
        nargs="+",
        choices=("snapshot", "gru60", "gru120", "repeat120"),
        default=["snapshot", "gru60", "gru120"],
    )
    memory.add_argument(
        "--control-from",
        type=Path,
        help="Optional completed repeat120 control with the same seeds and split",
    )
    audit = sub.add_parser(
        "audit", help="Reproduce audit labels and generate a blinded local evidence viewer"
    )
    for option in ("dataset", "sheet", "key", "reference-report", "raw-dir", "output"):
        audit.add_argument(f"--{option}", type=Path, required=True)
    review = sub.add_parser(
        "review",
        help="Validate two independent review exports and adjudication; never fabricate judgments",
    )
    review.add_argument("--audit", type=Path, required=True)
    review.add_argument("--review", type=Path, action="append", required=True)
    review.add_argument("--adjudication", type=Path)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "compare":
        from rift_research.compare import compare_bundles

        report = compare_bundles(
            args.dataset,
            args.bundles,
            args.output,
            samples=args.bootstrap_samples,
            unseen=args.unseen,
        )
        print(
            f"Compared {len(report['bundles'])} bundles on {len(report['evaluation_matches'])} matches"
        )
    elif args.command == "evaluate":
        from rift_research.evaluation import run_research

        run_research(args)
    elif args.command == "cohort":
        from rift_research.cohort import run_cohort

        run_cohort(args, args.config)
    elif args.command == "memory":
        from rift_research.memory import run_memory

        run_memory(args)
    elif args.command == "audit":
        from rift_research.audit import run_audit

        run_audit(args)
    elif args.command == "review":
        from rift_research.review import run_review

        run_review(args)


if __name__ == "__main__":
    main()
