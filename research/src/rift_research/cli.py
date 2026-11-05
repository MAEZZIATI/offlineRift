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
    args = parser.parse_args()
    report = (
        describe_dataset(args.dataset)
        if args.command == "dataset"
        else evaluate_policy(args.dataset, args.bundle, args.output, samples=args.bootstrap_samples)
    )
    print(json.dumps(report, indent=2))
