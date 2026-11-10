"""New-objective research: inspect decision evidence and evaluate frozen next-action policies."""

import argparse
import json
from pathlib import Path

from rift_research.evaluation import describe_dataset, evaluate_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    comparison = sub.add_parser(
        "compare-holdout", help="Paired match-bootstrap comparison of fixed products"
    )
    comparison.add_argument("--previous", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--bootstrap-samples", type=int, default=1000)
    game = sub.add_parser(
        "review-game", help="Replay all policies/players in one reserved European ranked game"
    )
    game.add_argument("--match", type=Path, required=True)
    game.add_argument("--bundle", type=Path, required=True)
    game.add_argument("--baseline", type=Path)
    game.add_argument("--study", type=Path)
    game.add_argument("--protocol", type=Path)
    game.add_argument("--catalogs", type=Path, default=Path("data/catalogs"))
    game.add_argument("--holdout-comparison", type=Path)
    game.add_argument(
        "--followup", type=Path, help="Retained first-test findings and adaptive development status"
    )
    game.add_argument("--output", type=Path, required=True)
    game.add_argument("--notebook", type=Path)
    game.add_argument("--execute-notebook", action="store_true")
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
    evaluate.add_argument(
        "--project-state",
        action="store_true",
        help="Explicit v2-to-v1 feature projection for an older baseline on new matches",
    )
    evaluate.add_argument(
        "--match-manifest",
        type=Path,
        help="Restrict --new-matches to a frozen manifest's holdout_matches",
    )
    evaluate.add_argument(
        "--stress",
        action="store_true",
        help="Also diagnose controlled missing-health/history inputs",
    )
    evaluate.add_argument(
        "--new-matches",
        action="store_true",
        help="Evaluate only matches absent from this frozen model's training/development corpus",
    )
    study = sub.add_parser("study", help="Run matched-split CQL/IQL/imitation comparisons")
    study.add_argument("--config", type=Path, default=Path("workflow.json"))
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--steps", type=int, default=10000)
    study.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    study.add_argument("--bootstrap-samples", type=int, default=500)
    study.add_argument(
        "--suite",
        choices=("algorithms", "robustness", "readiness", "evidence", "evidence-low-alpha"),
        default="algorithms",
    )
    study.add_argument(
        "--holdout-manifest",
        type=Path,
        help="JSON with match_ids reserved from fitting and development",
    )
    study.add_argument(
        "--corpus-manifest",
        type=Path,
        help="Freeze the exact raw match-ID membership for this study",
    )
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
    session = sub.add_parser("audit-session", help="Audit measured opt-in live samples and advice")
    session.add_argument("--session", type=Path, required=True)
    session.add_argument("--output", type=Path, required=True)
    session.add_argument("--reported-outcome", choices=("win", "loss"))
    adherence = sub.add_parser(
        "adherence",
        help="Review observed activity after served advice; unresolved actions remain explicit",
    )
    adherence.add_argument("--session", type=Path, required=True)
    adherence.add_argument("--output", type=Path, required=True)
    adherence.add_argument(
        "--match",
        type=Path,
        help="Optional cached match/timeline JSON; roster and game clock must match",
    )
    synthesis = sub.add_parser(
        "synthesize",
        help="Generate the current six-policy Markdown/PDF from completed offline artifacts",
    )
    synthesis.add_argument("--study", type=Path, required=True)
    synthesis.add_argument("--evaluation", type=Path, required=True)
    synthesis.add_argument("--baseline-evaluation", type=Path)
    synthesis.add_argument("--output", type=Path, default=Path("research/docs/SYNTHESIS.pdf"))
    args = parser.parse_args()
    if args.command == "compare-holdout":
        from rift_research.evaluation import compare_holdout

        report = compare_holdout(
            args.previous, args.candidate, args.output, samples=args.bootstrap_samples
        )
        print(
            json.dumps(
                {
                    "matches": len(report["matches"]),
                    "paired_differences": report["paired_differences"],
                }
            )
        )
        return
    if args.command == "review-game":
        from rift_research.game_review import review_game, write_notebook

        report = review_game(
            args.match,
            args.bundle,
            args.output,
            baseline=args.baseline,
            study=args.study,
            protocol=args.protocol,
            holdout_comparison=args.holdout_comparison,
            followup=args.followup,
            catalogs=args.catalogs,
        )
        if args.notebook:
            write_notebook(args.output, args.notebook)
            if args.execute_notebook:
                import sys

                import nbformat
                from nbclient import NotebookClient

                notebook = nbformat.read(args.notebook, as_version=4)
                client = NotebookClient(
                    notebook,
                    timeout=600,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(Path.cwd())}},
                )
                # Run with the same environment as the installed research command.
                client.create_kernel_manager().kernel_spec.argv[0] = sys.executable
                client.execute()
                nbformat.write(notebook, args.notebook)
        elif args.execute_notebook:
            raise ValueError("--execute-notebook requires --notebook")
        print(
            json.dumps(
                {
                    "match_id": report["match_id"],
                    "models": list(report["models"]),
                    "notebook": str(args.notebook) if args.notebook else None,
                }
            )
        )
        return
    if args.command == "synthesize":
        from rift_research.reporting import synthesize

        print(
            json.dumps(
                synthesize(
                    args.study,
                    args.evaluation,
                    args.output,
                    baseline_evaluation=args.baseline_evaluation,
                )
            )
        )
        return
    if args.command == "adherence":
        from rift_research.adherence import review_adherence

        report = review_adherence(args.session, args.output, match=args.match)
        print(json.dumps({"status": report["status"], "counts": report["counts"]}))
        return
    if args.command == "audit-session":
        from rift_research.session_audit import audit_session

        report = audit_session(args.session, args.output, reported_outcome=args.reported_outcome)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "samples",
                        "game_modes",
                        "mode_mismatch",
                        "voluntary_dominant_fraction",
                        "self_reported_outcome",
                    )
                }
            )
        )
        return
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
            holdout_matches=json.loads(args.holdout_manifest.read_text())["match_ids"]
            if args.holdout_manifest
            else (),
            corpus_matches=json.loads(args.corpus_manifest.read_text())["match_ids"]
            if args.corpus_manifest
            else None,
        )
        print(json.dumps({"status": report["status"], "runs": len(report["runs"])}))
        return
    report = (
        describe_dataset(args.dataset)
        if args.command == "dataset"
        else evaluate_policy(
            args.dataset,
            args.bundle,
            args.output,
            samples=args.bootstrap_samples,
            new_matches=args.new_matches,
            stress=args.stress,
            project_state=args.project_state,
            match_ids=json.loads(args.match_manifest.read_text())["holdout_matches"]
            if args.match_manifest
            else None,
        )
    )
    print(json.dumps(report, indent=2))
