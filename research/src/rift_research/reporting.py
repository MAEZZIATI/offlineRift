"""Reproducible current synthesis from completed coach and frozen offline evaluation."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape

from rift_common.artifacts import file_digest, read_json
from rift_common.coaching import POLICY_SCOPES


def synthesize(study, evaluation, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pymupdf
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    study, evaluation, output = Path(study), Path(evaluation), Path(output)
    source, measured = read_json(study / "study.json"), read_json(evaluation / "report.json")
    if source["status"] != "complete" or measured["bundle_sha256"] != file_digest(
        Path(source["bundle"]) / "coach.json"
    ):
        raise ValueError("Synthesis requires matching completed coach/evaluation artifacts")
    run = read_json(Path(source["bundle"]) / "run.json")
    training = run["report"]
    validation_path = study / "validation.json"
    validation = read_json(validation_path) if validation_path.exists() else {}
    checks_text = (
        f"Final validation recorded {validation['tests_passed']} passing tests, including {validation.get('browser_cases', 0)} synthetic Chromium cases. "
        if "tests_passed" in validation
        else "Final validation counts are not attached to this study. "
    )
    cold = measured.get("availability_stress", {}).get("cold_history", {})
    cold_text = (
        "Changes after cold-history omission: "
        + "; ".join(
            f"{scope} {cold[scope]['changed_fraction']:.1%}"
            for scope in POLICY_SCOPES
            if scope in cold
        )
        + ". These include actions made infeasible by lost prerequisites; the JSON report separates that mechanism. Omission-test retention: "
        + "; ".join(
            f"{scope} {cold[scope]['retention']:.1%}" for scope in POLICY_SCOPES if scope in cold
        )
        + ". Retention does not establish reliable advice."
    )
    if not cold:
        cold_text = "No controlled cold-history diagnostic was provided."
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    pdfmetrics.registerFont(TTFont("Rift", str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("RiftBold", str(fonts / "DejaVuSans-Bold.ttf")))
    body = ParagraphStyle("body", fontName="Rift", fontSize=9.2, leading=13, spaceAfter=9)
    heading = ParagraphStyle("heading", fontName="RiftBold", fontSize=21, leading=27, spaceAfter=14)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.5, leading=10, spaceAfter=0)
    width = A4[0] - 88
    coverage = [
        [
            scope,
            str(source["coverage"][scope]["identified_rows"]),
            str(source["coverage"][scope]["matches"]),
        ]
        for scope in POLICY_SCOPES
    ]
    development, heldout = [], []
    for scope in POLICY_SCOPES:
        chosen = source["selected"][scope]
        bc = next(item for item in source["candidates"][scope] if item["name"] == "bc")
        development.append(
            [
                scope,
                chosen["name"],
                f"{(chosen['identified'] or {}).get('macro_f1', 0):.3f}",
                f"{(bc['identified'] or {}).get('macro_f1', 0):.3f}",
                f"{chosen['voluntary_concentration']:.1%}",
            ]
        )
        group = measured["per_group"][scope]
        heldout.append(
            [
                scope,
                f"{(group['identified'] or {}).get('macro_f1', 0):.3f}",
                f"{group['retention_coverage']:.1%}",
                f"{group['largest_prediction_fraction']:.1%}",
                str(group["prediction_coverage"]),
            ]
        )
    sections = [
        (
            "Offline Rift: six-policy macro coach",
            [
                "Current product synthesis. One team-direction model and separate TOP, JUNGLE, MIDDLE, BOTTOM and UTILITY specialists share causal state/action contracts and terminal team victory as the objective. The ADC-only pilot is historical.",
                "This phase builds and tests the product offline. No active-game test was run. Saved-model correctness and behavior diagnostics do not establish optimal strategy, calibrated victory probabilities or increased win rate.",
                "Architecture: verified raw cache → causal v3 evidence and complete episodes → one maintained trainer → six independent selected networks → integrity-checked inference router → team proposal, role proposal and symbolic alignment/conflict. Production imports no research; inference imports no training orchestration.",
                "Coordination compares the actual proposals. It does not blend scores, silently override the specialist, or claim that the specialist was trained conditional on a hypothetical macro plan.",
            ],
            [
                ["Layer", "Boundary"],
                ["Common", "Causal state, evidence, transitions, scopes, compatibility"],
                ["Production training", "Optimizer and explicit TD/observed-return targets"],
                ["Production inference", "Six verified independent networks and routing"],
                ["Workflow / UI", "One orchestration path and optional future delivery"],
                ["Research", "Offline diagnostics and reproducible reporting"],
            ],
        ),
        (
            "Data, targets and exclusions",
            [
                f"The frozen dataset is {source['dataset_path']}. Its SHA-256 is {source['dataset_sha256']}. There are {len(training['training_matches'])} training, {len(training['validation_matches'])} development and {len(training['holdout_matches'])} reserved matches. Every perspective of a match remains together.",
                "The existing cache contained enough role and macro evidence, including previously downloaded games not used by the earlier study. No extra games were collected merely to increase volume. Reserved IDs were frozen before this comparison; pooled support audits inspected evidence availability, not holdout model performance.",
                "V3 retains representative labels, candidate activities, identification flags and complete stored transitions. Death-only combat and overlapping activity do not identify intent. TEAM targets require three identified matching executed macro options. No coherent direction is unresolved, not a positive continue strategy.",
                "Only identified rows in the requested scope fit its action-specific losses/support counts. Exclusions remain a selection bias. Complete episodes, exact next states, measured elapsed discounts and terminal rewards are preserved without stitching gaps.",
            ],
            [["Scope", "Identified rows (pooled)", "Matches with evidence"], *coverage],
        ),
        (
            "Training and development comparison",
            [
                "Each scope compares two conservative critics with a behavior-imitation baseline. Networks are independent and use one maintained optimizer path. BC is diagnostic only. Selection prefers noncollapsed CQL candidates, then within-scope identified development macro F1, with concentration as a tie breaker.",
                "cql-td uses supported next-action Bellman targets. cql-return variants regress complete observed episode returns with the same conservative penalty. Macro candidates use observed returns because identified macro intervals rarely coincide with terminal frames. This is an explicit conservative return-regression adaptation, not a replication or new theorem for standard CQL.",
                "Terminal team result is used only as supervision. Observed trajectory returns remain confounded by teammates, opponents and hidden state; they are not identified causal alternative-action values. Bounded heads constrain numerical range but do not calibrate values.",
                "F1 measures reconstructed-activity agreement. A higher imitation baseline or concentrated policy must be reported, not reinterpreted as a victory-rate comparison. All candidates, failures and source snapshots are retained in the study directory.",
            ],
            [
                ["Scope", "Selected critic", "Identified F1", "BC F1", "Voluntary top share"],
                *development,
            ],
        ),
        (
            "Reserved-match results",
            [
                f"This fixed selection was evaluated on {len(measured['evaluated_matches'])} reserved match IDs and {measured['evaluated_transitions']} measured transitions. These IDs were excluded from optimizer and development selection; repeated players and team dependence remain possible. This is an offline retrospective test.",
                "Identified F1 uses identified activities only. Retention is the fraction receiving advice; action diversity counts predicted classes. Top share includes forced respawn outputs in this table, while the development table separates voluntary concentration. None of these measures is policy win rate.",
                "Macro evidence is much thinner than role evidence. Even coherent executed team activity does not establish deliberate coordination. Sparse targets, action reconstruction ambiguity, hidden positioning/vision and selection bias limit interpretation.",
                "Controlled missing-health and missing-history tests are saved in the evaluation report when requested. Changes after removing evidence can be legitimate. They do not establish the frequency of real live missingness or uncertainty calibration.",
                cold_text,
            ],
            [["Scope", "Identified F1", "Retention", "Top share", "Action types"], *heldout],
        ),
        (
            "Engineering, cleanup and next tests",
            [
                "The coach manifest binds six child manifests, aggregate metadata and the completed reload-parity record. Children must agree on data digest, feature ordering/scaling and match partitions, and declare independent fitting/serving scopes. Corruption or incompatible children fail loading.",
                checks_text
                + "Tests exercise real small optimizer/save/reload runs, scope isolation, holdout exclusion, terminal discounting, routing, corruption rejection, coordination and synthetic local UI behavior. These are engineering checks, not evidence of strategic benefit.",
                "Superseded ADC-only utilities, dated reports and the old specialized synthesis renderer were checksummed into the local archive. Current documentation is concise; historical raw data, datasets, weights and failed/interrupted reports were not deleted. See research/docs/HISTORY.json for the archive manifest.",
                "Later live acceptance should validate each selected role and the macro view, actual input availability, advice expiry, conflicts and opt-in recording. A predefined prospective comparison with terminal victory and measured adherence is still required before an increased-win-rate claim. No live test was needed or performed for this phase.",
                f"Reproduce offline: rift verify --model {source['bundle']}; rift-research evaluate --new-matches --stress --dataset {source['dataset_path']} --bundle {source['bundle']} --output NEW_DIRECTORY. Current source and contracts: README.md, AGENTS.md, docs/ARCHITECTURE.md, docs/MODELING.md and research/docs/PROTOCOL.md.",
            ],
            None,
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    story, markdown = [], []
    for number, (title, paragraphs, rows) in enumerate(sections):
        if number:
            story.append(PageBreak())
        story.append(Paragraph(escape(title), heading))
        markdown.extend(["# " + title, ""])
        for paragraph in paragraphs:
            story.append(Paragraph(escape(paragraph), body))
            markdown.extend([paragraph, ""])
        if rows:
            table = Table(
                [[Paragraph(escape(str(value)), cell) for value in row] for row in rows],
                colWidths=[width / len(rows[0])] * len(rows[0]),
                repeatRows=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e9ef")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.HexColor("#f1f6f8"), colors.white],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story.append(table)
            markdown.extend(["| " + " | ".join(map(str, row)) + " |" for row in rows[:1]])
            markdown.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
            markdown.extend(["| " + " | ".join(map(str, row)) + " |" for row in rows[1:]])
            markdown.append("")
        if number == 1:
            figure, axis = plt.subplots(figsize=(6.8, 2.2))
            axis.barh(
                list(POLICY_SCOPES),
                [source["coverage"][scope]["identified_rows"] for scope in POLICY_SCOPES],
                color="#087f8c",
            )
            axis.set_xlabel("Pooled identified activity rows (not independent matches)")
            figure.tight_layout()
            svg = output.with_name(output.stem + "_support.svg")
            figure.savefig(svg, metadata={"Date": None})
            stream = BytesIO()
            figure.savefig(stream, format="png", dpi=200)
            plt.close(figure)
            story.extend(
                [Spacer(1, 14), Image(BytesIO(stream.getvalue()), width=width, height=165)]
            )
            markdown.extend([f"![Identified activity support]({svg.name})", ""])

    def footer(canvas, document):
        canvas.setFont("Rift", 7)
        canvas.drawString(44, 25, "OFFLINE RIFT / SIX-POLICY OFFLINE SYNTHESIS")
        canvas.drawRightString(A4[0] - 44, 25, str(document.page))

    with TemporaryDirectory(prefix="rift-synthesis-") as temporary:
        pdf = Path(temporary) / "report.pdf"
        SimpleDocTemplate(
            str(pdf),
            pagesize=A4,
            leftMargin=44,
            rightMargin=44,
            topMargin=42,
            bottomMargin=44,
            title="Offline Rift six-policy synthesis",
        ).build(story, onFirstPage=footer, onLaterPages=footer)
        with pymupdf.open(pdf) as document:
            pages = len(document)
            for page in document:
                for x0, y0, x1, y1, *_ in page.get_text("blocks"):
                    if min(x0, y0) < 0 or x1 > A4[0] + 1 or y1 > A4[1] + 1:
                        raise ValueError("Generated synthesis has text outside page bounds")
        output.write_bytes(pdf.read_bytes())
    output.with_suffix(".md").write_text("\n".join(markdown))
    return {
        "pdf": str(output.resolve()),
        "pages": pages,
        "study_sha256": file_digest(study / "study.json"),
        "evaluation_sha256": file_digest(evaluation / "report.json"),
    }
