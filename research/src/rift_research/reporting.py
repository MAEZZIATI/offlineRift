"""Reproducible current synthesis from completed coach and frozen offline evaluation."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape

from rift_common.artifacts import file_digest, read_json
from rift_common.coaching import POLICY_SCOPES


def synthesize(study, evaluation, output, *, baseline_evaluation=None):
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
    baseline = None
    if baseline_evaluation is not None:
        baseline = read_json(Path(baseline_evaluation) / "report.json")
        if (
            baseline["evaluated_matches"] != measured["evaluated_matches"]
            or baseline["dataset_sha256"] != measured["dataset_sha256"]
        ):
            raise ValueError("Baseline and candidate must use the same dataset and evaluation IDs")
    if source["status"] != "complete" or measured["bundle_sha256"] != file_digest(
        Path(source["bundle"]) / "coach.json"
    ):
        raise ValueError("Synthesis requires matching completed coach/evaluation artifacts")
    run = read_json(Path(source["bundle"]) / "run.json")
    training = run["report"]
    robust_study = source.get("study_design") == "coherent-robust-v1"
    protocol_file = study / "protocol.json"
    protocol = read_json(protocol_file) if protocol_file.exists() else {}
    collection = protocol.get("collection_observation", {})
    collection_text = (
        f"Bounded collection: {collection.get('verified_accounts', 'unreported')} configured verified accounts; "
        f"{collection.get('reused_match_references', 'unreported')} cache references and "
        f"{collection.get('new_matches', 'unreported')} new payloads from "
        f"{collection.get('accounts_contributing_new_games', 'unreported')} accounts; route {collection.get('region', 'unreported')}. "
        "All new IDs were reserved before fitting. Historical tests are now development evidence. "
        "Rebuilding uses retained raw payloads, not synthetic games. Configured-account distribution and recurring players limit generalization."
    )
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
        if baseline is not None:
            heldout[-1].insert(
                2, f"{(baseline['per_group'][scope]['identified'] or {}).get('macro_f1', 0):.3f}"
            )
    sections = [
        (
            "Offline Rift: six-policy macro coach",
            [
                "Current product synthesis. One team-direction model and separate TOP, JUNGLE, MIDDLE, BOTTOM and UTILITY specialists share causal state/action contracts and terminal team victory as the objective. The ADC-only pilot is historical.",
                "This phase builds and tests the product offline. No active-game test was run. Saved-model correctness and behavior diagnostics do not establish optimal strategy, calibrated victory probabilities or increased win rate.",
                "Architecture: verified raw cache → causal v3 evidence and complete episodes → one maintained trainer → six independent selected networks → integrity-checked inference router → team proposal, role proposal and symbolic alignment/conflict. Production imports no research; inference imports no training orchestration.",
                "Coordination compares actual proposals. A conflicting role destination is explicitly withheld while the macro proposal remains visible. No compatible alternative is invented. This constraint is not learned plan conditioning or evidence of strategic compatibility.",
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
                (
                    collection_text
                    if robust_study
                    else "The existing cache contained enough role and macro evidence, including previously downloaded games not used by the earlier study. No extra games were collected merely to increase volume. Reserved IDs were frozen before this comparison; pooled support audits inspected evidence availability, not holdout model performance."
                ),
                "V3 retains representative labels, candidate activities, identification flags and complete stored transitions. Death-only combat and overlapping activity do not identify intent. TEAM targets require three identified matching executed macro options. No coherent direction is unresolved, not a positive continue strategy.",
                "Only identified rows in the requested scope fit its action-specific losses/support counts. Exclusions remain a selection bias. Complete episodes, exact next states, measured elapsed discounts and terminal rewards are preserved without stitching gaps.",
            ],
            [["Scope", "Identified rows (pooled)", "Matches with evidence"], *coverage],
        ),
        (
            "Training and development comparison",
            [
                (
                    "Each scope compares ordinary imitation and conservative observed-return regression with availability-consistent versions: four candidates per scope, 24 total. Every algorithm is eligible for selection. The frozen score averages clean and cold-feasible identified F1 and subtracts 0.1 times still-feasible prediction switching. Gates require nonempty evidence, clean/cold retention at least 50% and voluntary concentration at most 90%. These exploratory gates do not prove policy quality."
                    if robust_study
                    else "Each scope compares two conservative critics with a behavior-imitation baseline. Networks are independent and use one maintained optimizer path. BC is diagnostic only. Selection prefers noncollapsed CQL candidates, then within-scope identified development macro F1, with concentration as a tie breaker."
                ),
                "cql-td uses supported next-action Bellman targets. cql-return variants regress complete observed episode returns with the same conservative penalty. Macro candidates use observed returns because identified macro intervals rarely coincide with terminal frames. This is an explicit conservative return-regression adaptation, not a replication or new theorem for standard CQL.",
                "Terminal team result is used only as supervision. Observed trajectory returns remain confounded by teammates, opponents and hidden state; they are not identified causal alternative-action values. Bounded heads constrain numerical range but do not calibrate values.",
                "A selected imitation child learns next-activity agreement, not terminal win/loss directly. The product's victory objective does not convert this supervised surrogate into a demonstrated winning policy. Return losses apply to critic candidates only.",
                "F1 measures reconstructed-activity agreement. A higher imitation baseline or concentrated policy must be reported, not reinterpreted as a victory-rate comparison. All candidates, failures and source snapshots are retained in the study directory.",
            ],
            [
                ["Scope", "Selected model", "Identified F1", "BC F1", "Voluntary top share"],
                *development,
            ],
        ),
        (
            "Reserved-match results",
            [
                f"This fixed selection was evaluated on {len(measured['evaluated_matches'])} reserved match IDs and {measured['evaluated_transitions']} measured transitions. These IDs were excluded from optimizer and development selection; repeated players and team dependence remain possible. This is an offline retrospective test.",
                "Identified F1 uses identified activities only. Retention here is raw per-policy support coverage BEFORE coordination withholding, not final paired-advice coverage. Action diversity counts predicted classes. Top share includes forced respawn outputs in this table, while the development table separates voluntary concentration. None of these measures is policy win rate.",
                "Macro evidence is much thinner than role evidence. Even coherent executed team activity does not establish deliberate coordination. Sparse targets, action reconstruction ambiguity, hidden positioning/vision and selection bias limit interpretation.",
                "Controlled missing-health and missing-history tests are saved in the evaluation report when requested. Changes after removing evidence can be legitimate. They do not establish the frequency of real live missingness or uncertainty calibration.",
                cold_text,
            ],
            [
                [
                    "Scope",
                    "Identified F1",
                    *(["Previous F1"] if baseline is not None else []),
                    "Retention",
                    "Top share",
                    "Action types",
                ],
                *heldout,
            ],
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
    if robust_study:
        robustness_rows = [["Scope", "Selected", "Cold F1", "Feasible switching", "Gate"]]
        for scope in POLICY_SCOPES:
            item = source["selected"][scope]
            switching = item["still_feasible_choice_changes"]
            robustness_rows.append(
                [
                    scope,
                    item["name"],
                    f"{(item['cold_identified'] or {}).get('macro_f1', 0):.3f}",
                    f"{switching:.1%}" if switching is not None else "unmeasured",
                    "pass" if item["development_gate_passed"] else "FAIL",
                ]
            )
        sections.insert(
            3,
            (
                "Mathematical design and robustness",
                [
                    "Partial observability: x = phi(H) summarizes causal history with explicit known flags; it is not a sufficient belief state. Terminal reward is Y in {-1,+1}. Complete-episode return satisfies G_t = r_t + gamma^(d_t/60) G_next. Hidden teammates, opponents and continuation behavior prevent identifying counterfactual action values from observed returns.",
                    "Imitation minimizes weighted cross entropy. Conservative regression minimizes Huber(q(x,a),G) + alpha[logsumexp q(x,b) - q(x,a)] + 0.25 cross entropy, with bounded q. The history-consistent variants add 0.25 times detached-teacher KL plus critic Huber disagreement on supported actions feasible before AND after omission. Joint history dropout is 50%; neither values nor known flags are imputed.",
                    "The regularizer assumes some ranking stability after information loss. This can suppress justified adaptation; the table therefore reports both cold-target agreement and switching conditional on the warm action remaining feasible. Recorded targets invalidated by omission are excluded from the cold F1 denominator, never silently relabeled.",
                    "Coherence is a constrained delivery rule: when macro and role choose different explicit destinations, withhold the role action. Other pairs remain independent rather than being claimed aligned. This adds abstention, not a learned alternative or a joint-value theorem. Recheck after advice persistence.",
                    "The full equations, assumptions and primary references are in docs/MATHEMATICAL_DESIGN.md. Related work: Kumar et al., CQL (arxiv.org/abs/2006.04779); Yang et al., RORL (arxiv.org/abs/2206.02829). Our availability omission experiment is not a replication of RORL.",
                ],
                robustness_rows,
            ),
        )
        coordination = measured.get("coordination", {})
        sections[-1][1].insert(
            0,
            f"Development gates: {'all six passed' if source.get('development_gate_passed') else 'at least one failed; candidate must not replace the default'}. "
            f"Coordination audit over {coordination.get('role_rows', 'unmeasured')} measured role rows: "
            f"{coordination.get('aligned', 'unmeasured')} matching actions, {coordination.get('independent', 'unmeasured')} independent proposals, "
            f"{coordination.get('withheld_conflict', 'unmeasured')} conflicting role directions withheld and {coordination.get('forced_respawn', 'unmeasured')} forced-respawn pairs. "
            f"Missing macro rows: {coordination.get('missing_macro_row', 'unmeasured')}; unsupported pairs: {coordination.get('unsupported_pair', 'unmeasured')}. "
            "These are instantaneous proposal counts; persistence and real-game benefit are not simulated.",
        )
    if baseline is not None:
        sections[-2][1].append(
            "Previous F1 is the former default evaluated on exactly the same frozen match IDs and dataset. This measures a product comparison, not an isolated algorithm ablation: the historical model used different training partitions and update counts. Test results were measured after candidate selection and did not change it. Baseline bundle: "
            + baseline["bundle"]
        )
        macro_rows = sum(measured["per_group"]["TEAM"]["identified"]["action_support"])
        sections[-2][1].append(
            f"The reserved macro F1 uses only {macro_rows} identified TEAM rows. This is thin evidence, not a reliable estimate of strategic utility. Read all positive and negative differences as exploratory. Activation, if performed, retains an experimental offline candidate selected by the frozen development rule; it does not establish superiority on every metric."
        )
        previous_cold = baseline.get("availability_stress", {}).get("cold_history", {})
        if cold and previous_cold:

            def percentage(value):
                return "unmeasured" if value is None else f"{value:.1%}"

            rows = [
                [
                    "Scope",
                    "Previous switch",
                    "New switch",
                    "Previous voluntary",
                    "New voluntary",
                    "New voluntary rows",
                ]
            ]
            for scope in POLICY_SCOPES:
                before, after = previous_cold[scope], cold[scope]
                rows.append(
                    [
                        scope,
                        percentage(before["changed_when_original_still_feasible"]),
                        percentage(after["changed_when_original_still_feasible"]),
                        percentage(before.get("voluntary_changed_when_original_still_feasible")),
                        percentage(after.get("voluntary_changed_when_original_still_feasible")),
                        str(after.get("voluntary_still_feasible_rows", "unmeasured")),
                    ]
                )
            sections.insert(
                len(sections) - 1,
                (
                    "History omission on reserved matches",
                    [
                        "Both fixed products are evaluated on the same measured states after controlled history omission. Switch rates condition on the original action remaining feasible; voluntary rates additionally remove forced respawn choices. The conditioning cohorts can differ because the two models choose different original actions.",
                        "Lower switching is not automatically better strategy: history can change the correct choice. Read this table together with identified agreement, action diversity and support coverage. The dropout experiment does not establish how often information will be absent in a real game.",
                        "The previous default used return critics; the new selection may use imitation. Gains therefore cannot be attributed solely to the consistency penalty. Same-study ordinary versus consistent candidates provide the controlled development comparison; no test retuning was performed.",
                    ],
                    rows,
                ),
            )
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
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
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
        "baseline_evaluation_sha256": file_digest(Path(baseline_evaluation) / "report.json")
        if baseline_evaluation is not None
        else None,
    }
