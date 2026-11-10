"""Causal all-player game replay and an explanatory, executable research notebook."""

from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
from rift_common.actions import ACTION_NAMES, Action
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.behavior import player_activity, team_activity
from rift_common.coaching import POLICY_SCOPES, withhold_conflicting_role
from rift_common.metrics import behavior_metrics
from rift_common.timeline import read_match


def replay_game(payload, coach, *, catalog=None):
    """Replay measured frames only. Future frames supply labels, never recommendations."""
    from offline_rift.inference.advice import AdviceController

    match, timeline = payload["match"], payload["timeline"]
    match_id = match["metadata"]["matchId"]
    split = coach.metadata["training_report"]
    if match_id in set(split["training_matches"]) | set(split["validation_matches"]):
        raise ValueError("Game review requires a match excluded from training and development")
    observations = read_match(match, timeline)
    controller = AdviceController()
    rows = []
    for index, (current, following) in enumerate(pairwise(observations)):
        if current.time < 90:
            continue
        previous = observations[index - 1] if index else None
        for team in (100, 200):
            actors = [None, *[p.id for p in current.players if p.team == team]]
            for actor in actors:
                player = current.player(actor) if actor is not None else None
                scope = player.role if player else "TEAM"
                if scope not in POLICY_SCOPES:
                    raise ValueError("All ten players need retrospective role metadata")
                child = coach.policies[scope]
                raw = child.recommend(
                    current, team, actor, role=scope, previous=previous, catalog=catalog
                )
                paired = coach.recommend(
                    current, team, actor, role=scope, previous=previous, catalog=catalog
                )
                served = withhold_conflicting_role(controller.select(paired))
                evidence = (
                    player_activity(current, following, actor)
                    if actor
                    else team_activity(current, following, team)
                )
                # Everything above this point uses observations at or before current.time.
                rows.append(
                    {
                        "match_id": match_id,
                        "time": current.time,
                        "end_time": following.time,
                        "team": team,
                        "actor": actor or "team",
                        "scope": scope,
                        "champion": player.champion if player else "Team",
                        "forced_respawn": bool(player and player.is_dead is True),
                        "raw_action": raw.get("action_id", -1),
                        "served_action": served.get("action_id", -1),
                        "status": served["status"],
                        "coordination": served.get("coordination", {}).get("status", "team_only"),
                        "macro_action": served.get("macro_recommendation", {}).get("action_id", -1),
                        "observed_action": int(evidence.representative.action) if evidence else -1,
                        "identified": bool(evidence and evidence.identified),
                        "candidate_actions": list(evidence.candidates) if evidence else [],
                        "evidence": evidence.representative.evidence if evidence else "unresolved",
                        "instruction": served.get("instruction", served.get("message", "")),
                        "check": served.get("check", ""),
                        "scores": raw.get("scores", []),
                        "method": child.algorithm,
                    }
                )
    return rows


def review_metrics(rows):
    result = {}
    for scope in POLICY_SCOPES:
        selected = [r for r in rows if r["scope"] == scope]
        identified = [r for r in selected if r["identified"]]
        voluntary = [r for r in selected if not r["forced_respawn"]]
        counts = Counter(r["served_action"] for r in voluntary if r["served_action"] >= 0)
        result[scope] = {
            "rows": len(selected),
            "identified_rows": len(identified),
            "raw_coverage": float(np.mean([r["raw_action"] >= 0 for r in selected])),
            "served_coverage": float(np.mean([r["served_action"] >= 0 for r in selected])),
            "voluntary_coverage": float(np.mean([r["served_action"] >= 0 for r in voluntary]))
            if voluntary
            else None,
            "voluntary_top_share": max(counts.values(), default=0) / max(1, sum(counts.values())),
            "identified_raw": behavior_metrics(
                [r["observed_action"] for r in identified],
                [r["raw_action"] for r in identified],
                len(Action),
            )
            if identified
            else None,
            "identified_served": behavior_metrics(
                [r["observed_action"] for r in identified],
                [r["served_action"] for r in identified],
                len(Action),
            )
            if identified
            else None,
            "statuses": dict(Counter(r["status"] for r in selected)),
            "coordination": dict(Counter(r["coordination"] for r in selected)),
        }
    return result


def review_game(
    match,
    bundle,
    output,
    *,
    baseline=None,
    study=None,
    protocol=None,
    holdout_comparison=None,
    followup=None,
    catalogs="data/catalogs",
):
    from offline_rift.catalogs import CatalogCache, patch_of
    from offline_rift.inference.coach import MacroCoach

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = read_json(Path(match))
    info = payload["match"]["info"]
    match_id = payload["match"]["metadata"]["matchId"]
    if (
        not match_id.startswith(("EUW1_", "EUN1_"))
        or info.get("queueId") != 420
        or info.get("mapId") != 11
    ):
        raise ValueError("Expected a European ranked solo Summoner's Rift game")
    cache = CatalogCache(catalogs)
    try:
        catalog, catalog_version, catalog_sha256 = cache.get(patch_of(info["gameVersion"]))
    finally:
        cache.close()
    atomic_json(output / "replay-item-catalog.json", catalog.data)
    verified_source = None
    if protocol:
        frozen = read_json(Path(protocol))
        if frozen.get("notebook_match") != match_id or match_id not in frozen["holdout_matches"]:
            raise ValueError("Notebook game must be reserved by the frozen protocol")
        if frozen.get("verified_roster_snapshot"):
            snapshot_path = Path(frozen["verified_roster_snapshot"])
            snapshot = read_json(snapshot_path)
            identities = {entry["puuid"] for entry in snapshot["entries"]}
            verified = [
                p["participantId"] for p in info["participants"] if p.get("puuid") in identities
            ]
            if not verified or snapshot["platform"] not in {"euw1", "eun1"}:
                raise ValueError("Game must contain an API-verified European roster player")
            verified_source = {
                "platform": snapshot["platform"],
                "current_tier": snapshot["tier"],
                "participant_ids": verified,
                "snapshot_sha256": file_digest(snapshot_path),
            }
    report = {
        "schema": "all-player-game-review-v1",
        "match_id": match_id,
        "raw_path": str(Path(match).resolve()),
        "raw_sha256": file_digest(match),
        "patch": info["gameVersion"],
        "queue": info["queueId"],
        "duration_seconds": info["gameDuration"],
        "game_creation_utc": datetime.fromtimestamp(
            info["gameCreation"] / 1000, timezone.utc
        ).isoformat()
        if info.get("gameCreation")
        else None,
        "winning_team": next(p["teamId"] for p in info["participants"] if p["win"]),
        "participants": [
            {k: p.get(k) for k in ("participantId", "teamId", "teamPosition", "championName")}
            for p in info["participants"]
        ],
        "protocol": str(Path(protocol).resolve()) if protocol else None,
        "verified_collection_account": verified_source,
        "replay_catalog": {
            "version": catalog_version,
            "upstream_sha256": catalog_sha256,
            "sha256": file_digest(output / "replay-item-catalog.json"),
            "patch": patch_of(info["gameVersion"]),
        },
        "study": str(Path(study).resolve()) if study else None,
        "models": {},
        "limitations": [
            "One retrospectively replayed game cannot establish strategic utility or a win-rate gain.",
            "Roles are retrospective; current Challenger membership does not certify historical rank of all ten players.",
            "Only measured frames are replayed, typically one minute apart; five-second live persistence is not simulated.",
            "Activity agreement describes behavior, not the correctness of alternative actions.",
            "No vision, wave, camp safety or intermediate actions are fabricated. Scores are not win probabilities.",
        ],
    }
    for name, directory in {
        "candidate": bundle,
        **({"previous": baseline} if baseline else {}),
    }.items():
        coach = MacroCoach(directory)
        rows = replay_game(payload, coach, catalog=catalog)
        atomic_json(output / (name + "-decisions.json"), rows)
        report["models"][name] = {
            "bundle": str(Path(directory).resolve()),
            "sha256": file_digest(Path(directory) / "coach.json"),
            "state_schema": coach.metadata["state_schema"],
            "per_scope": review_metrics(rows),
        }
    if holdout_comparison:
        comparison = read_json(Path(holdout_comparison))
        if match_id not in comparison["matches"] or any(
            comparison["models"].get(name, {}).get("bundle_sha256") != model["sha256"]
            for name, model in report["models"].items()
        ):
            raise ValueError("Holdout comparison does not bind this game and these models")
        atomic_json(output / "holdout-comparison.json", comparison)
    if followup:
        atomic_json(output / "followup.json", read_json(Path(followup)))
        report["followup_sha256"] = file_digest(output / "followup.json")
    atomic_json(output / "review.json", report)
    return report


def write_notebook(review_directory, destination):
    """Notebook calls maintained replay/plotting functions, with portable local paths."""
    import nbformat as nbf

    directory, destination = Path(review_directory).resolve(), Path(destination)
    root = Path.cwd().resolve()
    relative = str(directory.relative_to(root))
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (Offline Rift .venv)",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# European high-elo game: six-policy coach\n\nAn executed development report and one-game replay covering both teams, all ten players and every policy. **This is behavior and delivery evaluation, not a demonstrated win-rate gain.**"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Markdown\nfrom rift_research.game_review import notebook_section\nROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'pyproject.toml').exists() and (p / 'common').exists())\nREVIEW = ROOT / "
            + repr(relative)
            + "\nreport = json.loads((REVIEW / 'review.json').read_text())\ndecisions = pd.read_json(REVIEW / 'candidate-decisions.json')\nplt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})"
        ),
    ]
    sections = [
        (
            "What changed and what was frozen",
            "development",
            "The new state expresses public-stat leads and deficits explicitly and records measured alive fractions. Missing operands stay unknown. European games broaden the previous Korean corpus. Phase support screens out actions with too few examples at that stage of a game. The game below was reserved before fitting; selection uses development matches only. See [the mathematical design](../../docs/MATHEMATICAL_DESIGN.md) for the losses and assumptions.",
        ),
        (
            "Development comparison",
            "selection",
            "Imitation predicts recorded activities; conservative return regression fits observed discounted terminal outcomes; IQL uses critic/value learning and an advantage-weighted actor. The consistency variants penalize some changes under missing history. Every algorithm remains eligible. These metrics measure activity agreement and missing-history robustness. The previous-product comparison changes multiple factors, so it is not an isolated ablation.",
        ),
        (
            "Learning curves and early stopping",
            "learning",
            "Longer training can overfit the reconstructed labels. Every 500 updates, checkpoints are evaluated on development data under the same clean/cold score and gates. Training stops after three nonimproving checkpoints; the saved weights come from the best checkpoint, which can precede the last update.",
        ),
        (
            "Independent European holdout",
            "holdout",
            "The broader held-out European cohort is separate from training/development. Match-level uncertainty and sparse macro evidence limit conclusions.",
        ),
        (
            "What the first test taught us",
            "followup",
            "A failed evaluation remains part of the evidence. If a later development selection responds to that finding, it needs a fresh test before claiming generalization.",
        ),
        (
            "The reserved European game",
            "game",
            "Current verified Challenger roster membership establishes the collection account's current tier, not all participants' historical ranks. Roles shown here are retrospective.",
        ),
        (
            "Team direction over time",
            "teams",
            "Two team policies are evaluated, one for each side. Recorded future activity is shown only for evaluation and remains unresolved when evidence is ambiguous.",
        ),
        (
            "All ten players: recommendations versus measured activity",
            "players",
            "Each player's role specialist is evaluated alongside the team proposal. Light gray means abstention or unresolved evidence. An observed action is not necessarily the best action.",
        ),
        (
            "Coverage, concentration and coordination",
            "coverage",
            "Coverage is measured after conflict withholding and measured-frame persistence. Forced respawn is separated from voluntary advice.",
        ),
        (
            "Agreement by policy",
            "agreement",
            "F1 and confusion matrices use identified targets only. Missing support and unresolved team targets remain visible.",
        ),
        (
            "Inspect individual recommendations",
            "decisions",
            "This table includes the instruction, prerequisite checks, independent role proposal, team proposal and served result. Use the data frame to inspect any player or minute.",
        ),
        (
            "Conclusions and limits",
            "conclusion",
            "Read the measured improvements and regressions together. One game's result does not establish the causal value of any recommendation.",
        ),
    ]
    for title, section, description in sections:
        notebook.cells += [
            nbf.v4.new_markdown_cell("## " + title + "\n\n" + description),
            nbf.v4.new_code_cell("notebook_section(REVIEW, " + repr(section) + ")"),
        ]
    notebook.cells += [
        nbf.v4.new_markdown_cell(
            "## Reproduce the replay\n\nThe cell below reloads the saved models and raw game without any network access. It checks that every recorded recommendation is reproducible."
        ),
        nbf.v4.new_code_cell(
            "from rift_research.game_review import replay_game\nfrom offline_rift.inference.coach import MacroCoach\nfrom rift_common.artifacts import file_digest\nfrom rift_common.items import ItemCatalog\ncatalog_path = REVIEW / 'replay-item-catalog.json'\nassert file_digest(catalog_path) == report['replay_catalog']['sha256']\ncatalog = ItemCatalog(json.loads(catalog_path.read_text()))\nassert file_digest(Path(report['raw_path'])) == report['raw_sha256']\npayload = json.loads(Path(report['raw_path']).read_text())\nfor name, model in report['models'].items():\n    assert file_digest(Path(model['bundle']) / 'coach.json') == model['sha256']\n    replayed = replay_game(payload, MacroCoach(model['bundle']), catalog=catalog)\n    saved = json.loads((REVIEW / f'{name}-decisions.json').read_text())\n    assert replayed == saved, name\nprint('Verified: all policies, both teams, all ten players; no downloads or fitting.')"
        ),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    return destination


def notebook_section(directory, section):
    """Render tables and scientific figures from retained measured artifacts."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import Markdown, display
    from matplotlib.colors import BoundaryNorm, ListedColormap

    directory = Path(directory)
    report = read_json(directory / "review.json")
    rows = pd.DataFrame(read_json(directory / "candidate-decisions.json"))
    study = read_json(Path(report["study"])) if report.get("study") else None
    names = {**ACTION_NAMES, -1: "Abstain / unresolved"}
    colors = ["#d8dadd", *plt.get_cmap("tab20").colors[:15]]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-1.5, 15.5), cmap.N)

    def show():
        plt.tight_layout()
        plt.show()

    def summary():
        return pd.DataFrame(
            [
                {
                    "model": name,
                    "scope": scope,
                    "identified_n": m["identified_rows"],
                    "raw_coverage": m["raw_coverage"],
                    "served_coverage": m["served_coverage"],
                    "voluntary_coverage": m["voluntary_coverage"],
                    "voluntary_top_share": m["voluntary_top_share"],
                    "identified_F1": (m["identified_raw"] or {}).get("macro_f1"),
                }
                for name, model in report["models"].items()
                for scope, m in model["per_scope"].items()
            ]
        )

    if section == "development":
        protocol = read_json(Path(report["protocol"])) if report.get("protocol") else {}
        display(
            Markdown(
                "**Changes tested:** 24 masked relative-strength/alive features in state v2; EUW data alongside retained Korean games; phase-specific empirical action support; softer class weights; development checkpoint selection; ordinary/consistent imitation, conservative return regression and IQL. Six models fit independently. Team/role conflicts are withheld without inventing an alternative."
            )
        )
        display(
            pd.DataFrame(
                [{k: v for k, v in protocol.items() if not isinstance(v, (list, dict))}]
            ).T.rename(columns={0: "frozen setting"})
        )
        display(
            pd.DataFrame(
                [
                    {"model": k, "state_schema": v["state_schema"], "bundle_sha256": v["sha256"]}
                    for k, v in report["models"].items()
                ]
            )
        )
        if study:
            from rift_common.actions import GROUPS

            data_report = read_json(Path(study["dataset_path"]).with_name("report.json"))
            display(
                pd.DataFrame(
                    [
                        {
                            "accepted_matches": data_report["matches"],
                            "transitions": data_report["transitions"],
                            "accepted_european_matches": sum(
                                m.startswith(("EUW1_", "EUN1_"))
                                for m in data_report["accepted_matches"]
                            ),
                            "excluded_matches": len(data_report["skipped"]),
                            "exclusion_reasons": dict(
                                Counter(m["reason"] for m in data_report["skipped"])
                            ),
                        }
                    ]
                )
            )
            support = np.asarray(
                [
                    read_json(Path(study["selected"][scope]["bundle"]) / "metadata.json")[
                        "training_report"
                    ]["counts_by_group"][GROUPS.index(scope)]
                    for scope in POLICY_SCOPES
                ]
            )
            fig, ax = plt.subplots(figsize=(14, 4))
            image = ax.imshow(np.log1p(support), cmap="viridis", aspect="auto")
            ax.set(
                yticks=range(6),
                yticklabels=POLICY_SCOPES,
                xticks=range(15),
                xticklabels=[ACTION_NAMES[i] for i in range(15)],
                title="Identified training support by policy/action (log1p count)",
            )
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
            fig.colorbar(image, ax=ax, label="log(1 + examples)")
            show()
            display(
                pd.DataFrame(
                    {
                        "scope": POLICY_SCOPES,
                        "identified_training_rows": support.sum(1),
                        "supported_action_classes": (support >= 20).sum(1),
                    }
                )
            )
    elif section == "selection":
        if not study:
            display(Markdown("No development study supplied."))
            return
        data = pd.DataFrame(
            [
                {
                    "scope": scope,
                    "candidate": c["name"],
                    "clean_F1": (c["identified"] or {}).get("macro_f1"),
                    "cold_F1": (c["cold_identified"] or {}).get("macro_f1"),
                    "switching": c["still_feasible_choice_changes"],
                    "retention": c["retention"],
                    "concentration": c["voluntary_concentration"],
                    "score": c["selection_score"],
                    "passed": c["development_gate_passed"],
                    "selected": c["name"] == study["selected"][scope]["name"],
                }
                for scope, candidates in study["candidates"].items()
                for c in candidates
            ]
        )
        display(data.round(3))
        data.pivot(index="scope", columns="candidate", values="score").plot.bar(
            figsize=(13, 5),
            ylabel="Development selection score",
            title="Every candidate, including failed and unselected runs",
        )
        plt.legend(bbox_to_anchor=(1, 1))
        show()
    elif section == "learning":
        if not study:
            return
        if study.get("optimizer_updates_this_stage") == 0:
            display(
                Markdown(
                    "These curves belong to the original training stage. This revision reselects among those saved networks using equal-weight European/Korean development scores; it performs **zero additional optimizer updates** and does not retune checkpoints."
                )
            )
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, scope in zip(axes.flat, POLICY_SCOPES):
            for c in study["candidates"][scope]:
                path = Path(c["bundle"]) / "checkpoints.json"
                if path.exists():
                    cp = read_json(path)
                    ax.plot(
                        [x["step"] for x in cp["checkpoints"]],
                        [x["score"] for x in cp["checkpoints"]],
                        label=c["name"],
                        alpha=0.8,
                    )
            ax.set(title=scope, xlabel="Optimizer updates", ylabel="Development score")
        axes.flat[-1].legend(fontsize=7, bbox_to_anchor=(1, 1))
        show()
        display(
            pd.DataFrame(
                [
                    {
                        "scope": scope,
                        "selected": chosen["name"],
                        "completed_updates": metadata["optimizer_steps"],
                        "restored_checkpoint": metadata.get("selected_checkpoint_step"),
                        "development_score": chosen["selection_score"],
                    }
                    for scope, chosen in study["selected"].items()
                    for metadata in [
                        read_json(Path(chosen["bundle"]) / "metadata.json")["training_report"]
                    ]
                ]
            )
        )
    elif section == "holdout":
        path = directory / "holdout-comparison.json"
        if not path.exists():
            display(
                Markdown(
                    "No broader holdout comparison attached. The one-game replay below is descriptive."
                )
            )
            return
        comparison = read_json(path)
        table = pd.DataFrame(comparison["per_scope"])
        table["f1_lower"] = table.bootstrap.map(lambda b: b["macro_f1_95"][0] if b else None)
        table["f1_upper"] = table.bootstrap.map(lambda b: b["macro_f1_95"][1] if b else None)
        display(table.drop(columns="bootstrap").round(3))
        table.pivot(index="scope", columns="model", values="identified_f1").plot.bar(
            figsize=(10, 4),
            ylabel="Identified activity F1",
            title="Fresh European matches: fixed model comparison",
        )
        show()
        differences = pd.DataFrame(comparison["paired_differences"])
        display(differences)
        valid = differences[differences.f1_difference_95.notna()]
        if len(valid):
            intervals = np.asarray(valid.f1_difference_95.tolist())
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hlines(
                range(len(valid)), intervals[:, 0], intervals[:, 1], color="tab:blue", linewidth=2
            )
            ax.scatter(valid.point_difference, range(len(valid)), color="tab:blue")
            ax.axvline(0, color="gray", linestyle="--")
            ax.set(
                yticks=range(len(valid)),
                yticklabels=valid.scope,
                xlabel="Candidate minus previous F1: paired bootstrap 95% interval",
                title="Uncertainty in activity-agreement differences",
            )
            show()
        display(Markdown(comparison["meaning"]))
    elif section == "followup":
        path = directory / "followup.json"
        if not path.exists():
            display(Markdown("No adaptive follow-up is attached to this study."))
            return
        followup = read_json(path)
        revised_path = Path(followup["revised_study"])
        if file_digest(revised_path) != followup["revised_study_sha256"]:
            raise ValueError("Follow-up study integrity mismatch")
        revised = read_json(revised_path)
        display(Markdown(followup["explanation"]))
        display(
            Markdown("**First study's reserved European result (before the revised selection):**")
        )
        notebook_section(Path(followup["first_review_directory"]), "holdout")
        regional = pd.DataFrame(
            [
                {
                    "scope": scope,
                    "candidate": c["name"],
                    "region": region,
                    "identified_rows": d["identified_validation_rows"],
                    "clean_F1": (d["identified"] or {}).get("macro_f1"),
                    "cold_F1": (d["cold_identified"] or {}).get("macro_f1"),
                    "region_score": d["selection_score"],
                    "selected": c["name"] == revised["selected"][scope]["name"],
                }
                for scope, candidates in revised["candidates"].items()
                for c in candidates
                for region, d in c["development_regions"].items()
            ]
        )
        display(regional[regional.scope.eq("TEAM")].round(3))
        regional[regional.scope.eq("TEAM")].pivot(
            index="candidate", columns="region", values="clean_F1"
        ).plot.bar(
            figsize=(11, 4),
            ylabel="Identified development F1",
            title="TEAM development evidence: Europe versus Korea",
        )
        show()
        display(
            pd.DataFrame(
                [
                    {
                        "scope": scope,
                        "revised_selection": c["name"],
                        "balanced_development_score": c["selection_score"],
                        "passed_both_region_gates": c["development_gate_passed"],
                    }
                    for scope, c in revised["selected"].items()
                ]
            ).round(3)
        )
        display(Markdown("**Fresh-test status:** " + followup["fresh_test_status"]))
        if followup.get("selection_comparison"):
            comparison = read_json(Path(followup["selection_comparison"]))
            display(
                Markdown(
                    "**Fresh-cohort comparison of the two selection rules, using the same trained candidate bank:**"
                )
            )
            display(pd.DataFrame(comparison["paired_differences"]))
        display(Markdown("**Release decision:** " + followup["release_decision"]))
    elif section == "game":
        if report.get("replay_catalog"):
            display(pd.DataFrame([report["replay_catalog"]]))
        display(
            pd.DataFrame([report["verified_collection_account"]])
            if report.get("verified_collection_account")
            else Markdown("No roster evidence supplied.")
        )
        display(
            pd.DataFrame(
                [
                    {
                        k: report[k]
                        for k in (
                            "match_id",
                            "patch",
                            "queue",
                            "duration_seconds",
                            "game_creation_utc",
                            "winning_team",
                            "raw_sha256",
                        )
                    }
                ]
            ).T
        )
        display(pd.DataFrame(report["participants"]))
        payload = read_json(Path(report["raw_path"]))
        observed = read_match(payload["match"], payload["timeline"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for team in (100, 200):
            time = [o.time / 60 for o in observed]
            axes[0].plot(
                time,
                [sum(p.kills or 0 for p in o.players if p.team == team) for o in observed],
                label=str(team),
            )
            axes[1].plot(
                time,
                [sum(p.cs or 0 for p in o.players if p.team == team) for o in observed],
                label=str(team),
            )
        axes[0].set(title="Measured team kills", xlabel="Game minute", ylabel="Kills")
        axes[1].set(title="Measured team CS", xlabel="Game minute", ylabel="CS")
        axes[0].legend()
        axes[1].legend()
        show()
    elif section in {"teams", "players"}:
        subset = rows[rows.scope.eq("TEAM") if section == "teams" else rows.scope.ne("TEAM")].copy()
        subset["subject"] = subset.apply(lambda r: f"{r.team} · {r.scope} · {r.champion}", axis=1)
        times = sorted(subset.time.unique())
        subjects = sorted(subset.subject.unique())
        fig, axes = plt.subplots(3, 1, figsize=(15, 5 if section == "teams" else 12), sharex=True)
        for ax, column, title in zip(
            axes,
            ("raw_action", "served_action", "observed_action"),
            (
                "Independent model proposal",
                "Delivered after coordination / measured-frame persistence",
                "Identified recorded activity (future labels only)",
            ),
        ):
            selected = subset.copy()
            if column == "observed_action":
                selected.loc[~selected.identified, column] = -1
            matrix = (
                selected.pivot(index="subject", columns="time", values=column)
                .reindex(index=subjects, columns=times)
                .fillna(-1)
            )
            ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
            ax.set(yticks=range(len(subjects)), yticklabels=subjects, title=title)
        axes[-1].set(
            xticks=range(0, len(times), 2),
            xticklabels=[f"{t / 60:.0f}" for t in times[::2]],
            xlabel="Measured game minute",
        )
        show()
        display(pd.DataFrame([{"color_index": k, "activity": v} for k, v in names.items()]))
        fig, ax = plt.subplots(figsize=(13, 1.8))
        ax.imshow(np.arange(-1, 15)[None, :], cmap=cmap, norm=norm, aspect="auto")
        ax.set(yticks=[], xticks=range(16), xticklabels=[names[i] for i in range(-1, 15)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        show()
    elif section == "coverage":
        data = summary()
        display(data.round(3))
        data.pivot(index="scope", columns="model", values="served_coverage").plot.bar(
            ylim=(0, 1.05),
            figsize=(10, 4),
            ylabel="Delivered fraction",
            title="Coverage after coordination",
        )
        show()
        coordination = pd.crosstab(
            rows.loc[rows.scope.ne("TEAM"), "scope"],
            rows.loc[rows.scope.ne("TEAM"), "coordination"],
        )
        coordination.plot.bar(
            stacked=True,
            figsize=(10, 4),
            ylabel="Measured role decisions",
            title="Symbolic coordination; independent is not proof of compatibility",
        )
        show()
    elif section == "agreement":
        display(summary()[["model", "scope", "identified_n", "identified_F1"]].round(3))
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, scope in zip(axes.flat, POLICY_SCOPES):
            selected = rows[rows.scope.eq(scope) & rows.identified]
            matrix = np.zeros((15, 16), dtype=int)
            for r in selected.itertuples():
                matrix[r.observed_action, r.served_action if r.served_action >= 0 else 15] += 1
            image = ax.imshow(matrix, cmap="Blues", aspect="auto")
            ax.set(
                title=f"{scope}: {len(selected)} identified rows",
                xlabel="Delivered action (15 = abstain)",
                ylabel="Recorded action",
                xticks=[0, 5, 10, 15],
                yticks=[0, 5, 10, 14],
            )
            fig.colorbar(image, ax=ax, shrink=0.7)
        show()
    elif section == "decisions":
        table = rows.copy()
        for column in ("raw_action", "macro_action", "served_action", "observed_action"):
            table[column] = table[column].map(names)
        table["minute"] = table.time / 60
        display(
            table[
                [
                    "minute",
                    "team",
                    "scope",
                    "champion",
                    "raw_action",
                    "macro_action",
                    "served_action",
                    "observed_action",
                    "identified",
                    "coordination",
                    "instruction",
                    "check",
                ]
            ].head(48)
        )
        display(
            Markdown(
                "Load all decisions with `decisions = pd.read_json(REVIEW / 'candidate-decisions.json')`; filter `actor`, `scope`, `team` and `time` to inspect any player."
            )
        )
    elif section == "conclusion":
        parity_path = directory / "evaluation-parity.json"
        if parity_path.exists():
            display(
                Markdown(
                    "**Independent pipeline check:** raw one-game replay agrees exactly with saved batch evaluation predictions."
                )
            )
            display(pd.DataFrame(read_json(parity_path)).T)
        data = summary()
        if len(report["models"]) > 1:
            changes = data.pivot(index="scope", columns="model", values="identified_F1")
            changes["candidate_minus_previous"] = changes.candidate - changes.previous
            display(changes.round(3))
        display(Markdown("\n".join("- " + line for line in report["limitations"])))
        display(
            Markdown(
                "**Next evidence needed:** broader chronological/player-disjoint tests, decision review with only causal information, live-input acceptance and a predefined prospective outcome comparison. A successful replay does not certify winning strategy."
            )
        )
    else:
        raise ValueError("Unknown notebook section")
