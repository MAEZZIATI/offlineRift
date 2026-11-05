"""Optional annotation ingestion, agreement diagnostics and adjudication."""

import json
from pathlib import Path

import numpy as np
from rift_common.models.evaluation import classification_metrics
from rift_common.review import _label, file_digest, load_review

from rift_research.evaluation import match_bootstrap


def summarize_reviews(audit_dir, review_paths, output, *, adjudication=None, samples=1000):
    from rift_common.artifacts import atomic_json

    audit_dir = Path(audit_dir)
    manifest_path = audit_dir / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if file_digest(audit_dir / "review_packets.json") != manifest["packet_sha256"]:
        raise ValueError("Audit packets changed after manifest creation")
    if file_digest(audit_dir / "diagnostics.json") != manifest["diagnostics_sha256"]:
        raise ValueError("Unblinded audit key changed after manifest creation")
    if len(review_paths) != 2:
        raise ValueError("Exactly two independent review exports are required")
    documents, indexed = zip(*(load_review(path, manifest) for path in review_paths))
    if documents[0]["reviewer_id"] == documents[1]["reviewer_id"]:
        raise ValueError("Two distinct reviewer IDs are required")
    diagnostics = {
        row["sample_id"]: row for row in json.loads((audit_dir / "diagnostics.json").read_text())
    }
    decisions = {}
    adjudicator = None
    if adjudication is not None:
        adjudicator = json.loads(Path(adjudication).read_text())
        if adjudicator.get("packet_sha256") != manifest["packet_sha256"] or not adjudicator.get(
            "adjudicator_id"
        ):
            raise ValueError("Adjudication requires the same packet hash and an adjudicator ID")
        if adjudicator.get("adjudicator_kind") not in {"human", "ai"}:
            raise ValueError("Adjudicator kind must explicitly be human or ai")
        rows = adjudicator.get("answers", [])
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate adjudication samples")
        decisions = {row["sample_id"]: row for row in rows}
        if set(decisions) - set(manifest["sample_ids"]):
            raise ValueError("Adjudication contains unknown samples")
        for row in decisions.values():
            _label(row.get("label"))
            if not row.get("notes", "").strip():
                raise ValueError("Adjudication requires a rationale for each explicit decision")
    conflicts, unknown, truth, weak, matches, comparable = [], [], [], [], [], []
    for sample in manifest["sample_ids"]:
        left, right = indexed[0][sample], indexed[1][sample]
        a, b = _label(left["label"]), _label(right["label"])
        scorable = (
            a is not None
            and b is not None
            and left["evidence_sufficient"] == right["evidence_sufficient"] == "yes"
        )
        if scorable:
            comparable.append((a, b))
        if sample in decisions:
            target = _label(decisions[sample]["label"])
        elif scorable and a == b:
            target = a
        elif a is None and b is None:
            target = None
        else:
            conflicts.append(sample)
            continue
        if target is None:
            unknown.append(sample)
        else:
            truth.append(target)
            weak.append(diagnostics[sample]["weak_label"])
            matches.append(diagnostics[sample]["match_id"])
    agreement = kappa = None
    if comparable:
        a, b = np.asarray(comparable).T
        agreement = float(np.mean(a == b))
        chance = float(
            (np.bincount(a, minlength=12) * np.bincount(b, minlength=12)).sum() / len(a) ** 2
        )
        kappa = (agreement - chance) / (1 - chance) if chance < 1 else None
    decision = adjudicator.get("decision", "hold") if adjudicator else "hold"
    if decision not in {"hold", "revise_labels", "accepted_for_retraining"}:
        raise ValueError("Unknown review decision")
    if decision == "accepted_for_retraining" and (
        conflicts or not truth or not adjudicator.get("rationale", "").strip()
    ):
        raise ValueError(
            "Acceptance requires resolved disagreements, scorable judgments and a rationale"
        )
    report = {
        "status": "awaiting_adjudication" if conflicts else "complete",
        "decision": decision,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "dataset_sha256": manifest["dataset_sha256"],
        "objective_rules_version": manifest["objective_rules_version"],
        "objective_rules_sha256": manifest["objective_rules_sha256"],
        "labeler_sha256": manifest["labeler_sha256"],
        "packet_sha256": manifest["packet_sha256"],
        "reviewed_samples": len(manifest["sample_ids"]),
        "reviewers": [
            {"id": d["reviewer_id"], "kind": d["reviewer_kind"], "independence_attested": True}
            for d in documents
        ],
        "human_reviewers": sum(d["reviewer_kind"] == "human" for d in documents),
        "ai_reviewers": sum(d["reviewer_kind"] == "ai" for d in documents),
        "adjudicator": {
            "id": adjudicator["adjudicator_id"],
            "kind": adjudicator["adjudicator_kind"],
        }
        if adjudicator
        else None,
        "comparable_reviewer_samples": len(comparable),
        "raw_agreement": agreement,
        "cohens_kappa": kappa,
        "unresolved_samples": conflicts,
        "unknown_samples": unknown,
        "scorable_samples": len(truth),
        "limitations": [
            "reviewer identity/independence are attestations, not authenticated facts",
            "AI judgments are not independent human annotations",
            "stratified audit metrics are not population accuracy or strategic utility",
            "unknown judgments are excluded from precision, never counted as correct",
        ],
    }
    if truth:
        truth, weak = np.asarray(truth), np.asarray(weak)
        report["weak_label_metrics"] = classification_metrics(truth, weak, 12)
        report["bootstrap"] = match_bootstrap(
            truth, {"weak_labels": weak}, np.asarray(matches), 12, samples=samples, comparisons=()
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        ("review_manifest.json", manifest_path),
        ("diagnostics.json", audit_dir / "diagnostics.json"),
        *[(f"reviewer-{i}.json", Path(path)) for i, path in enumerate(review_paths, 1)],
    ]
    if adjudication:
        sources.append(("adjudication.json", Path(adjudication)))
    report["input_sha256"] = {}
    for name, source in sources:
        target = output / name
        target.write_bytes(source.read_bytes())
        report["input_sha256"][name] = file_digest(target)
    atomic_json(output / "report.json", report)
    return report


def run_review(args):
    result = summarize_reviews(
        args.audit,
        args.review,
        args.output,
        adjudication=args.adjudication,
        samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "decision",
                    "human_reviewers",
                    "ai_reviewers",
                    "scorable_samples",
                    "raw_agreement",
                    "cohens_kappa",
                )
            },
            indent=2,
        )
    )
