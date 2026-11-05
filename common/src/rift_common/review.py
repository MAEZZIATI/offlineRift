"""Review artifact contracts and acceptance checks, without research dependencies."""

import json
from pathlib import Path

from rift_common.artifacts import file_digest

RUBRIC_VERSION = "formation-review-v1"


def _label(value):
    if value == "unknown":
        return None
    if isinstance(value, bool) or str(value) not in {str(i) for i in range(12)}:
        raise ValueError("Review label must be 0..11 or 'unknown'; blank is unfinished")
    return int(value)


def load_review(path, manifest):
    document = json.loads(Path(path).read_text())
    if not str(document.get("reviewer_id", "")).strip():
        raise ValueError("Reviewer ID is required")
    if document.get("reviewer_kind") not in {"human", "ai"}:
        raise ValueError("Explicit human/ai reviewer kind is required")
    if document.get("independent_review") is not True:
        raise ValueError("Independent/blinded review attestation is required")
    if (
        document.get("packet_sha256") != manifest["packet_sha256"]
        or document.get("rubric_version") != RUBRIC_VERSION
    ):
        raise ValueError("Review packet hash or rubric version mismatch")
    answers = document.get("answers", [])
    ids = [row["sample_id"] for row in answers]
    if len(set(ids)) != len(ids) or set(ids) != set(manifest["sample_ids"]):
        raise ValueError("Review must contain each audit sample exactly once")
    for row in answers:
        _label(row.get("label"))
        for key in ("evidence_sufficient", "intent_observable"):
            if row.get(key) not in {"yes", "no", "uncertain"}:
                raise ValueError(f"Every sample requires {key}")
    return document, {row["sample_id"]: row for row in answers}


def require_promotion_review(dataset, path):
    """A completed fit is insufficient: corrected workflow publication needs review."""
    from rift_common.pipeline.features import objective_rule_digest, objective_rule_spec

    if dataset.get("objective_rules_version", "legacy-v1") == "legacy-v1":
        raise ValueError("Migrate legacy objective rules before reviewed promotion")
    if objective_rule_spec(dataset.get("objective_rules_version"))["lifecycle"] != "active":
        raise ValueError("Superseded objective rules cannot be promoted")

    if path is None:
        raise ValueError(
            "Label review is unfinished: supply label_review.report before training for promotion"
        )
    path = Path(path)
    report = json.loads(path.read_text())
    if report.get("status") != "complete" or report.get("decision") != "accepted_for_retraining":
        raise ValueError(
            "Label review has not accepted retraining; finish review/adjudication first"
        )
    if report.get("dataset_fingerprint") != dataset["fingerprint"]:
        raise ValueError("Label review belongs to a different immutable dataset")
    if report.get("objective_rules_version") != dataset.get(
        "objective_rules_version"
    ) or report.get("objective_rules_sha256") != objective_rule_digest(
        dataset.get("objective_rules_version")
    ):
        raise ValueError("Label review objective rules are stale")
    labeler = Path(__file__).parent / "pipeline" / "action_labeler.py"
    if report.get("labeler_sha256") != file_digest(labeler):
        raise ValueError("Label review labeler contract is stale")
    if len(report.get("reviewers", [])) != 2 or not report.get("input_sha256"):
        raise ValueError("Label review has no verifiable reviewer inputs")
    for name, digest in report["input_sha256"].items():
        if Path(name).name != name or file_digest(path.parent / name) != digest:
            raise ValueError("Label review inputs changed after completion")
    # Re-read exports as well as their hashes; incomplete/duplicate reviewers cannot pass.
    manifest = json.loads((path.parent / "review_manifest.json").read_text())
    if (
        manifest["dataset_fingerprint"] != dataset["fingerprint"]
        or manifest["packet_sha256"] != report["packet_sha256"]
    ):
        raise ValueError("Review manifest does not bind the target dataset and packets")
    if manifest.get("dataset_sha256") != report.get("dataset_sha256") or file_digest(
        Path(dataset["path"])
    ) != manifest.get("dataset_sha256"):
        raise ValueError("Reviewed dataset bytes changed")
    if not manifest.get("dataset_patches") or set(manifest.get("sampled_patches", [])) != set(
        manifest["dataset_patches"]
    ):
        raise ValueError("Review must cover all target dataset patches")
    if report.get("unresolved_samples") or report.get("scorable_samples", 0) < 1:
        raise ValueError("Review has unresolved or wholly unscorable judgments")
    decision_path = path.parent / "adjudication.json"
    if "adjudication.json" not in report["input_sha256"]:
        raise ValueError("An explicit adjudication/acceptance decision is required")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("decision") != "accepted_for_retraining"
        or not decision.get("rationale", "").strip()
        or decision.get("packet_sha256") != manifest["packet_sha256"]
        or decision.get("adjudicator_kind") not in {"human", "ai"}
    ):
        raise ValueError("Adjudicator has not accepted this packet set")
    reviews = [load_review(path.parent / f"reviewer-{i}.json", manifest)[0] for i in (1, 2)]
    if reviews[0]["reviewer_id"] == reviews[1]["reviewer_id"]:
        raise ValueError("Label review reviewers are not distinct")
    return {
        "report": str(path.resolve()),
        "sha256": file_digest(path),
        "reviewers": report["reviewers"],
    }
