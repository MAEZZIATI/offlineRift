"""Independent review completeness, provenance, disagreements and promotion guards."""

import copy
import json

import pytest
from offline_rift.workflow import atomic_json
from rift_common.pipeline.features import objective_rule_digest, objective_rule_spec
from rift_common.review import (
    RUBRIC_VERSION,
    file_digest,
    require_promotion_review,
)
from rift_research.review import summarize_reviews


def review_fixture(tmp_path, *, kind="human"):
    audit = tmp_path / "audit"
    atomic_json(audit / "review_packets.json", [{"sample_id": str(i)} for i in range(4)])
    atomic_json(
        audit / "diagnostics.json",
        [
            {"sample_id": str(i), "match_id": f"match-{i // 2}", "weak_label": label}
            for i, label in enumerate((0, 1, 1, 0))
        ],
    )
    from pathlib import Path

    from rift_common.pipeline import action_labeler

    (tmp_path / "dataset.bin").write_bytes(b"immutable test dataset")
    manifest = {
        "sample_ids": [str(i) for i in range(4)],
        "packet_sha256": file_digest(audit / "review_packets.json"),
        "diagnostics_sha256": file_digest(audit / "diagnostics.json"),
        "dataset_fingerprint": "dataset-fingerprint",
        "dataset_sha256": file_digest(tmp_path / "dataset.bin"),
        "sampled_patches": ["16.21"],
        "dataset_patches": ["16.21"],
        "objective_rules_version": objective_rule_spec()["version"],
        "objective_rules_sha256": objective_rule_digest(),
        "labeler_sha256": file_digest(Path(action_labeler.__file__)),
    }
    atomic_json(audit / "review_manifest.json", manifest)
    reviews = []
    for n in (1, 2):
        labels = (0, 1, 1, 1) if n == 1 else (0, 1, 0, 1)
        document = {
            "reviewer_id": f"reviewer-{n}",
            "reviewer_kind": kind,
            "independent_review": True,
            "packet_sha256": manifest["packet_sha256"],
            "rubric_version": RUBRIC_VERSION,
            "answers": [
                {
                    "sample_id": str(i),
                    "label": value,
                    "evidence_sufficient": "yes",
                    "intent_observable": "no",
                }
                for i, value in enumerate(labels)
            ],
        }
        path = tmp_path / f"review-{n}.json"
        atomic_json(path, document)
        reviews.append(path)
    decision = tmp_path / "adjudication.json"
    atomic_json(
        decision,
        {
            "packet_sha256": manifest["packet_sha256"],
            "adjudicator_id": "third-reviewer",
            "adjudicator_kind": "human",
            "decision": "accepted_for_retraining",
            "rationale": "Test fixture only; not a real quality judgment",
            "answers": [
                {
                    "sample_id": "2",
                    "label": 1,
                    "notes": "Fixture evidence resolves the disagreement",
                }
            ],
        },
    )
    return audit, manifest, reviews, decision


def test_review_requires_adjudication_and_binds_exact_dataset(tmp_path):
    audit, manifest, reviews, decision = review_fixture(tmp_path)
    pending = summarize_reviews(audit, reviews, tmp_path / "pending", samples=10)
    assert pending["status"] == "awaiting_adjudication"
    assert pending["unresolved_samples"] == ["2"]
    assert pending["raw_agreement"] == 0.75
    assert pending["cohens_kappa"] == 0.5
    report = summarize_reviews(
        audit, reviews, tmp_path / "complete", adjudication=decision, samples=10
    )
    assert report["status"] == "complete"
    assert report["human_reviewers"] == 2 and report["ai_reviewers"] == 0
    assert report["weak_label_metrics"]["heuristic_agreement"] == 0.75
    dataset = {
        "path": str(tmp_path / "dataset.bin"),
        "fingerprint": manifest["dataset_fingerprint"],
        "objective_rules_version": manifest["objective_rules_version"],
    }
    assert (
        require_promotion_review(dataset, tmp_path / "complete/report.json")["reviewers"]
        == report["reviewers"]
    )
    with pytest.raises(ValueError, match="different immutable dataset"):
        require_promotion_review(
            {**dataset, "fingerprint": "different"}, tmp_path / "complete/report.json"
        )
    with pytest.raises(FileExistsError):
        summarize_reviews(audit, reviews, tmp_path / "complete", adjudication=decision, samples=10)
    (tmp_path / "dataset.bin").write_bytes(b"changed dataset")
    with pytest.raises(ValueError, match="dataset bytes changed"):
        require_promotion_review(dataset, tmp_path / "complete/report.json")
    (tmp_path / "dataset.bin").write_bytes(b"immutable test dataset")
    forged = json.loads((tmp_path / "pending/report.json").read_text())
    forged.update(status="complete", decision="accepted_for_retraining", unresolved_samples=[])
    atomic_json(tmp_path / "pending/report.json", forged)
    with pytest.raises(ValueError, match="explicit adjudication"):
        require_promotion_review(dataset, tmp_path / "pending/report.json")
    (tmp_path / "complete/reviewer-1.json").write_text("{}")
    with pytest.raises(ValueError, match="inputs changed"):
        require_promotion_review(dataset, tmp_path / "complete/report.json")


@pytest.mark.parametrize(
    "problem", ["same_reviewer", "missing_sample", "wrong_packet", "no_attestation"]
)
def test_incomplete_or_nonindependent_reviews_rejected(tmp_path, problem):
    audit, _, reviews, _ = review_fixture(tmp_path)
    second = json.loads(reviews[1].read_text())
    if problem == "same_reviewer":
        second["reviewer_id"] = "reviewer-1"
    elif problem == "missing_sample":
        second["answers"].pop()
    elif problem == "wrong_packet":
        second["packet_sha256"] = "different"
    else:
        second["independent_review"] = False
    atomic_json(reviews[1], second)
    with pytest.raises(ValueError):
        summarize_reviews(audit, reviews, tmp_path / "output", samples=10)


def test_ai_reviews_never_claim_human_validation_and_unknowns_are_not_correct(tmp_path):
    audit, _, reviews, _ = review_fixture(tmp_path, kind="ai")
    first = json.loads(reviews[0].read_text())
    first["answers"][0]["label"] = "unknown"
    first["answers"][0]["evidence_sufficient"] = "no"
    second = copy.deepcopy(first)
    second["reviewer_id"] = "reviewer-2"
    atomic_json(reviews[0], first)
    atomic_json(reviews[1], second)
    report = summarize_reviews(audit, reviews, tmp_path / "ai", samples=10)
    assert report["human_reviewers"] == 0 and report["ai_reviewers"] == 2
    assert report["unknown_samples"] == ["0"]
    assert report["scorable_samples"] == 3
    assert report["decision"] == "hold"
