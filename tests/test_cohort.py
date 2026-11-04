"""Verified account sampling, configuration isolation and resumable pilot collection."""

import json
from unittest.mock import Mock

import pytest

from offline_rift.cohort import collect_cohort, prepare_cohort, prepare_ranked_cohort, select_cohort
from offline_rift.workflow import ResearchWorkflow, atomic_json


def test_cohort_uses_payload_identity_and_uniform_role_strata(
    tmp_path, match_payloads, monkeypatch
):
    match, timeline = match_payloads("KR_1")
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    for i, player in enumerate(match["info"]["participants"]):
        player.update(puuid=f"verified-{i}", teamPosition=roles[i % 5])
    match["info"].update(queueId=420, gameVersion="16.19.1")
    payload = {"match": match, "timeline": timeline}
    raw = tmp_path / "raw"
    atomic_json(raw / "KR_1.json", payload)
    workflow = ResearchWorkflow.initialize(
        tmp_path / "workflow.json",
        name="original",
        raw_dir="raw",
        puuid="verified-0",
        region="asia",
    )
    original = workflow.config_path.read_bytes()
    selected, sizes = select_cohort([(None, payload)], {"verified-0"}, per_role=1)
    assert len(selected) == 5
    assert sizes == {"TOP": 1, "JUNGLE": 2, "MIDDLE": 2, "BOTTOM": 2, "UTILITY": 2}
    assert "verified-0" not in {p["puuid"] for p in selected}
    assert all(p["region"] == "asia" and p["verified_in_match"] == "KR_1" for p in selected)
    assert select_cohort([(None, payload)], {"verified-0"}, per_role=1) == (selected, sizes)
    pilot = tmp_path / "pilot"
    prepare_cohort(workflow, pilot, per_role=1, count=1)
    assert workflow.config_path.read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare_cohort(workflow, pilot)
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    assert collect_cohort(pilot / "workflow.json")["status"] == "blocked_missing_credentials"
    # Exercise real workflow cache/manifest behavior through a mocked transport.
    api = Mock()
    api.fetch_match_ids.return_value = ["KR_1"]
    monkeypatch.setattr("offline_rift.workflow.RiotDataIngestor", Mock(return_value=api))
    monkeypatch.setenv("RIOT_API_KEY", "test-key-not-real")
    completed = collect_cohort(pilot / "workflow.json")
    assert completed["status"] == "complete"
    assert completed["new_unique_downloads"] == 0
    assert len(completed["batches"]) == 5
    api.fetch_match.assert_not_called()
    assert api.fetch_match_ids.call_count == 5
    assert collect_cohort(pilot / "workflow.json")["batches"] == completed["batches"]
    assert api.fetch_match_ids.call_count == 5  # Completed players are not fetched again.
    saved = json.loads((pilot / "cohort.json").read_text())
    assert saved["seed_match_ids"] == ["KR_1"]


def test_ranked_sampling_uses_frozen_rosters_not_appearance_counts(tmp_path):
    roster_dir = tmp_path / "rosters"
    for tier in ("master", "grandmaster", "challenger"):
        atomic_json(roster_dir / f"{tier}.json", [{"puuid": f"{tier}-{i}"} for i in range(5)])
    workflow_path = tmp_path / "workflow.json"
    atomic_json(
        workflow_path,
        {
            "version": 1,
            "name": "original",
            "player": {"puuid": "master-0", "region": "asia"},
            "paths": {"raw": "raw", "catalogs": "catalogs"},
            "collection": {"count": 2, "queue": 420},
        },
    )
    (tmp_path / "raw").mkdir()
    workflow = ResearchWorkflow(workflow_path)
    before = workflow_path.read_bytes()
    first = prepare_ranked_cohort(workflow, tmp_path / "one", roster_dir, per_tier=2)
    second = prepare_ranked_cohort(workflow, tmp_path / "two", roster_dir, per_tier=2)
    assert first["players"] == second["players"]
    assert len(first["players"]) == 6
    assert "master-0" not in {p["puuid"] for p in first["players"]}
    assert first["eligible_accounts_by_current_tier"]["MASTER"] == 4
    assert all(p["region"] == "asia" for p in first["players"])
    assert workflow_path.read_bytes() == before
