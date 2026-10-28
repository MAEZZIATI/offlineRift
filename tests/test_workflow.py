from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from action_labeler import load_dataset
from data_pipeline import TimelineProcessor
from features import FeatureConfig
from workflow import CatalogCache, ResearchWorkflow, atomic_json


def prepare(tmp_path, match_payloads):
    raw = tmp_path / "data" / "raw"
    for i in range(3):
        match, timeline = match_payloads(f"KR_{i}")
        match["info"].update(queueId=420, gameVersion="16.19.123.456")
        for p in match["info"]["participants"]:
            p["puuid"] = "common-player" if p["participantId"] == 1 else f"{i}-{p['participantId']}"
            p.update(riotIdGameName="Test Player", riotIdTagline="KR1")
        atomic_json(raw / f"KR_{i}.json", {"match": match, "timeline": timeline})
    workflow = ResearchWorkflow.initialize(tmp_path / "workflow.json", name="research")
    atomic_json(
        workflow.path("catalogs") / "items-16.19.json",
        {"version": "16.19.1", "data": {"1001": {"gold": {"total": 3000}}}},
    )
    return workflow


def test_adopt_existing_data_and_rebuild_only_on_changes(tmp_path, match_payloads):
    workflow = prepare(tmp_path, match_payloads)
    assert workflow.config["player"]["region"] == "asia"
    assert workflow.config["player"]["puuid"] == "common-player"
    assert workflow.status()["cached_matches"] == 3
    with CatalogMock(workflow) as cache:
        first = workflow.build(cache=cache)
        second = workflow.build(cache=cache)
        assert first == second
        assert first["matches"] == 3
        assert first["catalog_versions"] == {"16.19": "16.19.1"}
        assert not cache.session.get.called
        workflow.config["features"]["item_spike_gold"] = 4000
        changed = workflow.build(cache=cache)
    assert changed["fingerprint"] != first["fingerprint"]
    assert first["path"] != changed["path"]
    arrays, metadata = load_dataset(first["path"])
    assert len(set(arrays["match_ids"])) == 3
    assert metadata["provenance"]["catalog_versions"] == {"16.19": "16.19.1"}


class CatalogMock(CatalogCache):
    def __init__(self, workflow):
        super().__init__(workflow.path("catalogs"), session=Mock())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_catalog_auto_download_selects_matching_patch_and_caches(tmp_path):
    session = Mock()
    versions = Mock()
    versions.json.return_value = ["16.21.1", "16.19.2", "16.19.1"]
    items = Mock()
    items.json.return_value = {"version": "16.19.2", "data": {"1001": {"gold": {"total": 300}}}}
    session.get.side_effect = [versions, items]
    cache = CatalogCache(tmp_path, session=session)
    assert cache.get("16.19")[1] == "16.19.2"
    assert cache.get("16.19")[0].value(1001) == 300
    assert session.get.call_count == 2
    fresh = CatalogCache(tmp_path, session=Mock())
    assert fresh.get("16.19")[1] == "16.19.2"
    fresh.session.get.assert_not_called()


def test_collection_paginates_and_resumes_cached_downloads(tmp_path, match_payloads):
    workflow = prepare(tmp_path, match_payloads)
    api = Mock()
    # First attempt caches one new match, then fails. Next attempt reuses that file.
    match, timeline = match_payloads("KR_new")
    api.fetch_match_ids.return_value = ["KR_0", "KR_new", "KR_fail"]
    api.fetch_match.side_effect = [match, RuntimeError("connection interrupted")]
    api.fetch_timeline.return_value = timeline
    with pytest.raises(RuntimeError, match="interrupted"):
        workflow.collect(api=api)
    assert (workflow.path("raw") / "KR_new.json").is_file()
    failed = [
        json.loads(p.read_text()) for p in (workflow.path("artifacts") / "batches").glob("*.json")
    ]
    assert failed[0]["status"] == "failed"
    api.reset_mock(side_effect=True)
    api.fetch_match_ids.side_effect = [["KR_0", "KR_new"], []]
    # Request 101 so pagination has two pages; page 1 must be full to reach page 2.
    ids = ["KR_0", "KR_new"] + [f"KR_extra_{i}" for i in range(98)]
    api.fetch_match_ids.side_effect = [ids, []]
    api.fetch_match.side_effect = lambda mid: match_payloads(mid)[0]
    api.fetch_timeline.side_effect = lambda mid: match_payloads(mid)[1]
    manifest = workflow.collect(count=101, start=20, api=api)
    assert manifest["status"] == "complete"
    assert manifest["reused"] == ["KR_0", "KR_new"]
    assert len(manifest["downloaded"]) == 98
    assert [c.kwargs["start"] for c in api.fetch_match_ids.call_args_list] == [20, 120]
    assert [c.kwargs["count"] for c in api.fetch_match_ids.call_args_list] == [100, 1]


def test_real_monster_event_names(match_payloads):
    match, timeline = match_payloads()
    timeline["info"]["frames"][1]["events"] += [
        {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": 40000,
            "monsterType": "BARON_NASHOR",
            "killerId": 1,
        },
        {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": 45000,
            "monsterType": "RIFTHERALD",
            "killerId": 1,
        },
    ]
    snapshots = TimelineProcessor(FeatureConfig()).parse(match, timeline)
    state = next(s for s in snapshots if s.time == 50)
    assert state.objectives["baron"]["eta"] == 350
    assert state.objectives["baron_buff"]["team"] == 100
    assert not state.objectives["herald"]["enabled"]


def test_training_is_versioned_and_unchanged_runs_are_reused(tmp_path, match_payloads):
    pytest.importorskip("d3rlpy")
    workflow = prepare(tmp_path, match_payloads)
    workflow.config["training"].update(n_steps=3, n_steps_per_epoch=2, batch_size=4)
    dataset = workflow.build()
    first = workflow.train(dataset)
    assert first["status"] == "complete"
    assert workflow.train(dataset) == first
    assert first["report"]["optimizer_steps"] == 3  # No rounding down incomplete epochs.
    second = workflow.train(dataset, force=True)
    assert second["path"] != first["path"]
