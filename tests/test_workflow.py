from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from offline_rift.workflow import Workflow
from rift_common.artifacts import atomic_json


def prepare(tmp_path, match_payloads):
    config = {
        "version": 2,
        "name": "test",
        "primary_puuid": "player-1",
        "players": [{"puuid": "player-1", "region": "asia", "riot_id": "Example#TEST"}],
        "paths": {
            "raw": "raw",
            "catalogs": "catalogs",
            "artifacts": "artifacts",
            "models": "models",
        },
        "training": {"steps": 2, "minimum_support": 1},
    }
    atomic_json(tmp_path / "workflow.json", config)
    for i in range(3):
        match, timeline = match_payloads(f"KR_{i}")
        atomic_json(tmp_path / "raw" / f"KR_{i}.json", {"match": match, "timeline": timeline})
    return Workflow(tmp_path / "workflow.json")


def test_immutable_build_cache_reuse_and_changed_input(tmp_path, match_payloads, catalog):
    workflow = prepare(tmp_path, match_payloads)
    cache = SimpleNamespace(get=lambda patch: (catalog, "16.22.1", "catalog-hash"))
    first = workflow.build(cache=cache)
    assert first["matches"] == 3 and first["episodes"] == 36
    assert workflow.build(cache=cache)["path"] == first["path"]
    old = Path(first["path"]).read_bytes()
    match, timeline = match_payloads("KR_new")
    atomic_json(tmp_path / "raw/KR_new.json", {"match": match, "timeline": timeline})
    second = workflow.build(cache=cache)
    assert second["fingerprint"] != first["fingerprint"]
    assert Path(first["path"]).read_bytes() == old
    assert workflow.state()["dataset"]["path"] == second["path"]


def test_failed_training_does_not_publish_latest_model(
    tmp_path, match_payloads, catalog, monkeypatch
):
    workflow = prepare(tmp_path, match_payloads)
    data = workflow.build(cache=SimpleNamespace(get=lambda patch: (catalog, "16.22.1", "hash")))
    state = workflow.state()
    state["model"] = {"path": "prior-completed-model", "fingerprint": "prior"}
    atomic_json(workflow.state_path, state)
    monkeypatch.setattr(
        "offline_rift.training.trainer.PolicyTrainer.train",
        Mock(side_effect=RuntimeError("optimizer failed")),
    )
    with pytest.raises(RuntimeError, match="optimizer failed"):
        workflow.train(data, force=True)
    assert workflow.state()["model"] == state["model"]


def test_legacy_workflow_is_rejected_without_modification(tmp_path):
    path = tmp_path / "workflow.json"
    atomic_json(path, {"version": 1})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="version-2"):
        Workflow(path)
    assert path.read_bytes() == before


def test_reused_model_is_reloaded_and_rejects_corruption(tmp_path, match_payloads, catalog):
    workflow = prepare(tmp_path, match_payloads)
    data = workflow.build(cache=SimpleNamespace(get=lambda patch: (catalog, "16.22.1", "hash")))
    model = workflow.train(data)
    assert workflow.train(data) == model
    with (Path(model["path"]) / "policy.pt").open("ab") as stream:
        stream.write(b"corrupted")
    previous = workflow.state_path.read_bytes()
    with pytest.raises(ValueError, match="integrity"):
        workflow.train(data)
    assert workflow.state_path.read_bytes() == previous
