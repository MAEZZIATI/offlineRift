"""CPU acceptance gates for contracts, lineage, resume and offline staging."""

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from offlinerift.storage.local import LocalArtifactStore
from offlinerift.training.config import RunConfig
from offlinerift.training.run import Checkpoint, execute
from offlinerift.training.workflow import prepare, sync_outputs


@pytest.fixture
def config() -> RunConfig:
    return RunConfig.model_validate_json(Path("configs/smoke.json").read_text())


def test_resume_matches_uninterrupted(config: RunConfig, tmp_path: Path) -> None:
    full = execute(config, tmp_path / "full")
    first = execute(config, tmp_path / "partial", stop_after=4)
    assert first.completed_steps == 4
    resumed = execute(config, tmp_path / "resumed", tmp_path / "partial/checkpoint.json")
    assert full == resumed
    assert resumed.completed_steps == config.steps
    lineage = json.loads((tmp_path / "resumed/lineage.json").read_text())
    assert lineage["config_hash"] == config.digest()
    assert {"python", "gpu", "ram"} <= lineage["hardware"].keys()
    with pytest.raises(ValueError, match="explicit"):
        execute(config, tmp_path / "full")


def test_resume_rejects_changed_config_and_corrupt_cursor(
    config: RunConfig, tmp_path: Path
) -> None:
    execute(config, tmp_path / "first", stop_after=2)
    resume = tmp_path / "first/checkpoint.json"
    with pytest.raises(ValueError, match="hash"):
        execute(config.model_copy(update={"seed": 9}), tmp_path / "other", resume)
    broken = Checkpoint(resume_hash=config.resume_digest(), completed_steps=2, samples=(1,))
    resume.write_text(broken.model_dump_json())
    with pytest.raises(ValueError, match="cursor"):
        execute(config, tmp_path / "other", resume)


def test_resume_accepts_new_submission_with_previous_output(
    config: RunConfig, tmp_path: Path
) -> None:
    expected = execute(config, tmp_path / "full")
    execute(config, tmp_path / "partial", stop_after=4)
    submitted = config.model_copy(
        update={"kernel_version": "2", "kernel_sources": ("owner/previous-output",)}
    )
    assert submitted.digest() != config.digest()
    resumed = execute(submitted, tmp_path / "resumed", tmp_path / "partial/checkpoint.json")
    assert resumed == expected


def test_private_script_generation(config: RunConfig, tmp_path: Path) -> None:
    prepare(config, tmp_path, "/kaggle/input/previous/artifacts/checkpoint.json")
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["kernel_type"] == "script"
    assert metadata["dataset_sources"] == []  # No third-party data attached.
    script = (tmp_path / "entry.py").read_text()
    ast.parse(script)
    assert "--no-index" in script and "offlinerift.training.run" in script
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**config.model_dump(), "enable_internet": True})


def test_sync_is_allowlisted_idempotent_and_versioned(config: RunConfig, tmp_path: Path) -> None:
    source = tmp_path / "outputs"
    execute(config, source)
    (source / "unexpected.txt").write_text("must not be copied")
    tracking = tmp_path / "tracking"
    target = sync_outputs(source, tracking, "1")
    assert sync_outputs(source, tracking, "1") == target
    assert not (target / "unexpected.txt").exists()
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["kernel_version"] == "1" and receipt["sha256"]
    (source / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="overwrite"):
        sync_outputs(source, tracking, "1")
    with pytest.raises(ValueError, match="numeric"):
        sync_outputs(source, tracking, "../escape")


def test_store_rejects_escape_and_symlink(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.write_json("../escape", "{}")
    (tmp_path / "link.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError):
        store.write_json("link.json", "{}")
    store.write_json("ok.json", "{}")
    assert store.read_json("ok.json") == "{}"
