"""Offline foundation runner; validates plumbing without fitting a model."""

import argparse
import json
import platform
import random
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from pydantic import Field

from offlinerift.domain.contracts import ArtifactStore, Contract
from offlinerift.storage.local import LocalArtifactStore
from offlinerift.training.config import RunConfig


class Checkpoint(Contract):
    """Smoke state: seed and cursor reconstruct the deterministic stream."""

    resume_hash: str
    completed_steps: int = Field(ge=0)
    samples: tuple[int, ...]


def hardware() -> dict[str, str]:
    """Record hardware at run start, with explicit unavailable values."""
    gpu = "unavailable"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            gpu = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    ram = "unavailable"
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        ram = meminfo.read_text().splitlines()[0]
    return {"python": sys.version, "platform": platform.platform(), "gpu": gpu, "ram": ram}


def execute(
    config: RunConfig,
    output: Path,
    resume: Path | None = None,
    stop_after: int | None = None,
    *,
    artifacts: ArtifactStore | None = None,
) -> Checkpoint:
    """Exercise checkpoint/resume on CPU; never train or contact a service."""
    store: ArtifactStore = artifacts if artifacts is not None else LocalArtifactStore(output)
    checkpoint = Checkpoint(resume_hash=config.resume_digest(), completed_steps=0, samples=())
    if resume is not None:
        checkpoint = Checkpoint.model_validate_json(resume.read_text())
    if checkpoint.resume_hash != config.resume_digest():
        raise ValueError("Resume config hash mismatch")
    if checkpoint.completed_steps != len(checkpoint.samples):
        raise ValueError("Invalid checkpoint cursor")
    if checkpoint.completed_steps > config.steps:
        raise ValueError("Checkpoint exceeds configured steps")
    if (output / "checkpoint.json").exists() and resume is None:
        raise ValueError("Existing run requires explicit --resume")
    lineage = {
        "schema_version": "1",
        "config_hash": config.digest(),
        "config": config.model_dump(mode="json"),
        "hardware": hardware(),
        "libraries": {name: version(name) for name in ("pydantic", "offlinerift-training")},
        "resume_from": str(resume) if resume else None,
        "purpose": "foundation-smoke-only",
    }
    store.write_json("lineage.json", json.dumps(lineage, sort_keys=True, indent=2))
    rng = random.Random(config.seed)
    generated = tuple(rng.randrange(1_000_000) for _ in range(checkpoint.completed_steps))
    if generated != checkpoint.samples:
        raise ValueError("Checkpoint stream does not match seed/cursor")
    samples = list(checkpoint.samples)
    end = min(config.steps, stop_after) if stop_after is not None else config.steps
    if end < checkpoint.completed_steps:
        raise ValueError("Stop cursor precedes checkpoint")
    for step in range(checkpoint.completed_steps + 1, end + 1):
        samples.append(rng.randrange(1_000_000))
        checkpoint = Checkpoint(
            resume_hash=config.resume_digest(), completed_steps=step, samples=tuple(samples)
        )
        if step % config.checkpoint_every == 0:
            store.write_json("checkpoint.json", checkpoint.model_dump_json())
    store.write_json("checkpoint.json", checkpoint.model_dump_json())
    store.write_json("metrics.json", json.dumps({"completed_steps": end, "model_trained": False}))
    store.write_json("events.jsonl", json.dumps({"event": "smoke_complete", "step": end}) + "\n")
    return checkpoint


def main() -> None:
    """Identical entry point locally and inside the offline script kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    config = RunConfig.model_validate_json(args.config.read_text())
    execute(config, args.output, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
