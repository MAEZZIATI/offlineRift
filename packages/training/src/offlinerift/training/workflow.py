"""Generate a script kernel and import downloaded outputs without credentials."""

import argparse
import hashlib
import json
from pathlib import Path

from offlinerift.storage.local import LocalArtifactStore
from offlinerift.training.config import RunConfig
from offlinerift.training.run import Checkpoint


def prepare(config: RunConfig, destination: Path, resume: str | None = None) -> None:
    """Build an allowlisted kernel folder; never copy repository or raw data."""
    store = LocalArtifactStore(destination)
    metadata = {
        "id": config.kernel_id,
        "title": config.run_id,
        "code_file": "entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": config.enable_gpu,
        "enable_internet": False,
        "dataset_sources": config.dataset_sources,
        "kernel_sources": config.kernel_sources,
    }
    # Config is embedded because Kaggle pushes the designated script, not a source tree.
    script = f'''"""Generated offline bootstrap; no training logic."""
import json
from pathlib import Path
import subprocess
import sys

wheelhouse = Path({config.wheelhouse!r})
wheels = sorted(str(path) for path in wheelhouse.glob("*.whl"))
if not wheels:
    raise RuntimeError("Attach the private runtime wheelhouse before running")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
     str(wheelhouse), *wheels], check=True
)
config = Path("/kaggle/working/run-config.json")
config.write_text({config.model_dump_json()!r})
command = [sys.executable, "-m", "offlinerift.training.run", "--config", str(config),
           "--output", "/kaggle/working/artifacts"]
resume = {resume!r}
if resume is not None:
    command.extend(["--resume", resume])
subprocess.run(command, check=True)
'''
    store.write_json("kernel-metadata.json", json.dumps(metadata, indent=2))
    store.write_json("entry.py", script)


def sync_outputs(source: Path, tracking: Path, kernel_version: str) -> Path:
    """Import only validated run artifacts, idempotently, with actual kernel version."""
    lineage = json.loads((source / "lineage.json").read_text())
    config = RunConfig.model_validate_json(json.dumps(lineage["config"]))
    checkpoint = Checkpoint.model_validate_json((source / "checkpoint.json").read_text())
    if (
        checkpoint.resume_hash != config.resume_digest()
        or lineage["config_hash"] != config.digest()
    ):
        raise ValueError("Output lineage/config mismatch")
    if not kernel_version.strip():
        raise ValueError("Actual Kaggle kernel version is required")
    target = tracking / config.run_id / kernel_version
    # Only identifiers, not paths, are accepted for version directories.
    if not kernel_version.isdecimal():
        raise ValueError("Kernel version must be the numeric version from Kaggle")
    payloads = {
        name: (source / name).read_text()
        for name in ("lineage.json", "checkpoint.json", "metrics.json", "events.jsonl")
    }
    receipt = {
        "schema_version": "1",
        "kernel_id": config.kernel_id,
        "kernel_version": kernel_version,
        "config_hash": config.digest(),
        "sha256": {
            name: hashlib.sha256(data.encode()).hexdigest() for name, data in payloads.items()
        },
    }
    payloads["receipt.json"] = json.dumps(receipt, sort_keys=True, indent=2)
    for name, data in payloads.items():
        existing = target / name
        if existing.exists() and existing.read_text() != data:
            raise ValueError("Refusing to overwrite different tracked outputs")
    store = LocalArtifactStore(target)
    for name, data in payloads.items():
        store.write_json(name, data)
    return target


def main() -> None:
    """Offline preparation/import commands; Kaggle itself is Makefile-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--destination", type=Path, required=True)
    prepare_parser.add_argument("--resume")
    sync_parser = commands.add_parser("sync")
    sync_parser.add_argument("--source", type=Path, required=True)
    sync_parser.add_argument("--tracking", type=Path, required=True)
    sync_parser.add_argument("--kernel-version", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            RunConfig.model_validate_json(args.config.read_text()), args.destination, args.resume
        )
    else:
        sync_outputs(args.source, args.tracking, args.kernel_version)


if __name__ == "__main__":
    main()
