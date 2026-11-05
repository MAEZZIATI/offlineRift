"""Atomic JSON persistence, versioned run IDs and streamed artifact digests."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, data: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text())


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle_manifest(directory, filenames):
    directory = Path(directory)
    atomic_json(
        directory / "bundle.json",
        {"version": 1, "sha256": {name: file_digest(directory / name) for name in filenames}},
    )


def verify_bundle_manifest(directory, *, required=False):
    directory = Path(directory)
    path = directory / "bundle.json"
    if not path.exists() and not required:
        return
    manifest = read_json(path)
    if manifest.get("version") != 1 or not manifest.get("sha256"):
        raise ValueError("Invalid model bundle manifest")
    expected = {"metadata.json", "macro_predictor.pt"}
    if not expected <= manifest["sha256"].keys():
        raise ValueError("Model bundle manifest is incomplete")
    for name, digest in manifest["sha256"].items():
        if Path(name).name != name or file_digest(directory / name) != digest:
            raise ValueError(f"Model bundle integrity check failed: {name}")
