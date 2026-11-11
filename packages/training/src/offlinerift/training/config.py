"""Typed, credential-free configuration for the foundation smoke job."""

import hashlib
from typing import Literal

from pydantic import Field

from offlinerift.domain.contracts import Contract, DatasetRef


class ComputeBudget(Contract):
    """Explicit assumptions, never promises about the user's Kaggle account."""

    gpu_type: str = Field(min_length=1)
    vram_gb: float = Field(ge=0)
    ram_gb: float = Field(gt=0)
    session_hours: float = Field(gt=0)
    weekly_gpu_hours: float = Field(ge=0)
    output_gb: float = Field(gt=0)
    verified_on: str | None = None


class RunConfig(Contract):
    """Semantic config excludes output/resume paths so relocation is safe."""

    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset: DatasetRef
    seed: int = Field(ge=0)
    steps: int = Field(gt=0)
    checkpoint_every: int = Field(gt=0)
    kernel_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$")
    kernel_version: str = Field(min_length=1)
    enable_gpu: bool = False
    enable_internet: Literal[False] = False
    dataset_sources: tuple[str, ...] = ()
    kernel_sources: tuple[str, ...] = ()
    wheelhouse: str = "/kaggle/input/offlinerift-runtime/wheels"
    budget: ComputeBudget

    def digest(self) -> str:
        """Hash the full validated config, including lineage and assumptions."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def resume_digest(self) -> str:
        """Allow a new submission to attach previous output without changing computation."""
        payload = self.model_dump_json(exclude={"kernel_sources", "kernel_version"})
        return hashlib.sha256(payload.encode()).hexdigest()
