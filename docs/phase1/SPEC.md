# Phase 1 foundation specification v1

Established scope: repository layout, contracts, tooling/CI, ADRs, offline
training workflow scaffolding, lineage, checkpoint layout and local output sync.
No Phase 0 mathematical modeling decisions are inferred or implemented here.
No Phase 2 data client, splitting pipeline, features or later models are built.

## File tree

```text
AGENTS.md
Makefile, pyproject.toml, uv.lock, .python-version, .pre-commit-config.yaml
.github/workflows/ci.yml
configs/smoke.json
packages/
  domain/       # contracts.py; no I/O
  storage/      # local.py; atomic per-run artifact store
  training/     # config.py, run.py, workflow.py
  ingestion/    # typed namespace scaffold only
  features/     # typed namespace scaffold only
  models/       # typed namespace scaffold only
  evaluation/   # typed namespace scaffold only
  serving/      # typed namespace scaffold only
  interface/    # typed namespace scaffold only
# Each package: pyproject.toml, README.md, src/offlinerift/<name>/, py.typed
scripts/smoke.py
tests/{conftest,test_architecture,test_contracts,test_workflow}.py
docs/{adr,brief,math,phase1}/
```

## Interfaces and data contracts

Established: `ParticipantKey` V1 validates `EUW1|NA1|KR` match IDs and slots
1–10. `DatasetRef` V1 declares a version and SHA-256. `Review` V1 always abstains.
`ReviewProvider` and `ArtifactStore` are typed ports; `LocalArtifactStore` is the
only concrete adapter. All contracts forbid extra fields and mutation.
These contracts do not claim anonymization or Riot payload validation.

Established: `RunConfig` V1 holds seed, steps/checkpoint interval, immutable
source references, remote input references and explicit compute assumptions.
JSON config is the same locally/remotely; paths for output/resume are CLI concerns.
`Checkpoint` V1 records resume hash, completed-step cursor and synthetic stream.
The resume hash excludes only previous-kernel attachments and submission version;
the full config hash remains in lineage and output receipts.
Config mismatch, corrupt cursor, unsafe artifact paths, and conflicting output
imports fail closed. The stream is reconstructed from its seed on resume.
This simple smoke checkpoint is intentionally unsuitable for large training runs.

Established: JSON schemas can be generated directly from each contract's
`model_json_schema()`; V1 is carried in payloads. Breaking field or semantic
changes require a new contract version, migration decision, and tests.

Established: tensor shapes, trainable losses, optimizers, regret estimators and
training recipes are **not applicable** in this phase. The smoke runner produces
integers to check determinism and I/O; it does not estimate win probability,
implement a mathematical estimator, or produce coaching advice.

## Gates and definition of done

Established: spec → acceptance tests → implementation → docs → lint/types/tests
and CPU smoke. `make check` must pass Ruff lint/format, strict mypy and all tests;
`make smoke` must complete interrupt/resume through the CLI; `make build` must
build nine wheels. No test requires external data, credentials, GPU or network.

Established acceptance gates: V1 round trips; invalid fields/versions rejected;
domain has no I/O imports; package imports point inward; identical final state
after interruption/resume; changed config and broken cursor rejected; lineage
contains hardware/library observations; private/internet-off script metadata;
safe atomic store; versioned, hashed, idempotent local imports with conflict
rejection. These are infrastructure gates, not model quality benchmarks.

Established: performance validation uses tiny CPU fixtures and records measured
test duration. No throughput, coaching effectiveness, or predictive benchmark
is claimed. Heavy profiling waits for a concrete data/model workload.

## Uncertainties

Established open items: actual Kaggle account/session limits; runtime Python and
wheel compatibility; privacy/hosting rights; final retention policy; precise
Riot payload observability and rate limits. These do not block offline foundation
checks; they do block claiming a validated real-data training workflow.

Plausible revision triggers: offline bootstrap fails on an actual Kaggle runtime,
contract evolution needs cross-version readers, or local folders stop meeting
experiment audit needs. Add complexity only after such evidence.
