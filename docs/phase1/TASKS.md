# Phase 1 implementation tasks and ranked work backlog

Established: tasks below are limited to foundation work, implemented together
with tiny CPU acceptance tests. No task authorizes starting another phase.
GPU cost is zero for every local task; no Kaggle session is needed to pass.

| Rank/task | Spec and files | Acceptance | Expected gain / cost | Risk / kill criterion |
|---|---|---|---|---|
| 1. Toolchain/layout | Root configs, CI, package pyprojects and READMEs; SPEC.md tree | lint/types/tests, nine wheel builds | Reproducible checks / CPU only, 0 GPU-h, 0 sessions | Packaging friction; simplify if independent imports fail |
| 2. Boundary contracts | domain/contracts.py, tests/test_contracts.py, test_architecture.py | V1 keys, invalid payloads, inward imports | Stable minimal boundaries / tiny CPU tests, 0 GPU-h, 0 sessions | Premature schema breadth; add no speculative fields |
| 3. Offline runner | training/{config,run}.py, storage/local.py, scripts/smoke.py, test_workflow.py | CLI interruption/resume equals uninterrupted state, mismatch rejection | Catch workflow errors early / tiny CPU tests, 0 GPU-h, 0 sessions | False training claims; label smoke only |
| 4. CLI scaffolding | Makefile, training/workflow.py, KAGGLE.md, ADR 0002 | Private offline script metadata; idempotent allowlisted sync | Reviewable remote path / CPU tests, 0 GPU-h, 0 required sessions | Runtime incompatibility; defer upload until prerequisites verified |

Established: no predictive experiment is justified during Phase 1. Optional
future user-run verification of this scaffolding costs at most one CPU Kaggle
session with a requested 0.1-hour budget, 0 GPU-hours; actual account limits must
be checked first. It is not required for offline acceptance and was not run.

Established task completion recipe: read the linked spec/ADR, change only listed
files (plus corresponding tests/docs), run `make check smoke`, build packages
after packaging changes, record actual results, and stop. Tests precede or ship
with code. No secret reads, API calls, datasets, model experiments or public
uploads are permitted in a foundation task.

Plausible expected gains are qualitative engineering benefits, not measured
model improvements. Future experiment budgets and kill criteria must be frozen
before those phases; they are deliberately not invented here.
