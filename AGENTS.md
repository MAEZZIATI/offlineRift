# AGENTS.md

## Project
Post-game decision-review coach for League of Legends (macro model + 5 role
models). Post-game only, official Riot APIs only. No live features.
Train on EUW1; transfer study to NA1 and KR. Clean slate: no legacy code.

## Non-negotiable rules
- Never read, print, or commit the Riot API key or the Kaggle token.
  Commands needing them are Makefile targets that the user runs.
- Never call the live Riot API or Kaggle from tests or CI.
- Datasets are keyed on matchId + participant slot, never PUUID.
- Splits are temporal and grouped by match, never random.
- Claims in docs are labeled: established / plausible / speculative.
  Never invent benchmarks.

## Architecture
Hexagonal. Packages: domain, ingestion, storage, features, models,
training, evaluation, serving, interface. Dependencies point inward.
domain has no I/O. Contracts = typed Protocols + versioned pydantic schemas.

## Code standards
- Full type hints, strict type checking, linter/formatter, pre-commit.
- Pure functions for transforms, dependency injection, no global state,
  typed config-driven runs, deterministic seeds, idempotent pipelines.
- Write tests first or together with code. Every task must pass locally
  on CPU with tiny fixtures.
- Code that implements an equation cites its ID from docs/math/.

## Training (Kaggle)
- Entry point: `python -m <pkg>.training.run --config <file>`, same locally
  and in a Kaggle script kernel. No logic in notebooks.
- Kaggle is driven only via the Kaggle CLI through Makefile targets.
- Jobs checkpoint and resume. Training runs have no internet.

## Definition of done
spec -> tests -> implementation -> docs -> lint/types/tests pass.

## Working style
- Small diffs. One task at a time. Say what you changed and what you did not.
- If a spec is ambiguous, stop and ask instead of guessing.

## Phase 1 commands and scope
- Python 3.11+, uv workspace; `make setup` installs locked dependencies/hooks.
- `make check` runs Ruff lint/format, strict mypy and offline pytest.
- `make smoke` checks CPU CLI interruption/resume; `make build` builds all packages.
- Authenticated `kaggle-*` Makefile targets are USER-RUN ONLY. Never invoke them
  as an agent, in tests or CI; never inspect credential files or environment secrets.
- Kaggle is training compute for the user's own privately staged Riot data;
  do not import third-party Kaggle training datasets.
- Phase 1 is foundation only. No crawling, feature/model implementations,
  experiment results, serving endpoints or product UI. Stop after this phase.
- Read `docs/phase1/SPEC.md`, `TASKS.md`, `KAGGLE.md` and `docs/adr/` for foundation
  acceptance gates, exact files to touch, workflow and decisions.

Full project brief: docs/brief/FULL_BRIEF.md. Read only the sections relevant to the current phase.
