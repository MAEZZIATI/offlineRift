# Repository instructions

## Purpose and scope

Offline Rift is a Python research backend for learning team-level League of Legends
macro actions from real Match-V5 data. Preserve the real-data
workflow and describe model limitations accurately. A running training pipeline does
not establish useful strategy, optimality, or calibrated win probabilities.

## Architecture

- `production/src/offline_rift/workflow.py`: configured-player collection, patch catalogs, immutable dataset
  builds, versioned training runs, and latest-completed artifact references.
- `common/src/rift_common/pipeline/data_pipeline.py`: Riot API routing/retries and causal timeline reconstruction.
- `common/src/rift_common/pipeline/features.py`: the shared feature contract, item catalog, and objective estimates.
- `common/src/rift_common/pipeline/action_labeler.py`: weak macro labels, rewards, explicit transitions, and validation.
- `production/src/offline_rift/training/`: maintained CQL and imitation trainers.
- `production/src/offline_rift/inference/policy.py`: portable runtime policy loading and scoring.
- `production/src/offline_rift/inference/live_client.py`: local polling and live feature adaptation.
- `production/src/offline_rift/training/train_macro_model.py`: direct training CLI; prefer the workflow for normal operation.
- `tests/`: deterministic fixtures, mocked transport, and short real optimizer tests.
- `research/src/rift_research/`: optional experiments, cohort studies, audit and labeling.
- `common/src/rift_common/models/`: shared Torch models, scoring, metrics and match splits.
- `common/src/rift_common/artifacts.py`: atomic persistence and bundle integrity.
- `docs/`: operational workflow and backend contract.
- `research/docs/`: results, research plans and papers.
- `configs/`: tracked configuration examples; local `workflow.json` remains at the root.
- `scripts/` contains compatibility entry points. Keep application implementation in
  `production/`, reusable code in `common/`, and experimental code in `research/`.
  Production must never import research; common must import neither application.
  Keep every test in the single root `tests/` folder.

## Development conventions

- Target Python 3.10+ and the dependencies in `pyproject.toml`; use the existing `.venv`.
- Keep changes scoped and object interfaces clear. Avoid duplicate orchestration paths.
- Prefer `rift`/`python workflow.py` commands for end-user workflows. Collection is
  currently scoped to Faker's configured account; preserve that scope unless requested.
- Centralize feature definitions and availability masks in
  `common/src/rift_common/pipeline/features.py`. Preserve
  training/live feature order and scaling. Version incompatible schemas explicitly.
- Keep documentation and CLI help synchronized with behavior. Use real-data examples;
  synthetic fixtures belong in tests and must not be presented as trained strategy.

## Data and model invariants

- Never interpolate from future frames or use final match gold/items as state features.
- Treat the 10-second grid as causal resampling of coarse observations, not new telemetry.
- Preserve episode terminal boundaries and `(state, action, reward, next_state)` alignment.
- Deduplicate match IDs before training. Keep both teams
  from a match together when splitting train/validation data.
- Match item catalogs to each game's patch. Missing catalog versions must not silently
  fall back to a different patch. Objective timing rules remain explicit estimates.
- Do not invent live coordinates, team gold, cast history, cooldowns, or probabilities
  when the API does not expose them. Keep unavailable features masked consistently.
- Report action coverage, heuristic agreement, and majority baseline. Flag observed
  label imbalance or collapsed predictions when explaining experiment results.

## External APIs and persistence

- Keep Riot credentials in `RIOT_API_KEY`, using the token header rather than URLs.
- Use the correct Riot routing values; Match-V5 uses regional routing.
  Reuse bounded retry/error handling and honor Retry-After without early retries.
- Use the configured PUUID as the collection identity. Riot IDs are display identifiers;
  do not infer or replace the user's configured account from an unverified player name.
- Cache raw payloads by match ID. Persist progress and metadata atomically. Publish
  latest-artifact references only after successful validation or completed training.
- Retain previous datasets/models and interrupted-run reports. Never remove user data
  or overwrite completed experiment artifacts as routine cleanup.
- Never commit API keys, `.env`, local `workflow.json`, raw game data, catalogs, trained
  weights, generated logs, or `.venv`. Keep these covered by `.gitignore`.

## Validation and handoff

Run checks appropriate to the change from the repository root:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
git diff --check
```

Tests must not call Riot or Data Dragon over the network. Mock API responses,
pagination, rate limits, duplicate matches, and interrupted collection. Short CPU CQL tests
may train/save/reload to verify real library integration. Keep them small and reproducible.
For CLI changes, check the affected `--help` command and installed `rift` entry point.

Explain what changed, relevant validation, and any material limitations. Do not claim
live authenticated integration succeeded without actually exercising it. Update recorded
results only for the specific run observed; preserve historical findings.
