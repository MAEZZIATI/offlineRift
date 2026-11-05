# Offline Rift

A Python backend for causal Match-V5 datasets, team formation models, and local League
Live Client inference. Training uses real cached matches. Formation scores describe
patterns in weak labels; they do not establish good strategy or win probabilities.

## Source layout

| Folder | Responsibility |
|---|---|
| `production/src/offline_rift/` | Collection workflow, maintained trainers, policy serving, live adapter and dashboard |
| `common/src/rift_common/` | Shared feature/action contracts, causal timeline processing, datasets, neural components, splits and artifact integrity |
| `research/src/rift_research/` | Optional comparisons, bootstrap studies, cohort expansion, memory experiments and labeling |
| `tests/` | All tests, with one shared fixture module |
| `docs/` | Operational documentation and interfaces |
| `research/docs/` | Experiment results, label audits, research plans and papers |

Production and research depend on common code. Production never imports research.
Research is a separate optional package and is excluded from the production wheel.
See [the structure guide](docs/STRUCTURE.md) for installation and dependency boundaries.

## Install

Use the existing `.venv`; Python 3.10+ is required:

```bash
uv pip install --python .venv/bin/python -e '.[train,dev]'

# Optional: experiments and label review
uv pip install --python .venv/bin/python -e ./research
```

A classifier-only inference deployment can install `.[inference]` without d3rlpy or the
research package. Existing CQL/hybrid bundles require the `train` extra for d3rlpy.
No credentials, cached games or learned weights are included in these packages.

## Everyday workflow

Run from the repository root. The local `workflow.json` retains your configured PUUIDs,
regional routing, paths and settings; do not initialize it again.

```bash
.venv/bin/rift status
.venv/bin/rift build
.venv/bin/rift ui
```

`rift build` uses cached matches and patch-matched item catalogs. Missing catalogs need
internet access. `rift ui` uses the current completed model and serves the dashboard
at http://localhost:8765. Live access requires an active Summoner's Rift game; WSL can
use Windows loopback through the existing `auto` transport. See [live setup](docs/LIVE_UI.md).

To collect additional history from the configured account:

```bash
export RIOT_API_KEY='your-current-key'
.venv/bin/rift collect --count 100
```

Collection is resumable, caches raw payloads by match ID and deduplicates shared games.
Keep API keys in the environment. Local settings, data, catalogs, logs and models are
ignored by Git. Previous completed and interrupted experiments are retained.

## Iterate without label review

Independent labeling is deferred and stays in the optional research package. Train a
separate candidate from a development dataset using the maintained trainer:

```bash
.venv/bin/rift-train --algorithm imitation --encoder-type mlp \
  --dataset /path/to/development/transitions.npz \
  --output /path/to/new/candidate --steps 10000 --batch-size 128

# Preview a candidate explicitly, without replacing the current model:
.venv/bin/rift-ui --model /path/to/new/candidate
```

Each candidate uses a new directory and saves metadata, weights, checksums and a run
report. `--split-from /path/to/reference/bundle` reuses its exact match split for
comparisons. CPU training defaults to two threads; accelerator training transfers only
sampled batches. The imitation path trains no CQL model and serves without Q values.
CQL/hybrid training remains available through `--algorithm cql`.

`rift train` / `rift run` retain the reviewed-promotion gate for corrected objective
rules. Experimental training does not update the active reference. Review and later
promotion are separate from improving the training/inference implementation.

## Validate

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
git diff --check
```

Tests use deterministic fixtures and mocked external services. Real optimizer tests
remain short and local. The dashboard integration test opens a localhost socket.
Install `./research` to run the full suite, including optional experiment/review tests.

Further documentation: [workflow](docs/WORKFLOW.md), [backend](docs/BACKEND.md),
[execution](docs/EXECUTION.md), and [research workspace](research/README.md).

The [2026-11-05 iteration](research/docs/ITERATION_20261105.md) produced an inference-ready
experimental MLP candidate with 23.32% macro F1 on 846 matches excluded from fitting,
versus 21.98% for the corrected hybrid. Its overall label agreement is lower. Preview it
with `rift-ui --model models/candidates/20261105-imitation-mlp-seed43-v1`;
the active model reference is unchanged.
