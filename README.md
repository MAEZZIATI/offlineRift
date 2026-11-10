# Offline Rift

Offline Rift is an experimental League of Legends macro coach with **six policies**:
one team-direction model and independent TOP, JUNGLE, MIDDLE, BOTTOM and UTILITY
specialists. Every learned return target is derived from terminal team victory.
The product proposes a team direction and a selected player's next activity, exposes
agreement or conflicting directions, and abstains when support is missing.

The ADC-only pilot is historical. It does not define the product scope. Improved
victory rates and optimal strategy have not been established. Current work is offline;
active-game acceptance and prospective outcome testing come later.

## Architecture

| Layer | Location | Responsibility |
|---|---|---|
| Domain | `common/src/rift_common` | Causal observations, actions, evidence, complete transitions, scopes and compatibility |
| Acquisition | `production/src/offline_rift/collection.py`, `catalogs.py` | Verified accounts, bounded Riot requests, raw cache and exact-patch catalogs |
| Learning | `production/src/offline_rift/training` | One trainer, explicit critic targets, per-scope fitting and coach assembly |
| Serving | `production/src/offline_rift/inference` | Immutable bundle verification, six-model routing and advisory persistence |
| Delivery | `production/src/offline_rift/ui` | Local dashboard and optional session recording |
| Orchestration | `rift` / `workflow.py` | Collect, build, train, verify and activate |
| Research | `research/src/rift_research` | Optional offline comparisons, diagnostics, following evidence and synthesis |
| Verification | `tests` | All synthetic fixtures and tests; no real API requests |

Common imports neither production nor research. Production never imports research.
Serving never imports training orchestration. Six independent networks use one state,
reward and persistence contract; collection/build/train implementations are shared.
See [architecture](docs/ARCHITECTURE.md), [modeling](docs/MODELING.md) and
[action evidence](docs/ACTIONS.md) for contracts and limitations.
The [active task](task.md) defines the product work. The
[mathematical design](docs/MATHEMATICAL_DESIGN.md) gives the objective, losses,
missing-history regularizer, coherence constraint and their assumptions.

## Install

Use Python 3.10+ and the existing virtual environment:

```bash
uv pip install --python .venv/bin/python -e '.[train,dev,docs]' -e ./research
```

## Data and offline training

The existing local `workflow.json` contains verified collection identities. Do not
reinitialize it or silently change the primary account. The Riot key can be loaded
explicitly without printing it; only RIOT_API_KEY is imported from the file:

```bash
.venv/bin/rift status
.venv/bin/rift --env-file .env collect --count 100
.venv/bin/rift build
.venv/bin/rift train-coach --steps 10000 --output models/my-new-coach \
  --holdout-manifest /path/to/frozen-protocol.json
```

The holdout manifest contains `holdout_matches`, reserved before development selection.
Collection defaults to the primary verified PUUID. Use `--all-players` explicitly for
configured-account collection. Inspect support gaps before downloading more games.
Raw matches, catalogs, models and failed/interrupted reports remain local and ignored.

The coach trainer compares ordinary imitation and bounded conservative return regression
with availability-consistent versions per scope. Every algorithm is eligible; selection
uses clean/cold development evidence with explicit retention and concentration gates.
Role specialists fit only their role's rows. Macro targets require
coherent executed macro evidence; absence of consensus is unresolved. A conservative
Monte Carlo return-regression variant is distinct from Bellman-target CQL and is
reported explicitly. The algorithms do not establish counterfactual policy utility.

```bash
.venv/bin/rift verify --model models/my-new-coach/model
.venv/bin/rift-research evaluate --new-matches \
  --dataset /path/to/transitions.npz --bundle models/my-new-coach/model \
  --match-manifest /path/to/frozen-protocol.json \
  --output /path/to/new/evaluation
.venv/bin/rift activate --model models/my-new-coach/model
```

Activation verifies saved serving replay and retains the prior pointer. It publishes
an experimental candidate, not a calibrated win predictor. Compare support, majority
baselines, abstention, repetition and rare actions separately for every scope.

## Product behavior

The macro model suggests a team direction. A selected role specialist independently
suggests an activity. Different explicit destinations cause the role proposal to be
withheld, while the macro direction remains visible. No replacement action is invented.
Other nonmatching proposals remain independent, not proven strategically aligned.
The specialist was not trained conditional on a hypothetical plan; scores are not blended.

The observation contract is coarse. Vision, waves, camp safety, precise positions,
casts and fine mechanics are not invented. Current formation labels are not strategy
recommendations. Historical role metadata is retrospective; live role selection must
be reported or explicit. Macro options are reconstructed behavior, not verified intent.

Live testing is deferred. The eventual localhost UI uses:

```bash
.venv/bin/rift ui --actor active --role JUNGLE --interval 5
```

Choose the actual role. Optional `--capture-session NEW_DIRECTORY` records measured
inputs and served advice. [The session runbook](docs/RANKED_TEST.md) documents recording
and bounded following-evidence review. No live session is required for offline checks.

## Validation and research

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
git diff --check
```

Optional Chromium checks use synthetic localhost fixtures via RIFT_TEST_BROWSER.
Tests never call Riot or Data Dragon. Research is excluded from the production wheel.
The [research guide](research/README.md) covers optional diagnostics and reporting.
The new synthesis is [research/docs/SYNTHESIS.pdf](research/docs/SYNTHESIS.pdf).
The subsequent [European study](research/docs/EUROPEAN_STUDY.md) and
[executed game notebook](research/notebooks/european_high_elo_game.ipynb) include all
six policies, both teams, all ten players, plots and the first study's TEAM regression.
The revised regional selection is an experimental candidate: a fresh forty-game
comparison supports better MIDDLE/UTILITY activity agreement, with other changes
uncertain. Its activation retains the previous coach. Win-rate gains remain unproven.
Superseded prototypes and dated documents are checksummed in the local archive;
[history](research/docs/HISTORY.md) explains where they were preserved.
