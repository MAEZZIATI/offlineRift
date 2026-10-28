# Offline Rift

A Python backend for collecting League Match-V5 timelines, labeling five team macro
actions, training discrete offline CQL, and polling the local Live Client API.

The main workflow uses **your real cached matches**. There is no synthetic-data step.

## Your existing Faker dataset

`data/raw/` contains 100 ranked solo/duo games from the account `Hide on bush#KR1`,
spanning patches 16.18, 16.19, and 16.20. The initializer infers the player shared
across the files and the Korean routing region automatically.

```bash
cd /home/amine/offlineRift
source .venv/bin/activate

# One-time setup, only when workflow.json does not exist:
python workflow.py init --name faker

# Process cached matches and train only when inputs/settings have changed:
python workflow.py run

# Show data coverage and current artifact paths:
python workflow.py status
```

`workflow.json` is already initialized in this workspace. Your normal command is now
**`python workflow.py run`**. No Riot key is needed to process cached games. Data Dragon
downloads require internet only when patch catalogs are missing.

## Add games later

```bash
export RIOT_API_KEY='your-current-key'
python workflow.py run --refresh
```

This checks the latest 100 games, reuses cached matches, downloads new ones, builds
from **all** cached games for the configured player/queue, and trains when needed.
Increase `collection.count` in `workflow.json` to scan more history. For older games:

```bash
python workflow.py collect --start 100 --count 200
python workflow.py run
```

Requests paginate automatically. Batches get unique IDs and progress manifests.
Interrupted collection resumes by reusing completed downloads. Datasets get content-based
IDs; model runs get timestamp IDs. Earlier artifacts are retained and unchanged runs
reuse their outputs.

## Change settings

Edit `workflow.json`, then run `python workflow.py run` again.

- Training changes reuse the dataset and create a new model run.
- Data, preprocessing, feature, or catalog changes rebuild the dataset automatically.
- `run --retrain` forces another fit on unchanged data.
- `run --refresh-catalogs` checks for updated Data Dragon builds within each match's patch.

Patch-matched item downloads and output naming are automatic. The old `data/item.json`
and `data/macro.npz` are preserved but are not used by the new workflow. Objective timing
rules remain explicit configuration; catalog downloads do not validate them.

## Use the latest policy

```bash
python workflow.py live --team 100
```

Team 100 is blue/ORDER; 200 is red/CHAOS. No model path is needed. `--once` performs
a single poll. Outside an active local game, polling returns an offline status.

## Documentation

- **[Workflow guide](docs/WORKFLOW.md):** commands, editable configuration, patch updates,
  artifact layout, interruption recovery, reports, and limitations.
- **[Backend reference](docs/BACKEND.md):** modules, Python interfaces, features, MDP arrays.
- **[First real-data results](docs/RESULTS.md):** the 100-match baseline and its limitations.

| File | Responsibility |
|---|---|
| `workflow.py` | Collection/cache orchestration, patch catalogs, versioned datasets and runs |
| `data_pipeline.py` | Riot requests and causal 10-second snapshots |
| `features.py` | Shared feature schema, item valuation, objective estimates |
| `action_labeler.py` | Five weak macro labels, rewards, NPZ validation |
| `rl_agent.py` | Discrete CQL trainer and policy loader |
| `train_macro_model.py` | Direct CLI for training an explicit NPZ dataset |
| `live_client.py` | Local polling, features, and recommendations |

## Install and test

Python 3.10+:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e '.[train,dev]'
pytest -q
ruff check .
```

After reinstalling the updated project, `rift` is an alias for `python workflow.py`.
The Python command works immediately in the existing environment.

## Interpretation

These are weak team-positioning labels, not records of Faker's intended decisions.
Both teams contribute separate episodes, kept together in train/validation splits.
A single player's history is a small, biased research corpus. Validation agreement
with heuristic labels does not establish win-rate improvement or optimal play.

Minute-level frames are forward-filled onto the 10-second grid. The default live
profile masks features unavailable from the Live Client API in training and serving.
Objective timers are configurable estimates. Q-values are discounted-return estimates,
not calibrated win probabilities.
