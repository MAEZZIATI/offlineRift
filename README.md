# Offline Rift

A Python backend for collecting League Match-V5 timelines, labeling twelve team macro
actions, training discrete offline CQL, and polling the local Live Client API.

The main workflow uses **your real cached matches**. There is no synthetic-data step.

## Your existing Faker dataset

The original seed cache contained 100 ranked solo/duo games from `Hide on bush#KR1`,
spanning patches 16.18–16.20. The workspace has since expanded to additional accounts;
see the [data plan](docs/DATA_PLAN.md) and per-cohort manifests for collection counts.
Keep the existing configured PUUIDs. For a fresh mixed-account cache, initialize with
an explicit verified `--puuid` and `--region` when no single account is shared by every file.

```bash
cd /home/amine/offlineRift
source .venv/bin/activate
rift status

# One-time setup for the original single-account seed cache, if no config exists:
rift init --name faker

# Process cached matches and train only when inputs/settings have changed:
rift run

# Show data coverage and current artifact paths:
rift status
```

`workflow.json` is already initialized in this workspace. Your normal command is now
**`rift run`**. No Riot key is needed to process cached games. Data Dragon
downloads require internet only when patch catalogs are missing.

Use these commands in your WSL terminal from the repository root. If `rift` is not
found, run `.venv/bin/python -m offline_rift status` without activation, or refresh
the installation with `uv pip install --python .venv/bin/python -e '.[train,dev]'`.
The older `python workflow.py ...` command is also supported.

## Add games later

```bash
export RIOT_API_KEY='your-current-key'
rift run --refresh
```

This checks the latest 100 games, reuses cached matches, downloads new ones, builds
from **all** cached games for the configured player/queue, and trains when needed.
Increase `collection.count` in `workflow.json` to scan more history. For older games:

```bash
rift collect --start 100 --count 200
rift run
```

Requests paginate automatically. Batches get unique IDs and progress manifests.
Interrupted collection resumes by reusing completed downloads. Datasets get content-based
IDs; model runs get timestamp IDs. Earlier artifacts are retained and unchanged runs
reuse their outputs.

## Change settings

Edit `workflow.json`, then run `rift run` again.

- Training changes reuse the dataset and create a new model run.
- Data, preprocessing, feature, or catalog changes rebuild the dataset automatically.
- `run --retrain` forces another fit on unchanged data.
- `run --refresh-catalogs` checks for updated Data Dragon builds within each match's patch.

Patch-matched item downloads and output naming are automatic. The old `data/item.json`
and `data/macro.npz` are preserved but are not used by the new workflow. Objective timing
rules remain explicit configuration; catalog downloads do not validate them.

## Use the latest policy

```bash
rift ui
```

Open **http://localhost:8765** in your browser. Choose blue or red side to see the
latest model suggestion, twelve macro ranking scores, game clock, and model limitations.
The dashboard waits for an active 5v5 Summoner's Rift match; no model path or Riot
API key is needed. Stop the command with Ctrl-C.

The UI is plain HTML/CSS/JavaScript in `src/offline_rift/ui/static/`, served by a
small Python standard-library server. It loads one CPU model, shares one cached
game fetch across tabs, refreshes every 30 seconds, and pauses hidden tabs. No Node,
frontend build, cloud service, or new runtime dependency is required.

On WSL, `auto` transport uses Windows `curl.exe` when available to read the Windows
game's loopback API. On native Windows/Linux it uses local HTTP. See the
[live UI guide](docs/LIVE_UI.md) for setup, transport overrides, and limitations.
The terminal interface remains available: `rift live --team 100 --once`.

## Documentation

- **[Workflow guide](docs/WORKFLOW.md):** commands, editable configuration, patch updates,
  artifact layout, interruption recovery, reports, and limitations.
- **[Backend reference](docs/BACKEND.md):** modules, Python interfaces, features, MDP arrays.
- **[Current real-data results](docs/RESULTS_V2.md):** the 200-match model and controlled prediction comparisons.
- **[First real-data results](docs/RESULTS.md):** the historical 100-match baseline and its limitations.
- **[Research synthesis](docs/RESEARCH_SYNTHESIS.md):** frozen-model ablations, uncertainty,
  prioritized experiments, and two mathematical paper drafts.
- **[Data plan](docs/DATA_PLAN.md):** sample-size estimates, quality gates, selection bias,
  and reproducible collection from additional accounts.
- **[Label audit](docs/LABEL_AUDIT.md):** evidence checks, timing contradictions, and a
  blinded local review viewer.
- **[Causal memory study](docs/MEMORY_STUDY.md):** five-seed GRU comparisons with matched
  snapshot and repeated-current controls; experimental models are not automatically deployed.
- **[Project structure](docs/STRUCTURE.md):** package layout, compatibility entry points,
  and locations for configuration and generated artifacts.

| File | Responsibility |
|---|---|
| `src/offline_rift/workflow.py` | Collection/cache orchestration, patch catalogs, versioned datasets and runs |
| `src/offline_rift/pipeline/data_pipeline.py` | Riot requests and causal 10-second snapshots |
| `src/offline_rift/pipeline/features.py` | Shared feature schema, item valuation, objective estimates |
| `src/offline_rift/pipeline/action_labeler.py` | Twelve weak macro labels, rewards, NPZ validation |
| `src/offline_rift/training/rl_agent.py` | Discrete CQL trainer and policy loader |
| `src/offline_rift/training/train_macro_model.py` | Direct CLI for training an explicit NPZ dataset |
| `src/offline_rift/inference/live_client.py` | Local polling, features, and recommendations |
| `src/offline_rift/inference/transport.py` | Bounded local/Windows game API access |
| `src/offline_rift/ui/` | Browser assets, shared snapshot cache, and local HTTP serving |
| `scripts/` | Small compatibility launchers for the package commands |

## Install and test

Python 3.10+:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e '.[train,dev]'
pytest -q
ruff check .
```

Use `rift` for the workflow. Compatibility launchers are in `scripts/`, for example
`python scripts/workflow.py status`. Import backend APIs from `offline_rift`.

## Interpretation

These are weak team-positioning labels, not records of Faker's intended decisions.
Both teams contribute separate episodes, kept together in train/validation splits.
A single player's history is a small, biased research corpus. Validation agreement
with heuristic labels does not establish win-rate improvement or optimal play.

The [latest audit](docs/LABEL_AUDIT.md) found objective-timing contradictions that need
review before promoting another model. The [memory experiment](docs/MEMORY_STUDY.md)
finds a modest history signal but does not beat the published hybrid on macro F1;
the plugged model remains unchanged.

Minute-level frames are forward-filled onto the 10-second grid. The default live
profile masks features unavailable from the Live Client API in training and serving.
Objective timers are configurable estimates. Q-values are discounted-return estimates,
not calibrated win probabilities.
- **[Execution guide](docs/EXECUTION.md):** recollect, train, and run against an active game,
  including Windows/WSL setup.

The current `rift-v2` / `macro-v2` implementation requires rebuilding and retraining
old bundles with `rift run`. Historical artifacts remain on disk. See
[model improvement directions and research](docs/MODEL_IMPROVEMENTS.md) for the action
contract, deep set encoder, experiment protocol, limitations, and proposed innovations.
To add the requested second account, follow [the multi-player workflow](docs/WORKFLOW.md#add-chovys-requested-account).
