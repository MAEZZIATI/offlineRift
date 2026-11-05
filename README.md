# Offline Rift

A local, experimental next-action companion for League of Legends. It collects real
Match-V5 data, reconstructs executed options over measured observation intervals,
and trains an actor/role-conditioned policy against terminal team win/loss rewards.
It recommends **what to consider doing next**, rather than classifying the current
team formation. Improved live win rate and optimal play have not been established.

## Layout

| Folder | Responsibility |
|---|---|
| `production/` | Riot collection, patch catalogs, workflow, training, inference and local dashboard |
| `common/` | Action/state contracts, causal timeline reconstruction, explicit transitions, network and persistence |
| `research/` | New-objective dataset audits, frozen-policy diagnostics and study protocol |
| `tests/` | Every test, with synthetic fixtures used only for deterministic verification |

Production never imports research. Common imports neither application. Research is an
optional package excluded from the production wheel. There is one maintained workflow
and one policy network; there is no formation-classifier or classifier/Q-score blend.

## Install

Use the existing Python 3.10+ `.venv`:

```bash
uv pip install --python .venv/bin/python -e '.[train,dev]' -e ./research
```

Training and inference use Torch. The small discrete CQL implementation directly uses
stored next states, elapsed-time discounting and action masks; it does not need d3rlpy.
The `bc` algorithm is an imitation baseline for comparisons, not the RL default.

## Collect, build and train

The local `workflow.json` was migrated to version 2 using the previously verified player
identities. Raw games, catalogs and historical artifacts were preserved. Do not initialize
that existing configuration again.

```bash
.venv/bin/rift status
.venv/bin/rift build
.venv/bin/rift train --algorithm cql --steps 10000
```

Collection uses `RIOT_API_KEY` in the token header. It defaults to the configured primary
PUUID; collecting every configured account is explicit:

```bash
export RIOT_API_KEY='your-current-key'
.venv/bin/rift collect --count 100
.venv/bin/rift collect --all-players --count 25
.venv/bin/rift add-player --riot-id 'Name#TAG' --region asia
```

Pagination, bounded retries, Retry-After, deduplication, atomic caching and interrupted
batch reports are retained. No Riot request is needed to build from cached games and
cached patch catalogs. Missing catalog versions fail or download the exact patch;
there is no nearest-patch fallback. Objective estimates currently cover patches 16.16–16.22.

Fresh projects can use `rift init --puuid VERIFIED_PUUID --region asia --name my-project`.
Never infer a professional identity from a player name. Configuration examples are in
`configs/workflow.example.json`; the root `workflow.json` is local and ignored.

## Run the app

```bash
.venv/bin/rift ui --interval 10
```

Open **http://localhost:8765**. Select **Me**, a specific live player, or **Selected team**.
If the API does not identify your player or role, choose it explicitly; the app does not
guess from a champion. Role-specific advice includes the action, instruction, map checks
and a nominal next-minute horizon. Source data are coarse, so this is not a micro-mechanics
or spell-timing coach. “Continue current activity” and insufficient-support responses are
valid outputs; the app does not force a new macro every poll.

```bash
# Explicit completed bundle:
.venv/bin/rift-ui --model /path/to/next-action-policy --transport auto
# One local poll:
.venv/bin/rift live --once --actor active --role JUNGLE
```

The game API is available during an active 5v5 game. WSL can use the Windows bridge.
The dashboard is localhost-only, clears expired advice, supports focus/pause controls
and does not record sessions. Restart its command after updating browser assets.

## Research and limitations

```bash
.venv/bin/rift-research dataset --dataset /path/to/transitions.npz
.venv/bin/rift-research evaluate --dataset /path/to/transitions.npz \
  --bundle /path/to/policy --output /path/to/new/evaluation
```

See [the modeling contract](docs/MODELING.md), [action catalog](docs/ACTIONS.md),
[architecture](docs/ARCHITECTURE.md) and [new research protocol](research/README.md) and [initial observed study](research/docs/INITIAL_STUDY.md).
Actions are reconstructed from observed behavior, not logged player commands or known
intent. State history is causal. Live coordinates, vision, waves, camp availability and
cast history are not invented. Both teams and all players from a match stay together in
train/validation splits. Q values are return estimates, not win probabilities. Behavior
agreement, macro F1 and prediction coverage diagnose data/support; they are not policy
win-rate or counterfactual improvement estimates.

The former objective's source, research documents and local configuration were archived
under `data/archive/` and removed from maintained code. Old datasets/models remain stored
but are intentionally rejected by the new schema. New latest references live under
`data/projects/<name>/next-actions/`, separate from the old model pointer.

## Validation

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
git diff --check
```

Tests never call Riot or Data Dragon. Browser tests, when enabled with `RIFT_TEST_BROWSER`,
use local synthetic UI fixtures. Full tests need the training and research packages.
