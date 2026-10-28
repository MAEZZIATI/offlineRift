# Real-data workflow

## Start with your downloaded games

```bash
source .venv/bin/activate
python workflow.py status
python workflow.py run
```

The workspace's `workflow.json` selects the account observed in all 100 raw files,
uses Asia routing, and filters queue 420. No identity lookup or match re-download is
needed. `run` resolves patch catalogs, builds and validates transitions, and trains CQL
on real data. It skips completed work when inputs and settings are unchanged.

For a fresh checkout with cached files but no configuration:

```bash
python workflow.py init --name faker
```

Initialization infers a unique player common to every raw match. If that is ambiguous,
provide `--puuid YOUR_PUUID --region asia`. Without cached data, those options also allow
initialization before collection. The initializer never overwrites a configuration.
Edit it thereafter. `workflow.json` is ignored by Git; API keys stay in the environment.

## Commands

| Command | Effect |
|---|---|
| `python workflow.py status` | Show cached matches, patch coverage, dataset and model paths |
| `python workflow.py run` | Build from cache and train when changed |
| `python workflow.py run --refresh` | Collect recent history first, then build/train |
| `python workflow.py collect --count 300` | Scan latest 300 games, downloading only missing payloads |
| `python workflow.py collect --start 100 --count 200` | Backfill 200 games after the newest 100 |
| `python workflow.py build` | Build and validate a dataset without training |
| `python workflow.py train` | Build if needed, then train if needed |
| `python workflow.py run --retrain` | Force a new model run on unchanged data |
| `python workflow.py run --refresh-catalogs` | Refresh required catalogs; rebuild if changed |
| `python workflow.py live --team 200` | Poll with the latest model for red team |
| `python workflow.py live --once` | One poll; returns offline status if no local game |

For another configuration: `python workflow.py --config another-workflow.json run`.
Place the option before the subcommand. Relative artifact paths resolve against the
configuration file's directory. Run one writer per project at a time.

## Routine refresh and pagination

```bash
export RIOT_API_KEY='your-current-development-key'
python workflow.py run --refresh
```

`collection.count` is a recent-history window, not a required number of *new* games.
If 5 new games appeared in a window of 100, the workflow normally reuses 95 and downloads
5. Older cached games stay in the corpus when they leave that recent window.

Counts above 100 paginate automatically in pages of at most 100. A short page ends the
scan; Riot controls how much history is available. Older-history offsets refer to the
current match list, so repeat scans can overlap. Embedded match IDs prevent duplicates.
The dataset selects only the configured PUUID/queue, even if the raw directory contains
other players' histories.

## Configuration

`workflow.json` is the single editable source of defaults:

| Setting | Meaning |
|---|---|
| `name` | Project label for generated dataset IDs |
| `player.puuid`, `player.riot_id`, `player.region` | Collection identity and routing |
| `collection.count` | Recent-history window size |
| `collection.queue` | Queue filter; 420 is ranked solo/duo; null disables filtering |
| `paths.raw`, `paths.catalogs` | Raw JSON and patch catalog cache locations |
| `paths.artifacts`, `paths.models` | Dataset/manifests and versioned model locations |
| `features.profile` | `live` for stock Live Client, `full` for offline spatial experiments |
| `features.grid_size` | Full-profile spatial grid resolution |
| `features.item_spike_gold` | Expensive-item threshold for the power-spike proxy |
| `features.first_spawns`, `features.respawns` | Objective estimates in seconds |
| `features.herald_despawn` | Modeled Herald availability end |
| `features.epic_buff_durations` | Estimated Baron/Elder buff windows |
| `feature_overrides_by_patch` | Feature configuration overrides keyed by major.minor patch |
| `reward_mode` | `sparse` terminal outcome or `gold` shaping plus terminal outcome |
| `training.n_steps` | Requested optimizer updates |
| `training.n_steps_per_epoch` | Desired logging/checkpoint interval |
| `training.batch_size`, `training.device`, `training.cpu_threads` | Compute settings |
| `training.alpha`, `training.gamma`, `training.seed` | CQL penalty, discount, seed |
| `training.validation_fraction` | Match-level validation fraction |

Example: increase `training.n_steps` from 10000 to 20000 and run the same command.
The dataset is reused and a new model is fitted. Changing the feature profile rebuilds
the dataset too. Logging intervals are adjusted to a common divisor when needed so the
exact requested optimizer update count runs, rather than an incomplete epoch being dropped.

## Patch updates

Your original `data/item.json` was 16.21.1; your matches span 16.18–16.20. The workflow
ignores that catalog and resolves the newest published Data Dragon item build **within
each match's major.minor patch**. See [Riot's Data Dragon documentation](https://developer.riotgames.com/docs/lol#data-dragon).

Catalog files are cached as `data/catalogs/items-16.19.json`, etc. Versions and hashes
enter the dataset fingerprint; versions are recorded in reports and NPZ metadata.
Missing catalogs fail explicitly instead of falling back to the newest global patch.
Cached catalogs permit offline rebuilding. `--refresh-catalogs` checks for updated builds
within the required patches. Per-game inventory features use that game's catalog.

**Objective rules need separate maintenance.** Data Dragon item downloads do not verify
spawn timings. Override patch-specific settings, for example:

```json
{
  "feature_overrides_by_patch": {
    "16.20": {
      "first_spawns": {"dragon": 300, "baron": 1200, "herald": 840},
      "herald_despawn": 1170
    }
  }
}
```

Those values illustrate syntax, not verified 16.20 rules. Replaced timer dictionaries
must include every objective key. Overrides cannot change feature schema/profile across
patches. The bundled tracker models one Herald and generic repeating Dragon/Baron spawns;
it omits soul-to-Elder transitions, Void Grubs/other patch objectives, early buff loss on
death, and exact spawn delays.

The newest training patch's catalog/config is embedded for live inference. A live patch
outside training coverage requires updated data and retraining; item stability is not
assumed.

## Artifacts and reuse

```text
workflow.json                         # Local editable settings
data/raw/KR_<match-id>.json            # Real match + timeline payload
data/catalogs/items-<patch>.json       # Patch-matched Data Dragon catalog
data/projects/faker/
  batches/<UTC-time>-<unique-id>.json  # Collection progress/completion/failure
  datasets/faker-<content-hash>/
    transitions.npz                   # Validated MDP transitions
    report.json                       # Labels, patches, skipped games, provenance
  latest.json                         # Completed dataset/model pointers
models/faker/<UTC-time>-<unique-id>/
  policy.d3                          # Learned policy/configuration
  metadata.json                      # Feature/catalog contract and evaluation
  run.json                           # Dataset linkage, settings, run status
  logs/                              # Metrics and epoch checkpoints
```

The old `data/macro.npz` is preserved and is not concatenated into the new dataset.
The new dataset is rebuilt directly from raw games, preventing duplicate episodes.

Dataset fingerprints include raw payload bytes, feature/reward settings, catalogs, and
preprocessing code. Training fingerprints also include training settings and trainer code.
An unchanged latest completed artifact is reused. Earlier datasets and models remain
available for comparison; you never need to invent batch filenames.

## Recover from interruptions

- **Missing/expired key:** export a current key and repeat collection. Cached matches remain.
- **429/outage:** bounded retries honor Retry-After. After exhaustion, the batch records
  failure; repeat the same command later.
- **Interrupted collection:** completed payloads are retained atomically. Repeat the command;
  it records a new batch and downloads only missing matches.
- **Catalog failure:** already cached catalogs remain. No new completed dataset is published.
- **Unsupported timeline:** the build report records skipped filenames/reasons. Malformed
  raw JSON or catalog corruption fails explicitly rather than silently losing data.
- **Interrupted build:** a partial NPZ never becomes the current dataset. Repeat `run`.
- **Interrupted training:** the run records failure/interruption and retains checkpoints.
  The previous completed model stays current. Repeating `run` starts a fresh fit for the
  new inputs; automatic optimizer-checkpoint resumption is not implemented.

Raw data, catalogs, local configuration, and model outputs are ignored by Git. Back them
up separately if you need reproducibility. Do not run concurrent writers for one project.

## Evaluate the real-data run

Dataset `report.json` contains match/episode/transition counts, patch/catalog versions,
action counts, low-confidence label fraction, source match IDs, and skipped games.

Model `metadata.json` contains `training_report`: train/validation IDs, optimizer steps,
label agreement, majority-label baseline, and predicted action counts. Splits are by
match: both team perspectives always stay together. A high agreement score may reflect
the frequent fallback label 4, so compare it with the majority baseline.

These metrics measure heuristic agreement, not calibrated win probability, counterfactual
policy value, or win-rate lift. Both teams from the configured player's history contribute;
the dataset does not claim to capture that player's personal intended decisions.

## Live usage

```bash
python workflow.py live --team 100 --interval 10
```

The latest completed model is selected automatically. `--once` checks a single poll.
`--ca-bundle /path/to/riot-root.pem` enables certificate verification. The official endpoint
uses local HTTPS; see [Riot's Game Client API documentation](https://developer.riotgames.com/docs/lol#game-client-api).

The default live profile masks unavailable team gold, coordinates, role gaps, frame age,
and ability cooldown features in both training and serving. It retains champion levels,
inventory, time, and objective estimates. Full-profile policies need another feature
adapter and are rejected by the stock live client. Q-values are discounted returns,
not win probabilities. Failed polls never reuse previous recommendations.
