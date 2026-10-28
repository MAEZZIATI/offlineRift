# Collect, train, and run against your game

Run the commands below from the repository root. The workspace already contains Faker's
local configuration, 100 cached matches, and a trained policy. Those local artifacts are
ignored by Git and are not included when cloning from GitHub.

## 1. Install or refresh the package

In the existing WSL/Linux environment:

```bash
cd /home/amine/offlineRift
source .venv/bin/activate
uv pip install -e '.[train,dev]'
rift status
```

For a fresh environment, create the virtual environment first:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e '.[train,dev]'
```

If `workflow.json` is absent, restore your local configuration/cache backup, or initialize
with your chosen account PUUID and routing region before collecting:

```bash
rift init --name faker --puuid YOUR_FAKER_ACCOUNT_PUUID --region asia
```

With the existing raw Faker cache present, `rift init --name faker` can infer the account.
Do not initialize again when the configuration already exists.

## 2. Collect recent games

Get a current development key from the [Riot Developer Portal](https://developer.riotgames.com/).
Keep it in the environment:

```bash
export RIOT_API_KEY='YOUR_CURRENT_KEY'
rift collect --count 100
```

This scans the configured Faker account's latest 100 games, reusing cached matches and
downloading missing metadata/timelines. It does not delete older cached games. To backfill
history beyond the first 100:

```bash
rift collect --start 100 --count 200
```

The count applies to the requested history window, not a guarantee of that many new games.
Downloads paginate automatically. Re-run an interrupted command to reuse completed files.

## 3. Build and train

```bash
rift train
rift status
```

Training first builds from all selected cached matches. Patch item catalogs are downloaded
and cached automatically. Changed data/settings create new dataset/model artifacts;
unchanged completed work is reused. To force a fresh fit on unchanged data:

```bash
rift train --retrain
```

Change `training.n_steps`, batch size, or other settings in local `workflow.json` before
running the command when needed. Objective timing settings remain explicit patch estimates.
See [the workflow guide](WORKFLOW.md) for configuration and report interpretation.

The combined refresh/build/train command is:

```bash
rift run --refresh
```

No Riot API key is needed for training from cached games. Catalog downloads need internet
only if the required patch catalogs have not already been cached.

## 4. Run live recommendations

The current serving interface prints JSON recommendations to a terminal. It is not an
HTTP backend server or an in-game overlay.

Start a 5v5 Summoner's Rift match and wait until the map is loaded. Then, from a Python
environment on the same operating system as the game:

```bash
rift live --team 100 --once
rift live --team 100 --interval 10
```

Use team 100 for blue/ORDER and 200 for red/CHAOS. `--once` performs a single check; the
second command keeps polling every ten seconds until Ctrl-C. The latest completed local
policy is chosen automatically. Live polling does not require `RIOT_API_KEY`.

The API is `https://127.0.0.1:2999/liveclientdata/allgamedata`. It becomes available during
an active game, not simply when the launcher is open. The client uses local self-signed
TLS by default; `--ca-bundle` enables verification with a supplied certificate. See
[Riot's Game Client API documentation](https://developer.riotgames.com/docs/lol#game-client-api).

### Windows game with a WSL workspace

This workspace runs in Ubuntu WSL2. Windows and WSL loopback access can differ. An offline
result from WSL does not by itself establish whether the Windows game API is available.
From Windows PowerShell, check the Windows endpoint directly:

```powershell
curl.exe --insecure --noproxy "*" https://127.0.0.1:2999/liveclientdata/gamestats
```

If that fails, wait for an active match to load before retrying. If Windows can access it
but WSL cannot, use native Windows Python for live polling; do not point the client at a
public address or expose port 2999.

One option is to open the existing checkout through its WSL share from PowerShell:

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\amine\offlineRift'
```

Do not reuse the WSL virtual environment as a Windows environment. For a separate Windows
checkout, use its own `.venv`, and copy the completed model directory (`policy.d3` plus
`metadata.json`) there. Model weights are not pushed to GitHub.

When reading the existing WSL checkout, create the Windows environment outside that
checkout instead of overwriting its `.venv`, for example:

```powershell
py -3.11 -m venv "$env:LOCALAPPDATA\OfflineRift\venv"
$riftPython = "$env:LOCALAPPDATA\OfflineRift\venv\Scripts\python.exe"
& $riftPython -m pip install -e '.[train]'
& $riftPython scripts\live_client.py --model 'models\faker\YOUR_COMPLETED_RUN_ID' --team 100 --once
& $riftPython scripts\live_client.py --model 'models\faker\YOUR_COMPLETED_RUN_ID' --team 100 --interval 10
```

Use the completed run directory displayed by `rift status`. The direct model path avoids
using WSL absolute paths embedded in the workflow's latest-artifact pointer from Windows.
Use an installed supported Python version if 3.11 is unavailable; the project requires 3.10+.

## 5. Interpret a live result

Successful output contains an action ID/name, a selected Q-value, five action values, game
time, team, and `partial_observation` status. The live API lacks coordinates, team total
gold, and reliable ability cast history; those features are masked consistently with the
training profile. Q-values are discounted-return estimates, not win probabilities.

The current first real-data model predicted the fallback action on every held-out state.
It is a pipeline baseline, not evidence of useful tactical advice. See
[the recorded real-data results](RESULTS.md) before interpreting recommendations.
