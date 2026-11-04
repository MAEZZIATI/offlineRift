# Local live dashboard

From the repository root in WSL, Linux, or a native Windows Python environment:

```bash
rift ui
```

Open **http://localhost:8765** in a browser on the same machine. In WSL, you can use
your Windows browser. Choose blue/ORDER or red/CHAOS in the interface. The dashboard
waits for an active 5v5 Summoner's Rift game; the launcher alone does not expose live
game data. Press Ctrl-C in the terminal to stop the server.

With this workspace's existing environment, activation is optional:

```bash
.venv/bin/python -m offline_rift ui
```

If `rift ui` is not recognized after updating, refresh the editable install:

```bash
uv pip install --python .venv/bin/python -e '.[train,dev]'
```

## Connection and options

`auto` selects Windows `curl.exe` when running in WSL and that executable is on PATH.
The bounded subprocess queries the game's Windows loopback API, without a proxy or
redirects. This relies on [Windows executable interop](https://learn.microsoft.com/en-us/windows/wsl/interop).
In other environments, `auto` uses a direct local HTTP session. Neither mode makes a
remote Linux host able to observe a game on a different machine.

```bash
rift ui --team 200 --interval 15 --port 8766
rift ui --transport windows
rift ui --transport local
```

`--team` selects the initial side; switching sides in the browser reuses the same
observation. `--interval` defaults to 10 seconds and must be at least 2. If the port
is occupied, choose another port and open its printed URL.

When the connection is offline during a match, check the game API from Windows PowerShell:

```powershell
curl.exe --insecure --noproxy "*" https://127.0.0.1:2999/liveclientdata/gamestats
```

If this succeeds but WSL cannot launch `curl.exe`, enable Windows interop or use a
separate native Windows environment as described in [the execution guide](EXECUTION.md).
You can serve an explicit policy bundle without workflow configuration or WSL absolute
artifact pointers:

```bash
rift-ui --model models/faker/YOUR_COMPLETED_RUN_ID
# Equivalent, also usable with native Windows Python:
python -m offline_rift.ui.server --model models/faker/YOUR_COMPLETED_RUN_ID
```

Live requests use the game's self-signed TLS by default. `--ca-bundle /path/to/riot.pem`
enables certificate verification; in the WSL Windows transport the path is translated
with `wslpath`. The dashboard binds to `127.0.0.1` and only serves its fixed assets and
read-only `/api/state`. There is no remote hosting, authentication, telemetry, or
Riot credential requirement. Do not publish the server or expose the game's port.

## Resource use and separation

- `ui/static/`: plain HTML, CSS, and JavaScript. No framework, build step, CDN, or fonts to download.
- `ui/server.py`: Python standard-library HTTP serving and CLI. Assets are packaged with the module.
- `ui/service.py`: one in-memory cache shared by all tabs. One game fetch yields both team perspectives.
- `inference/transport.py`: bounded local HTTP/Windows subprocess access, independent of presentation.
- `inference/live_client.py`: the existing feature adapter and inference contract used by both interfaces.

The CPU model loads once. PyTorch uses one computation thread for this UI process.
The browser polls at the configured interval, stops while hidden or paused, and
discards expired recommendations. There is no background inference without requests,
match history collection, or new persistence. The server keeps only the latest state;
multiple viewers share its cache. The existing PyTorch/d3rlpy model remains the main
memory cost; a browser interface does not remove that dependency.

## Reading a suggestion

The card displays the model's action. Its accompanying sentence is static guidance
for that action, not an analysis of unseen player locations or vision. Q-values are
raw discounted-return estimates, never probabilities or confidence percentages.
Expired, unavailable, malformed, or failed observations clear the decision card.

The validation disclosure reports training action coverage, label counts, heuristic
agreement, majority baseline, and validation prediction counts from the loaded model's
metadata. Missing metrics are shown as unrecorded. A visible warning flags collapsed
validation predictions or a training label exceeding 80% of observations.

The current saved baseline chose the fallback action on every validation state. A
working live dashboard does not establish useful strategy or win-rate improvement.
Coordinates, team total gold, and reliable ability cooldowns remain unavailable and
masked; objective timing remains an explicit estimate. See [the recorded results](RESULTS.md).
