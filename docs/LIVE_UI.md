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
observation. `--interval` defaults to 30 seconds and must be at least 2. If the port
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
memory cost for hybrid bundles. Classifier-only bundles serve with Torch and do not
require d3rlpy.

## Layout and controls

The session bar identifies the loaded model as CQL, CQL + neural model, or neural
classifier. Team selection is explicit and remains selected across refreshes. The
main card shows one model estimate, a static map-check prompt and the observation's
game time. The clock is not advanced between observations.

Three other supported patterns appear in score order. All twelve raw scores and the
model's validation metrics are in expandable sections. No score is a probability.
The age and freshness indicator describe the cached observation's remaining lifetime;
expired, paused and failed observations clear suggestions, alternatives and scores.

Use **Focus view** (keyboard **F**) to hide secondary panels while retaining the main
suggestion, game clock and controls. Use **Pause** / **Resume** (keyboard **P**) to
control polling. Shortcuts apply only in the dashboard tab and do not intercept text
inputs or modified keys. Hidden tabs still pause requests. The layout stacks on narrow
screens, with visible focus states, labeled controls and reduced-motion support.

Assets are cached by the Python server at startup: restart the dashboard command and
reload the browser after updating the HTML, CSS or JavaScript.

## Visual identity

The interface uses original geometric SVG marks, a dark ink and muted brass palette,
system fonts, and an original layout. It includes no Riot logos, character/item art,
game maps, proprietary fonts or copied client panels. A visible footer carries the
developer-policy non-endorsement notice. This design does not establish legal approval
or approval of the app's other functionality.

The design references were checked on 2026-11-05:
[Riot's general developer policies](https://developer.riotgames.com/policies/general)
require distinct product styling and a visible legal notice;
[Legal Jibber Jabber](https://www.riotgames.com/en/legal) also addresses original
contributions and restrictions on branding. Publication, registration and other product
obligations remain separate from this local UI redesign.

Optional real-browser interaction tests are in `tests/test_dashboard_browser.py`.
Set `RIFT_TEST_BROWSER` to an installed Chromium/headless-shell executable to run them;
without it those tests skip. They use synthetic local UI fixtures, not live games,
cover desktop/mobile widths and all three model modes, and download no browser.

## Reading a suggestion

The card displays the model's action. Its accompanying sentence is static guidance
for that action, not an analysis of unseen player locations or vision. Q-values are
raw discounted-return estimates, never probabilities or confidence percentages.
Expired, unavailable, malformed, or failed observations clear the decision card.

The validation disclosure reports training action coverage, label counts, heuristic
agreement, majority baseline, and validation prediction counts from the loaded model's
metadata. Missing metrics are shown as unrecorded. A visible warning flags collapsed
validation predictions or a training label exceeding 80% of observations.

The historical five-action baseline chose the fallback on every validation state. A
working live dashboard does not establish useful strategy or win-rate improvement.
Coordinates, team total gold, and reliable ability cooldowns remain unavailable and
masked; objective timing remains an explicit estimate. See [current experiments](../research/docs/RESULTS_V2.md) and the [historical results](../research/docs/RESULTS.md).

Hybrid and classifier models display formation ranking scores. Expand all patterns to
inspect raw scores; hybrid rows also expose Q estimates on hover. Classifier-only rows
have no Q values. Model details include balanced accuracy and macro F1 alongside the
majority baseline and label agreement. Every validation metric concerns weak labels.
