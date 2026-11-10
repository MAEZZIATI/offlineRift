# Offline Rift desktop HUD

The Electron HUD displays the team proposal and the selected role specialist's proposal
from the completed six-policy coach. It starts click-through, stays above other windows,
and supports top-right or top-left placement. All renderer assets are bundled locally.

## Build and run

Requires Node.js 22.12+ and npm. From this folder:

```bash
npm ci
npm run package:win
# On Linux: npm run package:linux
```

From the repository root:

```bash
.venv/bin/rift hud --actor active --role BOTTOM --interval 5
```

The launcher loads the selected coach from workflow state. Override it with `--model PATH`,
the host with `--desktop PATH`, or the corner with `--position top-left`. Use League in
borderless or windowed mode. Ctrl+C closes the backend and the native window. Quitting
from the tray also stops the backend. Loss of the backend lifetime connection closes the
host, including when a WSL process proxy exits before its Windows application.

In WSL the launcher stages Electron on Windows-local storage under
`%LOCALAPPDATA%\OfflineRift\HUD\<package hash>`. It reuses completed copies and caches
file hashes while source file sizes and modification times remain unchanged. Package
updates get a new directory. Model bundles remain in the repository's configured paths.

## Controls

- Ctrl+Alt+M: expand or collapse recommendation details.
- Ctrl+Alt+I: toggle interaction and mouse click-through.
- Ctrl+Alt+S: player and role settings.
- Ctrl+Alt+H: hide or show the HUD.

The tray provides the same actions. Settings expose the selected coach fingerprint,
all six policy algorithms, training support, development gates, collapse and validation
abstentions. Select the role explicitly when the game does not report it. Conflicting
role destinations are withheld without inventing a replacement. Scores are neither
calibrated confidence nor win probabilities; win-rate improvement remains unproven.

## Runtime and performance

Python owns observation, inference and persistence. Electron makes one frame request per
backend refresh interval and broadcasts it to the overlay and settings through a narrow
preload bridge. Context changes invalidate older in-flight reads. The renderer has no
HTTP polling, WebSocket client, simulation controls or standalone browser launch mode.
The local HTTP server is internal to the desktop host and serves the compiled renderer.

Payloads coalesce once per animation frame. Stable recommendation objects retain identity;
memoized cards ignore telemetry-only updates. The objective countdown updates once per
second and details mount only when opened. Hidden windows use Electron's default background
throttling. Small CPU inference batches use a single Torch thread. These choices reduce
avoidable work; they do not guarantee a frame rate or CPU budget on another machine.

## Verification

```bash
npm test
npm run format:check
npm run build
RIFT_TEST_BROWSER=/absolute/path/to/chromium npm run test:browser
```

Renderer automation uses only synthetic fixtures under `tests/`; it is not a browser
product. Python compiled-renderer checks use the local Playwright installation and
Node.js 22.12+ (`RIFT_TEST_NODE` can override the Node path). The optional native test
runs in WSL when `RIFT_TEST_WINDOWS_HUD` points to the Windows executable. It checks
window creation, IPC, refresh request count and actual Windows process exit after Ctrl+C
using a synthetic localhost backend, without contacting the game API.

The superseded browser preview and WebSocket/library code were archived with checksums
under local `data/archive/hud-browser-*.zip`. Real model routing has been checked with
synthetic observations. Live-game overlay behavior and prospective outcomes remain untested.
