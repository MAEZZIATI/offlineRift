# The Macro HUD

A React / TypeScript / Tailwind / Lucide overlay with a 366 × 268 px compact footprint.
The team direction, objective countdown and selected role's action remain visible;
details expand underneath without moving the compact widget. Default position is the
top-right corner, with a top-left option. The library does not reset the host page's
styles or background. All assets are bundled locally.

## Use the trained coach

From the repository root, serve the compiled HUD with the selected completed coach:

```bash
.venv/bin/rift hud --serve-only --actor active --role JUNGLE --interval 5
```

Open `http://127.0.0.1:8765`. `--model PATH` selects another completed six-policy
bundle. The maintained live adapter and inference controller supply both the team
proposal and selected role proposal, including persistence and conflict withholding.
Settings show the coach fingerprint, each policy's algorithm, supported action count,
development gate, collapse diagnostic and validation abstentions. Action support is
the count meeting the training support minimum; state feasibility and phase support
can further restrict serving. These diagnostics do not establish strategic utility.

To build the native host, run from this folder with Node.js 22.12+:

```bash
npm ci
npm run package:win
# Or, on Linux:
npm run package:linux
```

Then run `.venv/bin/rift hud --actor active --role JUNGLE` from the repository root.
The launcher finds the packaged host under `release/`; use `--desktop PATH` to
override it. In WSL, package the Windows host for the Windows game desktop.
Use League in borderless or windowed mode. The overlay starts click-through:
Ctrl+Alt+I toggles interaction, Ctrl+Alt+M details, Ctrl+Alt+S settings and Ctrl+Alt+H
visibility. The tray also provides these controls. Local HTTP and desktop IPC use
the same versioned frames and explicit player/role selection. No simulation runs in
the production entry point. Rebuild it with `npm run build` after source changes.

Real model loading and all role routes have been checked with synthetic observations.
Authenticated collection and live-game overlay behavior have not been exercised.

## Run the local preview

Requires Node.js 22.12+ and npm. From this folder:

```bash
npm ci
npm run dev
```

Open the printed loopback URL. Preview controls cover all five roles, three urgency
levels, withheld/missing support, disconnect/recovery, corner selection, hiding and
a 120 Hz telemetry burst. All example game data lives in `tests/fixtures/macro-hud`;
the preview is synthetic and is not attached to League of Legends.

The preview's React p95 readout uses actual development Profiler observations. It is
not network latency, paint time, host IPC latency, or a guarantee for another machine.
Production React builds disable Profiler timing. A local Chromium check measured
2.30 ms p95 during the final 120 Hz burst check.

## Embed the component

Build the reusable ESM library and declarations:

```bash
npm run build:lib
```

The output is `dist-lib/macro-hud.js`, `dist-lib/macro-hud.css` and `dist-lib/types/`.
React and its JSX runtime are externalized so the host supplies one React instance.
The final build measured approximately 8.6 KB gzipped JavaScript and 2.9 KB CSS;
the standalone preview additionally bundles React DOM and its simulation controls.

```tsx
import { createRoot } from "react-dom/client";
import { MacroHUD, createHudStore, connectWebSocket } from "@offline-rift/macro-hud";
import "@offline-rift/macro-hud/style.css";

const store = createHudStore({ staleAfterMs: 8000 });
const detach = connectWebSocket(store, "ws://127.0.0.1:9000/recommendations");
const root = createRoot(document.getElementById("overlay")!);
root.render(<MacroHUD store={store} position="top-right" />);

// On actual host/window teardown:
// detach(); root.unmount(); store.dispose();
```

`metricsMode` defaults to `unverified`: impact displays as **Impact unverified** and
confidence as **Uncalibrated**. The current Offline Rift models do not supply causal
win-rate deltas or calibrated confidence. `demo` exposes the requested sample numbers
with visible DEMO labeling. Use `validated` only for an external backend whose impact
and confidence metrics have independent supporting evidence. A field named
`winrate_impact` or `ai_confidence` alone is not that evidence.

The production Python HUD uses HTTP frames and a dedicated desktop IPC bridge.
The reusable component additionally supplies WebSocket and IPC consumers for other
hosts, without a WebSocket server. An external producer must emit a payload or a
`{ type: "recommendation", payload }` envelope. The provided user example is retained
in the test fixtures. `macro` and `role` may be null to represent missing support.

Additional supported fields:

- `coordination`: `aligned`, `independent`, `withheld_conflict` or `missing_support`.
  Omission defaults to independent proposals when both recommendations exist.
- `role.status`: `recommendation`, `withheld_conflict`, `insufficient_support`,
  or `respawning`. Conflicts suppress both role instructions and role details.
- `valid_for_ms`: integer 0–60,000. Zero immediately withdraws the advice; absence
  uses the store's expiry. Invalid messages never extend advice lifetime.
- Role aliases `MID`, `ADC`, `SUPPORT` normalize to `MIDDLE`, `BOTTOM`, `UTILITY`.

The store drops events older than the highest accepted game-clock timestamp. Messages
in the same second use transport arrival order; the wire schema has no sequence number.
Call `store.beginSession()` for a new match if the same transport remains connected;
new WebSocket connections reset automatically. This avoids interpreting delayed packets
as a new game. A valid heartbeat may refresh advice freshness but does not restart a
countdown whose game-clock deadline is unchanged.

## Custom Electron / Overwolf host integration

For the synthetic preview entry point, a transparent web view uses `?overlay=1`.
It starts waiting without simulation data.
`?overlay=1&ws=ws://127.0.0.1:9000/recommendations` connects to a producer. External
sources always use unverified metrics in this entry point. `?ws=...` without overlay
mode retains the surrounding inspection stage.

Alternatively expose a narrow preload bridge as `window.macroHudBridge`:

```ts
import { contextBridge, ipcRenderer } from "electron";

function subscribe(channel: string, receive: (payload: unknown) => void) {
  const listener = (_event: Electron.IpcRendererEvent, payload: unknown) => receive(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

contextBridge.exposeInMainWorld("macroHudBridge", {
  onRecommendation: (receive: (payload: unknown) => void) =>
    subscribe("macro-hud:recommendation", receive),
  onToggleDetails: (toggle: () => void) => subscribe("macro-hud:toggle-details", () => toggle()),
});
```

The host sends `webContents.send("macro-hud:recommendation", payload)`. Keep
`contextIsolation: true` and `nodeIntegration: false`. The library's `connectIPC`
also supports an optional `onConnection(callback)` subscription; each callback must
return its unsubscribe function. `MacroHUD` accepts `enableKeyboard={false}` when
the host forwards shortcuts, to avoid double toggles.

Tab toggles from the web view's body or focused HUD and focuses the expand control;
subsequent Tab presses navigate controls normally. The details panel is keyboard
focusable for scrolling, and Escape collapses it. Inputs, selects,
links, buttons, modified chords and repeated keys retain their own keyboard behavior.
The browser cannot observe Tab while League has focus: register the chosen shortcut
with the Electron/Overwolf host and forward its event. Consider the game's scoreboard
binding when selecting a global shortcut.

Native always-on-top behavior, click-through and window sizing belong to the host.
CSS pointer-events only affects this DOM; it does not make an OS window click-through.
Resize or apply native hit testing as appropriate. The supplied desktop host packages
an Electron application; no Overwolf manifest or game injection is supplied.

## Rendering and checks

Incoming events coalesce to one latest snapshot per animation frame. Equal macro/role
objects preserve identity, `useSyncExternalStore` subscriptions select stable slices,
and memoized cards do not render for telemetry-only updates. The countdown updates its
own text, while CSS animates its small ring. Details mount only when expanded.
Expansion updates immediately; only opacity/transform animate over 120 ms. Reduced
motion disables transitions. Reconnect attempts back off with a finite retry budget,
and teardown cancels sockets, queued frames, timers and subscriptions.

```bash
npm test
npm run format:check
npm run build
npm run build:lib
RIFT_TEST_BROWSER=/absolute/path/to/chromium npm run test:browser
```

All tests and synthetic fixtures are under the repository's `tests/` folder. The
browser suite owns a local Vite fixture server, closes it after testing, and never
contacts Riot or uses training data. Playwright is pinned for the repository's Ubuntu
20.04 test host; `RIFT_TEST_BROWSER` may point at your installed compatible Chromium.
The Python compiled-HUD browser checks also use this local Playwright installation
and Node.js 22.12+. Set `RIFT_TEST_NODE` if the Node executable is not on PATH.
