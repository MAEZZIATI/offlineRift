const {
  app,
  BrowserWindow,
  globalShortcut,
  ipcMain,
  screen,
  Tray,
  Menu,
  nativeImage,
  session,
} = require("electron");
const path = require("node:path");
const fs = require("node:fs");

function argument(name, fallback) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline !== undefined) return inline.slice(prefix.length);
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
}
const backend = new URL(argument("--backend", "http://127.0.0.1:8765"));
if (
  backend.protocol !== "http:" ||
  !["127.0.0.1", "localhost"].includes(backend.hostname) ||
  backend.username ||
  backend.password ||
  backend.pathname !== "/" ||
  backend.search ||
  backend.hash
)
  throw Error("Expected a loopback HUD backend");
const position = argument("--position", "top-right");
if (!["top-right", "top-left"].includes(position)) throw Error("Invalid overlay position");
let overlay,
  settings,
  tray,
  timer,
  lastFrame = null;
let interactive = false,
  stopping = false,
  pointerOver = false;
let generation = 0,
  frameRequests = 0;
const lifetime = new AbortController();
const registered = {};
const windows = () => [overlay, settings].filter((win) => win && !win.isDestroyed());
const broadcast = (channel, value) =>
  windows().forEach((win) => win.webContents.send(channel, value));
function trusted(event) {
  return (
    windows().some(
      (win) => event.sender === win.webContents && event.senderFrame === win.webContents.mainFrame,
    ) && new URL(event.senderFrame.url).origin === backend.origin
  );
}
function secure(win) {
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (event, target) => {
    if (new URL(target).origin !== backend.origin) event.preventDefault();
  });
  win.webContents.on("will-attach-webview", (event) => event.preventDefault());
}
const webPreferences = {
  preload: path.join(__dirname, "preload.cjs"),
  contextIsolation: true,
  nodeIntegration: false,
  sandbox: true,
};
function updateMouseCapture() {
  overlay.setIgnoreMouseEvents(!(interactive || pointerOver), { forward: true });
}
function interaction(value) {
  interactive = value;
  updateMouseCapture();
  overlay.setFocusable(value);
  if (value) {
    overlay.show();
    overlay.focus();
  }
  broadcast("rift:interaction", value);
}
function toggleDetails() {
  if (!overlay.isVisible()) overlay.showInactive();
  overlay.webContents.send("rift:toggle");
}
function openSettings() {
  if (settings && !settings.isDestroyed()) {
    settings.show();
    settings.focus();
    return;
  }
  settings = new BrowserWindow({
    title: "Offline Rift · Coach settings",
    width: 510,
    height: 780,
    minWidth: 390,
    minHeight: 500,
    backgroundColor: "#080c14",
    autoHideMenuBar: true,
    webPreferences,
  });
  secure(settings);
  settings.loadURL(`${backend.origin}/?settings=1`);
  settings.on("closed", () => {
    settings = null;
  });
}
async function request(route, context) {
  const requestedGeneration = generation;
  frameRequests++;
  const response = await fetch(new URL(route, backend), {
    signal: AbortSignal.timeout(12000),
    redirect: "error",
    ...(context
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(context),
        }
      : {}),
  });
  if (!response.ok) throw Error(`HUD backend returned ${response.status}`);
  const frame = await response.json();
  if (frame.schema !== "macro-hud-frame-v1") throw Error("Incompatible HUD backend");
  if (stopping || requestedGeneration !== generation) return null;
  if (
    lastFrame?.session_id === frame.session_id &&
    lastFrame.context_version > frame.context_version
  )
    return null;
  lastFrame = frame;
  broadcast("rift:frame", frame);
  return frame;
}
async function poll() {
  if (stopping) return;
  const requestedGeneration = generation;
  let delay = 1000;
  try {
    const frame = await request("/api/hud");
    if (frame) delay = refreshDelay(frame);
  } catch {
    if (requestedGeneration === generation) {
      lastFrame = null;
      broadcast("rift:frame", null);
    }
  }
  if (!stopping && requestedGeneration === generation) timer = setTimeout(poll, delay);
}
function refreshDelay(frame) {
  return Number.isFinite(frame.valid_for_ms)
    ? Math.max(250, Math.min(60000, frame.valid_for_ms))
    : 1000;
}
async function watchBackend() {
  try {
    const response = await fetch(new URL("/api/lifecycle", backend), { signal: lifetime.signal });
    if (!response.ok || !response.body) throw Error("Backend lifetime unavailable");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    while (!stopping) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const lines = pending.split("\n");
      pending = lines.pop();
      if (lines.includes("shutdown")) break;
    }
  } catch (error) {
    if (!stopping) console.error("HUD backend lifetime ended:", error.message);
  } finally {
    if (!stopping) app.quit();
  }
}
function place() {
  const area = screen.getPrimaryDisplay().workArea;
  const width = Math.min(414, area.width);
  overlay.setBounds({
    x: position === "top-left" ? area.x : area.x + area.width - width,
    y: area.y,
    width,
    height: Math.min(760, area.height),
  });
}
if (!app.requestSingleInstanceLock()) app.quit();
else
  app
    .whenReady()
    .then(async () => {
      void watchBackend();
      session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) =>
        callback(false),
      );
      session.defaultSession.setPermissionCheckHandler(() => false);
      session.defaultSession.webRequest.onBeforeRequest((details, callback) => {
        const url = new URL(details.url);
        callback({ cancel: url.origin !== backend.origin && url.protocol !== "devtools:" });
      });
      overlay = new BrowserWindow({
        title: "The Macro HUD",
        width: 414,
        height: 760,
        transparent: true,
        frame: false,
        resizable: false,
        hasShadow: false,
        skipTaskbar: true,
        focusable: false,
        alwaysOnTop: true,
        show: false,
        webPreferences,
      });
      overlay.setAlwaysOnTop(true, "screen-saver");
      secure(overlay);
      place();
      overlay.setIgnoreMouseEvents(true, { forward: true });
      overlay.on("blur", () => {
        if (interactive) interaction(false);
      });
      overlay.on("closed", () => app.quit());
      screen.on("display-metrics-changed", place);
      ipcMain.on("rift:ready", (event) => {
        if (trusted(event)) {
          event.sender.send("rift:frame", lastFrame);
          event.sender.send("rift:interaction", interactive);
        }
      });
      ipcMain.on("rift:settings", (event) => {
        if (trusted(event)) openSettings();
      });
      ipcMain.on("rift:pointer", (event, value) => {
        if (trusted(event) && event.sender === overlay.webContents && typeof value === "boolean") {
          pointerOver = value;
          updateMouseCapture();
        }
      });
      ipcMain.handle("rift:configure", async (event, context) => {
        if (!trusted(event)) throw Error("Untrusted HUD window");
        generation++;
        clearTimeout(timer);
        lastFrame = null;
        broadcast("rift:frame", null);
        try {
          const frame = await request("/api/context", context);
          if (!stopping) timer = setTimeout(poll, frame ? refreshDelay(frame) : 1000);
          return frame;
        } catch (error) {
          if (!stopping) timer = setTimeout(poll, 1000);
          throw error;
        }
      });
      const shortcuts = {
        "CommandOrControl+Alt+M": toggleDetails,
        "CommandOrControl+Alt+I": () => interaction(!interactive),
        "CommandOrControl+Alt+S": openSettings,
        "CommandOrControl+Alt+H": () => {
          if (overlay.isVisible()) {
            pointerOver = false;
            interaction(false);
            overlay.hide();
          } else overlay.showInactive();
        },
      };
      for (const [key, callback] of Object.entries(shortcuts)) {
        registered[key] = globalShortcut.register(key, callback);
        if (!registered[key]) console.error(`HUD shortcut unavailable: ${key}. Use the tray menu.`);
      }
      const pixels = Buffer.alloc(32 * 32 * 4);
      for (let y = 5; y < 27; y++)
        for (let x = 5; x < 27; x++)
          if (x < 9 || x > 22 || Math.abs(x - 16) < (19 - y) / 2) {
            const offset = (y * 32 + x) * 4;
            pixels[offset] = 255;
            pixels[offset + 1] = 243;
            pixels[offset + 3] = 255;
          }
      tray = new Tray(nativeImage.createFromBitmap(pixels, { width: 32, height: 32 }));
      tray.setToolTip("Offline Rift · The Macro HUD");
      tray.setContextMenu(
        Menu.buildFromTemplate([
          { label: "Coach settings", click: openSettings },
          { label: "Show HUD", click: () => overlay.showInactive() },
          { label: "Interact / click through", click: () => interaction(!interactive) },
          { label: "Toggle details", click: shortcuts["CommandOrControl+Alt+M"] },
          { type: "separator" },
          { label: "Quit HUD", click: () => app.quit() },
        ]),
      );
      tray.on("double-click", openSettings);
      app.on("second-instance", () => overlay.showInactive());
      await overlay.loadURL(`${backend.origin}/?position=${position}`);
      overlay.showInactive();
      void poll();
      const report = argument("--smoke-report", null);
      if (report)
        setTimeout(async () => {
          const expanded = () => overlay.webContents.executeJavaScript(
            "document.querySelector('.hud-expand')?.getAttribute('aria-expanded') === 'true'",
          );
          const waitFor = async (read, expected) => {
            for (let attempt = 0; attempt < 80; attempt++) {
              if (await read() === expected) return true;
              await new Promise(resolve => setTimeout(resolve, 25));
            }
            return false;
          };
          toggleDetails();
          const shortcutOpens = await waitFor(expanded, true);
          toggleDetails();
          const shortcutCloses = await waitFor(expanded, false);
          const point = await overlay.webContents.executeJavaScript(
            "(() => {const r = document.querySelector('.hud-expand').getBoundingClientRect(); return {x: Math.round(r.x + r.width/2), y: Math.round(r.y+r.height/2)};})()",
          );
          overlay.webContents.sendInputEvent({type: "mouseMove", ...point});
          const hoverActivates = await waitFor(() => pointerOver, true);
          const click = () => {
            overlay.webContents.sendInputEvent({type:"mouseDown", button:"left", clickCount:1, ...point});
            overlay.webContents.sendInputEvent({type:"mouseUp", button:"left", clickCount:1, ...point});
          };
          click();
          const clickOpens = await waitFor(expanded, true);
          click();
          const clickCloses = await waitFor(expanded, false);
          overlay.webContents.sendInputEvent({type:"mouseMove",x:1,y:overlay.getContentBounds().height-2});
          const transparentPassThroughRestored = await waitFor(() => pointerOver, false);
          const renderer = await overlay.webContents.executeJavaScript(
            "({title: document.querySelector('.hud-empty h2')?.textContent, phase: document.querySelector('.macro-hud')?.dataset.phase, hasHud: !!document.querySelector('[data-macro-hud]'), hasBridge: !!window.riftDesktop, nodeExposed: typeof require !== 'undefined'})",
          );
          fs.writeFileSync(
            report,
            JSON.stringify(
              {
                platform: process.platform,
                pid: process.pid,
                alwaysOnTop: overlay.isAlwaysOnTop(),
                visible: overlay.isVisible(),
                focusable: overlay.isFocusable(),
                interactive,
                shortcuts: registered,
                frameRequests,
                expansion: {shortcutOpens, shortcutCloses, hoverActivates, clickOpens, clickCloses,
                  transparentPassThroughRestored},
                metrics: app.getAppMetrics(),
                model: lastFrame?.model,
                status: lastFrame?.status,
                renderer,
              },
              null,
              2,
            ),
          );
          if (!process.argv.includes("--smoke-keep-open")) app.quit();
        }, 6500);
    })
    .catch((error) => {
      console.error(error);
      app.exit(1);
    });
app.on("before-quit", () => {
  stopping = true;
  lifetime.abort();
  clearTimeout(timer);
  globalShortcut.unregisterAll();
  if (tray) tray.destroy();
});
app.on("window-all-closed", () => app.quit());
