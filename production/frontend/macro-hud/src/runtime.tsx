import { useCallback, useEffect, useRef, useState } from "react";
import { Settings, X } from "lucide-react";
import { MacroHUD } from "./MacroHUD";
import { createHudStore } from "./store";
import { PLAYER_ROLES } from "./types";
import "./runtime.css";

interface Frame {
  schema: "macro-hud-frame-v1";
  session_id: string;
  context_version: number;
  status: string;
  message: string;
  detail: string;
  hud: unknown;
  context: { team: number; actor: string; role: string | null };
  players: { id: string; champion: string; team: number; role: string }[];
  model: {
    display_name: string;
    fingerprint: string;
    policy_count: number;
    policies: Record<
      string,
      {
        algorithm: string;
        candidate: string;
        supported_actions: number;
        development_gate_passed: boolean | null;
        diagnostics: { collapsed: boolean | null; abstentions: number | null };
      }
    >;
  };
}
interface DesktopBridge {
  onFrame(callback: (frame: Frame | null) => void): () => void;
  onToggleDetails(callback: () => void): () => void;
  onInteraction(callback: (active: boolean) => void): () => void;
  configure(context: Frame["context"]): Promise<Frame>;
  openSettings(): void;
}
const bridge = (window as Window & { riftDesktop?: DesktopBridge }).riftDesktop;
const store = createHudStore();
const settingsWindow = new URLSearchParams(location.search).has("settings");
const position =
  new URLSearchParams(location.search).get("position") === "top-left" ? "top-left" : "top-right";

export function Runtime() {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [settings, setSettings] = useState(settingsWindow);
  const [interactive, setInteractive] = useState(!bridge);
  const [error, setError] = useState("");
  const [updating, setUpdating] = useState(false);
  const selecting = useRef(false);
  const receiveFrame = useRef<(next: Frame | null) => void>(() => {});
  useEffect(() => {
    let session = "";
    let latestVersion = -1;
    let latestSession = "";
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const receive = (next: Frame | null) => {
      if (stopped || selecting.current) return;
      if (!next || next.schema !== "macro-hud-frame-v1") {
        store.setConnection("offline");
        return;
      }
      if (next.session_id === latestSession && next.context_version < latestVersion) return;
      latestSession = next.session_id;
      latestVersion = next.context_version;
      const identity = `${next.session_id}/${next.context_version}`;
      if (identity !== session) {
        store.beginSession();
        session = identity;
      }
      setFrame(next);
      if (next.status === "live" && next.hud) {
        if (!store.ingest(next.hud)) store.setConnection("offline");
      } else store.setNotice(next.message, next.detail);
    };
    receiveFrame.current = receive;
    const poll = async () => {
      try {
        const response = await fetch("/api/hud", { signal: controller.signal, cache: "no-store" });
        if (!response.ok) throw Error("HUD backend unavailable");
        receive(await response.json());
      } catch {
        if (!stopped) store.setConnection("reconnecting");
      }
      if (!stopped) timer = setTimeout(poll, 1000);
    };
    const cleanup = bridge
      ? [
          bridge.onFrame(receive),
          bridge.onToggleDetails(() => store.toggleDetails()),
          bridge.onInteraction(setInteractive),
        ]
      : [];
    if (!bridge) void poll();
    return () => {
      stopped = true;
      controller.abort();
      clearTimeout(timer);
      cleanup.forEach((fn) => fn());
    };
  }, []);
  const configure = useCallback(async (context: Frame["context"]) => {
    if (selecting.current) return;
    selecting.current = true;
    setUpdating(true);
    setError("");
    // Withdraw the previous player's recommendation immediately, before the request returns.
    store.setNotice("Updating your player", "Waiting for the selected role's next read.");
    try {
      let next: Frame;
      if (bridge) next = await bridge.configure(context);
      else {
        const result = await fetch("/api/context", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(context),
        });
        if (!result.ok) throw Error("Selection was rejected. Refresh the observed player list.");
        next = await result.json();
      }
      selecting.current = false;
      receiveFrame.current(next);
    } catch (cause) {
      setError(String(cause));
      store.setConnection("offline");
    } finally {
      selecting.current = false;
      setUpdating(false);
    }
  }, []);
  if (settings)
    return (
      <main className="runtime-settings">
        <header>
          <div>
            <small>OFFLINE RIFT</small>
            <h1>Coach settings</h1>
          </div>
          {!settingsWindow && (
            <button aria-label="Close settings" onClick={() => setSettings(false)}>
              <X size={18} />
            </button>
          )}
        </header>
        <p className="runtime-muted">
          The team direction and your role specialist use the selected six-policy coach.
        </p>
        {frame ? (
          <>
            <section className="runtime-model">
              <small>SELECTED COACH · {frame.model.policy_count} POLICIES</small>
              <strong>{frame.model.display_name}</strong>
              <code title={frame.model.fingerprint}>{frame.model.fingerprint.slice(0, 16)}</code>
            </section>
            <label>
              Player
              <select
                disabled={updating}
                value={frame.context.actor}
                onChange={(event) =>
                  void configure({ ...frame.context, actor: event.target.value, role: null })
                }
              >
                <option value="active">My active player</option>
                <option value="team">Team direction only</option>
                {frame.players.map((player) => (
                  <option key={player.id} value={player.id}>
                    {player.champion} · {player.id} · {player.team === 100 ? "Blue" : "Red"}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Your role
              <select
                disabled={updating || frame.context.actor === "team"}
                value={frame.context.role ?? ""}
                onChange={(event) =>
                  void configure({ ...frame.context, role: event.target.value || null })
                }
              >
                <option value="">Use the game's reported role</option>
                {PLAYER_ROLES.map((role) => (
                  <option key={role} value={role}>
                    {role === "BOTTOM"
                      ? "ADC"
                      : role === "UTILITY"
                        ? "SUPPORT"
                        : role === "MIDDLE"
                          ? "MID"
                          : role}
                  </option>
                ))}
              </select>
            </label>
            {frame.context.actor === "team" && (
              <label>
                Team
                <select
                  disabled={updating}
                  value={frame.context.team}
                  onChange={(event) =>
                    void configure({ ...frame.context, team: Number(event.target.value) })
                  }
                >
                  <option value={100}>Blue</option>
                  <option value={200}>Red</option>
                </select>
              </label>
            )}
            <section className="runtime-policies">
              {Object.entries(frame.model.policies).map(([scope, policy]) => (
                <div key={scope}>
                  <span>{scope}</span>
                  <strong>{policy.algorithm.toUpperCase()}</strong>
                  <small>{policy.supported_actions ?? "?"} supported actions</small>
                  <small>
                    {policy.development_gate_passed === true
                      ? "Development gate passed"
                      : policy.development_gate_passed === false
                        ? "Development gate failed"
                        : "Development gate unavailable"}
                  </small>
                  <small>
                    {policy.diagnostics?.collapsed === true
                      ? "Collapsed predictions"
                      : policy.diagnostics?.collapsed === false
                        ? "No collapse detected"
                        : "Collapse diagnostic unavailable"}
                    {policy.diagnostics?.abstentions != null &&
                      ` · ${policy.diagnostics.abstentions} validation abstentions`}
                  </small>
                </div>
              ))}
            </section>
          </>
        ) : (
          <p>Connecting to the selected coach…</p>
        )}
        {error && <p role="alert">{error}</p>}
        <p className="runtime-muted">
          If League does not report your role, select it here. Advice expires between measured
          reads. Conflicting role destinations are withheld.
        </p>
        <div className="runtime-shortcuts">
          <p>
            <kbd>Ctrl Alt M</kbd> Details
          </p>
          <p>
            <kbd>Ctrl Alt I</kbd> Interact / click through
          </p>
          <p>
            <kbd>Ctrl Alt S</kbd> Settings
          </p>
          <p>
            <kbd>Ctrl Alt H</kbd> Hide / show
          </p>
        </div>
        <p className="runtime-muted">
          Use League in borderless or windowed mode. These experimental models do not establish a
          win-rate improvement.
        </p>
      </main>
    );
  return (
    <>
      <MacroHUD store={store} position={position} />
      <div className="runtime-toolbar" data-position={position}>
        <span title={frame?.model.fingerprint}>
          {frame?.model.display_name ?? "Connecting to coach"}
          {interactive && bridge ? " · INTERACTIVE" : ""}
        </span>
        <button
          aria-label="Open coach settings"
          title="Coach settings · Ctrl+Alt+S"
          onClick={() => (bridge ? bridge.openSettings() : setSettings(true))}
        >
          <Settings size={13} />
        </button>
      </div>
    </>
  );
}
