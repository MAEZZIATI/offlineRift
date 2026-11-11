import { useCallback, useEffect, useRef, useState } from "react";
import { Settings } from "lucide-react";
import { MacroHUD } from "./MacroHUD";
import { createHudStore } from "./store";
import { PLAYER_ROLES } from "./types";
import "./runtime.css";

export interface Frame {
  schema: "macro-hud-frame-v1";
  session_id: string;
  context_version: number;
  status: string;
  message: string;
  detail: string;
  hud: unknown;
  context: { team: number; actor: string; role: string | null };
  perspective?: {
    team: number | null;
    side: "blue" | "red" | null;
    actor: string | null;
    source: "reported_player" | "selected_team" | "unresolved";
  };
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
export interface DesktopBridge {
  onFrame(callback: (frame: Frame | null) => void): () => void;
  onToggleDetails(callback: () => void): () => void;
  onInteraction(callback: (active: boolean) => void): () => void;
  configure(context: Frame["context"]): Promise<Frame>;
  openSettings(): void;
  setPointerOver?(value: boolean): void;
}
const desktopBridge = (window as Window & { riftDesktop?: DesktopBridge }).riftDesktop;
const store = createHudStore();
const settingsWindow = new URLSearchParams(location.search).has("settings");
const position =
  new URLSearchParams(location.search).get("position") === "top-left" ? "top-left" : "top-right";

export function Runtime({
  bridge = desktopBridge,
  settings = settingsWindow,
}: {
  bridge?: DesktopBridge;
  settings?: boolean;
} = {}) {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [interactive, setInteractive] = useState(false);
  const [error, setError] = useState("");
  const [updating, setUpdating] = useState(false);
  const selecting = useRef(false);
  const receiveFrame = useRef<(next: Frame | null) => void>(() => {});
  const side = frame?.perspective?.side;
  const sideLabel = side === "red" ? "Red side" : side === "blue" ? "Blue side" : "Side unresolved";
  useEffect(() => {
    if (settings || !bridge?.setPointerOver) return;
    let over = false;
    const setOver = (value: boolean) => {
      if (over !== value) {
        over = value;
        bridge.setPointerOver?.(value);
      }
    };
    const move = (event: MouseEvent) => setOver(
      event.target instanceof Element && !!event.target.closest(
        ".macro-hud, .hud-details-panel:not([hidden]), .runtime-toolbar",
      ),
    );
    const leave = () => setOver(false);
    window.addEventListener("mousemove", move, {passive: true});
    window.addEventListener("mouseleave", leave);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseleave", leave);
      leave();
    };
  }, [bridge, settings]);
  useEffect(() => {
    let session = "";
    let latestVersion = -1;
    let latestSession = "";
    let stopped = false;
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
    const cleanup = bridge
      ? [
          bridge.onFrame(receive),
          bridge.onToggleDetails(() => store.toggleDetails()),
          bridge.onInteraction(setInteractive),
        ]
      : [];
    return () => {
      stopped = true;
      cleanup.forEach((fn) => fn());
    };
  }, [bridge]);
  const configure = useCallback(
    async (context: Frame["context"]) => {
      if (selecting.current) return;
      selecting.current = true;
      setUpdating(true);
      setError("");
      // Withdraw the previous player's recommendation immediately, before the request returns.
      store.setNotice("Updating your player", "Waiting for the selected role's next read.");
      try {
        if (!bridge) throw Error("The desktop host is unavailable");
        const next = await bridge.configure(context);
        selecting.current = false;
        receiveFrame.current(next);
      } catch (cause) {
        setError(String(cause));
        store.setConnection("offline");
      } finally {
        selecting.current = false;
        setUpdating(false);
      }
    },
    [bridge],
  );
  if (!bridge)
    return <main className="runtime-settings">Launch the desktop HUD with rift hud.</main>;
  if (settings)
    return (
      <main className="runtime-settings">
        <header>
          <div>
            <small>OFFLINE RIFT</small>
            <h1>Coach settings</h1>
          </div>
        </header>
        <p className="runtime-muted">
          The team direction and your role specialist use the selected six-policy coach.
        </p>
        <p className="runtime-side" data-side={side ?? "unknown"}>
          {sideLabel} · {frame?.perspective?.source === "reported_player" ? "Reported by League"
            : frame?.perspective?.source === "selected_team" ? "Selected team view" : "Waiting for player identity"}
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
                aria-label="Player"
                disabled={updating}
                value={frame.context.actor}
                onChange={(event) =>
                  void configure({ ...frame.context,
                    team: frame.perspective?.team ?? frame.context.team,
                    actor: event.target.value, role: null })
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
                aria-label="Your role"
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
                  aria-label="Team"
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
      <MacroHUD store={store} position={position} enableKeyboard={interactive} detailsShortcut="Control+Alt+M" />
      <div className="runtime-toolbar" data-position={position}>
        <span title={frame?.model.fingerprint}>
          {frame?.model.display_name ?? "Connecting to coach"}
          {interactive && bridge ? " · INTERACTIVE" : ""}
        </span>
        <span className="runtime-side" data-side={side ?? "unknown"} aria-label="Team perspective">
          {sideLabel}
        </span>
        <button
          aria-label="Open coach settings"
          title="Coach settings · Ctrl+Alt+S"
          onClick={() => bridge.openSettings()}
        >
          <Settings size={13} />
        </button>
      </div>
    </>
  );
}
