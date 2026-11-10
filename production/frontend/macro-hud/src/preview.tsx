import { Profiler, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowDownRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  Eye,
  EyeOff,
  FlaskConical,
  Layers3,
  Maximize2,
  Radio,
  RotateCcw,
  SlidersHorizontal,
  WifiOff,
  Zap,
} from "lucide-react";
import { MacroHUD } from "./MacroHUD";
import { createHudStore, useHudValue } from "./store";
import { connectIPC, connectWebSocket, type HudBridge } from "./transport";
import { clockSeconds, PLAYER_ROLES, type PlayerRole, type Urgency } from "./types";
import { ROLE_EXAMPLES, SCENARIOS } from "@fixtures/scenarios";
import "./preview.css";

declare global {
  interface Window {
    macroHudBridge?: HudBridge;
  }
}
const search = new URLSearchParams(window.location.search);
const endpoint = search.get("ws");
const overlay = search.get("overlay") === "1";
const external = !!endpoint || !!window.macroHudBridge || overlay;

function formatClock(seconds: number) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function Preview() {
  const store = useMemo(() => createHudStore(), []);
  const [scenario, setScenario] = useState("baron");
  const [role, setRole] = useState<PlayerRole>("JUNGLE");
  const [urgency, setUrgency] = useState<Urgency>("critical");
  const [visible, setVisible] = useState(true);
  const [position, setPosition] = useState<"top-right" | "top-left">("top-right");
  const [paused, setPaused] = useState(false);
  const [stress, setStress] = useState(false);
  const [sourceError, setSourceError] = useState("");
  const performanceText = useRef<HTMLSpanElement>(null);
  const durations = useRef<number[]>([]);
  const expanded = useHudValue(store, (state) => state.expanded);
  const lastPayload = useRef<unknown>(null);
  const onRender = useCallback((_id: string, _phase: string, actualDuration: number) => {
    durations.current.push(actualDuration);
    if (durations.current.length > 120) durations.current.shift();
  }, []);

  useEffect(() => {
    if (external) return;
    if (paused) {
      store.setConnection("offline");
      return;
    }
    store.beginSession();
    store.setConnection("connected");
    const started = performance.now();
    const base = SCENARIOS[scenario];
    const send = () => {
      const elapsed = Math.floor((performance.now() - started) / 1000);
      const original = base.macro?.timer ? clockSeconds(base.macro.timer) : 0;
      const payload = {
        ...base,
        urgency,
        timestamp: formatClock(clockSeconds(base.timestamp) + elapsed),
        macro: base.macro
          ? { ...base.macro, timer: formatClock(Math.max(0, original - elapsed)) }
          : null,
        role: base.role
          ? {
              ...base.role,
              player_role: role,
              ...(scenario === "baron" ? ROLE_EXAMPLES[role] : {}),
            }
          : null,
      };
      lastPayload.current = payload;
      store.ingest(payload);
    };
    send();
    const timer = window.setInterval(send, 1000);
    return () => clearInterval(timer);
  }, [scenario, role, urgency, paused, store]);

  useEffect(() => {
    if (!external) return;
    document.body.classList.toggle("preview-overlay-only", overlay);
    try {
      if (window.macroHudBridge) return connectIPC(store, window.macroHudBridge);
      if (endpoint) return connectWebSocket(store, endpoint);
    } catch {
      setSourceError(
        "The event source could not be opened. Use a valid WebSocket endpoint or the host IPC bridge.",
      );
      store.setConnection("offline");
    }
  }, [store]);

  useEffect(() => {
    if (!stress || external || paused) return;
    let ticks = 0;
    const timer = window.setInterval(() => {
      const data = lastPayload.current as typeof SCENARIOS.baron | null;
      if (data)
        store.ingest({
          ...data,
          telemetry: {
            ai_confidence: 75 + (ticks % 25),
            state_hash: `a8${(++ticks).toString(16).padStart(4, "0")}`,
          },
        });
    }, 8);
    return () => clearInterval(timer);
  }, [stress, paused, store]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const values = [...durations.current].sort((a, b) => a - b);
      if (performanceText.current)
        performanceText.current.textContent = values.length
          ? `${values[Math.floor((values.length - 1) * 0.95)].toFixed(2)} ms`
          : "—";
    }, 500);
    return () => {
      clearInterval(timer);
      store.dispose();
    };
  }, [store]);

  const hud = visible && (
    <Profiler id="macro-hud" onRender={onRender}>
      <MacroHUD
        store={store}
        position={position}
        metricsMode={external ? "unverified" : "demo"}
        enableKeyboard={!window.macroHudBridge?.onToggleDetails}
      />
    </Profiler>
  );
  if (overlay) return <>{hud}</>;

  return (
    <main className="preview" data-expanded={expanded}>
      <div className="preview-grain" aria-hidden="true" />
      <div className="preview-topbar">
        <a className="preview-brand" href="./">
          <span className="preview-brand-mark">
            <Layers3 size={19} strokeWidth={1.5} />
          </span>
          OFFLINE RIFT
          <span className="preview-brand-divider" />
          LABS
        </a>
        <span className="preview-build">
          INTERFACE STUDY <span>01 / MACRO HUD</span>
        </span>
      </div>
      <svg className="preview-map" viewBox="0 0 900 760" fill="none" aria-hidden="true">
        <defs>
          <radialGradient id="map-glow">
            <stop stopColor="#124453" stopOpacity=".24" />
            <stop offset="1" stopColor="#080c14" stopOpacity="0" />
          </radialGradient>
        </defs>
        <ellipse cx="430" cy="380" rx="430" ry="360" fill="url(#map-glow)" />
        <g stroke="#25404a" strokeWidth="1">
          <path d="m180 560 3-360 365-4 190 190-1 167-364 5Z" />
          <path d="m183 200 188 188 366-2M180 560l191-172 365 165M371 388l2 170M550 196 371 388" />
          <path strokeDasharray="3 9" d="m125 608 3-447 444-3 210 210-1 243-442 3Z" />
          <circle cx="371" cy="388" r="118" />
          <circle cx="371" cy="388" r="159" strokeDasharray="2 9" />
          <path d="m267 353 58-38 89 21 58 64-40 69-84-5-62-51Z" />
        </g>
        <g fill="#42656b">
          <circle cx="183" cy="200" r="4" />
          <circle cx="737" cy="553" r="4" />
          <circle cx="550" cy="196" r="3" />
          <circle cx="373" cy="558" r="3" />
        </g>
        <g stroke="#00f3ff" strokeOpacity=".22">
          <circle cx="492" cy="285" r="19" />
          <path d="M466 285h9m34 0h9m-26-26v9m0 34v9" />
        </g>
      </svg>
      <section className="preview-intro">
        <div className="preview-eyebrow">
          <span />
          LESS INTERFACE. MORE AWARENESS.
        </div>
        <h1>
          Your next move.
          <br />
          <span>At a glance.</span>
        </h1>
        <p>
          A team direction. A role-specific action.
          <br />A quieter way to keep the bigger picture.
        </p>
        <div className="preview-features">
          <span>
            <Check size={12} />
            Compact by default
          </span>
          <span>
            <Check size={12} />
            Context on demand
          </span>
        </div>
        <div className="preview-guide">
          <span className="preview-guide-line" />
          <span>YOUR OVERLAY LIVES HERE</span>
          <ArrowUpRight size={16} />
        </div>
      </section>
      {hud}
      <section className="preview-controls" aria-label="HUD preview controls">
        <div className="preview-controls-heading">
          <span>
            <FlaskConical size={13} />
            {external ? "EXTERNAL SOURCE" : "SIMULATION CONTROLS"}
          </span>
          <span className="preview-local">
            <span />
            {external ? "UNVERIFIED METRICS" : "SYNTHETIC DATA ONLY"}
          </span>
        </div>
        {sourceError && (
          <p role="alert" className="preview-source-error">
            {sourceError}
          </p>
        )}
        {!external && (
          <>
            <div className="preview-control-row">
              <label className="preview-select-label">
                SCENARIO
                <div className="preview-select-wrap">
                  <select
                    aria-label="Scenario"
                    value={scenario}
                    onChange={(event) => {
                      const value = event.target.value;
                      setScenario(value);
                      setUrgency(SCENARIOS[value].urgency);
                      setPaused(false);
                    }}
                  >
                    <option value="baron">Baron contest</option>
                    <option value="drake">Drake preparation</option>
                    <option value="crossmap">Cross-map pressure</option>
                    <option value="stall">Turtle & stall</option>
                    <option value="conflict">Conflicting directions</option>
                    <option value="unsupported">Missing support</option>
                  </select>
                  <ChevronDown size={12} />
                </div>
              </label>
              <div className="preview-urgency">
                <span className="preview-control-label">URGENCY</span>
                <div className="preview-segmented">
                  {(["normal", "high", "critical"] as const).map((level) => (
                    <button
                      type="button"
                      key={level}
                      aria-pressed={urgency === level}
                      data-tone={level}
                      onClick={() => setUrgency(level)}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="preview-role-row">
              <span className="preview-control-label">YOUR ROLE</span>
              <div className="preview-role-buttons">
                {PLAYER_ROLES.map((value) => (
                  <button
                    key={value}
                    type="button"
                    aria-pressed={role === value}
                    onClick={() => setRole(value)}
                  >
                    {
                      {
                        TOP: "TOP",
                        JUNGLE: "JUNGLE",
                        MIDDLE: "MID",
                        BOTTOM: "ADC",
                        UTILITY: "SUPPORT",
                      }[value]
                    }
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
        <div className="preview-tools">
          <button type="button" onClick={() => store.toggleDetails()} aria-pressed={expanded}>
            <Maximize2 size={13} />
            {expanded ? "Collapse" : "Expand"}
            <kbd>TAB</kbd>
          </button>
          <button type="button" onClick={() => setVisible(!visible)} aria-pressed={!visible}>
            {visible ? <EyeOff size={13} /> : <Eye size={13} />}
            {visible ? "Hide" : "Show"}
          </button>
          <button
            type="button"
            onClick={() => setPosition(position === "top-right" ? "top-left" : "top-right")}
            aria-label="Switch overlay corner"
          >
            <ArrowDownRight size={13} />
            Corner
          </button>
          {!external && (
            <>
              <button type="button" onClick={() => setPaused(!paused)} aria-pressed={paused}>
                {paused ? <RotateCcw size={13} /> : <WifiOff size={13} />}
                {paused ? "Reconnect" : "Disconnect"}
              </button>
              <button
                type="button"
                onClick={() => setStress(!stress)}
                aria-pressed={stress}
                disabled={paused}
              >
                <Zap size={13} />
                {stress ? "Stop burst" : "120 Hz burst"}
              </button>
            </>
          )}
        </div>
        <div className="preview-controls-footer">
          <span>
            <Radio size={11} />
            {external
              ? "External event adapter"
              : stress
                ? "High-frequency telemetry"
                : "1 Hz simulated observations"}
          </span>
          <span>
            REACT P95 <strong ref={performanceText}>—</strong>
          </span>
        </div>
      </section>
      <footer className="preview-footer">
        <span>
          <SlidersHorizontal size={12} />
          BUILT TO STAY OUT OF YOUR WAY
        </span>
        <p>
          Local component preview · No game client attached · Sample percentages are illustrative
        </p>
        <span className="preview-version">HUD / v0.1</span>
      </footer>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<Preview />);
