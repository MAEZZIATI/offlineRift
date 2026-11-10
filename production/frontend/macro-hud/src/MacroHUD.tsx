import { memo, useEffect, useId, useRef, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  ChevronDown,
  Compass,
  Crosshair,
  GitBranch,
  Layers3,
  Radio,
  Shield,
  Swords,
  Trees,
  TriangleAlert,
  Unplug,
  Waves,
} from "lucide-react";
import { useHudValue, type HudStore } from "./store";
import type { MetricsMode, PlayerRole } from "./types";
import "./hud.css";

const ROLE_LABEL: Record<PlayerRole, string> = {
  TOP: "TOP",
  JUNGLE: "JUNGLE",
  MIDDLE: "MID",
  BOTTOM: "ADC",
  UTILITY: "SUPPORT",
};
const ROLE_ICON = {
  TOP: Swords,
  JUNGLE: Trees,
  MIDDLE: Compass,
  BOTTOM: Crosshair,
  UTILITY: Shield,
};

export interface MacroHUDProps {
  store: HudStore;
  position?: "top-right" | "top-left";
  /** Default hides unsupported percentages. Demo is visibly marked; validated requires external evidence. */
  metricsMode?: MetricsMode;
  /** Only handles Tab in this web view. Global game hotkeys must be forwarded by the host. */
  enableKeyboard?: boolean;
  detailsShortcut?: "Tab" | "Control+Alt+M";
}

const Header = memo(function Header({ store, mode }: { store: HudStore; mode: MetricsMode }) {
  const phase = useHudValue(store, (state) => state.phase);
  const connection = useHudValue(store, (state) => state.connection);
  const timestamp = useHudValue(store, (state) => state.data?.timestamp ?? "--:--");
  return (
    <header className="hud-header flex items-center justify-between">
      <div className="flex items-center gap-2">
        <Layers3 size={13} className="hud-brand-icon" strokeWidth={1.8} />
        <span className="hud-wordmark">THE MACRO HUD</span>
      </div>
      <div className="flex items-center gap-2.5">
        {mode === "demo" && <span className="hud-demo-label">DEMO</span>}
        <span
          className="hud-source"
          data-live={phase === "live"}
          title={
            connection === "reconnecting"
              ? "Reconnecting"
              : phase === "live"
                ? "Receiving recommendations"
                : "No current recommendation"
          }
        >
          <span className="hud-source-dot" />
          <span className="sr-only">{phase === "live" ? "Receiving data" : connection}</span>
        </span>
        <time className="hud-game-clock" aria-label={`Game time ${timestamp}`}>
          {timestamp}
        </time>
      </div>
    </header>
  );
});

const Countdown = memo(function Countdown({ store }: { store: HudStore }) {
  const timer = useHudValue(store, (state) => state.timer);
  const [remaining, setRemaining] = useState(() =>
    timer ? Math.max(0, Math.ceil((timer.deadline - store.now()) / 1000)) : null,
  );
  useEffect(() => {
    if (!timer) {
      setRemaining(null);
      return;
    }
    let handle: number | undefined;
    const tick = () => {
      const milliseconds = Math.max(0, timer.deadline - store.now());
      setRemaining(Math.ceil(milliseconds / 1000));
      // Wake only when the displayed second changes; stop completely at zero.
      if (milliseconds > 0) handle = window.setTimeout(tick, (milliseconds % 1000 || 1000) + 1);
    };
    tick();
    return () => clearTimeout(handle);
  }, [timer, store]);
  const label =
    remaining === null
      ? "—"
      : remaining === 0
        ? "NOW"
        : `${String(Math.floor(remaining / 60)).padStart(2, "0")}:${String(remaining % 60).padStart(2, "0")}`;
  const progress =
    timer && remaining !== null
      ? 100 * (1 - Math.min(1, remaining / Math.max(1, timer.duration)))
      : 100;
  return (
    <div
      className="hud-countdown"
      role="timer"
      aria-live="off"
      aria-label={
        remaining === null
          ? "No objective timer provided"
          : `${remaining} seconds to objective window`
      }
    >
      <svg viewBox="0 0 56 56" aria-hidden="true">
        <circle className="hud-ring-track" cx="28" cy="28" r="25" />
        <circle
          className="hud-ring-progress"
          cx="28"
          cy="28"
          r="25"
          pathLength="100"
          strokeDasharray="100"
          strokeDashoffset={progress}
        />
      </svg>
      <span>{label}</span>
    </div>
  );
});

const MacroCard = memo(function MacroCard({ store, mode }: { store: HudStore; mode: MetricsMode }) {
  const macro = useHudValue(store, (state) => state.data?.macro ?? null);
  const urgency = useHudValue(store, (state) => state.data?.urgency ?? "normal");
  if (!macro)
    return (
      <div className="hud-macro hud-no-support">
        <Waves size={24} />
        <div>
          <span className="hud-label">TEAM DIRECTION</span>
          <h2>No supported team plan</h2>
        </div>
      </div>
    );
  const showImpact = mode !== "unverified" && !!macro.winrate_impact;
  return (
    <section className="hud-macro" aria-label="Team recommendation">
      <div className="hud-macro-top flex items-center justify-between">
        <span className="hud-label">TEAM DIRECTION</span>
        <span className="hud-urgency" data-urgency={urgency}>
          {urgency === "critical" && <TriangleAlert size={10} />}
          {urgency === "critical"
            ? "CRITICAL WINDOW"
            : urgency === "high"
              ? "HIGH PRIORITY"
              : "NORMAL PRIORITY"}
        </span>
      </div>
      <div className="hud-action-row flex items-center gap-3">
        <div className="hud-objective-icon">
          <Swords size={22} strokeWidth={1.45} />
        </div>
        <div className="hud-action-copy min-w-0 flex-1">
          <h2
            className="hud-action-title"
            title={macro.title}
            aria-live="polite"
            aria-atomic="true"
          >
            {macro.title}
          </h2>
        </div>
        <Countdown store={store} />
      </div>
      <div className="hud-impact-row flex items-center gap-2">
        <span
          className="hud-impact"
          data-verified={showImpact}
          data-negative={!!macro.winrate_impact?.startsWith("-")}
          title={
            mode === "demo"
              ? "Illustrative sample, not a measured win-rate improvement"
              : showImpact
                ? "Backend-reported validated estimate; inspect its evidence before relying on it"
                : "The current Offline Rift policies do not supply calibrated win-rate impact"
          }
        >
          {showImpact ? (
            macro.winrate_impact?.startsWith("-") ? (
              <ArrowDownRight size={12} />
            ) : (
              <ArrowUpRight size={12} />
            )
          ) : (
            <Activity size={12} />
          )}
          {showImpact ? macro.winrate_impact : "Impact unverified"}
          {showImpact && (
            <span className="hud-impact-caption">
              {mode === "demo" ? "DEMO IMPACT" : "WIN-RATE IMPACT"}
            </span>
          )}
        </span>
      </div>
    </section>
  );
});

const RoleCard = memo(function RoleCard({ store }: { store: HudStore }) {
  const role = useHudValue(store, (state) => state.data?.role ?? null);
  const coordination = useHudValue(store, (state) => state.data?.coordination ?? "missing_support");
  const notice = useHudValue(store, (state) => state.data?.notice);
  const conflict = coordination === "withheld_conflict" || role?.status === "withheld_conflict";
  const supported = role && role.status !== "insufficient_support" && !conflict;
  const Icon = role ? ROLE_ICON[role.player_role] : Crosshair;
  return (
    <section
      className="hud-role flex items-center gap-3"
      aria-label="Role-specific recommendation"
      data-withheld={!supported}
    >
      <div className="hud-role-icon">
        <Icon size={18} strokeWidth={1.65} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="hud-role-badge">{role ? ROLE_LABEL[role.player_role] : "ROLE"}</span>
          <span className="hud-label">YOUR NEXT ACTION</span>
        </div>
        <p
          className="hud-role-action"
          title={supported ? role.action : undefined}
          aria-live="polite"
          aria-atomic="true"
        >
          {conflict
            ? "Role direction withheld · conflict"
            : supported
              ? role.action ||
                (role.status === "respawning" ? "Wait to respawn" : "No supported role action")
              : notice || "No supported role action"}
        </p>
      </div>
    </section>
  );
});

const Details = memo(function Details({ store, mode }: { store: HudStore; mode: MetricsMode }) {
  const macro = useHudValue(store, (state) => state.data?.macro ?? null);
  const role = useHudValue(store, (state) => state.data?.role ?? null);
  const telemetry = useHudValue(store, (state) => state.data?.telemetry ?? null);
  const coordination = useHudValue(store, (state) => state.data?.coordination ?? "missing_support");
  const knownConfidence = mode !== "unverified" && telemetry?.ai_confidence != null;
  const withheld =
    coordination === "withheld_conflict" ||
    role?.status === "withheld_conflict" ||
    role?.status === "insufficient_support";
  return (
    <>
      <div className="hud-detail-heading flex items-center justify-between">
        <span className="hud-label">THE READ</span>
        <span className="hud-detail-hint">CONTEXT, NOT CERTAINTY</span>
      </div>
      <p className="hud-rationale">
        {macro?.subtext || "The backend has not supplied a team rationale."}
      </p>
      {role && (
        <div className="hud-role-details">
          <span className="hud-label">{ROLE_LABEL[role.player_role]} CHECK</span>
          <p>
            {withheld
              ? "Role advice is withheld. No replacement action has been inferred."
              : role.details || "No additional role prerequisite was supplied."}
          </p>
        </div>
      )}
      <div className="hud-confidence">
        <div className="flex items-center justify-between">
          <span>{mode === "demo" ? "Demo confidence" : "AI confidence"}</span>
          <strong>{knownConfidence ? `${telemetry.ai_confidence}%` : "Uncalibrated"}</strong>
        </div>
        <div
          className="hud-confidence-track"
          role={knownConfidence ? "meter" : "img"}
          aria-label={mode === "demo" ? "Illustrative confidence" : "AI confidence"}
          aria-valuemin={knownConfidence ? 0 : undefined}
          aria-valuemax={knownConfidence ? 100 : undefined}
          aria-valuenow={knownConfidence ? telemetry.ai_confidence! : undefined}
          aria-valuetext={
            knownConfidence
              ? `${telemetry.ai_confidence} percent${mode === "demo" ? ", demonstration data" : ""}`
              : "Unavailable: the backend's scores are not calibrated confidence"
          }
        >
          <span
            style={{ transform: `scaleX(${knownConfidence ? telemetry.ai_confidence! / 100 : 0})` }}
          />
        </div>
        <p className="hud-metric-note">
          {mode === "demo"
            ? "Illustrative percentages. No policy win-rate gain is established."
            : mode === "unverified"
              ? "Model scores are not confidence or win probabilities."
              : "Backend-supplied metric. Calibration is the provider's responsibility."}
        </p>
      </div>
      <div className="hud-telemetry flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <GitBranch size={11} />
          {coordination === "aligned"
            ? "Matching direction"
            : coordination === "withheld_conflict"
              ? "Conflict withheld"
              : coordination === "missing_support"
                ? "Missing support"
                : "Independent proposals"}
        </span>
        <code title="Backend state hash">#{telemetry?.state_hash ?? "unknown"}</code>
      </div>
    </>
  );
});

const EmptyState = memo(function EmptyState({ store }: { store: HudStore }) {
  const phase = useHudValue(store, (state) => state.phase);
  const connection = useHudValue(store, (state) => state.connection);
  const notice = useHudValue(store, (state) => state.notice);
  const Icon = phase === "offline" ? Unplug : Radio;
  const title =
    phase === "stale"
      ? "Waiting for a fresh read."
      : phase === "offline"
        ? connection === "reconnecting"
          ? "Reconnecting to your coach."
          : "Connection interrupted."
        : "Listening for your coach.";
  return (
    <div className="hud-empty" role="status">
      <div className="hud-empty-icon">
        <Icon size={24} strokeWidth={1.4} />
      </div>
      <h2>{phase === "waiting" && notice ? notice.title : title}</h2>
      <p>
        {phase === "waiting"
          ? notice?.detail || "The next recommendation will appear here."
          : "Previous advice has been withdrawn."}
      </p>
      <span className="hud-empty-label">
        {phase === "stale" ? "ADVICE EXPIRED" : "NO CURRENT RECOMMENDATION"}
      </span>
    </div>
  );
});

export const MacroHUD = memo(function MacroHUD({
  store,
  position = "top-right",
  metricsMode = "unverified",
  enableKeyboard = true,
  detailsShortcut = "Tab",
}: MacroHUDProps) {
  const phase = useHudValue(store, (state) => state.phase);
  const urgency = useHudValue(store, (state) => state.data?.urgency ?? "normal");
  const expanded = useHudValue(store, (state) => state.expanded);
  const id = useId();
  const toggleButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!enableKeyboard) return;
    const handle = (event: KeyboardEvent) => {
      if (
        event.repeat ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        event.isComposing
      )
        return;
      const target = event.target as HTMLElement | null;
      if (
        event.key === "Escape" &&
        store.getSnapshot().expanded &&
        !target?.closest("input, textarea, select, [contenteditable='true']")
      ) {
        event.preventDefault();
        store.collapse();
        toggleButton.current?.focus({ preventScroll: true });
        return;
      }
      if (target?.closest("input, textarea, select, button, a, [contenteditable='true']")) return;
      if (
        event.key === "Tab" &&
        (target === document.body || target?.matches("[data-macro-hud]"))
      ) {
        event.preventDefault();
        store.toggleDetails();
        toggleButton.current?.focus({ preventScroll: true });
      }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [store, enableKeyboard]);
  return (
    <aside
      className="macro-hud-anchor"
      data-position={position}
      aria-label="The Macro HUD"
      data-macro-hud
      tabIndex={0}
    >
      <div
        className="macro-hud"
        data-urgency={phase === "live" ? urgency : "normal"}
        data-phase={phase}
      >
        <div className="hud-accent-rail" />
        <Header store={store} mode={metricsMode} />
        {phase === "live" ? (
          <>
            <MacroCard store={store} mode={metricsMode} />
            <RoleCard store={store} />
          </>
        ) : (
          <EmptyState store={store} />
        )}
        <button
          ref={toggleButton}
          className="hud-expand flex items-center justify-between"
          type="button"
          aria-expanded={expanded}
          aria-controls={id}
          aria-keyshortcuts={detailsShortcut}
          onClick={() => store.toggleDetails()}
        >
          <span className="flex items-center gap-1.5">
            <span className="hud-detail-dot" />
            {expanded ? "Less noise" : "The full picture"}
          </span>
          <span className="flex items-center gap-2">
            <kbd>{detailsShortcut === "Tab" ? "TAB" : "CTRL ALT M"}</kbd>
            <ChevronDown size={13} className={expanded ? "hud-chevron-open" : ""} />
          </span>
        </button>
      </div>
      <section
        id={id}
        className="hud-details-panel"
        hidden={!expanded}
        tabIndex={expanded ? 0 : -1}
        aria-label="Recommendation context and telemetry"
      >
        {expanded &&
          (phase === "live" ? (
            <Details store={store} mode={metricsMode} />
          ) : (
            <p className="hud-rationale">
              Context is available when a fresh recommendation arrives.
            </p>
          ))}
      </section>
    </aside>
  );
});
