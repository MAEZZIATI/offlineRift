export const PLAYER_ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] as const;
export type PlayerRole = (typeof PLAYER_ROLES)[number];
export type Urgency = "critical" | "high" | "normal";
export type Coordination = "aligned" | "independent" | "withheld_conflict" | "missing_support";
export type MetricsMode = "unverified" | "demo" | "validated";

export interface MacroRecommendation {
  readonly title: string;
  readonly subtext: string;
  readonly timer?: string;
  readonly winrate_impact?: string;
}

export interface RoleRecommendation {
  readonly player_role: PlayerRole;
  readonly action: string;
  readonly details: string;
  readonly status: "recommendation" | "withheld_conflict" | "insufficient_support" | "respawning";
}

export interface HudPayload {
  readonly notice?: string;
  readonly timestamp: string;
  readonly urgency: Urgency;
  readonly macro: Readonly<MacroRecommendation> | null;
  readonly role: Readonly<RoleRecommendation> | null;
  readonly telemetry: Readonly<{ ai_confidence: number | null; state_hash: string }>;
  readonly coordination: Coordination;
  readonly valid_for_ms?: number;
}

const roles: Record<string, PlayerRole> = {
  TOP: "TOP",
  JUNGLE: "JUNGLE",
  MID: "MIDDLE",
  MIDDLE: "MIDDLE",
  ADC: "BOTTOM",
  BOTTOM: "BOTTOM",
  SUPPORT: "UTILITY",
  UTILITY: "UTILITY",
};

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function text(value: unknown, maximum: number, optional = false): string {
  if (optional && (value === undefined || value === null)) return "";
  if (typeof value !== "string" || value.length > maximum) throw new Error("Invalid text");
  return value.trim();
}

export function clockSeconds(clock: string): number {
  if (!/^\d{1,3}:[0-5]\d$/.test(clock)) return NaN;
  const [minutes, seconds] = clock.split(":").map(Number);
  return minutes * 60 + seconds;
}

/** Normalize untrusted WebSocket/IPC input. Never interpret model scores as calibrated metrics. */
export function parsePayload(input: unknown): HudPayload | null {
  try {
    const data = record(input);
    if (!data) return null;
    const timestamp = text(data.timestamp, 6);
    if (!Number.isFinite(clockSeconds(timestamp))) return null;
    if (!["critical", "high", "normal"].includes(String(data.urgency))) return null;
    const telemetry = record(data.telemetry);
    if (!telemetry) return null;
    const confidence = telemetry.ai_confidence ?? null;
    if (
      confidence !== null &&
      (typeof confidence !== "number" ||
        !Number.isFinite(confidence) ||
        confidence < 0 ||
        confidence > 100)
    )
      return null;
    const stateHash = text(telemetry.state_hash, 64);
    if (!stateHash) return null;
    let macro: MacroRecommendation | null = null;
    if (data.macro !== null) {
      const item = record(data.macro);
      if (!item) return null;
      const title = text(item.title, 160);
      if (!title) return null;
      const timer = text(item.timer, 6, true) || undefined;
      if (timer && !Number.isFinite(clockSeconds(timer))) return null;
      const impact = text(item.winrate_impact, 10, true) || undefined;
      if (
        impact &&
        (!/^[+-]?\d{1,3}(\.\d{1,2})?%$/.test(impact) || Math.abs(parseFloat(impact)) > 100)
      )
        return null;
      macro = Object.freeze({
        title,
        subtext: text(item.subtext, 640, true),
        timer,
        winrate_impact: impact,
      });
    }
    let role: RoleRecommendation | null = null;
    if (data.role !== null) {
      const item = record(data.role);
      if (!item) return null;
      const name = roles[text(item.player_role, 12).toUpperCase()];
      if (!name) return null;
      const status = item.status ?? "recommendation";
      if (
        !["recommendation", "withheld_conflict", "insufficient_support", "respawning"].includes(
          String(status),
        )
      )
        return null;
      const action = text(item.action, 180, status !== "recommendation");
      if (status === "recommendation" && !action) return null;
      role = Object.freeze({
        player_role: name,
        action,
        details: text(item.details, 640, true),
        status: status as RoleRecommendation["status"],
      });
    }
    const coordination =
      role?.status === "withheld_conflict"
        ? "withheld_conflict"
        : (data.coordination ?? (macro && role ? "independent" : "missing_support"));
    if (
      !["aligned", "independent", "withheld_conflict", "missing_support"].includes(
        String(coordination),
      )
    )
      return null;
    const validity = data.valid_for_ms;
    if (
      validity !== undefined &&
      (typeof validity !== "number" ||
        !Number.isInteger(validity) ||
        validity < 0 ||
        validity > 60_000)
    )
      return null;
    return Object.freeze({
      notice: text(data.notice, 180, true) || undefined,
      timestamp,
      urgency: data.urgency as Urgency,
      macro,
      role,
      telemetry: Object.freeze({
        ai_confidence: confidence as number | null,
        state_hash: stateHash,
      }),
      coordination: coordination as Coordination,
      valid_for_ms: validity as number | undefined,
    });
  } catch {
    return null;
  }
}
