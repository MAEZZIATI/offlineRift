import type {
  HudPayload,
  PlayerRole,
} from "../../../production/frontend/macro-hud/src/types";

// Synthetic presentation fixtures only. Never imported by the distributed HUD library.
export const SCENARIOS: Record<string, HudPayload> = {
  baron: {
    timestamp: "18:42",
    urgency: "critical",
    macro: {
      title: "Contest Baron Nashor",
      subtext: "Enemy Jungler spotted Bot. Secure vision and start.",
      timer: "00:30",
      winrate_impact: "+14.2%",
    },
    role: {
      player_role: "JUNGLE",
      action: "Sweep Baron Pit & Smite Prep",
      details: "Buy 2 Control Wards before pathing North.",
      status: "recommendation",
    },
    telemetry: { ai_confidence: 94, state_hash: "a8f9c2" },
    coordination: "independent",
  },
  drake: {
    timestamp: "14:15",
    urgency: "high",
    macro: {
      title: "Prep Drake in 45s",
      subtext:
        "The next objective window is approaching. Check lane pressure and available vision before committing.",
      timer: "00:45",
      winrate_impact: "+8.6%",
    },
    role: {
      player_role: "JUNGLE",
      action: "Reset, then approach river",
      details:
        "Check your inventory and the objective timer before choosing a route.",
      status: "recommendation",
    },
    telemetry: { ai_confidence: 86, state_hash: "d4e612" },
    coordination: "independent",
  },
  crossmap: {
    timestamp: "22:08",
    urgency: "normal",
    macro: {
      title: "Cross-map Top",
      subtext:
        "Consider pressure on the opposite side. Confirm enemy positions and your team's intent on the map.",
      timer: "01:10",
      winrate_impact: "+4.3%",
    },
    role: {
      player_role: "JUNGLE",
      action: "Assess the top-side route",
      details:
        "Camp availability, path safety and hidden opponents are not established by this example.",
      status: "recommendation",
    },
    telemetry: { ai_confidence: 79, state_hash: "e07ab1" },
    coordination: "independent",
  },
  stall: {
    timestamp: "26:31",
    urgency: "normal",
    macro: {
      title: "Turtle & Stall",
      subtext:
        "Group defensively and reassess the next measured state. This example does not identify a safe engage.",
      timer: "00:55",
      winrate_impact: "+6.1%",
    },
    role: {
      player_role: "JUNGLE",
      action: "Regroup before the next play",
      details: "Use your own map assessment to check immediate threats.",
      status: "recommendation",
    },
    telemetry: { ai_confidence: 82, state_hash: "4c08d1" },
    coordination: "independent",
  },
  conflict: {
    timestamp: "18:42",
    urgency: "high",
    macro: {
      title: "Prep Drake in 45s",
      subtext:
        "The team and role proposals point at different destinations. The role direction is withheld.",
      timer: "00:45",
    },
    role: {
      player_role: "JUNGLE",
      action: "Rotate toward top",
      details:
        "This incompatible direction must not appear as delivered advice.",
      status: "withheld_conflict",
    },
    telemetry: { ai_confidence: null, state_hash: "c09f41" },
    coordination: "withheld_conflict",
  },
  unsupported: {
    timestamp: "18:42",
    urgency: "normal",
    macro: null,
    role: null,
    telemetry: { ai_confidence: null, state_hash: "f00000" },
    coordination: "missing_support",
  },
};

export const ROLE_EXAMPLES: Record<
  PlayerRole,
  { action: string; details: string }
> = {
  TOP: {
    action: "Pressure top. Hold your teleport.",
    details:
      "Check the wave and team pings before leaving your lane. This is sample content.",
  },
  JUNGLE: {
    action: "Sweep Baron Pit & Smite Prep",
    details: "Buy 2 Control Wards before pathing North.",
  },
  MIDDLE: {
    action: "Push mid, then assess river",
    details:
      "Check your lane opponent and available vision. This is sample content.",
  },
  BOTTOM: {
    action: "Finish the wave. Move with team.",
    details: "Assess nearby threats before moving. This is sample content.",
  },
  UTILITY: {
    action: "Check the approach with your team",
    details: "Actual vision coverage is unobserved in this example.",
  },
};
