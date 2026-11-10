export { MacroHUD, type MacroHUDProps } from "./MacroHUD";
export { createHudStore, useHudValue, type HudStore, type HudSnapshot } from "./store";
export { connectIPC, connectWebSocket, type HudBridge } from "./transport";
export {
  parsePayload,
  clockSeconds,
  PLAYER_ROLES,
  type HudPayload,
  type PlayerRole,
  type Urgency,
  type Coordination,
  type MetricsMode,
} from "./types";
