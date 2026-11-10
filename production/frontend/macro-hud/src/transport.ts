import type { HudStore } from "./store";

export interface HudBridge {
  onRecommendation(callback: (payload: unknown) => void): () => void;
  onConnection?(callback: (connected: boolean) => void): () => void;
  onToggleDetails?(callback: () => void): () => void;
}

/** Host exposes narrow preload callbacks; no Node APIs are exposed to the renderer. */
export function connectIPC(store: HudStore, bridge: HudBridge): () => void {
  store.beginSession();
  store.setConnection("connected");
  const subscriptions = [
    bridge.onRecommendation((payload) => {
      store.ingest(payload);
    }),
  ];
  if (bridge.onConnection)
    subscriptions.push(
      bridge.onConnection((connected) => {
        if (connected) {
          store.beginSession();
          store.setConnection("connected");
        } else store.setConnection("offline");
      }),
    );
  if (bridge.onToggleDetails)
    subscriptions.push(bridge.onToggleDetails(() => store.toggleDetails()));
  return () => {
    subscriptions.forEach((unsubscribe) => unsubscribe());
    store.setConnection("offline");
  };
}

interface WebSocketOptions {
  maxRetries?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  Socket?: typeof WebSocket;
}

/** Bounded reconnects; no stale advice is retained while disconnected. */
export function connectWebSocket(
  store: HudStore,
  address: string,
  options: WebSocketOptions = {},
): () => void {
  const url = new URL(address);
  if (!["ws:", "wss:"].includes(url.protocol) || url.username || url.password)
    throw new Error("Expected a ws:// or wss:// endpoint without embedded credentials");
  const maximum = options.maxRetries ?? 6;
  const initial = options.initialDelayMs ?? 500;
  const cap = options.maxDelayMs ?? 15_000;
  if (
    !Number.isInteger(maximum) ||
    maximum < 0 ||
    !Number.isFinite(initial) ||
    initial <= 0 ||
    !Number.isFinite(cap) ||
    cap < initial
  )
    throw new Error("Invalid reconnect settings");
  const Socket = options.Socket ?? WebSocket;
  let active: WebSocket | null = null;
  let retry: ReturnType<typeof setTimeout> | undefined;
  let attempts = 0;
  let stopped = false;

  function reconnect() {
    if (stopped) return;
    if (attempts >= maximum) {
      store.setConnection("offline");
      return;
    }
    const delay = Math.min(cap, initial * 2 ** attempts++);
    store.setConnection("reconnecting");
    retry = setTimeout(connect, delay);
  }

  function connect() {
    if (stopped) return;
    let socket: WebSocket;
    try {
      socket = new Socket(url.href);
    } catch {
      reconnect();
      return;
    }
    active = socket;
    socket.onopen = () => {
      if (stopped || active !== socket) return;
      store.beginSession();
      store.setConnection("connected");
    };
    socket.onmessage = (event) => {
      if (
        stopped ||
        active !== socket ||
        typeof event.data !== "string" ||
        event.data.length > 32_768
      )
        return;
      try {
        const decoded = JSON.parse(event.data);
        const payload = decoded?.type === "recommendation" ? decoded.payload : decoded;
        if (store.ingest(payload)) attempts = 0;
      } catch {
        /* Malformed events neither replace advice nor extend its lifetime. */
      }
    };
    socket.onerror = () => {
      if (!stopped && active === socket) {
        store.setConnection("offline");
        socket.close();
      }
    };
    socket.onclose = () => {
      if (stopped || active !== socket) return;
      active = null;
      store.setConnection("offline");
      reconnect();
    };
  }
  store.beginSession();
  connect();
  return () => {
    stopped = true;
    clearTimeout(retry);
    if (active) {
      active.onclose = null;
      active.onerror = null;
      active.onmessage = null;
      active.onopen = null;
      active.close(1000, "HUD detached");
    }
    active = null;
    store.setConnection("offline");
  };
}
