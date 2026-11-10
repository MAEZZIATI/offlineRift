import { useCallback, useSyncExternalStore } from "react";
import { clockSeconds, parsePayload, type HudPayload } from "./types";

export interface HudSnapshot {
  readonly data: HudPayload | null;
  readonly notice?: Readonly<{ title: string; detail: string }>;
  readonly phase: "waiting" | "live" | "stale" | "offline";
  readonly connection: "connecting" | "connected" | "reconnecting" | "offline";
  readonly expanded: boolean;
  readonly timer: Readonly<{ deadline: number; duration: number; target: number }> | null;
}

interface StoreOptions {
  staleAfterMs?: number;
  now?: () => number;
  schedule?: (callback: () => void) => number;
  cancel?: (handle: number) => void;
}

function shared<T extends object>(old: T | null, next: T | null): T | null {
  if (old === null || next === null) return next;
  const keys = Object.keys(next) as (keyof T)[];
  return keys.length === Object.keys(old).length && keys.every((key) => old[key] === next[key])
    ? old
    : next;
}

/** Latest-frame store: transport traffic never becomes a React prop stream. */
export function createHudStore(options: StoreOptions = {}) {
  const ttl = options.staleAfterMs ?? 8_000;
  if (!Number.isFinite(ttl) || ttl <= 0 || ttl > 60_000) throw new Error("Invalid advice expiry");
  const now = options.now ?? (() => performance.now());
  const schedule = options.schedule ?? ((fn: () => void) => requestAnimationFrame(fn));
  const cancel = options.cancel ?? ((handle: number) => cancelAnimationFrame(handle));
  const initial: HudSnapshot = Object.freeze({
    data: null,
    phase: "waiting",
    connection: "connecting",
    expanded: false,
    timer: null,
  });
  let snapshot = initial;
  let pending: HudPayload | null = null;
  let frame: number | null = null;
  let expiry: ReturnType<typeof setTimeout> | undefined;
  let watermark = -1;
  let disposed = false;
  const listeners = new Set<() => void>();

  function update(next: HudSnapshot) {
    if (disposed) return;
    snapshot = Object.freeze(next);
    listeners.forEach((fn) => fn());
  }

  function clearPending() {
    if (frame !== null) cancel(frame);
    frame = null;
    pending = null;
    clearTimeout(expiry);
  }

  function flush() {
    frame = null;
    if (!pending || disposed) return;
    const incoming = pending;
    pending = null;
    const previous = snapshot.data;
    const data = Object.freeze({
      ...incoming,
      macro: shared(previous?.macro ?? null, incoming.macro),
      role: shared(previous?.role ?? null, incoming.role),
      telemetry: shared(previous?.telemetry ?? null, incoming.telemetry)!,
    });
    const seconds = incoming.macro?.timer ? clockSeconds(incoming.macro.timer) : NaN;
    const target = clockSeconds(incoming.timestamp) + seconds;
    const timer = Number.isFinite(target)
      ? snapshot.timer?.target === target && previous?.macro?.title === data.macro?.title
        ? snapshot.timer
        : Object.freeze({ deadline: now() + seconds * 1000, duration: seconds, target })
      : null;
    const validity = incoming.valid_for_ms ?? ttl;
    update({
      ...snapshot,
      data,
      notice: undefined,
      timer,
      phase: validity === 0 ? "stale" : "live",
      connection: "connected",
    });
    clearTimeout(expiry);
    expiry = setTimeout(() => update({ ...snapshot, phase: "stale" }), validity);
  }

  return {
    now,
    getSnapshot: () => snapshot,
    getServerSnapshot: () => initial,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    ingest(input: unknown): boolean {
      if (disposed) return false;
      const data = parsePayload(input);
      if (!data) return false;
      const time = clockSeconds(data.timestamp);
      if (time < watermark) return false;
      watermark = time;
      pending = data;
      if (frame === null) frame = schedule(flush);
      return true;
    },
    setConnection(connection: HudSnapshot["connection"]) {
      if (connection !== "connected") clearPending();
      update({
        ...snapshot,
        connection,
        phase:
          connection === "connected"
            ? snapshot.phase
            : connection === "connecting"
              ? "waiting"
              : "offline",
      });
    },
    setNotice(title: string, detail: string) {
      clearPending();
      update({
        ...snapshot,
        data: null,
        timer: null,
        phase: "waiting",
        connection: "connected",
        notice: Object.freeze({ title, detail }),
      });
    },
    beginSession() {
      clearPending();
      watermark = -1;
      update({ ...initial, expanded: snapshot.expanded });
    },
    toggleDetails() {
      update({ ...snapshot, expanded: !snapshot.expanded });
    },
    collapse() {
      if (snapshot.expanded) update({ ...snapshot, expanded: false });
    },
    dispose() {
      clearPending();
      disposed = true;
      listeners.clear();
    },
  };
}

export type HudStore = ReturnType<typeof createHudStore>;

/** Select stable slices or primitives, never newly allocated objects. */
export function useHudValue<T>(store: HudStore, selector: (state: HudSnapshot) => T): T {
  const read = useCallback(() => selector(store.getSnapshot()), [store, selector]);
  const server = useCallback(() => selector(store.getServerSnapshot()), [store, selector]);
  return useSyncExternalStore(store.subscribe, read, server);
}
