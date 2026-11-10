import { expect, it, vi } from "vitest";
import { createHudStore } from "../../../production/frontend/macro-hud/src/store";
import {
  connectIPC,
  connectWebSocket,
} from "../../../production/frontend/macro-hud/src/transport";
import { SCENARIOS } from "../../fixtures/macro-hud/scenarios";

it("detaches all IPC subscriptions and forwards host hotkeys", () => {
  vi.useFakeTimers();
  const store = createHudStore();
  const unsubscribers = [vi.fn(), vi.fn(), vi.fn()];
  let receive: (value: unknown) => void = () => {};
  let toggle = () => {};
  const detach = connectIPC(store, {
    onRecommendation(callback) {
      receive = callback;
      return unsubscribers[0];
    },
    onConnection() {
      return unsubscribers[1];
    },
    onToggleDetails(callback) {
      toggle = callback;
      return unsubscribers[2];
    },
  });
  receive(SCENARIOS.baron);
  vi.advanceTimersByTime(17);
  expect(store.getSnapshot().phase).toBe("live");
  toggle();
  expect(store.getSnapshot().expanded).toBe(true);
  detach();
  unsubscribers.forEach((fn) => expect(fn).toHaveBeenCalledOnce());
  expect(store.getSnapshot().phase).toBe("offline");
  store.dispose();
});

it("validates socket endpoints before connecting", () => {
  const store = createHudStore();
  expect(() => connectWebSocket(store, "https://example.com")).toThrow();
  expect(() =>
    connectWebSocket(store, "wss://user:secret@example.com"),
  ).toThrow();
  store.dispose();
});

it("uses bounded socket reconnects, processes envelopes, and never reconnects after teardown", () => {
  vi.useFakeTimers();
  class Socket {
    static instances: Socket[] = [];
    onopen: (() => void) | null = null;
    onclose: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    constructor(_address: string) {
      Socket.instances.push(this);
    }
    close() {
      this.onclose?.();
    }
  }
  const store = createHudStore();
  const detach = connectWebSocket(store, "ws://127.0.0.1:9000", {
    Socket: Socket as unknown as typeof WebSocket,
    maxRetries: 2,
    initialDelayMs: 100,
    maxDelayMs: 200,
  });
  Socket.instances[0].onopen?.();
  Socket.instances[0].onmessage?.({
    data: JSON.stringify({ type: "recommendation", payload: SCENARIOS.baron }),
  });
  vi.advanceTimersByTime(17);
  expect(store.getSnapshot().phase).toBe("live");
  Socket.instances[0].close();
  expect(store.getSnapshot().phase).toBe("offline");
  vi.advanceTimersByTime(100);
  expect(Socket.instances).toHaveLength(2);
  Socket.instances[1].close();
  vi.advanceTimersByTime(200);
  expect(Socket.instances).toHaveLength(3);
  Socket.instances[2].close();
  vi.advanceTimersByTime(1000);
  expect(Socket.instances).toHaveLength(3);
  detach();
  vi.runAllTimers();
  expect(Socket.instances).toHaveLength(3);
  store.dispose();
});
