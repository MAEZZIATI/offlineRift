import { act, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  createHudStore,
  useHudValue,
} from "../../../production/frontend/macro-hud/src/store";
import { parsePayload } from "../../../production/frontend/macro-hud/src/types";
import { SCENARIOS } from "../../fixtures/macro-hud/scenarios";

const fixture = SCENARIOS.baron;
const flush = () =>
  act(() => {
    vi.advanceTimersByTime(17);
  });

describe("payload boundary", () => {
  it("handles the requested payload and normalizes role aliases", () => {
    expect(parsePayload(fixture)?.macro?.title).toBe("Contest Baron Nashor");
    expect(
      parsePayload({
        ...fixture,
        role: { ...fixture.role, player_role: "ADC" },
      })?.role?.player_role,
    ).toBe("BOTTOM");
    expect(parsePayload(SCENARIOS.unsupported)?.macro).toBeNull();
    expect(
      parsePayload({
        ...fixture,
        role: { ...fixture.role, status: "withheld_conflict" },
      })?.coordination,
    ).toBe("withheld_conflict");
  });
  it.each([
    null,
    [],
    { ...fixture, timestamp: "18:99" },
    { ...fixture, urgency: "urgent" },
    { ...fixture, telemetry: { ai_confidence: 101, state_hash: "abc" } },
    { ...fixture, macro: { ...fixture.macro, timer: "soon" } },
    { ...fixture, role: { ...fixture.role, player_role: "UNKNOWN" } },
    { ...fixture, valid_for_ms: -1 },
    { ...fixture, valid_for_ms: Infinity },
    { ...fixture, macro: { ...fixture.macro, winrate_impact: "+140%" } },
  ])("rejects malformed data without creating a recommendation", (value) => {
    expect(parsePayload(value)).toBeNull();
  });
});

describe("frame-coalesced store", () => {
  it("commits only the newest of 300 ticks and preserves unchanged recommendation slices", () => {
    vi.useFakeTimers();
    const store = createHudStore({ now: () => Date.now() });
    store.ingest(fixture);
    flush();
    const before = store.getSnapshot();
    const listener = vi.fn();
    store.subscribe(listener);
    for (let i = 0; i < 300; i++)
      store.ingest({
        ...fixture,
        telemetry: { ai_confidence: i % 100, state_hash: `tick-${i}` },
      });
    expect(listener).not.toHaveBeenCalled();
    flush();
    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().data?.telemetry.state_hash).toBe("tick-299");
    expect(store.getSnapshot().data?.macro).toBe(before.data?.macro);
    expect(store.getSnapshot().data?.role).toBe(before.data?.role);
    expect(store.getSnapshot().timer).toBe(before.timer);
    store.dispose();
  });
  it("does not redraw a macro subscriber for telemetry-only traffic", () => {
    vi.useFakeTimers();
    const store = createHudStore();
    store.ingest(fixture);
    flush();
    const rendered = vi.fn();
    function MacroSubscriber() {
      const macro = useHudValue(store, (state) => state.data?.macro);
      rendered();
      return <span>{macro?.title}</span>;
    }
    render(<MacroSubscriber />);
    const count = rendered.mock.calls.length;
    act(() => {
      for (let i = 0; i < 250; i++)
        store.ingest({
          ...fixture,
          telemetry: { ai_confidence: i % 100, state_hash: String(i) },
        });
    });
    flush();
    expect(rendered).toHaveBeenCalledTimes(count);
    store.dispose();
  });
  it("expires advice and rejects delayed or malformed events without extending freshness", () => {
    vi.useFakeTimers();
    const store = createHudStore({ staleAfterMs: 1000 });
    store.ingest(fixture);
    flush();
    vi.advanceTimersByTime(500);
    expect(store.ingest({ ...fixture, timestamp: "18:41" })).toBe(false);
    expect(store.ingest({ broken: true })).toBe(false);
    vi.advanceTimersByTime(501);
    expect(store.getSnapshot().phase).toBe("stale");
    store.dispose();
  });
  it("keeps one objective deadline across clock ticks and allows explicit new sessions", () => {
    vi.useFakeTimers();
    const store = createHudStore({ now: () => Date.now() });
    store.ingest(fixture);
    flush();
    const timer = store.getSnapshot().timer;
    vi.advanceTimersByTime(1000);
    store.ingest({
      ...fixture,
      timestamp: "18:43",
      macro: { ...fixture.macro, timer: "00:29" },
    });
    flush();
    expect(store.getSnapshot().timer).toBe(timer);
    expect(store.ingest({ ...fixture, timestamp: "01:00" })).toBe(false);
    store.beginSession();
    expect(store.ingest({ ...fixture, timestamp: "01:00" })).toBe(true);
    flush();
    expect(store.getSnapshot().data?.timestamp).toBe("01:00");
    store.dispose();
  });
  it("cancels pending updates on disconnect and disposal", () => {
    vi.useFakeTimers();
    const store = createHudStore();
    store.ingest(fixture);
    store.setConnection("offline");
    flush();
    expect(store.getSnapshot().phase).toBe("offline");
    expect(store.getSnapshot().data).toBeNull();
    store.dispose();
    expect(store.ingest(fixture)).toBe(false);
    vi.runAllTimers();
  });
});
