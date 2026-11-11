import { act, fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { MacroHUD } from "../../../production/frontend/macro-hud/src/MacroHUD";
import { createHudStore } from "../../../production/frontend/macro-hud/src/store";
import { SCENARIOS } from "../../fixtures/macro-hud/scenarios";

it("stops countdown wakeups at zero and clears the remaining expiry on teardown", () => {
  vi.useFakeTimers();
  const store = createHudStore({ now: () => Date.now() });
  render(<MacroHUD store={store} />);
  act(() => {
    store.ingest({
      ...SCENARIOS.baron,
      macro: { ...SCENARIOS.baron.macro!, timer: "00:02" },
      valid_for_ms: 60000,
    });
    vi.advanceTimersByTime(17);
  });
  expect(vi.getTimerCount()).toBe(2);
  act(() => vi.advanceTimersByTime(2100));
  expect(screen.getByRole("timer").textContent).toBe("NOW");
  expect(vi.getTimerCount()).toBe(1);
  store.dispose();
  expect(vi.getTimerCount()).toBe(0);
});

it("defaults to unverified metrics and supports keyboard expansion without hijacking form navigation", () => {
  vi.useFakeTimers();
  const store = createHudStore();
  render(
    <>
      <input aria-label="Other control" />
      <MacroHUD store={store} />
    </>,
  );
  act(() => {
    store.ingest(SCENARIOS.baron);
    vi.advanceTimersByTime(17);
  });
  expect(screen.getByText("Contest Baron Nashor")).toBeTruthy();
  expect(screen.queryByText("+14.2%")).toBeNull();
  fireEvent.keyDown(screen.getByLabelText("Other control"), { key: "Tab" });
  expect(store.getSnapshot().expanded).toBe(false);
  fireEvent.keyDown(document.body, { key: "Tab" });
  expect(screen.getByText("Uncalibrated")).toBeTruthy();
  expect(screen.queryByText("94%")).toBeNull();
  expect(screen.getByText(/Enemy Jungler spotted Bot/)).toBeTruthy();
  fireEvent.keyDown(document.body, { key: "Escape" });
  expect(store.getSnapshot().expanded).toBe(false);
  store.dispose();
});

it("marks illustrative percentages as demo data and renders all five role badges", () => {
  vi.useFakeTimers();
  const store = createHudStore();
  render(<MacroHUD store={store} metricsMode="demo" />);
  act(() => {
    store.ingest(SCENARIOS.baron);
    vi.advanceTimersByTime(17);
  });
  expect(screen.getByText("+14.2%")).toBeTruthy();
  expect(screen.getByText("DEMO IMPACT")).toBeTruthy();
  const roles = {
    TOP: "TOP",
    JUNGLE: "JUNGLE",
    MIDDLE: "MID",
    BOTTOM: "ADC",
    UTILITY: "SUPPORT",
  };
  for (const [name, label] of Object.entries(roles)) {
    act(() => {
      store.ingest({
        ...SCENARIOS.baron,
        role: { ...SCENARIOS.baron.role, player_role: name },
      });
      vi.advanceTimersByTime(17);
    });
    expect(screen.getByText(label)).toBeTruthy();
  }
  store.dispose();
});

it("withdraws stale or disconnected instructions, including rationale", () => {
  vi.useFakeTimers();
  const store = createHudStore({ staleAfterMs: 100 });
  render(<MacroHUD store={store} />);
  act(() => {
    store.ingest(SCENARIOS.baron);
    vi.advanceTimersByTime(17);
    store.toggleDetails();
  });
  act(() => {
    vi.advanceTimersByTime(101);
  });
  expect(screen.queryByText("Contest Baron Nashor")).toBeNull();
  expect(screen.queryByText(/Enemy Jungler spotted Bot/)).toBeNull();
  expect(screen.getByText("Waiting for a fresh read.")).toBeTruthy();
  act(() => store.setConnection("offline"));
  expect(screen.getByText("Connection interrupted.")).toBeTruthy();
  store.dispose();
});

it("withholds conflicting role destinations without inventing an alternative", () => {
  vi.useFakeTimers();
  const store = createHudStore();
  render(<MacroHUD store={store} />);
  act(() => {
    store.ingest(SCENARIOS.conflict);
    vi.advanceTimersByTime(17);
    store.toggleDetails();
  });
  expect(screen.getByText("Prep Drake in 45s")).toBeTruthy();
  expect(screen.getByText("Role direction withheld · conflict")).toBeTruthy();
  expect(screen.queryByText("Rotate toward top")).toBeNull();
  expect(screen.queryByText(/incompatible direction must not/)).toBeNull();
  store.dispose();
});
