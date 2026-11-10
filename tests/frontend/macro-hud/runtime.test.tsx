import { act, fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { Runtime } from "../../../production/frontend/macro-hud/src/runtime";

it("withdraws old advice during selection and ignores delayed responses for the previous role", async () => {
  vi.useFakeTimers();
  const frame = (version: number, role: string, action: string) => ({
    schema: "macro-hud-frame-v1",
    session_id: "fixture",
    context_version: version,
    status: "live",
    hud: {
      timestamp: "10:00",
      urgency: "normal",
      macro: { title: "Prepare dragon", subtext: "Check the map." },
      role: { player_role: role, action, details: "Check the map." },
      telemetry: { ai_confidence: null, state_hash: "fixture" },
      coordination: "independent",
      valid_for_ms: 6000,
    },
    context: { team: 100, actor: "active", role },
    players: [],
    model: {
      display_name: "fixture-coach",
      fingerprint: "abc",
      policy_count: 6,
      policies: {
        TEAM: {
          algorithm: "bc",
          candidate: "fixture",
          supported_actions: 3,
          development_gate_passed: false,
          diagnostics: { collapsed: true, abstentions: 7 },
        },
      },
    },
  });
  const old = frame(0, "TOP", "Farm top lane");
  const next = frame(1, "JUNGLE", "Clear jungle camps");
  let finishSelection!: (value: unknown) => void;
  const fetcher = vi.fn(async (_url: string, options?: RequestInit) => {
    if (options?.method === "POST")
      return new Promise((resolve) => {
        finishSelection = resolve;
      });
    return { ok: true, json: async () => old };
  });
  vi.stubGlobal("fetch", fetcher);
  try {
    render(<Runtime />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20);
    });
    expect(screen.getByText("Farm top lane")).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Open coach settings"));
    expect(screen.getByText("Development gate failed")).toBeTruthy();
    expect(
      screen.getByText(/Collapsed predictions.*7 validation abstentions/),
    ).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Your role"), {
      target: { value: "JUNGLE" },
    });
    expect(
      (screen.getByLabelText("Your role") as HTMLSelectElement).disabled,
    ).toBe(true);
    fireEvent.click(screen.getByLabelText("Close settings"));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1100);
    });
    expect(screen.queryByText("Farm top lane")).toBeNull();
    expect(screen.getByText("Updating your player")).toBeTruthy();
    await act(async () => {
      finishSelection({ ok: true, json: async () => next });
      await vi.advanceTimersByTimeAsync(20);
    });
    expect(screen.getByText("Clear jungle camps")).toBeTruthy();
    // A poll already in flight may still deliver the previous context afterward.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1100);
    });
    expect(screen.queryByText("Farm top lane")).toBeNull();
    expect(screen.getByText("Clear jungle camps")).toBeTruthy();
  } finally {
    vi.unstubAllGlobals();
  }
});
