import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0),
);
vi.stubGlobal("cancelAnimationFrame", (handle: number) =>
  window.clearTimeout(handle),
);
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});
