import {
  expect,
  test,
} from "../../../production/frontend/macro-hud/node_modules/@playwright/test/index.mjs";
import { fileURLToPath } from "node:url";

const fixture =
  "/@fs" + fileURLToPath(new URL("./fixture.html", import.meta.url));
test.skip(
  !process.env.RIFT_TEST_BROWSER,
  "Set RIFT_TEST_BROWSER; fixtures never contact the game API",
);
test.beforeEach(async ({ page }) => {
  page.on("pageerror", (error) =>
    console.error("HUD fixture error:", error.message),
  );
  page.on("console", (message) => {
    if (message.type() === "error") console.error(message.text());
  });
});

test("native HUD cards expand, withhold conflicts and fit a narrow viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(fixture);
  await expect(
    page.getByRole("heading", { name: "Contest Baron Nashor" }),
  ).toBeVisible();
  await expect(page.getByText("+14.2%")).toHaveCount(0);
  await page.keyboard.press("Tab");
  await expect(page.getByText("Uncalibrated")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.evaluate(() => {
    const { store, scenarios } = (window as any).fixtureHud;
    store.ingest({
      ...scenarios.baron,
      coordination: "withheld_conflict",
      role: { ...scenarios.baron.role, status: "withheld_conflict" },
    });
  });
  await expect(
    page.getByText("Role direction withheld · conflict"),
  ).toBeVisible();
  await expect(page.getByText("Sweep Baron Pit & Smite Prep")).toHaveCount(0);
  await page.evaluate(() =>
    (window as any).fixtureHud.store.setConnection("offline"),
  );
  await expect(
    page.getByRole("heading", { name: "Contest Baron Nashor" }),
  ).toHaveCount(0);
});

test("120 Hz telemetry coalesces without changing the macro card", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(fixture);
  await expect(
    page.getByRole("heading", { name: "Contest Baron Nashor" }),
  ).toBeVisible();
  const result = await page.evaluate(async () => {
    const { store, scenarios, durations } = (window as any).fixtureHud;
    durations.length = 0;
    const before = document.querySelector(".hud-action-title");
    for (let index = 0; index < 120; index++) {
      store.ingest({
        ...scenarios.baron,
        telemetry: {
          ...scenarios.baron.telemetry,
          state_hash: `burst-${index}`,
        },
      });
      await new Promise((resolve) => setTimeout(resolve, 1000 / 120));
    }
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const sorted = [...durations].sort((a: number, b: number) => a - b);
    return {
      sameCard: before === document.querySelector(".hud-action-title"),
      p95: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
    };
  });
  expect(result.sameCard).toBe(true);
  expect(errors).toEqual([]);
  console.log(
    `HUD React p95 during 120 Hz fixture: ${result.p95.toFixed(2)} ms`,
  );
});
