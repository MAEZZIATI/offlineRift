import {
  expect,
  test,
} from "../../../production/frontend/macro-hud/node_modules/@playwright/test/index.mjs";
import { SCENARIOS } from "../../fixtures/macro-hud/scenarios.ts";

test.skip(
  !process.env.RIFT_TEST_BROWSER,
  "Set RIFT_TEST_BROWSER to a Chromium executable; only local synthetic fixtures are used",
);

test("desktop glance view, keyboard expansion, urgency and layout stability", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  const hud = page.getByRole("complementary", { name: "The Macro HUD" });
  await expect(
    hud.getByRole("heading", { name: "Contest Baron Nashor" }),
  ).toBeVisible();
  await expect(hud.getByText("DEMO", { exact: true })).toBeVisible();
  await expect(hud.getByText("+14.2%")).toBeVisible();
  await hud.screenshot({ path: "/tmp/macro-hud-widget.png" });
  const compact = await hud.boundingBox();
  expect(compact!.width).toBe(366);
  await page.evaluate(() => (document.activeElement as HTMLElement)?.blur());
  await page.keyboard.press("Tab");
  await expect(hud.getByText("94%")).toBeVisible();
  const expanded = await hud.boundingBox();
  expect(expanded).toEqual(compact);
  await expect(hud.getByText(/Enemy Jungler spotted Bot/)).toBeVisible();
  await page.keyboard.press("Tab");
  await expect(hud.locator(".hud-details-panel")).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(hud.getByText("94%")).not.toBeVisible();
  for (const urgency of ["normal", "high", "critical"]) {
    await page.getByRole("button", { name: urgency, exact: true }).click();
    await expect(page.locator(".macro-hud")).toHaveAttribute(
      "data-urgency",
      urgency,
    );
  }
  await page
    .getByRole("button", { name: "Expand", exact: false })
    .first()
    .click();
  await page.waitForTimeout(160);
  await page.screenshot({ path: "/tmp/macro-hud-preview.png", fullPage: true });
  expect(errors).toEqual([]);
});

test("role selection, withheld conflicts and disconnect recovery", async ({
  page,
}) => {
  await page.goto("/");
  const hud = page.getByRole("complementary", { name: "The Macro HUD" });
  for (const label of ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]) {
    await page.getByRole("button", { name: label, exact: true }).click();
    await expect(hud.locator(".hud-role-badge")).toHaveText(label);
  }
  await page
    .getByRole("combobox", { name: "Scenario" })
    .selectOption("conflict");
  await expect(
    hud.getByText("Role direction withheld · conflict"),
  ).toBeVisible();
  await expect(hud.getByText("Rotate toward top")).toHaveCount(0);
  await page.getByRole("button", { name: "Disconnect", exact: true }).click();
  await expect(hud.getByText("Connection interrupted.")).toBeVisible();
  await expect(hud.getByText("Prep Drake in 45s")).toHaveCount(0);
  await page.getByRole("button", { name: "Reconnect", exact: true }).click();
  await expect(hud.getByText("Prep Drake in 45s")).toBeVisible();
});

test("narrow viewport and reduced motion keep the widget usable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const hud = page.getByRole("complementary", { name: "The Macro HUD" });
  await expect(
    hud.getByRole("heading", { name: "Contest Baron Nashor" }),
  ).toBeVisible();
  await hud.getByRole("button", { name: /The full picture/ }).click();
  await expect(hud.getByText("94%")).toBeVisible();
  const bounds = await hud.boundingBox();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await page
      .locator(".hud-details-panel")
      .evaluate((el) => getComputedStyle(el).animationName),
  ).toBe("none");
  await page.screenshot({ path: "/tmp/macro-hud-mobile.png", fullPage: true });
});

test("120 Hz telemetry burst stays error-free with a stable macro card", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".hud-action-title")).toHaveText(
    "Contest Baron Nashor",
  );
  await page.evaluate(() => {
    const card = document.querySelector(".hud-macro")!;
    (window as unknown as { originalMacro: Element }).originalMacro = card;
  });
  await page.getByRole("button", { name: "120 Hz burst" }).click();
  await page.waitForTimeout(1200);
  expect(
    await page.evaluate(
      () =>
        document.querySelector(".hud-macro") ===
        (window as unknown as { originalMacro: Element }).originalMacro,
    ),
  ).toBe(true);
  await expect(page.locator(".hud-action-title")).toHaveText(
    "Contest Baron Nashor",
  );
  await page.getByRole("button", { name: "Stop burst" }).click();
  const measured = await page
    .locator(".preview-controls-footer strong")
    .textContent();
  console.log("Measured preview React p95:", measured);
  expect(measured).toMatch(/\d+\.\d+ ms/);
});

test("transparent host surface starts without synthetic recommendations", async ({
  page,
}) => {
  await page.goto("/?overlay=1");
  await expect(page.getByText("Listening for your coach.")).toBeVisible();
  await expect(page.getByText("Contest Baron Nashor")).toHaveCount(0);
  await expect(page.locator("body")).toHaveClass("preview-overlay-only");
  expect(
    await page
      .locator("body")
      .evaluate((el) => getComputedStyle(el).backgroundColor),
  ).toBe("rgba(0, 0, 0, 0)");
});

test("browser WebSocket events render, suppress unsupported percentages, and expire", async ({
  page,
}) => {
  await page.routeWebSocket("ws://127.0.0.1:9998/recommendations", (socket) => {
    socket.send(
      JSON.stringify({
        type: "recommendation",
        payload: { ...SCENARIOS.baron, valid_for_ms: 1200 },
      }),
    );
    socket.send("malformed data must not renew advice");
  });
  await page.goto(
    "/?overlay=1&ws=ws%3A%2F%2F127.0.0.1%3A9998%2Frecommendations",
  );
  await expect(
    page.getByRole("heading", { name: "Contest Baron Nashor" }),
  ).toBeVisible();
  await expect(page.getByText("+14.2%")).toHaveCount(0);
  await expect(page.getByText("Impact unverified")).toBeVisible();
  await page.getByRole("button", { name: /The full picture/ }).click();
  await expect(page.getByText("Uncalibrated")).toBeVisible();
  await expect(page.getByText("Waiting for a fresh read.")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Contest Baron Nashor" }),
  ).toHaveCount(0);
});
