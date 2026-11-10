import { defineConfig } from "../../../production/frontend/macro-hud/node_modules/@playwright/test/index.mjs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = fileURLToPath(new URL(".", import.meta.url));
const frontend = path.resolve(
  directory,
  "../../../production/frontend/macro-hud",
);

export default defineConfig({
  testDir: directory,
  testMatch: "*.spec.ts",
  workers: 1,
  outputDir: path.join(frontend, "test-results"),
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
    launchOptions: {
      executablePath: process.env.RIFT_TEST_BROWSER,
      args: ["--no-sandbox", "--disable-dev-shm-usage"],
    },
  },
  webServer: {
    command:
      "node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 4173 --strictPort",
    cwd: frontend,
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
