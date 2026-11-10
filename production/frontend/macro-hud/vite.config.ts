import { fileURLToPath } from "node:url";
import path from "node:path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const directory = fileURLToPath(new URL(".", import.meta.url));
const fixtures = path.resolve(directory, "../../../tests/fixtures/macro-hud");
const repository = path.resolve(directory, "../../..");

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@fixtures": fixtures,
      "@hud": path.join(directory, "src"),
      react: path.join(directory, "node_modules/react"),
      "react-dom": path.join(directory, "node_modules/react-dom"),
      "@testing-library/react": path.join(directory, "node_modules/@testing-library/react"),
      vitest: path.join(directory, "node_modules/vitest/dist/index.js"),
    },
  },
  server: {
    fs: { allow: [directory, fixtures, path.join(repository, "tests/frontend/macro-hud")] },
  },
  build: {
    outDir: path.join(repository, "production/src/offline_rift/ui/static"),
    emptyOutDir: true,
  },
  test: {
    root: directory,
    environment: "jsdom",
    restoreMocks: true,
    include: [path.join(repository, "tests/frontend/macro-hud/**/*.test.{ts,tsx}")],
    setupFiles: [path.join(repository, "tests/frontend/macro-hud/setup.ts")],
  },
});
