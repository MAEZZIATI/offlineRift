import { packager } from "@electron/packager";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
const require = createRequire(import.meta.url);
const platform = process.argv[2];
if (!["win32", "linux"].includes(platform)) throw Error("Choose win32 or linux");
const output = await packager({
  dir: fileURLToPath(new URL(".", import.meta.url)),
  out: fileURLToPath(new URL("../release", import.meta.url)),
  name: "OfflineRiftHUD",
  executableName: "OfflineRiftHUD",
  platform,
  arch: "x64",
  electronVersion: require("electron/package.json").version,
  asar: true,
  overwrite: true,
  ignore: [/package\.mjs$/],
});
console.log(output.join("\n"));
