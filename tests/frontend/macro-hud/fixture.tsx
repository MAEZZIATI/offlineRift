import { Profiler } from "react";
import { createRoot } from "react-dom/client";
import { MacroHUD } from "@hud/MacroHUD";
import { createHudStore } from "@hud/store";
import { SCENARIOS } from "../../fixtures/macro-hud/scenarios";

const store = createHudStore();
const durations: number[] = [];
Object.assign(window, {
  fixtureHud: { store, durations, scenarios: SCENARIOS },
});
createRoot(document.getElementById("root")!).render(
  <Profiler id="hud" onRender={(_id, _phase, actual) => durations.push(actual)}>
    <MacroHUD store={store} />
  </Profiler>,
);
store.ingest(SCENARIOS.baron);
