"use strict";

const guidance = [
  "Check the dragon window, vision and your team’s readiness before committing.",
  "Check the Baron window, vision and your team’s readiness before committing.",
  "Check the Herald window and nearby teammates before committing.",
  "Check what your team can trade elsewhere. The model cannot see enemy positions.",
  "Check side waves, escape routes and whether the rest of your team can hold.",
  "Check mid waves and nearby teammates before grouping.",
  "Check top-side waves and nearby teammates before grouping.",
  "Check bottom-side waves and nearby teammates before grouping.",
  "Check incoming waves and enemy pressure before choosing what to defend.",
  "Check purchases and regroup timing. This pattern does not establish a recall window.",
  "Check lane coverage and pressure before choosing your next commitment.",
  "There is not enough evidence for a specific pattern. Use your own map assessment.",
];
const $ = (id) => document.getElementById(id);
let team = "100";
let choseTeam = false;
let paused = false;
let latest = null;
let expiresAt = 0;
let lifetime = 0;
let pollTimer;
let expiryTimer;
let pending;

function connection(text, live = false) {
  $("connection-text").textContent = text;
  $("connection").classList.toggle("live", live);
}

function clearAlternatives(message) {
  const row = document.createElement("li");
  row.className = "empty-options";
  row.textContent = message;
  $("alternatives").replaceChildren(row);
}

function clearDecision(title, description, label = "WAITING") {
  document.querySelector(".decision").classList.remove("active");
  $("decision-title").textContent = title;
  $("decision-description").textContent = description;
  $("decision-tag").textContent = label;
  $("decision-kicker").textContent = "WAITING FOR FRESH DATA";
  $("decision-time").textContent = "No current observation";
  $("game-clock").textContent = "--:--";
  clearAlternatives("No current observation to rank.");
  for (const row of $("action-list").children) {
    row.classList.remove("selected");
    row.removeAttribute("aria-current");
    row.removeAttribute("title");
    row.querySelector(".action-value").textContent = "—";
  }
  updateFreshness();
}

function buildActions(actions) {
  const keys = Object.keys(actions);
  const existing = [...$("action-list").children];
  if (existing.length === keys.length && existing.every((row) => row.querySelector(".action-name").textContent === actions[row.dataset.action])) return;
  $("action-list").replaceChildren();
  for (const [key, name] of Object.entries(actions)) {
    const row = document.createElement("li");
    row.className = "action-row";
    row.dataset.action = key;
    for (const [className, text] of [
      ["action-number", String(Number(key) + 1).padStart(2, "0")],
      ["action-name", name], ["action-value", "—"],
    ]) {
      const span = document.createElement("span");
      span.className = className;
      span.textContent = text;
      row.append(span);
    }
    $("action-list").append(row);
  }
}

function percentage(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Not recorded";
}

function renderModel(model) {
  const counts = model.action_counts || [];
  const predictions = model.prediction_counts || [];
  const types = { hybrid: "CQL + neural model", balanced_bc: "Neural classifier", cql: "CQL model" };
  $("model-kind").textContent = types[model.prediction_method] || "Model loaded";
  $("coverage").textContent = counts.length ? `${model.action_coverage} / ${Object.keys(latest.actions).length} patterns` : "Not recorded";
  $("agreement").textContent = percentage(model.heuristic_agreement);
  $("baseline").textContent = percentage(model.majority_baseline);
  $("balanced-accuracy").textContent = percentage(model.balanced_accuracy);
  $("macro-f1").textContent = percentage(model.macro_f1);
  $("distribution").textContent = counts.length ? `Training labels, in pattern order: ${counts.join(" / ")}.` : "Training counts not recorded.";
  $("predictions").textContent = predictions.length ? `Validation predictions, in pattern order: ${predictions.join(" / ")}.` : "Validation counts not recorded.";
  const total = counts.reduce((sum, value) => sum + value, 0);
  const imbalanced = total > 0 && Math.max(...counts) / total > 0.8;
  $("model-warning").hidden = !model.collapsed && !imbalanced;
  $("model-warning-text").textContent = model.collapsed
    ? "This model predicted the same pattern on every validation observation. It has not demonstrated useful macro discrimination."
    : "More than 80% of training labels belong to one pattern. This can bias the model’s suggestions.";
}

function updateFreshness() {
  const remaining = Math.max(0, expiresAt - Date.now());
  const live = latest && latest.status === "live" && remaining > 0;
  $("freshness").value = live && lifetime > 0 ? remaining / lifetime : 0;
  $("observed-age").textContent = live
    ? `Observed ${Math.max(0, Math.floor((Date.now() - latest.checked_at * 1000) / 1000))}s ago`
    : paused ? "Updates paused" : "No current observation";
  $("refresh-countdown").textContent = latest && remaining > 0
    ? `Next check in ${Math.ceil(remaining / 1000)}s` : "—";
}

function renderAlternatives(scores, selected) {
  const ranked = Object.entries(latest.actions)
    .map(([key, name]) => ({ id: Number(key), name, score: scores[Number(key)] }))
    .filter((row) => Number.isFinite(row.score))
    .sort((left, right) => right.score - left.score || left.id - right.id);
  const alternatives = ranked.filter((row) => row.id !== selected).slice(0, 3);
  if (!alternatives.length) {
    clearAlternatives("No other supported pattern in this observation.");
    return;
  }
  $("alternatives").replaceChildren();
  for (const option of alternatives) {
    const row = document.createElement("li");
    row.className = "alternative-row";
    const rank = document.createElement("span");
    rank.className = "alternative-rank";
    rank.textContent = String(ranked.indexOf(option) + 1).padStart(2, "0");
    const label = document.createElement("span");
    label.textContent = option.name;
    row.append(rank, label);
    $("alternatives").append(row);
  }
}

function render() {
  document.body.classList.toggle("red", team === "200");
  $("perspective").textContent = `${team === "100" ? "Blue" : "Red"} team perspective`;
  $("perspective-dot").className = `team-dot ${team === "100" ? "blue" : "red"}`;
  if (!latest || paused || document.hidden || Date.now() >= expiresAt) return;
  const result = latest.recommendations?.[team];
  if (latest.status !== "live" || !result) {
    const offline = latest.status === "offline";
    connection(offline ? "Waiting for game" : "Check connection");
    clearDecision(offline ? "Waiting for your game." : "No live suggestion.", latest.message || "Waiting for a fresh game observation.");
    return;
  }
  const patternScores = Array.isArray(result.macro_scores);
  const scores = patternScores ? result.macro_scores : result.q_values;
  if (!Array.isArray(scores) || !Number.isFinite(result.game_time) || !Object.hasOwn(latest.actions, result.action_id)) {
    throw new Error("Incomplete model response");
  }
  connection("Receiving live data", true);
  document.querySelector(".decision").classList.add("active");
  $("decision-title").textContent = result.action;
  $("decision-description").textContent = guidance[result.action_id];
  $("decision-tag").textContent = "MODEL ESTIMATE";
  $("decision-kicker").textContent = "CHECK AGAINST YOUR MAP";
  const seconds = Math.floor(result.game_time);
  const clock = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  $("game-clock").textContent = clock;
  $("decision-time").textContent = `Observed at ${clock}`;
  $("score-label").textContent = patternScores ? "MODEL SCORE" : "Q-VALUE";
  $("score-note").textContent = patternScores
    ? "Scores rank formation patterns. They are not probabilities or calibrated confidence."
    : "Q-values estimate discounted returns. They are not win probabilities or confidence scores.";
  for (const row of $("action-list").children) {
    const index = Number(row.dataset.action);
    const selected = index === result.action_id;
    row.classList.toggle("selected", selected);
    if (selected) row.setAttribute("aria-current", "true");
    else row.removeAttribute("aria-current");
    const value = scores[index];
    row.title = Number.isFinite(result.q_values?.[index])
      ? `Discounted-return Q estimate: ${result.q_values[index].toFixed(3)}`
      : "Formation evidence score; not a probability";
    row.querySelector(".action-value").textContent = Number.isFinite(value) ? value.toFixed(3) : "—";
  }
  renderAlternatives(scores, result.action_id);
  updateFreshness();
}

function stop() {
  clearTimeout(pollTimer);
  clearTimeout(expiryTimer);
  if (pending) pending.abort();
  pending = null;
  latest = null;
}

async function poll() {
  if (paused || document.hidden || pending) return;
  const request = new AbortController();
  pending = request;
  const timeout = setTimeout(() => request.abort(), 12000);
  let delay = 5000;
  try {
    const response = await fetch("/api/state", { cache: "no-store", signal: request.signal });
    if (!response.ok) throw new Error("Dashboard unavailable");
    const data = await response.json();
    if (pending !== request || paused || document.hidden) return;
    if (!Number.isFinite(data.valid_for_ms) || !Number.isFinite(data.checked_at)) throw new Error("Invalid observation timestamp");
    latest = data;
    delay = Math.max(250, data.valid_for_ms);
    lifetime = Math.max(0, data.valid_for_ms);
    expiresAt = Date.now() + lifetime;
    if (!choseTeam) {
      team = data.default_team === 200 ? "200" : "100";
      document.querySelector(`input[name="team"][value="${team}"]`).checked = true;
    }
    $("transport").textContent = data.transport === "windows" ? "Windows bridge" : "Local API";
    $("interval").textContent = `${data.interval_seconds}s`;
    $("last-check").textContent = new Date(data.checked_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    buildActions(data.actions);
    renderModel(data.model);
    if (lifetime > 0) render();
    else {
      latest = null;
      connection("Refreshing observation");
      clearDecision("Refreshing the picture.", "Waiting for a fresh observation.", "UPDATING");
    }
    clearTimeout(expiryTimer);
    expiryTimer = setTimeout(() => {
      latest = null;
      connection("Refreshing observation");
      clearDecision("Refreshing the picture.", "The previous suggestion has expired. Waiting for fresh data.", "UPDATING");
    }, lifetime);
  } catch (error) {
    if (pending !== request || paused || document.hidden) return;
    clearTimeout(expiryTimer);
    latest = null;
    connection("Dashboard disconnected");
    clearDecision("Connection interrupted.", "Keep the dashboard command running. We’ll retry automatically.", "OFFLINE");
  } finally {
    clearTimeout(timeout);
    if (pending === request) {
      pending = null;
      if (!paused && !document.hidden) pollTimer = setTimeout(poll, delay);
    }
  }
}

function renderPaused() {
  connection("Updates paused");
  clearDecision("Updates paused.", "Resume for a fresh observation when you’re ready.", "PAUSED");
}

document.querySelectorAll('input[name="team"]').forEach((input) => {
  input.addEventListener("change", () => { team = input.value; choseTeam = true; render(); });
});
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause-label").textContent = paused ? "Resume" : "Pause";
  $("pause").setAttribute("aria-pressed", String(paused));
  stop();
  if (paused) renderPaused();
  else poll();
});
$("focus").addEventListener("click", () => {
  const focused = document.body.classList.toggle("focus");
  $("focus").setAttribute("aria-pressed", String(focused));
  $("focus").querySelector("span").textContent = focused ? "Full view" : "Focus view";
  $("focus-note").hidden = !focused;
});
document.addEventListener("keydown", (event) => {
  if (event.repeat || event.altKey || event.ctrlKey || event.metaKey || event.target.isContentEditable || (event.target instanceof Element && event.target.closest("input, textarea, select"))) return;
  const id = { f: "focus", p: "pause" }[event.key.toLowerCase()];
  if (id) { event.preventDefault(); $(id).click(); }
});
document.addEventListener("visibilitychange", () => {
  stop();
  if (document.hidden) {
    connection("Tab inactive");
    clearDecision("Updates on hold.", "Return to this tab for a fresh observation.", "PAUSED");
  } else if (paused) renderPaused();
  else poll();
});
window.addEventListener("pagehide", stop);
window.addEventListener("pageshow", (event) => { if (event.persisted && !paused && !document.hidden) poll(); });
setInterval(() => { if (!document.hidden) updateFreshness(); }, 1000);
poll();
