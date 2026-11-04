"use strict";

const guidance = [
  "Check dragon timing and your team’s readiness before committing.",
  "Check baron timing and your team’s readiness before committing.",
  "Check herald timing and your team’s readiness before committing.",
  "Assess what your team can trade on the opposite side of the map.",
  "Consider side pressure with a group holding elsewhere. Check escape routes.",
  "Assess a window to group mid. Check waves and team readiness.",
  "Assess a window to group on the top side. Check waves and team readiness.",
  "Assess a window to group on the bottom side. Check waves and team readiness.",
  "Consider defending your half of the map. Check enemy pressure.",
  "Check purchases and regroup timing. This label describes base presence.",
  "Assess lane coverage and pressure before choosing the next commitment.",
  "The model has insufficient evidence for a specific macro. Use your own map assessment.",
];
const $ = (id) => document.getElementById(id);
let team = "100";
let choseTeam = false;
let paused = false;
let latest = null;
let expiresAt = 0;
let pollTimer;
let expiryTimer;
let pending;

function buildActions(actions) {
  $("action-list").replaceChildren();
  for (const [key, name] of Object.entries(actions)) {
    const index = Number(key);
    const row = document.createElement("li");
    row.className = "action-row";
    for (const [className, text] of [
      ["action-number", String(index + 1).padStart(2, "0")],
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


function connection(text, live = false) {
  $("connection-text").textContent = text;
  $("connection").className = `connection ${live ? "live" : "offline"}`;
}

function clearDecision(title, description, label = "STANDBY") {
  document.querySelector(".decision").classList.remove("active");
  $("decision-title").textContent = title;
  $("decision-description").textContent = description;
  $("decision-tag").textContent = label;
  $("decision-kicker").textContent = "THE MAP COMES FIRST";
  $("decision-time").textContent = "No active observation";
  $("game-clock").textContent = "--:--";
  $("observation").textContent = "Waiting";
  for (const row of $("action-list").children) {
    row.classList.remove("selected");
    row.removeAttribute("aria-current");
    row.querySelector(".action-value").textContent = "—";
  }
}

function percentage(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "Not recorded";
}

function renderModel(model) {
  $("coverage").textContent = model.action_counts.length ? `${model.action_coverage} / ${Object.keys(latest.actions).length} actions` : "Not recorded";
  $("agreement").textContent = percentage(model.heuristic_agreement);
  $("baseline").textContent = percentage(model.majority_baseline);
  $("balanced-accuracy").textContent = percentage(model.balanced_accuracy);
  $("macro-f1").textContent = percentage(model.macro_f1);
  $("distribution").textContent = model.action_counts.length
    ? `Training labels, in option order: ${model.action_counts.join(" / ")}.` : "Training counts not recorded.";
  $("predictions").textContent = model.prediction_counts.length
    ? `Validation predictions, in option order: ${model.prediction_counts.join(" / ")}.` : "Validation prediction counts not recorded.";
  const total = model.action_counts.reduce((sum, value) => sum + value, 0);
  const imbalanced = total > 0 && Math.max(...model.action_counts) / total > 0.8;
  $("model-warning").hidden = !model.collapsed && !imbalanced;
  $("model-warning-text").textContent = model.collapsed
    ? "Research baseline: this model chose the same action on every validation state. Useful macro discrimination has not been demonstrated."
    : "Research baseline: more than 80% of training labels belong to one action. This imbalance can bias the suggestions.";
}

function render() {
  document.body.classList.toggle("red", team === "200");
  $("perspective").textContent = `${team === "100" ? "Blue" : "Red"} team perspective`;
  if (!latest || paused || document.hidden || Date.now() >= expiresAt) return;
  const result = latest.recommendations[team];
  if (latest.status !== "live" || !result) {
    const offline = latest.status === "offline";
    connection(offline ? "Waiting for game" : "Check connection");
    clearDecision(offline ? "Waiting for the Rift." : "No live recommendation.", latest.message || "Waiting for the next game observation.");
    return;
  }
  connection("Receiving live data", true);
  document.querySelector(".decision").classList.add("active");
  $("decision-title").textContent = result.action;
  $("decision-description").textContent = guidance[result.action_id];
  $("decision-tag").textContent = "EXPERIMENTAL";
  $("decision-kicker").textContent = "CONSIDER THIS NEXT";
  const seconds = Math.floor(result.game_time);
  const clock = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  $("game-clock").textContent = clock;
  $("decision-time").textContent = `Observation at ${clock}`;
  $("observation").textContent = "Partial · masked features";
  const patternScores = Array.isArray(result.macro_scores);
  $("score-label").textContent = patternScores ? "MACRO SCORE" : "Q-VALUE";
  $("score-note").textContent = patternScores
    ? "Scores rank predicted macro patterns with a bounded return tie-breaker. They are not win probabilities or calibrated confidence."
    : "Q-values estimate discounted returns. They are not win probabilities or confidence scores.";
  for (const [index, row] of [...$("action-list").children].entries()) {
    const selected = index === result.action_id;
    row.classList.toggle("selected", selected);
    if (selected) row.setAttribute("aria-current", "true");
    else row.removeAttribute("aria-current");
    const value = patternScores ? result.macro_scores[index] : result.q_values[index];
    row.title = `Discounted-return Q estimate: ${result.q_values[index].toFixed(3)}`;
    row.querySelector(".action-value").textContent = Number.isFinite(value) ? value.toFixed(3) : "—";
  }
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
  let delay = 30000;
  try {
    const response = await fetch("/api/state", { cache: "no-store", signal: request.signal });
    if (!response.ok) throw new Error("Dashboard unavailable");
    const data = await response.json();
    if (pending !== request || paused || document.hidden) return;
    latest = data;
    delay = Math.max(250, data.valid_for_ms);
    expiresAt = Date.now() + data.valid_for_ms;
    if (!choseTeam) {
      team = String(data.default_team);
      document.querySelector(`input[name="team"][value="${team}"]`).checked = true;
    }
    $("transport").textContent = data.transport === "windows" ? "Windows bridge" : "Local API";
    $("interval").textContent = `${data.interval_seconds} seconds`;
    $("last-check").textContent = new Date(data.checked_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    buildActions(data.actions);
    renderModel(data.model);
    render();
    clearTimeout(expiryTimer);
    expiryTimer = setTimeout(() => {
      latest = null;
      connection("Updating observation");
      clearDecision("Updating the picture.", "Waiting for a fresh observation. The previous suggestion has expired.", "UPDATING");
    }, delay);
  } catch (error) {
    if (pending !== request || paused || document.hidden) return;
    clearTimeout(expiryTimer);
    latest = null;
    connection("Dashboard disconnected");
    clearDecision("Connection interrupted.", "Keep the dashboard command running in your terminal. Retrying automatically.", "OFFLINE");
  } finally {
    clearTimeout(timeout);
    if (pending === request) {
      pending = null;
      if (!paused && !document.hidden) pollTimer = setTimeout(poll, delay);
    }
  }
}

document.querySelectorAll('input[name="team"]').forEach((input) => {
  input.addEventListener("change", () => { team = input.value; choseTeam = true; render(); });
});
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume updates" : "Pause updates";
  $("pause").setAttribute("aria-pressed", String(paused));
  stop();
  if (paused) {
    connection("Updates paused");
    clearDecision("Take your time.", "Updates are paused. Resume when you want a fresh perspective.", "PAUSED");
  } else poll();
});
document.addEventListener("visibilitychange", () => {
  stop();
  if (document.hidden) {
    connection("Tab inactive");
    clearDecision("Updates on hold.", "Return to this tab to resume live observations.", "PAUSED");
  } else if (!paused) poll();
});
window.addEventListener("pagehide", stop);
poll();
