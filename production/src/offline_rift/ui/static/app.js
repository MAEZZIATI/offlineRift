"use strict";

const $ = id => document.getElementById(id);
const ROLE_NAMES = { TOP: "Top", JUNGLE: "Jungle", MIDDLE: "Mid", BOTTOM: "Bottom", UTILITY: "Support", TEAM: "Team" };
let team = "100";
let actor = "active";
let role = "";
let paused = false;
let latest = null;
let expiresAt = 0;
let lifetime = 0;
let pending;
let pollTimer;
let expiryTimer;
let rosterKey = "";

function connection(message, live = false) {
  $("connection-text").textContent = message;
  $("connection").classList.toggle("live", live);
}

function emptyOptions(message) {
  const item = document.createElement("li");
  item.className = "empty-options";
  item.textContent = message;
  $("alternatives").replaceChildren(item);
  for (const row of $("action-list").children) {
    row.classList.remove("selected");
    row.removeAttribute("aria-current");
    row.querySelector(".action-value").textContent = "—";
  }
}

function clearDecision(title, message, tag = "WAITING") {
  document.querySelector(".decision").classList.remove("active");
  $("decision-title").textContent = title;
  $("decision-description").textContent = message;
  $("decision-tag").textContent = tag;
  $("decision-kicker").textContent = "WAITING FOR A SUPPORTED DECISION";
  $("decision-check").hidden = true;
  $("decision-horizon").hidden = true;
  $("macro-direction").hidden = true;
  $("game-clock").textContent = "--:--";
  $("decision-time").textContent = "No current observation";
  emptyOptions("No current action ranking.");
  updateFreshness();
}

function updateFreshness() {
  const left = Math.max(0, expiresAt - Date.now());
  const live = latest && latest.status === "live" && left > 0;
  $("freshness").value = live && lifetime > 0 ? left / lifetime : 0;
  $("observed-age").textContent = live ? `Observed ${Math.max(0, Math.floor((Date.now() - latest.checked_at * 1000) / 1000))}s ago` : paused ? "Updates paused" : "No current observation";
  $("refresh-countdown").textContent = latest && left > 0 ? `Next check in ${Math.ceil(left / 1000)}s` : "—";
}

function buildOptions(actions) {
  $("action-list").replaceChildren();
  for (const [key, name] of Object.entries(actions)) {
    const row = document.createElement("li"); row.className = "action-row"; row.dataset.action = key;
    for (const [className, text] of [["action-number", String(Number(key) + 1).padStart(2, "0")], ["action-name", name], ["action-value", "—"]]) {
      const span = document.createElement("span"); span.className = className; span.textContent = text; row.append(span);
    }
    $("action-list").append(row);
  }
}

function updateActors(data) {
  const players = data.players || [];
  const signature = JSON.stringify(players);
  if (signature !== rosterKey) {
    rosterKey = signature;
    $("actor").replaceChildren();
    for (const [value, label] of [["active", "Me · identify from game"], ["team", "Selected team"], ...players.map(p => [p.id, `${p.champion} · ${p.team === 100 ? "Blue" : "Red"}`])]) {
      const option = document.createElement("option"); option.value = value; option.textContent = label; $("actor").append(option);
    }
    if (!["active", "team", ...players.map(p => p.id)].includes(actor)) { actor = "active"; role = ""; $("role").value = ""; }
    $("actor").value = actor;
  }
  $("role").disabled = actor === "team";
}

function percentage(value) { return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Not recorded"; }
function renderModel(model) {
  $("model-kind").textContent = { cql: "Offline RL · CQL", iql: "Offline RL · IQL", bc: "Behavior imitation baseline" }[model.prediction_method] || "Unknown policy";
  $("model-kind").title = model.name || "";
  if (model.policy_count === 6) $("model-kind").textContent += " · 6 policies";
  if (model.advisory_groups?.length === 1) $("model-kind").textContent += ` · ${ROLE_NAMES[model.advisory_groups[0]] || model.advisory_groups[0]} only`;
  const counts = model.action_counts || [];
  $("coverage").textContent = `${model.action_coverage} / ${Object.keys(latest.actions).length} actions`;
  $("agreement").textContent = percentage(model.behavior_agreement);
  $("baseline").textContent = percentage(model.majority_baseline);
  $("balanced-accuracy").textContent = percentage(model.balanced_accuracy);
  $("macro-f1").textContent = percentage(model.macro_f1);
  $("distribution").textContent = `Recorded training actions: ${counts.join(" / ")}.`;
  $("predictions").textContent = `Held-out predictions: ${(model.prediction_counts || []).join(" / ")}.`;
  const total = counts.reduce((sum, count) => sum + count, 0);
  const imbalanced = total > 0 && Math.max(...counts) / total > 0.8;
  $("model-warning").hidden = !model.collapsed && !imbalanced;
  $("model-warning-text").textContent = model.collapsed ? "This policy selects at most one action on the development split. Diverse useful decisions have not been demonstrated." : "More than 80% of recorded actions belong to one class; recommendations may be biased.";
}

function render() {
  if (!latest || paused || document.hidden || Date.now() >= expiresAt) return;
  if (latest.status !== "live") {
    connection(latest.status === "offline" ? "Waiting for game" : "Check connection");
    clearDecision("Waiting for your game.", latest.message || "Waiting for a fresh observation.");
    return;
  }
  const result = latest.recommendation;
  if (!result) throw Error("Missing policy response");
  connection("Receiving live data", true);
  if (actor === "active" && result.team) { team = String(result.team); document.querySelector(`input[name="team"][value="${team}"]`).checked = true; }
  document.body.classList.toggle("red", team === "200");
  $("perspective-dot").className = `team-dot ${team === "100" ? "blue" : "red"}`;
  const subject = result.actor === "team" ? "Team" : result.subject || "Player";
  $("perspective").textContent = `${subject} · ${ROLE_NAMES[result.role] || "Select role"} · ${team === "100" ? "Blue" : "Red"}`;
  const groupMetrics = latest.model.per_group?.[result.role];
  if (groupMetrics) {
    const predictions = groupMetrics.voluntary_prediction_counts || groupMetrics.prediction_counts;
    const total = predictions.reduce((sum, count) => sum + count, 0);
    const largest = Math.max(...predictions);
    if (total && (largest / total >= 0.8 || groupMetrics.prediction_coverage < 3)) {
      $("model-warning").hidden = false;
      $("model-warning-text").textContent = groupMetrics.prediction_coverage < 3 ? `For this subject/role, the policy selected only ${groupMetrics.prediction_coverage} distinct actions on the development split. Broad strategic choice has not been demonstrated.` : `For this subject/role, the policy chose “${latest.actions[predictions.indexOf(largest)]}” on ${(largest / total * 100).toFixed(0)}% of development observations. Useful choice diversity is not established.`;
    }
  }
  const ready = ["recommendation", "respawning"].includes(result.status);
  if (!ready) {
    const titles = { needs_role: "Choose your role.", needs_player: "Choose a player.", opening: "Follow your opening plan.", insufficient_support: "No supported action yet.", invalid_context: "Check your player selection." };
    clearDecision(titles[result.status] || "No current recommendation.", result.message || "Waiting for enough context.", "CONTEXT NEEDED");
  } else {
    if (!Array.isArray(result.scores) || !Object.hasOwn(latest.actions, result.action_id)) throw Error("Invalid action response");
    document.querySelector(".decision").classList.add("active");
    $("decision-title").textContent = result.action;
    $("decision-description").textContent = result.instruction;
    $("decision-check").textContent = result.check; $("decision-check").hidden = false;
    $("decision-horizon").textContent = `Next ~${result.horizon_seconds}s · Reassess when the situation changes`; $("decision-horizon").hidden = false;
    $("decision-tag").textContent = result.status === "respawning" ? "RESPAWNING" : "POLICY ESTIMATE";
    $("decision-kicker").textContent = "NEXT ACTION · CHECK AGAINST YOUR MAP";
    $("score-label").textContent = result.critic_target === "episode_return" ? "RETURN SCORE" : result.method === "cql" ? "Q-VALUE" : result.method === "iql" ? "POLICY SCORE" : "BEHAVIOR SCORE";
    $("score-note").textContent = result.score_meaning;
    for (const row of $("action-list").children) {
      const id = Number(row.dataset.action); const selected = id === result.action_id;
      row.classList.toggle("selected", selected);
      if (selected) row.setAttribute("aria-current", "true"); else row.removeAttribute("aria-current");
      row.querySelector(".action-value").textContent = Number.isFinite(result.scores[id]) ? result.scores[id].toFixed(3) : "—";
    }
    const alternatives = Object.entries(latest.actions).map(([key, name]) => ({ id: Number(key), name, score: result.scores[Number(key)] })).filter(value => value.id !== result.action_id && Number.isFinite(value.score)).sort((a, b) => b.score - a.score).slice(0, 3);
    $("alternatives").replaceChildren();
    for (const [index, option] of alternatives.entries()) {
      const row = document.createElement("li"); row.className = "alternative-row";
      const rank = document.createElement("span"); rank.className = "alternative-rank"; rank.textContent = String(index + 2).padStart(2, "0");
      const label = document.createElement("span"); label.textContent = option.name; row.append(rank, label); $("alternatives").append(row);
    }
    if (!alternatives.length) { const item = document.createElement("li"); item.className = "empty-options"; item.textContent = "No other supported action for this observation."; $("alternatives").append(item); }
  }
  const macro = result.macro_recommendation;
  $("macro-direction").hidden = !macro;
  if (macro) $("macro-direction").textContent = `Team direction: ${macro.action || macro.message || "insufficient evidence"}. ${result.coordination?.status === "conflict" ? "The proposals point in different directions; review both." : result.coordination?.status === "aligned" ? "Your role proposal supports this direction." : "Your role proposal is assessed separately."}`;
  if (Number.isFinite(result.game_time)) {
    const seconds = Math.floor(result.game_time);
    const clock = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    $("game-clock").textContent = clock; $("decision-time").textContent = `Observed at ${clock}`;
  }
  $("actor-hint").textContent = actor === "team" ? "Team-level options describe a shared direction, not individual micro commands." : "Individual advice uses this player's reported role or your explicit role selection.";
  updateFreshness();
}

function stop() {
  clearTimeout(pollTimer); clearTimeout(expiryTimer);
  if (pending) pending.abort(); pending = null; latest = null;
}
async function poll() {
  if (paused || document.hidden || pending) return;
  const request = new AbortController(); pending = request;
  const timeout = setTimeout(() => request.abort(), 12000);
  let delay = 5000;
  try {
    const query = new URLSearchParams({ team, actor }); if (role && actor !== "team") query.set("role", role);
    const response = await fetch(`/api/state?${query}`, { cache: "no-store", signal: request.signal });
    if (!response.ok) throw Error("Dashboard unavailable");
    const data = await response.json();
    if (pending !== request || paused || document.hidden) return;
    if (!Number.isFinite(data.valid_for_ms) || !Number.isFinite(data.checked_at)) throw Error("Invalid observation clock");
    latest = data; lifetime = Math.max(0, data.valid_for_ms); expiresAt = Date.now() + lifetime; delay = Math.max(250, lifetime);
    $("transport").textContent = data.transport === "windows" ? "Windows bridge" : "Local API";
    $("interval").textContent = `${data.interval_seconds}s`;
    $("last-check").textContent = new Date(data.checked_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    updateActors(data); buildOptions(data.actions); renderModel(data.model);
    if (lifetime > 0) render(); else { latest = null; connection("Refreshing"); clearDecision("Refreshing the picture.", "Waiting for a fresh observation.", "UPDATING"); }
    expiryTimer = setTimeout(() => { latest = null; connection("Refreshing"); clearDecision("Refreshing the picture.", "The previous advice has expired. Waiting for fresh data.", "UPDATING"); }, lifetime);
  } catch (error) {
    if (pending !== request || paused || document.hidden) return;
    clearTimeout(expiryTimer); latest = null; connection("Dashboard disconnected"); clearDecision("Connection interrupted.", "Keep the dashboard command running. Retrying automatically.", "OFFLINE");
  } finally {
    clearTimeout(timeout);
    if (pending === request) { pending = null; if (!paused && !document.hidden) pollTimer = setTimeout(poll, delay); }
  }
}
function refreshContext() { stop(); clearDecision("Updating your context.", "Loading advice for the selected subject.", "UPDATING"); if (!paused && !document.hidden) poll(); }
document.querySelectorAll('input[name="team"]').forEach(input => input.addEventListener("change", () => { team = input.value; actor = "team"; $("actor").value = "team"; $("role").disabled = true; refreshContext(); }));
$("actor").addEventListener("change", () => { actor = $("actor").value; role = ""; $("role").value = ""; $("role").disabled = actor === "team"; const player = latest?.players?.find(value => value.id === actor); if (player) { team = String(player.team); document.querySelector(`input[name="team"][value="${team}"]`).checked = true; } refreshContext(); });
$("role").addEventListener("change", () => { role = $("role").value; refreshContext(); });
$("pause").addEventListener("click", () => { paused = !paused; $("pause-label").textContent = paused ? "Resume" : "Pause"; $("pause").setAttribute("aria-pressed", String(paused)); stop(); if (paused) { connection("Updates paused"); clearDecision("Updates paused.", "Resume for fresh advice.", "PAUSED"); } else poll(); });
$("focus").addEventListener("click", () => { const focused = document.body.classList.toggle("focus"); $("focus").setAttribute("aria-pressed", String(focused)); $("focus").querySelector("span").textContent = focused ? "Full view" : "Focus view"; $("focus-note").hidden = !focused; });
document.addEventListener("keydown", event => { if (event.repeat || event.altKey || event.ctrlKey || event.metaKey || event.target.isContentEditable || (event.target instanceof Element && event.target.closest("input, textarea, select"))) return; const id = { f: "focus", p: "pause" }[event.key.toLowerCase()]; if (id) { event.preventDefault(); $(id).click(); } });
document.addEventListener("visibilitychange", () => { stop(); if (document.hidden || paused) { connection("Updates paused"); clearDecision("Updates on hold.", "Return and resume for fresh advice.", "PAUSED"); } else poll(); });
window.addEventListener("pagehide", stop);
window.addEventListener("pageshow", event => { if (event.persisted && !paused && !document.hidden) poll(); });
setInterval(() => { if (!document.hidden) updateFreshness(); }, 1000);
poll();
