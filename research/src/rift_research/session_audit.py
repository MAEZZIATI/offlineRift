"""Audit measured live sessions without interpreting recommendations as interventions."""

from collections import Counter
from pathlib import Path

import numpy as np
from rift_common.artifacts import atomic_json, file_digest, read_json


def audit_session(session, output, *, reported_outcome=None):
    session, output = Path(session), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if reported_outcome not in (None, "win", "loss"):
        raise ValueError("Outcome must be win or loss")
    capture = read_json(session / "report.json")
    if capture.get("schema") != "live-audit-session-v1":
        raise ValueError("Unsupported live session schema")
    paths = sorted((session / "samples").glob("*.json"))
    samples = [read_json(path) for path in paths]
    if not samples or len(samples) != capture["samples"]:
        raise ValueError("Missing or inconsistent captured samples")
    times = np.asarray([sample["game"]["gameTime"] for sample in samples], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Non-finite or rolled-back session clock")
    modes = sorted({sample["game"].get("gameMode", "unreported") for sample in samples})
    decisions = [sample.get("decision", {}) for sample in samples]
    statuses = Counter(d.get("status", "missing") for d in decisions)
    actions = Counter(d["action"] for d in decisions if "action" in d)
    voluntary = Counter(
        d["action"] for d in decisions if d.get("status") == "recommendation" and "action" in d
    )
    features = [sample.get("features", {}) for sample in samples]
    availability = {
        key: {
            "known_polls": sum(f.get(key, 0) > 0 for f in features),
            "total_polls": len(samples),
        }
        for key in sorted(
            {key for f in features for key in f if key.endswith(("_known", "_enabled"))}
        )
    }
    dead_rows = [
        i for i, f in enumerate(features) if f.get("self_dead_known", 0) and f.get("self_dead", 0)
    ]
    scores = [value for d in decisions for value in d.get("scores", []) if value is not None]
    active = []
    for sample in samples:
        matches = [p for p in sample["players"] if p["slot"] == sample.get("active_slot")]
        if len(matches) == 1:
            active.append(matches[0])
    events = [
        event
        for sample in samples
        for event in sample.get("events", [])
        if event.get("EventName") == "GameEnd"
    ]
    report = {
        "schema": "live-session-audit-v1",
        "status": "complete",
        "source_session": str(session.resolve()),
        "capture_report_sha256": file_digest(session / "report.json"),
        "sample_sha256": {path.name: file_digest(path) for path in paths},
        "bundle_sha256": capture.get("bundle_sha256"),
        "samples": len(samples),
        "first_game_time": float(times[0]),
        "last_game_time": float(times[-1]),
        "observed_span_seconds": float(times[-1] - times[0]),
        "positive_poll_gap_seconds": np.quantile(
            np.diff(times)[np.diff(times) > 0], [0, 0.5, 1]
        ).tolist()
        if np.any(np.diff(times) > 0)
        else [],
        "game_modes": modes,
        "ranked_training_scope_confirmed": False,
        "mode_mismatch": any(mode not in {"CLASSIC", "unreported"} for mode in modes),
        "statuses": dict(statuses),
        "action_counts": dict(actions),
        "voluntary_action_counts": dict(voluntary),
        "voluntary_dominant_fraction": max(voluntary.values(), default=0)
        / max(1, sum(voluntary.values())),
        "availability": availability,
        "dead_polls": len(dead_rows),
        "dead_polls_with_respawn_advice": sum(
            decisions[i].get("status") == "respawning" for i in dead_rows
        ),
        "last_observed_active_player": {
            k: active[-1][k] for k in ("championName", "team", "level", "scores") if k in active[-1]
        }
        if active
        else None,
        "game_end_observed": bool(events),
        "game_end_events": events[-1:],
        "self_reported_outcome": reported_outcome,
        "outcome_source": "user_report" if reported_outcome else "not_established",
        "finite_score_range": [min(scores), max(scores)] if scores else [],
        "policy_winrate": None,
        "limitations": [
            "One correlated session with an uncaptured opening; polls are not independent games",
            "Last observed scores are not verified final match statistics",
            "No recommendation adherence, action safety or causal win-rate improvement established",
            "Live samples cannot be substituted for measured Match-V5 training transitions",
            "Objective estimates from ranked rules are not validated for a different mode or unknown patch",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)
    summary = [
        "# Live session audit",
        "",
        f"Captured {len(samples)} polls from {times[0]:.1f}s to {times[-1]:.1f}s.",
        f"Reported game mode: {', '.join(modes)}. Outcome: {reported_outcome or 'unknown'} (user report only).",
        f"GameEnd captured: {bool(events)}. Capture status: {capture['status']}.",
        "",
        "## Recorded recommendations",
        "",
        "| Action | Polls |",
        "|---|---:|",
        *[f"| {action} | {count} |" for action, count in actions.items()],
        "",
        f"Dominant share among voluntary recommendations: {report['voluntary_dominant_fraction']:.2%}.",
        f"Respawn advice on observed-dead polls: {report['dead_polls_with_respawn_advice']}/{len(dead_rows)}.",
        "",
        "## Interpretation",
        "",
        "The reported win establishes no causal effect of the advice. The player may not have followed it.",
        "The captured mode differs from ranked training."
        if report["mode_mismatch"]
        else "Ranked queue eligibility is not established from the live game mode alone.",
        "This is useful evidence of live input",
        "availability and serving failure modes, not an independent policy-quality evaluation.",
        "GameEnd was not captured; the last sample and its scores are not a verified terminal state.",
        "",
        "## Next corrections",
        "",
        "Withhold policy advice for known unsupported modes while preserving opt-in observations.",
        "Audit combat target ambiguity and repetition before another model promotion.",
        "Use the measured availability counts for training/serving parity studies.",
        "Preserve the raw session; do not turn these polls into fabricated training transitions.",
        "",
        "The JSON report contains feature availability, source hashes and observed score diagnostics.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    return report
