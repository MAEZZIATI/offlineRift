"""Measured activity consistency after advice; attention and causal adherence are unknown."""

from collections import Counter
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from rift_common.actions import Action
from rift_common.artifacts import atomic_json, file_digest, read_json


def canonical_champion(player):
    return (
        str(player.get("rawChampionName") or player.get("championName", ""))
        .removeprefix("game_character_displayname_")
        .casefold()
    )


def review_adherence(session, output, *, horizon=60, tolerance=10, match=None):
    session, output = Path(session), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if horizon <= 0 or tolerance < 0:
        raise ValueError("Positive horizon and nonnegative tolerance required")
    capture = read_json(session / "report.json")
    samples = [read_json(p) for p in sorted((session / "samples").glob("*.json"))]
    if not samples or len(samples) != capture["samples"]:
        raise ValueError("Missing or inconsistent live samples")
    times = [s["game"]["gameTime"] for s in samples]
    if any(b < a for a, b in pairwise(times)):
        raise ValueError("Session clock rolled back; review sessions separately")
    measured = None
    payload = None
    if match is not None:
        from rift_common.timeline import read_match

        payload = read_json(Path(match))
        info = payload["match"]["info"]
        expected = Counter((canonical_champion(p), p.get("teamId")) for p in info["participants"])
        actual = Counter(
            (canonical_champion(p), {"ORDER": 100, "CHAOS": 200}.get(p.get("team")))
            for p in samples[0]["players"]
        )
        start = datetime.fromisoformat(samples[0]["observed_utc"]).timestamp() - times[0]
        game_start = info.get("gameStartTimestamp", info.get("gameCreation"))
        if expected != actual or game_start is None or abs(start - game_start / 1000) > 180:
            raise ValueError("Post-game match does not bind the recorded roster and game clock")
        if info.get("gameDuration", 0) < times[-1] - 1:
            raise ValueError("Post-game match ends before the captured session")
        measured = read_match(payload["match"], payload["timeline"])
    rows, seen = [], set()
    next_eligible = {}
    for i, sample in enumerate(samples):
        decision = sample.get("decision", {})
        if decision.get("status") not in {"recommendation", "respawning"}:
            continue
        slot = decision.get("subject_slot", sample.get("active_slot"))
        key = (slot, decision.get("role"))
        if times[i] < next_eligible.get(key, -1):
            continue
        action = decision.get("action_id")
        identifier = (slot, decision.get("issued_game_time", times[i]), action)
        if identifier in seen:
            continue
        seen.add(identifier)
        next_eligible[key] = times[i] + horizon
        row = {
            "start_game_time": times[i],
            "action_id": action,
            "subject_slot": slot,
            "status": "unverifiable",
            "reason": "incomplete_horizon",
        }
        rows.append(row)
        j = next((j for j in range(i + 1, len(samples)) if times[j] >= times[i] + horizon), None)
        if j is None or times[j] > times[i] + horizon + tolerance:
            continue
        row["end_game_time"] = times[j]
        interval = samples[i : j + 1]
        if any(b - a > 15 for a, b in zip(times[i:j], times[i + 1 : j + 1])):
            row["reason"] = "observation_gap"
            continue
        if any(
            s.get("decision", {}).get("subject_slot", s.get("active_slot")) != slot
            for s in interval
        ):
            row["reason"] = "subject_changed"
            continue
        before = next((p for p in sample["players"] if p["slot"] == slot), None)
        after = next((p for p in samples[j]["players"] if p["slot"] == slot), None)
        if (
            before is None
            or after is None
            or before.get("championName") != after.get("championName")
        ):
            row["reason"] = "subject_unresolved"
            continue
        if action == int(Action.WAIT_RESPAWN):
            if before.get("isDead") is True:
                row.update(
                    status="supported",
                    reason="dead_at_issue_forced_wait; subsequent_activity_not_assessed",
                )
            elif before.get("isDead") is False:
                row.update(status="contradicted", reason="alive_at_issue")
            continue
        if before.get("isDead") is True:
            row.update(status="contradicted", reason="voluntary_advice_issued_while_dead")
            continue
        if any(
            s.get("decision", {}).get("status") not in {"recommendation", "respawning"}
            or s.get("decision", {}).get("action_id") != action
            for s in interval[:-1]
        ):
            row["reason"] = "advice_withdrawn_or_superseded"
            continue
        if measured is not None:
            from rift_common.behavior import player_activity

            pid = next(
                p["participantId"]
                for p in payload["match"]["info"]["participants"]
                if canonical_champion(p) == canonical_champion(before)
                and p["teamId"] == {"ORDER": 100, "CHAOS": 200}[before["team"]]
            )
            a = min(range(len(measured)), key=lambda k: abs(measured[k].time - times[i]))
            b = min(range(len(measured)), key=lambda k: abs(measured[k].time - times[j]))
            if (
                b != a + 1
                or abs(measured[a].time - times[i]) > tolerance
                or abs(measured[b].time - times[j]) > tolerance
            ):
                row["reason"] = "no_aligned_measured_match_interval"
                continue
            evidence = player_activity(measured[a], measured[b], str(pid))
            row["match_interval"] = [measured[a].time, measured[b].time]
            if evidence is None:
                row["reason"] = "postgame_activity_unresolved"
            elif evidence.identified:
                row.update(
                    status="supported" if action in evidence.candidates else "contradicted",
                    reason="identified_reconstructed_activity; intention_and_attention_unverified",
                )
            elif action in evidence.candidates:
                row.update(
                    status="consistent_evidence",
                    reason="overlapping_activity_candidates; unique_action_unresolved",
                )
            else:
                row["reason"] = "ambiguous_activity_does_not_verify_target"
            continue
        row["reason"] = "required_position_or_intent_unobserved"
        score_before, score_after = before.get("scores", {}), after.get("scores", {})
        gains = {
            k: score_after[k] - score_before[k]
            for k in ("creepScore", "kills", "deaths", "assists")
            if k in score_before and k in score_after
        }
        row["observed_score_changes"] = gains
        if action in {
            int(Action.FARM_TOP),
            int(Action.FARM_MID),
            int(Action.FARM_BOTTOM),
            int(Action.CLEAR_JUNGLE),
        }:
            if gains.get("creepScore", 0) > 0:
                row["reason"] = "farming_evidence_present; lane_or_jungle_unverifiable"
        elif action == int(Action.SKIRMISH):
            events = [
                e
                for e in samples[j].get("events", [])
                if e.get("EventName") == "ChampionKill"
                and times[i] < e.get("EventTime", -1) <= times[j]
            ]
            credited = any(
                e.get("KillerNameSlot") == slot or slot in e.get("AssisterSlots", [])
                for e in events
            )
            if credited:
                row.update(
                    status="consistent_evidence",
                    reason="combat_participation_observed; voluntary_intent_unverified",
                )
            elif any(e.get("VictimNameSlot") == slot for e in events):
                row["reason"] = "death_participation_only; voluntary_fight_unverified"
        elif action == int(Action.RESET) and before.get("items") != after.get("items"):
            row["reason"] = "inventory_changed; return_to_base_unverifiable"
    counts = Counter(row["status"] for row in rows)
    result = {
        "schema": "live-adherence-evidence-v1",
        "status": "complete",
        "source_session": str(session.resolve()),
        "postgame_match_sha256": file_digest(match) if match is not None else None,
        "capture_sha256": file_digest(session / "report.json"),
        "samples": len(samples),
        "horizon_seconds": horizon,
        "nonoverlapping_windows_per_subject": True,
        "counts": dict(counts),
        "windows": rows,
        "verified_following_rate": None,
        "policy_winrate": None,
        "limitations": [
            "Consistency is not proof that advice was seen, followed or caused behavior",
            "No positions, waves, camps, voluntary combat intent or safe paths inferred",
            "Most macro actions remain unverifiable from live API polls alone",
            "Supported respawn states are forced actions and not voluntary adherence",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", result)
    lines = [
        "# Advice and observed activity",
        "",
        f"Reviewed {len(rows)} nonoverlapping windows from {len(samples)} polls.",
        "",
        "| Evidence result | Windows |",
        "|---|---:|",
        *[f"| {key} | {value} |" for key, value in counts.items()],
        "",
        "No verified following percentage can be inferred from these inputs.",
        "",
        "| Start (seconds) | Action ID | Result | Evidence |",
        "|---:|---:|---|---|",
        *[
            f"| {row['start_game_time']:.1f} | {row['action_id']} | {row['status']} | {row['reason']} |"
            for row in rows
        ],
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return result
