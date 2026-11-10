"""Audit combat-label overlap using observed events/endpoints; never infer initiation."""

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from rift_common.actions import GROUPS, Action
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.behavior import player_behavior, zone
from rift_common.dataset import load_dataset
from rift_common.timeline import read_match


def audit_combat(dataset, raw, output, *, samples_per_stratum=5):
    output, raw = Path(output), Path(raw)
    if output.exists():
        raise FileExistsError(output)
    if samples_per_stratum < 1:
        raise ValueError("Positive review sample count required")
    arrays, metadata = load_dataset(dataset)
    players = arrays["actors"] != "team"
    counts = Counter(player_transitions=int(players.sum()), combat_transitions=0)
    by_role = {
        role: Counter(transitions=int(np.sum(players & (arrays["groups"] == i))))
        for i, role in enumerate(GROUPS)
        if role not in {"UNKNOWN", "TEAM"}
    }
    rows = defaultdict(list)
    for index in np.flatnonzero(players & (arrays["actions"] == Action.SKIRMISH)):
        rows[str(arrays["match_ids"][index])].append(index)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "dataset_sha256": file_digest(dataset),
        "source_sha256": file_digest(__file__),
        "matches_processed": 0,
        "matches_in_dataset": len(set(arrays["match_ids"])),
        "counts": {},
        "by_role": {},
        "without_combat_priority": {},
        "limitations": [
            "Death-only participation does not identify forced or ambushed combat",
            "CS gains and endpoint movement can coexist with fighting; neither reveals precise timing",
            "Removing combat-event priority is a label diagnostic, not a validated replacement target",
            "Future interval observations are used only in this target-evidence audit",
        ],
    }
    atomic_json(output / "report.json", report)
    alternative = Counter()
    samples, seen = defaultdict(list), Counter()
    fractions = []
    rng = np.random.default_rng(20261105)
    try:
        for match_id, indices in sorted(rows.items()):
            if Path(match_id).name != match_id:
                raise ValueError("Invalid cached match ID")
            path = raw / (match_id + ".json")
            if file_digest(path) != metadata["provenance"]["input_sha256"][match_id]:
                raise ValueError("Raw input differs from dataset provenance")
            payload = read_json(path)
            observed = read_match(payload["match"], payload["timeline"])
            frames = {round(frame.time, 3): i for i, frame in enumerate(observed)}
            for row in indices:
                index = frames[round(float(arrays["timestamps"][row]), 3)]
                current, following = observed[index : index + 2]
                if not np.isclose(following.time, arrays["next_timestamps"][row]):
                    raise ValueError("Audit interval differs from stored transition")
                actor = str(arrays["actors"][row])
                before, after = current.player(actor), following.player(actor)
                events = [
                    e
                    for e in following.events
                    if e["kind"] == "kill"
                    and current.time < e["time"] <= following.time
                    and actor in (e.get("actor"), e.get("victim"), *e.get("assists", []))
                ]
                if not events:
                    raise ValueError("Combat label has no observed participation evidence")
                died = any(e.get("victim") == actor for e in events)
                credited = any(
                    e.get("actor") == actor or actor in e.get("assists", []) for e in events
                )
                category = (
                    "death_only"
                    if died and not credited
                    else "death_and_kill_or_assist"
                    if died
                    else "kill_or_assist_without_death"
                )
                lane_gain = (
                    before.lane_cs is not None
                    and after.lane_cs is not None
                    and after.lane_cs > before.lane_cs
                )
                jungle_gain = (
                    before.jungle_cs is not None
                    and after.jungle_cs is not None
                    and after.jungle_cs > before.jungle_cs
                )
                first = (min(e["time"] for e in events) - current.time) / (
                    following.time - current.time
                )
                fractions.append(first)
                masked = replace(
                    following, events=tuple(e for e in following.events if e["kind"] != "kill")
                )
                other = player_behavior(current, masked, actor)
                alternative[Action(other.action).name if other is not None else "UNKNOWN"] += 1
                flags = {
                    category: 1,
                    "combat_transitions": 1,
                    "lane_cs_also_increased": int(lane_gain),
                    "jungle_cs_also_increased": int(jungle_gain),
                    "first_combat_in_final_quarter": int(first > 0.75),
                    "coarse_area_changed": int(
                        zone(before.position, before.team) != zone(after.position, after.team)
                    ),
                }
                counts.update(flags)
                role = GROUPS[int(arrays["groups"][row])]
                by_role[role].update(flags)
                stratum = role + ":" + category
                seen[stratum] += 1
                sample = {
                    "match_id": match_id,
                    "actor": actor,
                    "role": role,
                    "category": category,
                    "start": current.time,
                    "end": following.time,
                    "start_position": before.position,
                    "end_position": after.position,
                    "lane_cs": [before.lane_cs, after.lane_cs],
                    "jungle_cs": [before.jungle_cs, after.jungle_cs],
                    "events": events,
                    "alternative_without_combat_priority": Action(other.action).name
                    if other is not None
                    else None,
                }
                if len(samples[stratum]) < samples_per_stratum:
                    samples[stratum].append(sample)
                else:
                    chosen = int(rng.integers(seen[stratum]))
                    if chosen < samples_per_stratum:
                        samples[stratum][chosen] = sample
            report["matches_processed"] += 1
            if report["matches_processed"] % 50 == 0:
                report["counts"] = dict(counts)
                atomic_json(output / "report.json", report)
        report.update(
            status="complete",
            first_combat_interval_fraction_quantiles=np.quantile(
                fractions, [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            if fractions
            else [],
        )
        atomic_json(
            output / "review-sample.json", {key: value for key, value in sorted(samples.items())}
        )
        return report
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        report.update(
            counts=dict(counts),
            by_role={role: dict(value) for role, value in by_role.items()},
            without_combat_priority=dict(alternative),
        )
        atomic_json(output / "report.json", report)
