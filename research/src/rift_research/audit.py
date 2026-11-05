"""Reproduce weak labels and expose causal evidence for independent manual review.

Diagnostics flag fragility, not human-verified label correctness or player intent.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from rift_common.pipeline import action_labeler
from rift_common.pipeline.action_labeler import MacroActionLabeler, load_dataset
from rift_common.pipeline.data_pipeline import TimelineProcessor
from rift_common.pipeline.features import FeatureConfig, ObjectiveTracker


def audit_objective_events(raw_dir, match_ids, config):
    """Check kill events against the pre-kill clock; unknown windows are not failures."""
    violations, uncertain, counts, ignored = [], [], Counter(), Counter()
    for match in sorted(set(match_ids)):
        payload = json.loads((Path(raw_dir) / f"{match}.json").read_text())
        config.validate_patch(payload["match"]["info"]["gameVersion"])
        teams = {p["participantId"]: p["teamId"] for p in payload["match"]["info"]["participants"]}
        tracker = ObjectiveTracker(config)
        events = sorted(
            (
                e
                for f in payload["timeline"]["info"]["frames"]
                for e in f.get("events", [])
                if e.get("type") == "ELITE_MONSTER_KILL"
            ),
            key=lambda e: e["timestamp"],
        )
        for event in events:
            monster = event.get("monsterType", "").lower()
            kind = {"riftherald": "herald", "rift_herald": "herald", "baron_nashor": "baron"}.get(
                monster, monster
            )
            if kind not in {"dragon", "baron", "herald"}:
                ignored[monster] += 1
                continue
            time = event["timestamp"] / 1000
            state = tracker.state(time)[kind]
            counts[kind] += 1
            observation = {
                "match_id": match,
                "objective": kind,
                "timestamp": time,
                "pre_kill_state": state,
            }
            if not state.get("availability_known", True):
                uncertain.append(observation)
            elif not state["enabled"] or not state["available"]:
                violations.append(observation)
            team = event.get("killerTeamId", teams.get(event.get("killerId")))
            subtype = event.get("monsterSubType")
            tracker.consume(
                kind,
                time,
                team,
                is_elder=None
                if kind == "dragon" and subtype is None
                else subtype == "ELDER_DRAGON",
            )
    return {
        "objective_rules_version": config.objective_rules_version,
        "matches": len(set(match_ids)),
        "kill_counts": dict(counts),
        "known_state_contradictions": violations,
        "kills_in_unknown_availability_windows": uncertain,
        "unmodeled_monster_kill_counts": dict(ignored),
        "limitations": "One-sided consistency check: no contradictions does not prove exact spawn/expiry mechanics, absence, or label correctness",
    }


def herald_timing_audit(raw_dir, match_ids, config):
    """Recorded kills contradict an earlier hard cutoff, without inferring spawn telemetry."""
    kills, late, per_match = [], [], Counter()
    for match in sorted(set(match_ids)):
        payload = json.loads((Path(raw_dir) / f"{match}.json").read_text())
        events = [
            e
            for frame in payload["timeline"]["info"]["frames"]
            for e in frame.get("events", [])
            if e.get("type") == "ELITE_MONSTER_KILL"
            and e.get("monsterType") in {"RIFTHERALD", "RIFT_HERALD"}
        ]
        per_match[len(events)] += 1
        for event in events:
            timestamp = event["timestamp"] / 1000
            kills.append(timestamp)
            if timestamp >= config.herald_despawn:
                late.append(
                    {
                        "match_id": match,
                        "timestamp_seconds": timestamp,
                        "patch": ".".join(payload["match"]["info"]["gameVersion"].split(".")[:2]),
                    }
                )
    return {
        "matches": len(set(match_ids)),
        "herald_kills": len(kills),
        "kill_count_distribution": dict(per_match),
        "configured_first_spawn_seconds": config.first_spawns["herald"],
        "configured_cutoff_seconds": config.herald_despawn,
        "earliest_recorded_kill_seconds": min(kills) if kills else None,
        "latest_recorded_kill_seconds": max(kills) if kills else None,
        "kills_at_or_after_configured_cutoff": late,
        "meaning": "observed kill events contradict an earlier hard disable; kill times do not identify exact spawn/despawn rules",
        "documented_spawn_reference_seconds": 900,
        "spawn_reference": "https://www.leagueoflegends.com/en-au/news/game-updates/patch-25-09-notes/",
        "reference_limit": "official documented spawn change; each target patch still needs a versioned mechanics review",
    }


def audit_corpus(raw_dir, match_ids, *, baseline_ids=(), sampled_accounts=()):
    """Structural quality and selection-overlap diagnostics; never deletes raw matches."""
    raw_dir = Path(raw_dir)
    previous_players = set()
    for match in sorted(set(baseline_ids)):
        payload = json.loads((raw_dir / f"{match}.json").read_text())
        previous_players.update(
            p.get("puuid") for p in payload["match"]["info"]["participants"] if p.get("puuid")
        )
    failures, patches, roles, durations = [], Counter(), Counter(), []
    exposure = Counter()
    accounts = {p["puuid"]: p for p in sampled_accounts}
    strata, missing_positions, early_ends, overlapping = Counter(), 0, 0, 0
    all_players, old_players_in_new = set(), set()
    participant_disjoint_ids = []
    for match in sorted(set(match_ids)):
        payload = json.loads((raw_dir / f"{match}.json").read_text())
        info, timeline = payload["match"]["info"], payload["timeline"]
        players = info["participants"]
        puuids = {p.get("puuid") for p in players if p.get("puuid")}
        frames = timeline["info"]["frames"]
        issues = []
        if (
            payload["match"]["metadata"]["matchId"] != match
            or timeline.get("metadata", {}).get("matchId") != match
        ):
            issues.append("identity_mismatch")
        if info.get("queueId") != 420 or info.get("mapId") != 11:
            issues.append("wrong_queue_or_map")
        if (
            len(players) != 10
            or len(puuids) != 10
            or any(sum(p["teamId"] == team for p in players) != 5 for team in (100, 200))
        ):
            issues.append("invalid_roster")
        if (
            not frames
            or frames[0]["timestamp"] != 0
            or any(b["timestamp"] <= a["timestamp"] for a, b in pairwise(frames))
        ):
            issues.append("missing_or_nonmonotone_frames")
        expected = {str(p["participantId"]) for p in players}
        if any(set(frame.get("participantFrames", {})) != expected for frame in frames):
            issues.append("incomplete_participant_frames")
        if issues:
            failures.append({"match_id": match, "issues": issues})
        missing_positions += any(
            not pf.get("position")
            for frame in frames
            for pf in frame.get("participantFrames", {}).values()
        )
        early_ends += any(p.get("gameEndedInEarlySurrender", False) for p in players)
        overlap = puuids & previous_players
        overlapping += bool(overlap)
        if previous_players and not overlap:
            participant_disjoint_ids.append(match)
        old_players_in_new.update(overlap)
        all_players.update(puuids)
        exposure.update(puuids & set(accounts))
        strata.update(
            {
                accounts[p].get("sampling_tier", accounts[p].get("sampling_role", "unknown"))
                for p in puuids & set(accounts)
            }
        )
        patches[".".join(info["gameVersion"].split(".")[:2])] += 1
        roles.update(p.get("teamPosition", "unknown") for p in players)
        ends = [
            e["timestamp"] / 1000
            for f in frames
            for e in f.get("events", [])
            if e.get("type") == "GAME_END"
        ]
        if ends:
            durations.append(min(ends))
    return {
        "matches": len(set(match_ids)),
        "structural_failure_matches": failures,
        "patches": dict(patches),
        "participant_roles": dict(roles),
        "unique_participant_puuids": len(all_players),
        "matches_with_missing_position_values": missing_positions,
        "early_surrender_matches_retained": early_ends,
        "duration_seconds_from_game_end_events": {
            "minimum": min(durations) if durations else None,
            "maximum": max(durations) if durations else None,
        },
        "matches_sharing_any_baseline_participant": overlapping,
        "baseline_participants_seen_again": len(old_players_in_new),
        "participant_disjoint_match_ids": participant_disjoint_ids,
        "sampled_account_participation_counts": sorted(exposure.values()),
        "matches_with_sampled_current_tier_anchor": dict(strata),
        "limitations": [
            "rank strata refer to sampled current-tier anchors, not historical rank of every player",
            "stratum counts may overlap within a match",
            "new match IDs do not establish participant disjointness",
            "structural checks do not validate labels or objective mechanics",
        ],
    }


def diagnose_snapshot(snapshot, team, events):
    """Use events at/before t only; death flags do not infer respawn or current life state."""
    label, confidence = MacroActionLabeler().classify(snapshot, team)
    variants = {
        MacroActionLabeler(pit_radius=radius, mid_width=width).label(snapshot, team)
        for radius in (1800, 2000, 2200)
        for width in (1620, 1800, 1980)
    }
    frame_time = snapshot.time - snapshot.frame_age
    deaths = sorted(
        {
            int(event["victimId"])
            for event in events
            if event.get("type") == "CHAMPION_KILL"
            and event.get("victimId")
            and frame_time < event["timestamp"] / 1000 <= snapshot.time
        }
    )
    positions = [p.get("position") for p in snapshot.players]
    return {
        "reconstructed_label": label,
        "heuristic_confidence": confidence,
        "position_source_time_seconds": frame_time,
        "frame_age_seconds": snapshot.frame_age,
        "missing_positions": sum(p is None for p in positions),
        "out_of_map_positions": sum(
            p is not None and not all(0 <= x <= 15000 for x in p) for p in positions
        ),
        "threshold_variant_labels": sorted(variants),
        "threshold_sensitive": len(variants) > 1,
        "observed_deaths_since_position_frame": deaths,
        "death_flag_meaning": "position predates a recorded death; current alive/respawn state unknown",
    }


def _view(snapshot):
    return {
        "time": snapshot.time,
        "position_source_time": snapshot.time - snapshot.frame_age,
        "frame_age_seconds": snapshot.frame_age,
        "players": [
            {
                "id": p["id"],
                "team": p["team"],
                "champion": p["champion"],
                "position": p.get("position"),
            }
            for p in snapshot.players
        ],
        "objective_estimates": snapshot.objectives,
    }


def audit_labels(dataset, sheet, key, reference_report, raw_dir, output):
    from rift_common.artifacts import atomic_json

    arrays, metadata = load_dataset(dataset)
    reference = json.loads(Path(reference_report).read_text())
    training = set(reference["training_matches"])
    with Path(sheet).open(newline="") as source:
        rows = list(csv.DictReader(source))
    with Path(key).open(newline="") as source:
        keys = {row["sample_id"]: row for row in csv.DictReader(source)}
    if len({r["sample_id"] for r in rows}) != len(rows) or set(keys) != {
        r["sample_id"] for r in rows
    }:
        raise ValueError("Duplicate/mismatched audit sample IDs")
    if any(row["match_id"] not in training for row in rows):
        raise ValueError("Audit sheet contains a non-training match")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["match_id"]].append(row)
    diagnostics, packets = [], []
    for match, windows in grouped.items():
        payload = json.loads((Path(raw_dir) / f"{match}.json").read_text())
        if payload["match"]["metadata"]["matchId"] != match:
            raise ValueError("Raw audit match identity mismatch")
        patch = ".".join(payload["match"]["info"]["gameVersion"].split(".")[:2])
        config = dict(metadata["feature_config"])
        config.update(metadata["provenance"].get("patch_feature_overrides", {}).get(patch, {}))
        snapshots = TimelineProcessor(FeatureConfig(**config)).parse(
            payload["match"], payload["timeline"]
        )
        by_time = {s.time: s for s in snapshots}
        events = [
            e for frame in payload["timeline"]["info"]["frames"] for e in frame.get("events", [])
        ]
        for row in windows:
            timestamp, team = int(row["timestamp_seconds"]), int(row["team"])
            snapshot = by_time[timestamp]
            expected = int(keys[row["sample_id"]]["weak_label"])
            mask = (
                (arrays["match_ids"] == match)
                & (arrays["teams"] == team)
                & (arrays["timestamps"] == timestamp)
            )
            if mask.sum() != 1 or int(arrays["actions"][mask][0]) != expected:
                raise ValueError("Audit key does not match original dataset")
            diagnostic = diagnose_snapshot(snapshot, team, events)
            diagnostics.append(
                {
                    "sample_id": row["sample_id"],
                    "match_id": match,
                    "team": team,
                    "timestamp": timestamp,
                    "patch": patch,
                    "weak_label": expected,
                    "label_reproduced": diagnostic["reconstructed_label"] == expected,
                    **diagnostic,
                }
            )
            # Reviewer packets contain neither weak labels nor sensitivity diagnostics.
            packets.append(
                {
                    "sample_id": row["sample_id"],
                    "match_id": match,
                    "team": team,
                    "patch": patch,
                    "timestamp": timestamp,
                    "causal_events": [
                        {
                            key: event[key]
                            for key in (
                                "type",
                                "timestamp",
                                "victimId",
                                "killerId",
                                "monsterType",
                                "monsterSubType",
                                "buildingType",
                            )
                            if key in event
                        }
                        for event in events
                        if event.get("type")
                        in {"CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL"}
                        and max(0, timestamp - 60) * 1000 <= event["timestamp"] <= timestamp * 1000
                    ],
                    "causal_views": [
                        _view(s) for s in snapshots if max(0, timestamp - 60) <= s.time <= timestamp
                    ],
                }
            )
    diagnostics.sort(key=lambda r: r["sample_id"])
    packets.sort(key=lambda r: r["sample_id"])
    report = {
        "status": "complete",
        "kind": "automated weak-label evidence audit; not human annotation",
        "windows": len(diagnostics),
        "unique_matches": len(grouped),
        "human_annotations_completed": 0,
        "reproduction_mismatches": sum(not r["label_reproduced"] for r in diagnostics),
        "threshold_sensitive_windows": sum(r["threshold_sensitive"] for r in diagnostics),
        "windows_with_deaths_since_position_frame": sum(
            bool(r["observed_deaths_since_position_frame"]) for r in diagnostics
        ),
        "windows_with_missing_positions": sum(bool(r["missing_positions"]) for r in diagnostics),
        "windows_with_out_of_map_positions": sum(
            bool(r["out_of_map_positions"]) for r in diagnostics
        ),
        "frame_age_seconds": {
            "mean": float(np.mean([r["frame_age_seconds"] for r in diagnostics])),
            "maximum": max(r["frame_age_seconds"] for r in diagnostics),
        },
        "per_label": {
            str(label): {
                "windows": sum(r["weak_label"] == label for r in diagnostics),
                "threshold_sensitive": sum(
                    r["weak_label"] == label and r["threshold_sensitive"] for r in diagnostics
                ),
                "deaths_since_frame": sum(
                    r["weak_label"] == label and bool(r["observed_deaths_since_position_frame"])
                    for r in diagnostics
                ),
            }
            for label in range(12)
        },
        "class_counts_in_audit": dict(Counter(r["weak_label"] for r in diagnostics)),
        "limitations": [
            "stratified training audit is not population-frequency sampling",
            "threshold perturbations are a sensitivity test, not evidence of an incorrect label",
            "death events flag stale position risk, not current death state",
            "no replay video, intent validation or independent human agreement is inferred",
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "diagnostics.json", diagnostics)
    atomic_json(output / "review_packets.json", packets)
    from rift_common.pipeline.features import objective_rule_digest
    from rift_common.review import RUBRIC_VERSION, file_digest

    review_manifest = {
        "rubric_version": RUBRIC_VERSION,
        "packet_sha256": file_digest(output / "review_packets.json"),
        "diagnostics_sha256": file_digest(output / "diagnostics.json"),
        "dataset_fingerprint": metadata["provenance"].get("fingerprint"),
        "dataset_sha256": file_digest(Path(dataset)),
        "objective_rules_version": metadata["feature_config"].get(
            "objective_rules_version", "legacy-v1"
        ),
        "objective_rules_sha256": objective_rule_digest(
            metadata["feature_config"]["objective_rules_version"]
        )
        if metadata["feature_config"].get("objective_rules_version", "legacy-v1") != "legacy-v1"
        else None,
        "labeler_sha256": file_digest(Path(action_labeler.__file__)),
        "sample_ids": [row["sample_id"] for row in packets],
        "sampled_patches": sorted({row["patch"] for row in packets}),
        "dataset_patches": sorted(metadata["provenance"].get("patches", {})),
        "independence": "Two separate exports, no weak-label/model/other-review access before completion",
        "reviewer_kind": "Each reviewer must explicitly identify human or AI; no implied human validation",
    }
    atomic_json(output / "review_manifest.json", review_manifest)
    data = json.dumps(packets).replace("<", "\\u003c")
    template = (Path(__file__).parent / "static" / "audit.html").read_text()
    html = template.replace("__AUDIT_PACKETS__", data).replace(
        "__REVIEW_METADATA__", json.dumps(review_manifest).replace("<", "\\u003c")
    )
    (output / "review.html").write_text(html)
    atomic_json(output / "report.json", report)
    return report


def run_audit(args):
    report = audit_labels(
        args.dataset, args.sheet, args.key, args.reference_report, args.raw_dir, args.output
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"per_label", "class_counts_in_audit"}},
            indent=2,
        )
    )
