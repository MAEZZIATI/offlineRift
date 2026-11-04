"""Automated flags must remain causal and must not pretend to be correctness labels."""

import copy

from offline_rift.audit import audit_corpus, diagnose_snapshot, herald_timing_audit
from offline_rift.pipeline.data_pipeline import TimelineProcessor
from offline_rift.pipeline.features import FeatureConfig


def test_audit_death_flag_ignores_future_events(match_payloads):
    match, timeline = match_payloads()
    snapshot = TimelineProcessor(FeatureConfig()).parse(match, timeline)[5]
    assert snapshot.time == 50
    events = [
        {"type": "CHAMPION_KILL", "timestamp": 40000, "victimId": 1},
        {"type": "CHAMPION_KILL", "timestamp": 60000, "victimId": 2},
    ]
    diagnosis = diagnose_snapshot(snapshot, 100, events)
    assert diagnosis["observed_deaths_since_position_frame"] == [1]
    assert diagnosis["frame_age_seconds"] == 50
    assert diagnosis == diagnose_snapshot(snapshot, 100, events[:1])
    changed = copy.deepcopy(snapshot)
    changed.players[0]["position"] = None
    assert diagnose_snapshot(changed, 100, events)["missing_positions"] == 1


def test_corpus_audit_preserves_failures_and_detects_timing_contradiction(tmp_path, match_payloads):
    from offline_rift.workflow import atomic_json

    match, timeline = match_payloads("KR_audit")
    timeline["metadata"] = {"matchId": "KR_audit"}
    match["info"].update(queueId=420, mapId=11, gameVersion="16.20.1")
    for player in match["info"]["participants"]:
        player["puuid"] = f"player-{player['participantId']}"
    timeline["info"]["frames"][-1]["events"].append(
        {"type": "ELITE_MONSTER_KILL", "monsterType": "RIFTHERALD", "timestamp": 120000}
    )
    atomic_json(tmp_path / "KR_audit.json", {"match": match, "timeline": timeline})
    report = audit_corpus(tmp_path, ["KR_audit"], baseline_ids=["KR_audit"])
    assert report["structural_failure_matches"] == []
    assert report["matches_sharing_any_baseline_participant"] == 1
    config = FeatureConfig(herald_despawn=100)
    timing = herald_timing_audit(tmp_path, ["KR_audit"], config)
    assert len(timing["kills_at_or_after_configured_cutoff"]) == 1
    timeline["metadata"]["matchId"] = "WRONG"
    atomic_json(tmp_path / "KR_audit.json", {"match": match, "timeline": timeline})
    assert audit_corpus(tmp_path, ["KR_audit"])["structural_failure_matches"][0]["issues"] == [
        "identity_mismatch"
    ]
    assert (tmp_path / "KR_audit.json").exists()  # Diagnostics never clean up user data.
