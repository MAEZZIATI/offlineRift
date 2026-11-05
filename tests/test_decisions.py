import copy
from dataclasses import replace

import numpy as np
import pytest
from rift_common.actions import Action, feasible_actions
from rift_common.behavior import player_behavior
from rift_common.dataset import DatasetBuilder, load_dataset
from rift_common.state import StateEncoder
from rift_common.timeline import read_match


def test_actual_frames_causal_items_and_no_final_stats(match_payloads, catalog):
    match, timeline = match_payloads()
    observed = read_match(match, timeline)
    assert [s.time for s in observed] == list(range(0, 481, 60))
    assert observed[0].player("1").items == ()
    assert observed[1].player("1").items == (1001,)
    assert observed[4].player("1").deaths == 0
    assert observed[5].player("1").deaths == 1
    before = StateEncoder(catalog).encode(observed[3], 100, "1", previous=observed[2])
    match["info"]["participants"][0].update(goldEarned=999999, item0=999999)
    timeline["info"]["frames"][4]["participantFrames"]["1"]["position"] = {"x": 14000, "y": 14000}
    after = read_match(match, timeline)
    np.testing.assert_array_equal(
        before, StateEncoder(catalog).encode(after[3], 100, "1", previous=after[2])
    )
    assert not any(
        "position" in name or name.endswith("_x") or "total_gold" in name
        for name in StateEncoder(catalog).names
    )


def test_recorded_actions_use_following_interval_and_count_failed_combat(match_payloads):
    match, timeline = match_payloads()
    observations = read_match(match, timeline)
    assert player_behavior(observations[2], observations[3], "1").action == Action.FARM_TOP
    assert player_behavior(observations[4], observations[5], "1").action == Action.SKIRMISH
    assert player_behavior(observations[5], observations[6], "1").action == Action.WAIT_RESPAWN
    before = observations[2]
    after = replace(
        observations[3],
        players=tuple(
            replace(p, position=(7000, 7000)) if p.id == "1" else p for p in observations[3].players
        ),
    )
    assert player_behavior(before, after, "1").action == Action.ROTATE_MID
    assert before.player("1").position == observations[2].player("1").position


def test_explicit_duration_terminal_rewards_all_roles_and_integrity(
    tmp_path, match_payloads, catalog
):
    builder = DatasetBuilder()
    match, timeline = match_payloads()
    builder.add_match(match, timeline, catalog)
    with pytest.raises(ValueError, match="Duplicate"):
        builder.add_match(match, timeline, catalog)
    path = tmp_path / "data.npz"
    builder.save(path, catalog=catalog, provenance={"fingerprint": "test"})
    arrays, metadata = load_dataset(path)
    assert len(set(arrays["groups"])) == 6  # Five player roles plus team.
    assert arrays["terminals"].sum() == 12
    assert (arrays["durations"] == 60).all()
    assert (arrays["rewards"][~arrays["terminals"]] == 0).all()
    assert sorted(arrays["rewards"][arrays["terminals"]].tolist()) == [-1] * 6 + [1] * 6
    assert metadata["action_schema"] == "next-action-v1"
    with np.load(path, allow_pickle=False) as archive:
        invalid = dict(archive)
    invalid["next_observations"] = invalid["next_observations"].copy()
    invalid["next_observations"][0, 0] = 99
    np.savez(tmp_path / "bad.npz", **invalid)
    with pytest.raises(ValueError, match="continuity"):
        load_dataset(tmp_path / "bad.npz")
    # Dropping an unresolved window must not splice a made-up transition across it.
    broken = copy.deepcopy(timeline)
    broken["info"]["frames"][3]["participantFrames"]["1"].pop("position")
    other = DatasetBuilder()
    other.add_match(match, broken, catalog)
    assert other.excluded["incomplete_behavior_episodes"] == 2
    assert len({r["episode_ids"] for r in other.rows}) == 10


def test_role_and_scope_conditioning_and_feasibility(match_payloads, catalog):
    observations = read_match(*match_payloads())
    encoder = StateEncoder(catalog)
    top = encoder.encode(observations[2], 100, "1")
    jungle = encoder.encode(observations[2], 100, "2")
    assert top[encoder.names.index("role_TOP")] == 1
    assert jungle[encoder.names.index("role_JUNGLE")] == 1
    team = encoder.encode(observations[2], 100)
    assert not feasible_actions(team, encoder.names)[0, 1:5].any()
    dead = encoder.encode(observations[5], 100, "1")
    assert np.flatnonzero(feasible_actions(dead, encoder.names)[0]).tolist() == [
        Action.WAIT_RESPAWN
    ]
    assert not feasible_actions(top, encoder.names)[0, Action.PREPARE_BARON]


def test_unknown_event_attribution_stays_masked(match_payloads, catalog):
    observed = read_match(*match_payloads())[2]
    observed = replace(
        observed,
        events=(
            {"kind": "tower", "team": None, "time": 100},
            {"kind": "kill", "victim_team": None, "time": 100},
        ),
    )
    encoder = StateEncoder(catalog)
    state = encoder.encode(observed, 100, "1")
    assert state[encoder.names.index("own_towers_lost_known")] == 0
    assert state[encoder.names.index("ally_deaths_60_known")] == 0
