from __future__ import annotations

import copy
from unittest.mock import Mock

import numpy as np
import pytest
import requests

from action_labeler import MacroActionLabeler, MDPDatasetBuilder, load_dataset
from data_pipeline import RiotAPIError, RiotDataIngestor, TimelineProcessor
from features import (
    PITS,
    ROLES,
    FeatureConfig,
    FeatureEncoder,
    ItemCatalog,
    ObjectiveTracker,
    Snapshot,
)


def test_causal_frames_events_and_terminal_boundary(match_payloads):
    match, timeline = match_payloads()
    config = FeatureConfig(profile="full")
    snapshots = TimelineProcessor(config).parse(match, timeline)
    by_time = {s.time: s for s in snapshots}
    assert [s.time for s in snapshots] == list(range(0, 125, 10)) + [125]
    assert by_time[50].players[0]["gold"] == by_time[0].players[0]["gold"]
    assert by_time[50].frame_age == 50
    assert by_time[60].players[0]["level"] == 2
    assert by_time[10].players[0]["items"] == []
    assert by_time[20].players[0]["items"] == [1001]
    assert by_time[30].players[0]["items"] == []
    assert by_time[30].objectives["dragon"]["eta"] == 270
    assert by_time[40].objectives["dragon"]["eta"] == 295
    assert by_time[20].players[0]["cooldown_remaining"] is None
    # Final metadata fields must never leak into earlier inventory or gold.
    match["info"]["participants"][0].update(goldEarned=999999, item0=9999)
    assert TimelineProcessor(config).parse(match, timeline)[0].players[0]["gold"] == 500


def test_mdp_alignment_rewards_validation_and_duplicates(tmp_path, match_payloads):
    config = FeatureConfig()
    builder = MDPDatasetBuilder(config, ItemCatalog())
    for match_id in ("EUW1_1", "EUW1_2"):
        match, timeline = match_payloads(match_id)
        snapshots = TimelineProcessor(config).parse(match, timeline)
        builder.add_match(match, snapshots)
    with pytest.raises(ValueError, match="Duplicate"):
        builder.add_match(match, snapshots)
    path = tmp_path / "dataset.npz"
    builder.save(path)
    arrays, metadata = load_dataset(path)
    assert len(arrays["actions"]) == 52
    assert arrays["terminals"].sum() == 4
    assert arrays["rewards"][arrays["terminals"]].tolist() == [1, -1, 1, -1]
    assert np.all(arrays["rewards"][~arrays["terminals"]] == 0)
    assert metadata["feature_config"]["profile"] == "live"
    with np.load(path, allow_pickle=False) as data:
        invalid = dict(data)
    invalid["next_observations"] = invalid["next_observations"].copy()
    invalid["next_observations"][0, 0] = 77
    np.savez(tmp_path / "invalid.npz", **invalid)
    with pytest.raises(ValueError, match="continuity"):
        load_dataset(tmp_path / "invalid.npz")


def test_gold_reward_and_full_feature_masks(match_payloads, tmp_path):
    config = FeatureConfig(profile="full")
    match, timeline = match_payloads()
    snapshots = TimelineProcessor(config).parse(match, timeline)
    snapshots[-1].players[0]["gold"] += 1000
    builder = MDPDatasetBuilder(config, ItemCatalog(), reward_mode="gold")
    builder.add_match(match, snapshots)
    path = tmp_path / "gold.npz"
    builder.save(path)
    arrays, _ = load_dataset(path)
    np.testing.assert_allclose(arrays["rewards"][arrays["terminals"]], [1.1, -1.1])
    encoder = FeatureEncoder(config, ItemCatalog())
    state = encoder.encode(snapshots[0], 100)
    assert state[encoder.names.index("gold_known")] == 1
    assert state[encoder.names.index("ally_0_position_known")] == 1
    assert state[encoder.names.index("ally_0_cooldown_known")] == 0


def make_geometry(allies, enemies):
    players = [
        {"id": i + 1, "team": 100 if i < 5 else 200, "position": point, "role": ROLES[i % 5]}
        for i, point in enumerate(allies + enemies)
    ]
    return Snapshot(1500, players, ObjectiveTracker(FeatureConfig()).state(1500))


@pytest.mark.parametrize(
    "allies,enemies,expected",
    [
        ([PITS["dragon"]] * 3 + [(7000, 7000)] * 2, [(10000, 10000)] * 5, 0),
        ([(3000, 14000)] + [(2000, 2000)] * 4, [PITS["dragon"]] * 5, 1),
        ([(3000, 14000)] + [(7000, 7000)] * 4, [(11000, 11000)] * 5, 2),
        ([(7000, 7000)] * 5, [(11000, 11000)] * 5, 3),
        ([(500, 500)] * 5, [(11000, 11000)] * 5, 4),
    ],
)
def test_all_five_macro_rules(allies, enemies, expected):
    assert MacroActionLabeler().label(make_geometry(allies, enemies), 100) == expected


def test_stale_or_missing_positions_lower_confidence():
    state = make_geometry([(7000, 7000)] * 5, [(11000, 11000)] * 5)
    state.frame_age = 95
    assert MacroActionLabeler().classify(state, 100) == (4, 0.1)
    state.frame_age = 0
    state.players[0]["position"] = None
    assert MacroActionLabeler().classify(state, 100) == (4, 0.1)


def response(status, data=None, headers=None):
    result = Mock(status_code=status, ok=200 <= status < 300, headers=headers or {})
    result.json.return_value = data
    return result


def test_riot_rate_limit_retry_routing_and_auth():
    session, sleep = Mock(), Mock()
    session.get.side_effect = [
        response(429, headers={"Retry-After": "3"}),
        response(200, ["EUW1_1"]),
    ]
    api = RiotDataIngestor("test-secret", "euw1", session=session, sleep=sleep)
    assert api.fetch_match_ids("a/b", count=1) == ["EUW1_1"]
    sleep.assert_called_once_with(3)
    args, kwargs = session.get.call_args
    assert args[0].startswith("https://europe.api.riotgames.com/")
    assert "a%2Fb" in args[0]
    assert kwargs["headers"] == {"X-Riot-Token": "test-secret"}


def test_retries_bounded_and_no_early_retry():
    for result in (response(429, headers={"Retry-After": "999"}), response(403)):
        session, sleep = Mock(), Mock()
        session.get.return_value = result
        with pytest.raises(RiotAPIError):
            RiotDataIngestor("key", session=session, sleep=sleep).fetch_match("id")
        sleep.assert_not_called()
    session = Mock()
    session.get.side_effect = requests.ConnectionError()
    with pytest.raises(RiotAPIError):
        RiotDataIngestor("key", session=session, max_retries=2, sleep=Mock()).fetch_match("id")
    assert session.get.call_count == 3


def test_objective_cooldown_and_buff_expiry():
    tracker = ObjectiveTracker(FeatureConfig())
    tracker.consume("baron", 1300, 100)
    assert tracker.state(1350)["baron"]["eta"] == 310
    assert tracker.state(1350)["baron_buff"]["remaining"] == 130
    assert tracker.state(1490)["baron_buff"]["remaining"] == 0
    tracker.consume("herald", 900)
    assert not tracker.state(1000)["herald"]["enabled"]


def test_rejects_incomplete_timeline(match_payloads):
    match, timeline = match_payloads()
    bad = copy.deepcopy(timeline)
    bad["info"]["frames"] = bad["info"]["frames"][:1]
    with pytest.raises(ValueError, match="Incomplete"):
        TimelineProcessor(FeatureConfig()).parse(match, bad)
