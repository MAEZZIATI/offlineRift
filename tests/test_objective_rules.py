"""Pinned mechanics, uncertainty masks, team-specific soul counts and legacy fidelity."""

import numpy as np
import pytest
from rift_common.pipeline.features import (
    FeatureConfig,
    FeatureEncoder,
    ItemCatalog,
    ObjectiveTracker,
    objective_rule_spec,
)


def test_versioned_rules_are_pinned_and_patch_scoped():
    config = FeatureConfig.corrected_2026()
    assert config.objective_rules_version == "sr-2026-v3"
    assert config.first_spawns["herald"] == 900
    assert config.epic_buff_durations["elder"] == 150  # 9.24b supersedes 9.23.
    config.validate_patch("16.22.123.456")
    with pytest.raises(ValueError, match="No .* objective rules"):
        config.validate_patch("16.23.1")
    with pytest.raises(ValueError, match="does not match pinned"):
        FeatureConfig.corrected_2026(first_spawns={"dragon": 300, "baron": 1200, "herald": 840})
    assert objective_rule_spec("sr-2026-v2")["lifecycle"] == "superseded"
    assert FeatureConfig().objective_rules_version == "legacy-v1"


def test_herald_uncertainty_is_not_a_false_hard_despawn(match_payloads):
    config = FeatureConfig.corrected_2026()
    tracker = ObjectiveTracker(config)
    assert tracker.state(839)["herald"]["eta"] == 61
    assert tracker.state(840)["herald"]["eta"] == 60
    assert tracker.state(900)["herald"]["available"]
    assert tracker.state(1169)["herald"]["availability_known"]
    for timestamp in (1170, 1185, 1195, 1199):
        obj = tracker.state(timestamp)["herald"]
        assert not obj["availability_known"]
        assert not obj["enabled"] and not obj["available"]
    assert tracker.state(1200)["herald"]["availability_known"]
    assert not tracker.state(1200)["herald"]["enabled"]
    tracker.consume("herald", 1190.933, 100)
    assert tracker.state(1191)["herald"]["availability_known"]
    assert not tracker.state(1191)["herald"]["enabled"]
    from rift_common.pipeline.data_pipeline import TimelineProcessor

    match, timeline = match_payloads()
    match["info"].update(gameVersion="16.22.1", queueId=420)
    snapshot = TimelineProcessor(config).parse(match, timeline)[0]
    snapshot.objectives = ObjectiveTracker(config).state(1180)
    encoder = FeatureEncoder(config, ItemCatalog())
    values = encoder.encode(snapshot, 100)
    assert values.shape == (177,)
    np.testing.assert_array_equal(
        values[[encoder.names.index(f"herald_{key}") for key in ("eta", "available", "enabled")]], 0
    )


def test_soul_is_four_elementals_for_one_team_not_four_globally():
    tracker = ObjectiveTracker(FeatureConfig.corrected_2026())
    for time, team in ((310, 100), (620, 200), (930, 100), (1240, 200)):
        tracker.consume("dragon", time, team)
    assert not tracker.elder_phase
    assert tracker.state(1240)["dragon"]["eta"] == 300
    tracker.consume("dragon", 1550, 100)
    tracker.consume("dragon", 1860, 100)
    assert tracker.elder_phase
    assert tracker.state(1860)["dragon"]["eta"] == 360
    assert not tracker.state(2219)["dragon"]["available"]
    assert tracker.state(2220)["dragon"]["available"]
    tracker.consume("dragon", 2230, 200, is_elder=True)
    assert tracker.elemental_kills == {100: 4, 200: 2}
    assert tracker.state(2230)["dragon"]["eta"] == 360
    assert tracker.state(2230)["elder_buff"]["remaining"] == 150
    assert tracker.state(2380)["elder_buff"]["remaining"] == 0


def test_unknown_killer_or_type_cannot_invent_dragon_phase():
    tracker = ObjectiveTracker(FeatureConfig.corrected_2026())
    for time in (310, 620, 930, 1240):
        tracker.consume("dragon", time, None)
    assert not tracker.state(1240)["dragon"]["availability_known"]
    tracker.consume("dragon", 1610, 100, is_elder=True)
    assert tracker.state(1610)["dragon"]["availability_known"]
    assert tracker.state(1610)["dragon"]["eta"] == 360
    other = ObjectiveTracker(FeatureConfig.corrected_2026())
    other.consume("dragon", 310, 100, is_elder=None)
    assert not other.state(400)["dragon"]["enabled"]


def test_legacy_rules_preserve_historical_values():
    tracker = ObjectiveTracker(FeatureConfig())
    assert tracker.state(840)["herald"]["available"]
    assert not tracker.state(1170)["herald"]["enabled"]
    for time in (310, 620, 930, 1240):
        tracker.consume("dragon", time, 100)
    assert tracker.state(1240)["dragon"]["eta"] == 300
    tracker.consume("dragon", 1610, 100, is_elder=True)
    assert tracker.state(1610)["elder_buff"]["remaining"] == 150
