from rift_common.objectives import ObjectiveRules, ObjectiveTracker


def test_soul_counts_same_team_and_unknown_subtype_masks_availability():
    tracker = ObjectiveTracker(ObjectiveRules())
    for t, team in ((310, 100), (620, 200), (930, 100), (1240, 200)):
        tracker.consume("dragon", t, team, is_elder=False)
    assert not tracker.elder_phase
    tracker.consume("dragon", 1500, 100, is_elder=None)
    assert not tracker.state(1501)["dragon"]["enabled"]
    tracker.consume("dragon", 1900, 100, is_elder=True)
    assert tracker.state(1901)["dragon"]["eta"] == 359
    assert tracker.state(1901)["elder_buff"]["remaining"] == 149


def test_herald_single_spawn_and_conservative_uncertainty():
    tracker = ObjectiveTracker(ObjectiveRules())
    assert tracker.state(899)["herald"]["eta"] == 1
    assert tracker.state(900)["herald"]["available"]
    assert not tracker.state(1180)["herald"]["enabled"]
    assert not tracker.state(1200)["herald"]["enabled"]
    tracker.consume("herald", 1000, 100)
    assert not tracker.state(1001)["herald"]["enabled"]
