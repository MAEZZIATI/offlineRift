import numpy as np
import pytest
from offline_rift.inference.advice import AdviceController
from offline_rift.inference.coach import MacroCoach
from offline_rift.training.coach import reselect_coach, select_candidate, train_coach
from rift_common.artifacts import atomic_json, file_digest, read_json
from rift_common.behavior import team_activity
from rift_common.coaching import POLICY_SCOPES, alignment, withhold_conflicting_role
from rift_common.dataset import DatasetBuilder, episode_returns, load_dataset
from rift_common.timeline import read_match


def macro_match(factory, name):
    match, timeline = factory(name)
    for player in (2, 3, 4, 7, 8, 9):
        timeline["info"]["frames"][3]["participantFrames"][str(player)]["position"] = {
            "x": 1400,
            "y": 11000,
        }
    return match, timeline


def test_selection_allows_imitation_and_rejects_collapsed_high_scoring_candidate():
    ordinary = {
        "name": "cql",
        "development_gate_passed": True,
        "selection_score": 0.1,
        "voluntary_concentration": 0.3,
    }
    imitation = {**ordinary, "name": "bc", "selection_score": 0.2}
    collapsed = {
        **ordinary,
        "name": "collapsed",
        "development_gate_passed": False,
        "selection_score": 0.9,
    }
    assert select_candidate([ordinary, imitation, collapsed]) is imitation


def test_macro_requires_coherent_executed_evidence_not_formation_fallback(match_payloads):
    observations = read_match(*macro_match(match_payloads, "KR_macro"))
    assert not team_activity(observations[1], observations[2], 100).identified
    evidence = team_activity(observations[2], observations[3], 100)
    assert evidence.identified and evidence.representative.action == 5
    assert (
        alignment(
            {"status": "recommendation", "action_id": 8},
            {"status": "recommendation", "action_id": 9},
        )
        == "conflict"
    )


def test_retained_role_advice_uses_fresh_macro_direction_and_rechecks_alignment():
    controller = AdviceController()
    first = {
        "status": "recommendation",
        "action_id": 3,
        "team": 100,
        "actor": "p",
        "role": "BOTTOM",
        "game_time": 100,
        "scores": [0.1] * 15,
        "macro_recommendation": {"status": "recommendation", "action_id": 8},
        "coordination": {"status": "independent"},
    }
    controller.select(first)
    changed = {
        **first,
        "action_id": 6,
        "game_time": 105,
        "macro_recommendation": {"status": "recommendation", "action_id": 6},
        "coordination": {"status": "aligned"},
    }
    result = controller.select(changed)
    assert result["action_id"] == 3
    assert result["macro_recommendation"]["action_id"] == 6
    assert result["coordination"]["status"] == "independent"


def test_conflicting_role_is_withheld_without_inventing_an_alternative():
    original = {
        "status": "recommendation",
        "action_id": 8,
        "action": "Dragon",
        "scores": [0.1] * 15,
        "macro_recommendation": {"status": "recommendation", "action_id": 9},
    }
    guarded = withhold_conflicting_role(original)
    assert guarded["status"] == "coordination_conflict"
    assert "action_id" not in guarded and "scores" not in guarded
    assert guarded["macro_recommendation"]["action_id"] == 9
    assert guarded["coordination"]["withheld_action_id"] == 8
    assert original["action_id"] == 8
    independent = {**original, "action_id": 3}
    assert withhold_conflicting_role(independent) == independent


def test_six_real_models_are_scope_trained_reload_route_and_detect_corruption(
    tmp_path, match_payloads, catalog, monkeypatch
):
    builder = DatasetBuilder()
    for i in range(4):
        builder.add_match(*macro_match(match_payloads, f"KR_{i}"), catalog)
    path = tmp_path / "data.npz"
    builder.save(path, catalog=catalog, provenance={"fingerprint": "fixture"})
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, {"holdout_matches": ["KR_3"], "steps": 1})
    result = train_coach(
        path,
        tmp_path / "coach",
        steps=1,
        holdout_matches=("KR_3",),
        protocol_path=protocol,
        study_design="relative-context-v2",
    )
    assert result["status"] == "complete" and set(result["selected"]) == set(POLICY_SCOPES)
    assert (
        result["protocol_sha256"]
        == file_digest(tmp_path / "coach/protocol.json")
        == file_digest(protocol)
    )
    coach = MacroCoach(result["bundle"])
    arrays, _ = load_dataset(path)
    predictions, _ = coach.predict_batch(
        arrays["observations"], arrays["groups"], arrays["action_masks"]
    )
    for scope, policy in coach.policies.items():
        assert policy.config["training_groups"] == [scope]
        other_groups = [
            i
            for i, name in enumerate(
                ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", "UNKNOWN", "TEAM")
            )
            if name != scope
        ]
        assert not policy.counts[other_groups].any()
        assert policy.metadata["training_report"]["holdout_matches"] == ["KR_3"]
        assert not any(
            "KR_3" in policy.metadata["training_report"][k]
            for k in ("training_matches", "validation_matches")
        )
    returns = episode_returns(arrays, 0.9)
    np.testing.assert_array_equal(
        returns[arrays["terminals"]], arrays["rewards"][arrays["terminals"]]
    )
    assert predictions.shape == arrays["actions"].shape
    from copy import deepcopy

    from rift_research.game_review import replay_game, review_metrics

    match, timeline = macro_match(match_payloads, "EUW1_reserved")
    payload = {"match": match, "timeline": timeline}
    decisions = replay_game(payload, coach)
    assert {r["scope"] for r in decisions} == set(POLICY_SCOPES)
    assert len({(r["team"], r["actor"]) for r in decisions}) == 12
    assert len(review_metrics(decisions)) == 6
    from rift_common.items import ItemCatalog

    altered_items = deepcopy(catalog.data)
    altered_items["1001"]["gold"]["total"] += 10000
    exact_catalog = ItemCatalog(altered_items)
    top = coach.policies["TOP"]
    original_predict = top.predict_batch
    encoded = []

    def capture(observed, groups, masks):
        encoded.append(observed.copy())
        return original_predict(observed, groups, masks)

    observations = read_match(match, timeline)
    with monkeypatch.context() as patch:
        patch.setattr(top, "predict_batch", capture)
        coach.recommend(
            observations[3], 100, "1", role="TOP", previous=observations[2], catalog=exact_catalog
        )
    index = top.encoder.names.index("self_items")
    expected = exact_catalog.power(list(observations[3].player("1").items), 2500)[0] / 20000
    assert encoded[-1][0, index] == pytest.approx(expected)
    assert top.encoder.catalog.data == catalog.data  # Frozen default is not mutated.
    changed = deepcopy(payload)
    changed["timeline"]["info"]["frames"][5]["participantFrames"]["1"]["level"] = 18
    later = replay_game(changed, coach)
    assert [r for r in decisions if r["time"] < 300] == [r for r in later if r["time"] < 300]
    # A training match may never be presented as an unseen one-game review.
    match["metadata"]["matchId"] = "KR_0"
    with pytest.raises(ValueError, match="excluded"):
        replay_game(payload, coach)
    with pytest.raises(ValueError, match="No development"):
        reselect_coach(tmp_path / "coach", tmp_path / "missing-region", protocol_path=protocol)
    assert read_json(tmp_path / "missing-region/study.json")["status"] == "failed"
    assert (
        MacroCoach(result["bundle"]).metadata["development_gate_passed"]
        == result["development_gate_passed"]
    )
    terminal = np.flatnonzero(arrays["terminals"])[0]
    assert returns[terminal - 1] == pytest.approx(0.9 * returns[terminal])
    manifest = read_json(tmp_path / "coach/model/coach.json")
    assert len(manifest["policies"]) == 6
    assert coach.metadata["training_report"]["optimizer_steps"] == 6
    with (tmp_path / "coach/model/policies/TOP/policy.pt").open("ab") as stream:
        stream.write(b"invalid")
    with pytest.raises(ValueError, match="integrity"):
        MacroCoach(result["bundle"])
