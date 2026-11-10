from dataclasses import replace

import numpy as np
import pytest
from offline_rift.inference.advice import AdviceController
from rift_common.actions import Action
from rift_common.behavior import player_activity
from rift_common.dataset import DatasetBuilder, load_dataset
from rift_common.timeline import read_match


def test_overlapping_death_and_farming_retains_both_without_inventing_reset(match_payloads):
    observed = read_match(*match_payloads())
    evidence = player_activity(observed[4], observed[5], "1")
    assert evidence.death_participation and not evidence.combat_credit
    assert not evidence.identified
    assert set(evidence.candidates) == {int(Action.FARM_TOP), int(Action.SKIRMISH)}
    at_base = replace(
        observed[5],
        players=tuple(
            replace(p, position=(0, 0)) if p.id == "1" else p for p in observed[5].players
        ),
    )
    evidence = player_activity(observed[4], at_base, "1")
    assert evidence.candidates == (int(Action.SKIRMISH),)
    assert not evidence.identified


def test_ambiguous_rows_keep_exact_episodes_and_are_not_training_support(
    tmp_path, match_payloads, catalog
):
    from offline_rift.training.trainer import PolicyTrainer, TrainConfig

    builder = DatasetBuilder()
    for i in range(5):
        builder.add_match(*match_payloads(f"KR_{i}"), catalog)
    data = tmp_path / "data.npz"
    builder.save(data, catalog=catalog, provenance={"fingerprint": "fixture"})
    arrays, metadata = load_dataset(data)
    assert metadata["dataset_schema"] == "option-transitions-v3"
    assert (~arrays["action_identified"]).any()
    assert arrays["terminals"].sum() == 60
    report = PolicyTrainer(
        TrainConfig(
            steps=2, batch_size=8, minimum_support=1, cpu_threads=1, holdout_matches=("KR_4",)
        )
    ).train(data, tmp_path / "model")
    assert "KR_4" not in report["training_matches"] + report["validation_matches"]
    assert report["holdout_matches"] == ["KR_4"]
    assert report["report"]["excluded_ambiguous_training_updates"] > 0
    assert report["report"]["action_counts"][Action.SKIRMISH] == 0
    assert (
        report["report"]["retained_training_transitions"] > report["report"]["training_transitions"]
    )
    broken = dict(arrays)
    broken["action_identified"] = np.ones_like(arrays["action_identified"])
    import json

    np.savez(tmp_path / "bad.npz", **broken, metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="Ambiguous activity"):
        load_dataset(tmp_path / "bad.npz")


def test_advice_persistence_requires_support_and_clears_on_death():
    controller = AdviceController(minimum_hold=20, confirmations=2)

    def advice(t, action):
        return {
            "game_time": t,
            "team": 100,
            "actor": "p",
            "role": "BOTTOM",
            "status": "recommendation",
            "action_id": action,
            "horizon_seconds": 60,
            "scores": [0.1] * 15,
        }

    assert controller.select(advice(100, 3))["action_id"] == 3
    assert controller.select(advice(105, 6))["action_id"] == 3
    assert controller.select(advice(110, 6))["action_id"] == 3
    switched = controller.select(advice(120, 6))
    assert switched["action_id"] == 6 and switched["advice_issued_game_time"] == 120
    unsupported = advice(125, 3)
    unsupported["scores"][6] = None
    assert controller.select(unsupported)["action_id"] == 3
    dead = {**advice(130, 13), "status": "respawning"}
    assert controller.select(dead)["action_id"] == 13
    assert controller.select(advice(135, 6))["action_id"] == 6


def test_environment_loader_does_not_execute_or_replace_existing_keys(tmp_path, monkeypatch):
    from offline_rift.collection import load_api_key

    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("RIOT_API_KEY = 'literal-$(do-not-run)' # comment\nIGNORED=other\n")
    load_api_key(path)
    import os

    assert os.environ["RIOT_API_KEY"] == "literal-$(do-not-run)"
    path.write_text("RIOT_API_KEY=replacement")
    load_api_key(path)
    assert os.environ["RIOT_API_KEY"] != "replacement"


def test_score_gap_abstains_on_ties_but_never_erases_a_forced_single_option():
    from rift_common.network import rank_options

    values = np.ones((2, 15)) * 0.1
    logits = np.zeros_like(values)
    masks = np.ones((2, 15), dtype=bool)
    masks[1] = False
    masks[1, Action.WAIT_RESPAWN] = True
    predicted, _, _ = rank_options(
        values,
        logits,
        np.full((7, 15), 100),
        np.array([0, 0]),
        masks,
        algorithm="cql",
        minimum_score_gap=0.02,
    )
    assert predicted.tolist() == [-1, int(Action.WAIT_RESPAWN)]
