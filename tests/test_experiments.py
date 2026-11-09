import numpy as np
import pytest
from offline_rift.training.iql import expectile_loss
from offline_rift.training.trainer import TrainConfig
from rift_common.network import OptionNetwork


def test_expectile_asymmetry_and_independent_actor_gradients():
    torch = pytest.importorskip("torch")
    torch.testing.assert_close(expectile_loss(torch.tensor([-2.0, 2.0]), 0.7), torch.tensor(2.0))
    torch.testing.assert_close(expectile_loss(torch.tensor([2.0]), 0.7), torch.tensor(2.8))
    network = OptionNetwork(4, separate_actor=True)
    _, logits = network(torch.ones(2, 4))
    logits.sum().backward()
    assert all(parameter.grad is None for parameter in network.encoder.parameters())
    assert any(parameter.grad is not None for parameter in network.actor_encoder.parameters())


@pytest.mark.parametrize(
    "suite,run_count", [("algorithms", 12), ("robustness", 8), ("readiness", 10)]
)
def test_study_reuses_split_reloads_all_algorithms_and_preserves_latest(
    tmp_path, match_payloads, catalog, monkeypatch, suite, run_count
):
    from offline_rift.workflow import Workflow
    from rift_common.dataset import DatasetBuilder
    from rift_research.experiments import run_study
    from rift_research.sensitivity import analyze_study

    builder = DatasetBuilder()
    for i in range(10):
        builder.add_match(*match_payloads(f"KR_{i}"), catalog)
    path = tmp_path / "data.npz"
    builder.save(path, catalog=catalog, provenance={"fingerprint": "fixture"})
    monkeypatch.setattr(Workflow, "__init__", lambda self, config: None)
    monkeypatch.setattr(Workflow, "build", lambda self: {"path": str(path)})
    report = run_study(
        "unused", tmp_path / "study", steps=2, seeds=(42, 43), samples=2, suite=suite
    )
    assert report["status"] == "complete" and len(report["runs"]) == run_count
    assert "offline_rift.training.iql" in report["source_sha256"]
    assert (tmp_path / "study/sources/offline_rift/training/iql.py").is_file()
    for run in report["runs"]:
        assert run["per_group"]
        assert np.isfinite(run["score_quantiles"]).all()
    assert report["policy_winrate"] is None
    stress = analyze_study(tmp_path / "study", tmp_path / "sensitivity")
    assert stress["status"] == "complete" and len(stress["runs"]) == run_count
    assert stress["seed_agreement"][next(iter(report["variants"]))]["43"]["agreement"] >= 0
    assert set(stress["runs"][0]["scenarios"]) == {"unobserved_health", "cold_start_history"}
    for run in stress["runs"]:
        for scenario in run["scenarios"].values():
            assert all(
                len(group["targets_made_infeasible"]) == 15
                for group in scenario["per_group"].values()
            )
    if suite == "readiness":
        assert len(stress["development_screen"]) == 3
        assert all(
            not result["production_strategy_validated"]
            for result in stress["development_screen"].values()
        )
    assert not list(tmp_path.rglob("latest.json"))
    with pytest.raises(FileExistsError):
        run_study("unused", tmp_path / "study", steps=2, samples=2)
    if suite == "robustness":
        from offline_rift.inference.policy import NextActionPolicy
        from rift_common.artifacts import read_json
        from rift_research.calibration import calibrate_coverage

        calibrated = calibrate_coverage(tmp_path / "study", tmp_path / "coverage", targets=(1, 0.5))
        calibration, evaluation = map(
            set, (calibrated["calibration_matches"], calibrated["evaluation_matches"])
        )
        policy = NextActionPolicy(report["runs"][0]["bundle"])
        assert calibration and evaluation and calibration.isdisjoint(evaluation)
        assert calibration | evaluation == set(
            policy.metadata["training_report"]["validation_matches"]
        )
        assert (calibration | evaluation).isdisjoint(
            policy.metadata["training_report"]["training_matches"]
        )
        for run in calibrated["runs"]:
            scenarios = run["scenarios"]
            for original, transferred in zip(
                scenarios["observed"], scenarios["cold_start_observed_thresholds"]
            ):
                assert original["thresholds_by_group"] == transferred["thresholds_by_group"]
                assert original["thresholds_by_group"]["UNKNOWN"] is None
            for curves in scenarios.values():
                assert curves[0]["evaluation"]["coverage"] >= curves[1]["evaluation"]["coverage"]
        assert calibrated["status"] == "complete" and calibrated["policy_winrate"] is None
        from rift_research.decision_regret import study_decision_regret

        regret = study_decision_regret(tmp_path / "study", tmp_path / "regret")
        assert regret["status"] == "complete" and len(regret["variants"]) == 4
        for variant in regret["variants"]:
            assert (
                variant["methods"]["single_observed"]["curves"][0]["evaluation"]
                == variant["baseline"]
            )
        conditional = calibrate_coverage(
            tmp_path / "study", tmp_path / "conditional", targets=(0.9,), conditional_evidence=True
        )
        for run in conditional["runs"]:
            for curves in run["conditional_evidence"]["scenarios"].values():
                assert curves[0]["evaluation"]["coverage"] == 0
                assert not curves[0]["coverage_screen_pass"]
        with pytest.raises(FileExistsError):
            calibrate_coverage(tmp_path / "study", tmp_path / "coverage")
        original_init = NextActionPolicy.__init__

        def overlapping_split(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.metadata["training_report"]["training_matches"] += list(calibration)

        monkeypatch.setattr(NextActionPolicy, "__init__", overlapping_split)
        with pytest.raises(ValueError, match="splits differ"):
            calibrate_coverage(tmp_path / "study", tmp_path / "invalid-split")
        assert read_json(tmp_path / "invalid-split/report.json")["status"] == "failed"


@pytest.mark.parametrize(
    "settings",
    [
        {"expectile": 1},
        {"target_tau": 0},
        {"advantage_clip": 0},
        {"learning_rate": float("nan")},
        {"alpha": float("inf")},
        {"steps": 1.5},
    ],
)
def test_invalid_iql_settings_are_rejected(settings):
    with pytest.raises(ValueError, match="configuration"):
        TrainConfig(**settings)


def test_new_match_evaluation_excludes_training_and_development(tmp_path, match_payloads, catalog):
    from offline_rift.training.trainer import PolicyTrainer
    from rift_common.dataset import DatasetBuilder
    from rift_research.evaluation import evaluate_policy

    builder = DatasetBuilder()
    for i in range(4):
        builder.add_match(*match_payloads(f"KR_{i}"), catalog)
    data, bundle = tmp_path / "old.npz", tmp_path / "policy"
    builder.save(data, catalog=catalog, provenance={"fingerprint": "old"})
    PolicyTrainer(TrainConfig(steps=2, minimum_support=1, cpu_threads=1)).train(data, bundle)
    with pytest.raises(ValueError, match="No new matches"):
        evaluate_policy(data, bundle, tmp_path / "empty", new_matches=True, samples=2)
    assert not (tmp_path / "empty").exists()
    builder.add_match(*match_payloads("KR_new"), catalog)
    expanded = tmp_path / "expanded.npz"
    builder.save(expanded, catalog=catalog, provenance={"fingerprint": "expanded"})
    with pytest.raises(ValueError, match="model match split"):
        evaluate_policy(expanded, bundle, tmp_path / "wrong", samples=2)
    result = evaluate_policy(expanded, bundle, tmp_path / "external", new_matches=True, samples=2)
    assert result["evaluated_matches"] == ["KR_new"]
    assert result["excluded_previously_seen_matches"] == [f"KR_{i}" for i in range(4)]
    assert result["evaluation_mode"] == "new_matches"
    assert result["policy_winrate"] is None
    assert 0 <= result["majority_baseline"] <= 1
    assert all("training_action_support" in group for group in result["per_group"].values())
    with np.load(tmp_path / "external/predictions.npz") as archive:
        assert set(archive["match_ids"]) == {"KR_new"}
