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


def test_study_reuses_split_reloads_all_algorithms_and_preserves_latest(
    tmp_path, match_payloads, catalog, monkeypatch
):
    from offline_rift.workflow import Workflow
    from rift_common.dataset import DatasetBuilder
    from rift_research.experiments import run_study
    from rift_research.sensitivity import analyze_study

    builder = DatasetBuilder()
    for i in range(3):
        builder.add_match(*match_payloads(f"KR_{i}"), catalog)
    path = tmp_path / "data.npz"
    builder.save(path, catalog=catalog, provenance={"fingerprint": "fixture"})
    monkeypatch.setattr(Workflow, "__init__", lambda self, config: None)
    monkeypatch.setattr(Workflow, "build", lambda self: {"path": str(path)})
    report = run_study("unused", tmp_path / "study", steps=2, seeds=(42, 43), samples=2)
    assert report["status"] == "complete" and len(report["runs"]) == 12
    assert "offline_rift.training.iql" in report["source_sha256"]
    assert (tmp_path / "study/sources/offline_rift/training/iql.py").is_file()
    for run in report["runs"]:
        assert run["per_group"]
        assert np.isfinite(run["score_quantiles"]).all()
    assert report["policy_winrate"] is None
    stress = analyze_study(tmp_path / "study", tmp_path / "sensitivity")
    assert stress["status"] == "complete" and len(stress["runs"]) == 12
    assert stress["seed_agreement"]["bc"]["43"]["agreement"] >= 0
    assert set(stress["runs"][0]["scenarios"]) == {"unobserved_health", "cold_start_history"}
    assert not list(tmp_path.rglob("latest.json"))
    with pytest.raises(FileExistsError):
        run_study("unused", tmp_path / "study", steps=2, samples=2)


@pytest.mark.parametrize("settings", [{"expectile": 1}, {"target_tau": 0}, {"advantage_clip": 0}])
def test_invalid_iql_settings_are_rejected(settings):
    with pytest.raises(ValueError, match="configuration"):
        TrainConfig(**settings)
