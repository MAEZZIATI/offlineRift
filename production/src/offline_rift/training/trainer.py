"""Explicit-transition CQL, discrete IQL and imitation with measured-time discount."""

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from rift_common import network as network_contract
from rift_common import observation_masks
from rift_common import support as support_contract
from rift_common.actions import ACTION_NAMES, GROUPS
from rift_common.artifacts import atomic_json, file_digest, write_bundle_manifest
from rift_common.dataset import episode_returns, load_dataset
from rift_common.metrics import behavior_metrics
from rift_common.network import OptionNetwork, rank_options
from rift_common.support import phase_counts, phase_support
from torch.nn import functional as F

from offline_rift.training import augmentation
from offline_rift.training.augmentation import ObservationAugmenter
from offline_rift.training.consistency import history_consistency
from offline_rift.training.targets import critic_targets


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "cql"
    steps: int = 10000
    batch_size: int = 256
    learning_rate: float = 3e-4
    alpha: float = 0.1
    gamma: float = 0.99
    seed: int = 42
    split_seed: int | None = None
    validation_fraction: float = 0.2
    minimum_support: int = 20
    behavior_floor: float = 0.02
    cpu_threads: int = 2
    device: str = "cpu"
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    advantage_clip: float = 100.0
    target_tau: float = 0.005
    joint_history_dropout: float = 0.0
    bounded_values: bool = False
    target_mode: str = "identified"
    minimum_score_gap: float = 0.0
    holdout_matches: tuple[str, ...] = ()
    advisory_groups: tuple[str, ...] = GROUPS
    training_groups: tuple[str, ...] = GROUPS
    critic_target: str = "td"
    history_consistency_weight: float = 0.0
    minimum_phase_support: int = 0
    evaluation_interval: int = 0
    early_stopping_patience: int = 3
    class_weight_exponent: float = 0.5

    def __post_init__(self):
        for name in (
            "steps",
            "batch_size",
            "seed",
            "minimum_support",
            "cpu_threads",
            "minimum_phase_support",
            "evaluation_interval",
            "early_stopping_patience",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError("Invalid training configuration: integer settings required")
        if self.split_seed is not None and type(self.split_seed) is not int:
            raise ValueError("Invalid training configuration: split seed must be an integer")
        for name in (
            "learning_rate",
            "alpha",
            "gamma",
            "validation_fraction",
            "behavior_floor",
            "expectile",
            "advantage_temperature",
            "advantage_clip",
            "target_tau",
            "joint_history_dropout",
            "minimum_score_gap",
            "history_consistency_weight",
            "class_weight_exponent",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError("Invalid training configuration: finite settings required")
        if (
            self.algorithm not in {"cql", "bc", "iql"}
            or self.target_mode not in {"identified", "representative"}
            or self.critic_target not in {"td", "episode_return"}
            or self.minimum_score_gap < 0
            or self.minimum_phase_support < 0
            or self.evaluation_interval < 0
            or self.early_stopping_patience < 1
            or not 0 <= self.class_weight_exponent <= 1
            or self.history_consistency_weight < 0
            or (self.history_consistency_weight > 0 and self.algorithm == "iql")
            or not self.advisory_groups
            or any(group not in GROUPS for group in self.advisory_groups)
            or not self.training_groups
            or any(group not in GROUPS for group in self.training_groups)
            or self.steps < 1
            or self.batch_size < 1
            or self.cpu_threads < 1
            or self.learning_rate <= 0
            or self.alpha < 0
            or not 0 < self.gamma <= 1
            or not 0 < self.validation_fraction < 1
            or self.minimum_support < 1
            or not 0 <= self.behavior_floor < 1
            or not 0 < self.expectile < 1
            or self.advantage_temperature <= 0
            or self.advantage_clip < 1
            or not 0 < self.target_tau <= 1
            or not 0 <= self.joint_history_dropout <= 1
            or not isinstance(self.bounded_values, bool)
            or (self.bounded_values and self.algorithm != "cql")
        ):
            raise ValueError("Invalid training configuration")


def predict(
    network, observations, groups, masks, counts, config, *, names=None, context_counts=None
):
    predictions = []
    values = []
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(observations), 1024):
            q, logits = network(
                torch.as_tensor(observations[start : start + 1024], device=config.device)
            )
            p, _, _ = rank_options(
                q.cpu().numpy(),
                logits.cpu().numpy(),
                counts,
                groups[start : start + 1024],
                masks[start : start + 1024],
                algorithm=config.algorithm,
                minimum_support=config.minimum_support,
                behavior_floor=config.behavior_floor,
                minimum_score_gap=config.minimum_score_gap,
                contextual_support=phase_support(
                    observations[start : start + 1024],
                    groups[start : start + 1024],
                    names,
                    context_counts,
                    config.minimum_phase_support,
                )
                if config.minimum_phase_support
                else None,
            )
            p[
                ~np.isin(
                    groups[start : start + 1024],
                    [GROUPS.index(group) for group in config.advisory_groups],
                )
            ] = -1
            predictions.append(p)
            values.append(q.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(values)


def checkpoint_diagnostics(network, arrays, metadata, selected, counts, config, context_counts):
    """Select checkpoints on development only, under the coach's clean/cold rule."""
    from rift_common.actions import Action, feasible_actions
    from rift_common.observation_masks import COLD_START_FIELDS

    observed = arrays["observations"][selected]
    groups = arrays["groups"][selected]
    truth = arrays["actions"][selected]
    identified = arrays["action_identified"][selected]
    arguments = {"names": metadata["feature_names"], "context_counts": context_counts}
    clean, _ = predict(
        network, observed, groups, arrays["action_masks"][selected], counts, config, **arguments
    )
    cold = observed.copy()
    cold[:, [metadata["feature_names"].index(n) for n in COLD_START_FIELDS]] = 0
    masks = feasible_actions(cold, metadata["feature_names"])
    omitted, _ = predict(network, cold, groups, masks, counts, config, **arguments)
    eligible = identified & masks[np.arange(len(truth)), truth]
    still = (clean >= 0) & masks[np.arange(len(truth)), np.maximum(clean, 0)]
    f1 = (
        behavior_metrics(truth[identified], clean[identified], len(Action))["macro_f1"]
        if identified.any()
        else 0
    )
    cold_f1 = (
        behavior_metrics(truth[eligible], omitted[eligible], len(Action))["macro_f1"]
        if eligible.any()
        else 0
    )
    switching = float(np.mean(clean[still] != omitted[still])) if still.any() else 1
    voluntary = np.bincount(
        clean[(clean >= 0) & (clean != Action.WAIT_RESPAWN)], minlength=len(Action)
    )
    concentration = float(voluntary.max() / max(1, voluntary.sum()))
    retention = min(float(np.mean(clean >= 0)), float(np.mean(omitted >= 0)))
    gate = bool(identified.any() and eligible.any() and retention >= 0.5 and concentration <= 0.9)
    score = (f1 + cold_f1) / 2 - 0.1 * switching
    diagnostic = {
        "identified_f1": f1,
        "cold_f1": cold_f1,
        "switching": switching,
        "concentration": concentration,
        "retention": retention,
        "gate": gate,
        "score": score,
    }
    return (gate, score, -concentration), diagnostic


class PolicyTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config

    def _explicit_update(
        self, network, target, optimizer, batch, class_weights, support, consistency_loss, step
    ):
        config = self.config
        q, logits = network(batch["observations"])
        observed = q.gather(1, batch["actions"][:, None]).squeeze(1)
        bc_loss = F.cross_entropy(
            logits.masked_fill(~batch["action_masks"], -1e9),
            batch["actions"],
            weight=class_weights,
        )
        if config.algorithm == "cql":
            targets = critic_targets(network, target, batch, support, config)
            td_loss = F.smooth_l1_loss(observed, targets)
            conservative = (
                torch.logsumexp(q.masked_fill(~batch["action_masks"], -1e9), 1) - observed
            ).mean()
            loss = td_loss + config.alpha * conservative + 0.25 * bc_loss
        else:
            td_loss = conservative = torch.zeros((), device=config.device)
            loss = bc_loss
        loss = loss + config.history_consistency_weight * consistency_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite optimizer loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if step % 200 == 0:
            target.load_state_dict(network.state_dict())
        return {
            "step": step,
            "loss": float(loss.detach()),
            "td_loss": float(td_loss.detach()),
            "cql_penalty": float(conservative.detach()),
            "behavior_loss": float(bc_loss.detach()),
            "history_consistency_loss": float(consistency_loss.detach()),
        }

    def train(self, dataset, output):
        config = self.config
        arrays, metadata = load_dataset(dataset)
        holdout = np.isin(arrays["match_ids"], config.holdout_matches)
        matches = np.unique(arrays["match_ids"][~holdout])
        if len(matches) < 2:
            raise ValueError("At least two independent matches are required")
        rng = np.random.default_rng(config.seed)
        split_rng = np.random.default_rng(
            config.seed if config.split_seed is None else config.split_seed
        )
        split_rng.shuffle(matches)
        # Preserve prior sampling when the split follows the optimizer seed.
        if config.split_seed is None:
            rng = split_rng
        count = max(1, int(len(matches) * config.validation_fraction))
        validation = np.isin(arrays["match_ids"], matches[:count])
        identified = arrays.get("action_identified", np.ones(len(arrays["actions"]), dtype=bool))
        eligible = (
            identified
            if config.target_mode == "identified"
            else np.ones(len(identified), dtype=bool)
        )
        training_scope = np.isin(
            arrays["groups"], [GROUPS.index(group) for group in config.training_groups]
        )
        rows = np.flatnonzero(~validation & ~holdout & eligible & training_scope)
        if not len(rows):
            raise ValueError("No identified training transitions")
        counts = np.zeros((len(GROUPS), len(ACTION_NAMES)), dtype=np.int64)
        np.add.at(counts, (arrays["groups"][rows], arrays["actions"][rows]), 1)
        context_counts = phase_counts(
            arrays["observations"][rows],
            arrays["groups"][rows],
            arrays["actions"][rows],
            metadata["feature_names"],
        )
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        run = {
            "status": "running",
            "settings": asdict(config),
            "dataset_sha256": file_digest(dataset),
            "dataset_path": str(Path(dataset).resolve()),
            "training_matches": matches[count:].tolist(),
            "validation_matches": matches[:count].tolist(),
            "holdout_matches": sorted(set(arrays["match_ids"][holdout])),
            "stage": "experimental_next_action_policy",
            "source_sha256": {
                "trainer": file_digest(__file__),
                "network": file_digest(network_contract.__file__),
                "augmentation": file_digest(augmentation.__file__),
                "observation_masks": file_digest(observation_masks.__file__),
                "targets": file_digest(Path(__file__).with_name("targets.py")),
                "consistency": file_digest(Path(__file__).with_name("consistency.py")),
                "support": file_digest(Path(support_contract.__file__)),
            },
        }
        if config.algorithm == "iql":
            run["source_sha256"]["iql"] = file_digest(Path(__file__).with_name("iql.py"))
        atomic_json(output / "run.json", run)
        try:
            torch.set_num_threads(config.cpu_threads)
            torch.manual_seed(config.seed)
            network = OptionNetwork(
                arrays["observations"].shape[1],
                separate_actor=config.algorithm == "iql",
                bounded_values=config.bounded_values,
            ).to(config.device)
            implicit = None
            if config.algorithm == "iql":
                from offline_rift.training.iql import IQLUpdate

                implicit = IQLUpdate(network, config)
            target = copy.deepcopy(network).eval()
            optimizer = torch.optim.AdamW(
                network.parameters(), lr=config.learning_rate, weight_decay=1e-4
            )
            tensors = {
                key: torch.as_tensor(arrays[key])
                for key in (
                    "observations",
                    "next_observations",
                    "actions",
                    "rewards",
                    "terminals",
                    "durations",
                    "groups",
                    "action_masks",
                    "next_action_masks",
                )
            }
            tensors["episode_returns"] = torch.as_tensor(episode_returns(arrays, config.gamma))
            support = torch.as_tensor(counts >= config.minimum_support, device=config.device)
            class_counts = counts.sum(axis=0)
            weights = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            weights[class_counts > 0] = (
                class_counts.sum() / class_counts[class_counts > 0]
            ) ** config.class_weight_exponent
            weights[class_counts > 0] /= weights[class_counts > 0].mean()
            class_weights = torch.as_tensor(weights, device=config.device)
            augmenter = ObservationAugmenter(
                metadata["feature_names"], joint_dropout=config.joint_history_dropout
            )

            last = {}
            checkpoints = []
            best_state, best_key, best_step, stale = None, None, 0, 0
            for step in range(1, config.steps + 1):
                chosen = rng.choice(rows, size=config.batch_size)
                batch = {key: value[chosen].to(config.device) for key, value in tensors.items()}
                consistency_loss = torch.zeros((), device=config.device)
                if config.history_consistency_weight:
                    consistency_loss = history_consistency(
                        network,
                        batch["observations"],
                        batch["action_masks"],
                        batch["actions"],
                        batch["groups"],
                        support,
                        metadata["feature_names"],
                        critic=config.algorithm == "cql",
                    )
                batch["observations"], batch["action_masks"] = augmenter(
                    batch["observations"], batch["action_masks"], actions=batch["actions"]
                )
                batch["next_observations"], batch["next_action_masks"] = augmenter(
                    batch["next_observations"], batch["next_action_masks"]
                )
                if implicit is not None:
                    last = {"step": step, **implicit.step(batch)}
                    if step % 1000 == 0 or step == config.steps:
                        atomic_json(output / "progress.json", last)
                else:
                    last = self._explicit_update(
                        network,
                        target,
                        optimizer,
                        batch,
                        class_weights,
                        support,
                        consistency_loss,
                        step,
                    )
                if config.evaluation_interval and (
                    step % config.evaluation_interval == 0 or step == config.steps
                ):
                    key, diagnostic = checkpoint_diagnostics(
                        network,
                        arrays,
                        metadata,
                        validation & training_scope,
                        counts,
                        config,
                        context_counts,
                    )
                    checkpoints.append({"step": step, **diagnostic})
                    if best_key is None or key > best_key:
                        best_state, best_key, best_step, stale = (
                            copy.deepcopy(network.state_dict()),
                            key,
                            step,
                            0,
                        )
                    else:
                        stale += 1
                    atomic_json(
                        output / "checkpoints.json",
                        {"checkpoints": checkpoints, "best_step": best_step},
                    )
                    if stale >= config.early_stopping_patience:
                        break
                if step % 1000 == 0 or step == config.steps:
                    atomic_json(output / "progress.json", last)
                network.train()
            completed_steps = step
            if best_state is not None:
                network.load_state_dict(best_state)
            network.eval()
            selected = validation
            predicted, values = predict(
                network,
                arrays["observations"][selected],
                arrays["groups"][selected],
                arrays["action_masks"][selected],
                counts,
                config,
                names=metadata["feature_names"],
                context_counts=context_counts,
            )
            truth = arrays["actions"][selected]
            metrics = behavior_metrics(truth, predicted, len(ACTION_NAMES))
            majority = int(class_counts.argmax())
            report = {
                "training_matches": matches[count:].tolist(),
                "validation_matches": matches[:count].tolist(),
                "training_transitions": len(rows),
                "retained_training_transitions": int(
                    (~validation & ~holdout & training_scope).sum()
                ),
                "stored_training_transitions": int((~validation & ~holdout).sum()),
                "critic_target": config.critic_target,
                "excluded_ambiguous_training_updates": int(
                    (~validation & ~holdout & ~eligible & training_scope).sum()
                ),
                "identified_validation": behavior_metrics(
                    truth[identified[selected]], predicted[identified[selected]], len(ACTION_NAMES)
                )
                if identified[selected].any()
                else None,
                "identified_training_terminal_rows": int(np.sum(arrays["terminals"][rows])),
                "holdout_matches": sorted(set(arrays["match_ids"][holdout])),
                "validation_transitions": int(validation.sum()),
                "optimizer_steps": completed_steps,
                "selected_checkpoint_step": best_step or completed_steps,
                "checkpoints": checkpoints,
                "counts_by_phase": context_counts.tolist(),
                "counts_by_group": counts.tolist(),
                "action_counts": class_counts.tolist(),
                "validation": metrics,
                "majority_baseline": float(np.mean(truth == majority)),
                "per_group": {
                    group: behavior_metrics(
                        truth[arrays["groups"][selected] == i],
                        predicted[arrays["groups"][selected] == i],
                        len(ACTION_NAMES),
                    )
                    for i, group in enumerate(GROUPS)
                    if np.any(arrays["groups"][selected] == i)
                },
                "last_optimizer": last,
                "augmentation": dict(augmenter.counts),
                "observed_action_q_mean": float(values[np.arange(len(truth)), truth].mean())
                if config.algorithm in {"cql", "iql"}
                else None,
                "limitations": [
                    "Inferred executed options, not logged commands or proven intent",
                    "Offline return estimates are not calibrated win probabilities",
                    "Behavior agreement is diagnostic, not policy quality",
                    "Team outcomes give shared rewards; individual causal credit is unverified",
                    "Unknown behavior causes whole actor-episode exclusion; selection bias remains",
                ],
            }
            torch.save(network.state_dict(), output / "policy.pt")
            metadata.update(
                bundle_schema="next-action-policy-v1",
                algorithm=config.algorithm,
                network_architecture="independent_actor_v1"
                if config.algorithm == "iql"
                else "bounded_shared_v1"
                if config.bounded_values
                else "shared_v1",
                train_config=asdict(config),
                training_report=report,
                dataset_sha256=run["dataset_sha256"],
                advisory_groups=list(config.advisory_groups),
            )
            atomic_json(output / "metadata.json", metadata)
            write_bundle_manifest(output, ("metadata.json", "policy.pt"))
            # Completion requires the actual serving loader and held-out prediction parity.
            from offline_rift.inference.policy import NextActionPolicy

            reloaded = NextActionPolicy(output, device=config.device)
            restored, _ = reloaded.predict_batch(
                arrays["observations"][selected],
                arrays["groups"][selected],
                arrays["action_masks"][selected],
            )
            if not np.array_equal(predicted, restored):
                raise RuntimeError("Saved policy failed serving prediction parity")
            run["serving_reload_verified"] = True
            run.update(status="complete", report=report)
            return run
        except BaseException as error:
            run.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
            )
            raise
        finally:
            atomic_json(output / "run.json", run)


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("cql", "bc", "iql"), default="cql")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--minimum-support", type=int, default=20)
    args = parser.parse_args()
    result = PolicyTrainer(
        TrainConfig(
            algorithm=args.algorithm,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            alpha=args.alpha,
            minimum_support=args.minimum_support,
        )
    ).train(args.dataset, args.output)
    print(json.dumps(result["report"], indent=2))
