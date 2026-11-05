"""Explicit-transition discrete CQL with a behavior-support head and elapsed-time discount."""

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from rift_common import network as network_contract
from rift_common.actions import ACTION_NAMES, GROUPS
from rift_common.artifacts import atomic_json, file_digest, write_bundle_manifest
from rift_common.dataset import load_dataset
from rift_common.metrics import behavior_metrics
from rift_common.network import OptionNetwork, bellman_target, rank_options
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "cql"
    steps: int = 10000
    batch_size: int = 256
    learning_rate: float = 3e-4
    alpha: float = 0.1
    gamma: float = 0.99
    seed: int = 42
    validation_fraction: float = 0.2
    minimum_support: int = 20
    behavior_floor: float = 0.02
    cpu_threads: int = 2
    device: str = "cpu"

    def __post_init__(self):
        if (
            self.algorithm not in {"cql", "bc"}
            or self.steps < 1
            or self.batch_size < 1
            or self.cpu_threads < 1
            or self.learning_rate <= 0
            or self.alpha < 0
            or not 0 < self.gamma <= 1
            or not 0 < self.validation_fraction < 1
            or self.minimum_support < 1
            or not 0 <= self.behavior_floor < 1
        ):
            raise ValueError("Invalid training configuration")


def predict(network, observations, groups, masks, counts, config):
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
            )
            predictions.append(p)
            values.append(q.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(values)


class PolicyTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config

    def train(self, dataset, output):
        config = self.config
        arrays, metadata = load_dataset(dataset)
        matches = np.unique(arrays["match_ids"])
        if len(matches) < 2:
            raise ValueError("At least two independent matches are required")
        rng = np.random.default_rng(config.seed)
        rng.shuffle(matches)
        count = max(1, int(len(matches) * config.validation_fraction))
        validation = np.isin(arrays["match_ids"], matches[:count])
        rows = np.flatnonzero(~validation)
        counts = np.zeros((len(GROUPS), len(ACTION_NAMES)), dtype=np.int64)
        np.add.at(counts, (arrays["groups"][rows], arrays["actions"][rows]), 1)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        run = {
            "status": "running",
            "settings": asdict(config),
            "dataset_sha256": file_digest(dataset),
            "dataset_path": str(Path(dataset).resolve()),
            "training_matches": matches[count:].tolist(),
            "validation_matches": matches[:count].tolist(),
            "stage": "experimental_next_action_policy",
            "source_sha256": {
                "trainer": file_digest(__file__),
                "network": file_digest(network_contract.__file__),
            },
        }
        atomic_json(output / "run.json", run)
        try:
            torch.set_num_threads(config.cpu_threads)
            torch.manual_seed(config.seed)
            network = OptionNetwork(arrays["observations"].shape[1]).to(config.device)
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
            support = torch.as_tensor(counts >= config.minimum_support, device=config.device)
            class_counts = counts.sum(axis=0)
            weights = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            weights[class_counts > 0] = np.sqrt(class_counts.sum() / class_counts[class_counts > 0])
            weights[class_counts > 0] /= weights[class_counts > 0].mean()
            class_weights = torch.as_tensor(weights, device=config.device)
            optional = [
                metadata["feature_names"].index(name)
                for name in (
                    "self_health",
                    "self_cs_rate",
                    "self_level_rate",
                    "own_towers_lost",
                    "enemy_towers_lost",
                )
            ]

            def mask_optional(observations):
                # Public history can start late; own-health fields may be unavailable.
                observed = observations.clone()
                for index in optional:
                    missing = torch.rand(len(observed), device=config.device) < 0.2
                    observed[missing, index : index + 2] = 0
                return observed

            last = {}
            for step in range(1, config.steps + 1):
                chosen = rng.choice(rows, size=config.batch_size)
                batch = {key: value[chosen].to(config.device) for key, value in tensors.items()}
                batch["observations"] = mask_optional(batch["observations"])
                batch["next_observations"] = mask_optional(batch["next_observations"])
                q, logits = network(batch["observations"])
                observed = q.gather(1, batch["actions"][:, None]).squeeze(1)
                bc_loss = F.cross_entropy(
                    logits.masked_fill(~batch["action_masks"], -1e9),
                    batch["actions"],
                    weight=class_weights,
                )
                if config.algorithm == "cql":
                    with torch.no_grad():
                        online, _ = network(batch["next_observations"])
                        future, future_behavior = target(batch["next_observations"])
                        legal = batch["next_action_masks"] & support[batch["groups"]]
                        legal &= (
                            future_behavior.masked_fill(~batch["next_action_masks"], -1e9).softmax(
                                1
                            )
                            >= config.behavior_floor
                        )
                        next_action = online.masked_fill(~legal, -1e9).argmax(1)
                        next_value = future.gather(1, next_action[:, None]).squeeze(1)
                        next_value = torch.where(
                            legal.any(1), next_value, torch.full_like(next_value, -1)
                        )
                        targets = bellman_target(
                            batch["rewards"],
                            batch["terminals"],
                            batch["durations"],
                            next_value,
                            config.gamma,
                        )
                    td_loss = F.smooth_l1_loss(observed, targets)
                    conservative = (
                        torch.logsumexp(q.masked_fill(~batch["action_masks"], -1e9), 1) - observed
                    ).mean()
                    loss = td_loss + config.alpha * conservative + 0.25 * bc_loss
                else:
                    td_loss = conservative = torch.zeros((), device=config.device)
                    loss = bc_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite optimizer loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                if step % 200 == 0:
                    target.load_state_dict(network.state_dict())
                if step % 1000 == 0 or step == config.steps:
                    last = {
                        "step": step,
                        "loss": float(loss.detach()),
                        "td_loss": float(td_loss.detach()),
                        "cql_penalty": float(conservative.detach()),
                        "behavior_loss": float(bc_loss.detach()),
                    }
                    atomic_json(output / "progress.json", last)
            network.eval()
            selected = validation
            predicted, values = predict(
                network,
                arrays["observations"][selected],
                arrays["groups"][selected],
                arrays["action_masks"][selected],
                counts,
                config,
            )
            truth = arrays["actions"][selected]
            metrics = behavior_metrics(truth, predicted, len(ACTION_NAMES))
            majority = int(class_counts.argmax())
            report = {
                "training_matches": matches[count:].tolist(),
                "validation_matches": matches[:count].tolist(),
                "training_transitions": int((~validation).sum()),
                "validation_transitions": int(validation.sum()),
                "optimizer_steps": config.steps,
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
                "observed_action_q_mean": float(values[np.arange(len(truth)), truth].mean())
                if config.algorithm == "cql"
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
                train_config=asdict(config),
                training_report=report,
            )
            atomic_json(output / "metadata.json", metadata)
            write_bundle_manifest(output, ("metadata.json", "policy.pt"))
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
    parser.add_argument("--algorithm", choices=("cql", "bc"), default="cql")
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
