"""Experimental causal-history BC; fixed match splits and matched-capacity controls.

This separate research bundle is not a stock MacroPolicy or a deployed live model.
All inputs retain the shared feature contract; extra history age/presence are known
from observation timestamps. Historical frame age remains masked in the live profile.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from rift_common.models.encoders import TeamSetEncoder
from rift_common.models.evaluation import classification_metrics
from rift_common.models.predictor import hybrid_scores
from rift_common.pipeline.action_labeler import load_dataset
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from rift_research.evaluation import match_bootstrap

METHODS = {
    "snapshot": (0, False),
    "gru60": (60, False),
    "gru120": (120, False),
    "repeat120": (120, True),
}


def history_indices(episodes, timestamps, horizon, *, interval=10):
    """Past/current observations only, with right padding and resets on clock rollback."""
    if horizon < 0 or horizon % interval:
        raise ValueError("History horizon must be a nonnegative multiple of the grid interval")
    width = horizon // interval + 1
    indices = np.full((len(episodes), width), -1, dtype=np.int64)
    lengths = np.empty(len(episodes), dtype=np.int64)
    history = deque()
    for i, (episode, timestamp) in enumerate(zip(episodes, timestamps)):
        if i == 0 or episode != episodes[i - 1] or timestamp <= timestamps[i - 1]:
            history.clear()
        while history and (timestamps[history[0]] < timestamp - horizon or len(history) >= width):
            history.popleft()
        history.append(i)
        lengths[i] = len(history)
        indices[i, : len(history)] = history
    return indices, lengths


class HistoryClassifier(nn.Module):
    def __init__(self, feature_size, grid_size, *, recurrent, repeat_current=False):
        super().__init__()
        self.encoder = TeamSetEncoder(feature_size, grid_size, 64)
        self.recurrent = recurrent
        self.repeat_current = repeat_current
        # GRU(66,64): 25,344 parameters. The snapshot control has 25,347.
        self.temporal = (
            nn.GRU(66, 64, batch_first=True)
            if recurrent
            else nn.Sequential(nn.Linear(66, 193), nn.ReLU(), nn.Linear(193, 64), nn.ReLU())
        )
        self.head = nn.Linear(64, 12)

    def forward(self, observations, ages, lengths):
        batch, steps, size = observations.shape
        if not self.recurrent:
            current = observations[torch.arange(batch), lengths - 1]
            context = torch.cat(
                (self.encoder(current), torch.zeros(batch, 1), torch.ones(batch, 1)), dim=1
            )
            return self.head(self.temporal(context))
        if self.repeat_current:
            current = observations[torch.arange(batch), lengths - 1]
            observations = current[:, None, :].expand(-1, steps, -1)
        encoded = self.encoder(observations.reshape(-1, size)).reshape(batch, steps, 64)
        present = torch.arange(steps)[None, :] < lengths[:, None]
        inputs = torch.cat((encoded, ages[:, :, None], present[:, :, None].float()), dim=2)
        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, final = self.temporal(packed)
        return self.head(final[-1])


def batch_history(observations, timestamps, indices, lengths, rows, horizon):
    index = indices[rows]
    valid = index >= 0
    chosen = index.clamp(min=0)
    values = observations[chosen] * valid[:, :, None]
    current = chosen.gather(1, (lengths[rows] - 1)[:, None])
    age = (timestamps[current] - timestamps[chosen]) / max(horizon, 1)
    return values, age.float() * valid, lengths[rows]


def predict_history(model, arrays, horizon, *, batch_size=256):
    indices, lengths = history_indices(arrays["episode_ids"], arrays["timestamps"], horizon)
    observations = torch.as_tensor(arrays["observations"])
    timestamps = torch.as_tensor(arrays["timestamps"])
    indices, lengths = torch.as_tensor(indices), torch.as_tensor(lengths)
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(observations)))
            logits.append(
                model(
                    *batch_history(observations, timestamps, indices, lengths, rows, horizon)
                ).numpy()
            )
    return np.concatenate(logits)


def fixed_match_split(match_ids, *, seed=20261105, fraction=0.2):
    matches = np.unique(match_ids)
    if len(matches) < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least two matches and a positive train/development split")
    np.random.default_rng(seed).shuffle(matches)
    count = min(len(matches) - 1, max(1, int(len(matches) * fraction)))
    return matches[count:].tolist(), matches[:count].tolist()


def train_memory_experiment(
    dataset,
    output,
    *,
    seeds=(42, 43, 44, 45, 46),
    n_steps=5000,
    batch_size=64,
    split_seed=20261105,
    bootstrap_samples=1000,
    methods=("snapshot", "gru60", "gru120"),
):
    from rift_common.artifacts import atomic_json

    if n_steps < 1 or batch_size < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive steps/batch size and distinct seeds required")
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(name not in METHODS for name in methods)
    ):
        raise ValueError("Choose distinct supported memory methods")
    torch.set_num_threads(2)
    arrays, metadata = load_dataset(dataset)
    if metadata["feature_config"]["profile"] != "live":
        raise ValueError("This experiment requires live-masked inputs")
    train, validation = fixed_match_split(arrays["match_ids"], seed=split_seed)
    mask = np.isin(arrays["match_ids"], validation)
    train_rows = np.flatnonzero(~mask)
    truth = arrays["actions"][mask]
    counts = np.bincount(arrays["actions"][~mask], minlength=12)
    weights = np.zeros(12, dtype=np.float32)
    weights[counts > 0] = np.sqrt(counts.sum() / counts[counts > 0])
    weights[counts > 0] /= weights[counts > 0].mean()
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights), reduction="none")
    observations = torch.as_tensor(arrays["observations"])
    timestamps = torch.as_tensor(arrays["timestamps"])
    targets = torch.as_tensor(arrays["actions"])
    confidence = torch.as_tensor(arrays["label_confidence"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "bundle_schema": "history-bc-v1",
        "dataset": str(Path(dataset).resolve()),
        "metadata": metadata,
        "training_matches": train,
        "validation_matches": validation,
        "train_transitions": len(train_rows),
        "validation_transitions": int(mask.sum()),
        "split_seed": split_seed,
        "optimizer_seeds": list(seeds),
        "methods": list(methods),
        "steps_per_fit": n_steps,
        "batch_size": batch_size,
        "feature_profile": "live",
        "action_counts": counts.tolist(),
        "loss": "inverse_sqrt_frequency_cross_entropy_x_label_confidence",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 5,
        "history_features": "relative observed timestamp age and presence; no unmasked frame age",
        "label_status": "weak labels; independent human audit incomplete",
        "runs": [],
    }
    atomic_json(output / "report.json", report)
    all_predictions = {}
    try:
        for seed in seeds:
            predictions = {}
            for name in methods:
                horizon, repeat = METHODS[name]
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                model = HistoryClassifier(
                    observations.shape[1],
                    metadata["feature_config"]["grid_size"],
                    recurrent=horizon > 0,
                    repeat_current=repeat,
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                index, length = history_indices(
                    arrays["episode_ids"], arrays["timestamps"], horizon
                )
                index, length = torch.as_tensor(index), torch.as_tensor(length)
                model.train()
                start = time.monotonic()
                for step in range(n_steps):
                    rows = torch.as_tensor(rng.choice(train_rows, size=batch_size))
                    logits = model(
                        *batch_history(observations, timestamps, index, length, rows, horizon)
                    )
                    loss = (criterion(logits, targets[rows]) * confidence[rows]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite memory training loss")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    if (step + 1) % 1000 == 0:
                        print(f"{name} seed {seed}: {step + 1}/{n_steps} updates", flush=True)
                elapsed = time.monotonic() - start
                model.eval()
                torch.save(model.state_dict(), output / f"{name}-{seed}.pt")
                # Reload before evaluation to validate the saved experimental artifact.
                restored = HistoryClassifier(
                    observations.shape[1],
                    metadata["feature_config"]["grid_size"],
                    recurrent=horizon > 0,
                    repeat_current=repeat,
                )
                restored.load_state_dict(
                    torch.load(output / f"{name}-{seed}.pt", weights_only=True)
                )
                logits = predict_history(
                    restored, {key: value[mask] for key, value in arrays.items()}, horizon
                )
                scores = hybrid_scores(
                    logits,
                    np.zeros_like(logits),
                    counts.tolist(),
                    arrays["observations"][mask],
                    metadata["feature_names"],
                )
                predicted = scores.argmax(axis=1)
                predictions[name] = predicted
                all_predictions[f"{name}_{seed}"] = predicted
                metrics = classification_metrics(truth, predicted, 12)
                row = {
                    "method": name,
                    "horizon_seconds": horizon,
                    "optimizer_seed": seed,
                    "optimizer_steps": n_steps,
                    "train_seconds": elapsed,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "metrics": metrics,
                }
                report["runs"].append(row)
                atomic_json(output / "report.json", report)
                print(
                    f"Completed {name} seed {seed}: macro F1 {metrics['macro_f1']:.4f}, balanced accuracy {metrics['balanced_accuracy']:.4f}",
                    flush=True,
                )
            bootstrap = match_bootstrap(
                truth,
                predictions,
                arrays["match_ids"][mask],
                12,
                samples=bootstrap_samples,
                seed=split_seed,
                comparisons=(("gru60", "snapshot"), ("gru120", "snapshot")),
            )
            report.setdefault("per_seed_bootstrap", {})[str(seed)] = bootstrap
            atomic_json(output / "report.json", report)
        report["summary"] = {}
        for name in methods:
            runs = [row for row in report["runs"] if row["method"] == name]
            report["summary"][name] = {
                metric: {
                    "mean": float(np.mean([r["metrics"][metric] for r in runs])),
                    "seed_std": float(np.std([r["metrics"][metric] for r in runs])),
                }
                for metric in ("heuristic_agreement", "balanced_accuracy", "macro_f1")
            }
        majority = np.full(len(truth), counts.argmax())
        report["majority"] = classification_metrics(truth, majority, 12)
        np.savez_compressed(
            output / "validation_predictions.npz",
            truth=truth,
            match_ids=arrays["match_ids"][mask],
            **all_predictions,
        )
        report["status"] = "complete"
        return report
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = str(error)
        raise
    finally:
        atomic_json(output / "report.json", report)


def run_memory(args):
    if args.evaluate_from:
        report = evaluate_memory_experiment(
            args.evaluate_from,
            args.dataset,
            args.output,
            bootstrap_samples=args.bootstrap_samples,
            reference_bundle=args.reference_bundle,
            control_from=args.control_from,
        )
        print(json.dumps(report["summary"], indent=2))
        return
    report = train_memory_experiment(
        args.dataset,
        args.output,
        seeds=args.seeds,
        n_steps=args.steps,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        bootstrap_samples=args.bootstrap_samples,
        methods=args.methods,
    )
    print(json.dumps(report["summary"], indent=2))


def evaluate_memory_experiment(
    experiment, dataset, output, *, bootstrap_samples=1000, reference_bundle=None, control_from=None
):
    """Frozen multi-seed evaluation excluding ALL experiment train/development matches."""
    from rift_common.artifacts import atomic_json

    torch.set_num_threads(2)
    experiment = Path(experiment)
    source = json.loads((experiment / "report.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Memory experiment must be complete before frozen evaluation")
    arrays, metadata = load_dataset(dataset)
    for key in ("schema_version", "feature_names", "feature_config", "action_schema"):
        if metadata[key] != source["metadata"][key]:
            raise ValueError(f"Memory evaluation contract mismatch: {key}")
    old_matches = source["training_matches"] + source["validation_matches"]
    methods = source.get("methods", ["snapshot", "gru60", "gru120"])
    if control_from is not None:
        control_from = Path(control_from)
        control = json.loads((control_from / "report.json").read_text())
        if control["status"] != "complete" or control.get("methods") != ["repeat120"]:
            raise ValueError("Repeat control must be a completed repeat120-only experiment")
        for key in (
            "training_matches",
            "validation_matches",
            "optimizer_seeds",
            "steps_per_fit",
            "batch_size",
        ):
            if source[key] != control[key]:
                raise ValueError(f"Control protocol mismatch: {key}")
        methods = [*methods, "repeat120"]
    mask = ~np.isin(arrays["match_ids"], old_matches)
    if not mask.any():
        raise ValueError("No unseen matches after excluding all training/development IDs")
    excluded_count = len(np.unique(arrays["match_ids"][~mask]))
    test = {
        key: arrays[key][mask]
        for key in ("observations", "actions", "episode_ids", "timestamps", "match_ids")
    }
    del arrays  # Validated next-state/reward arrays are not inputs to this classification test.
    truth = test["actions"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "kind": "frozen multi-seed memory evaluation on unseen matches",
        "experiment": str(experiment.resolve()),
        "dataset": str(Path(dataset).resolve()),
        "test_matches": np.unique(test["match_ids"]).tolist(),
        "test_transitions": len(truth),
        "excluded_previously_seen_matches": excluded_count,
        "optimizer_seeds": source["optimizer_seeds"],
        "runs": [],
        "per_seed_bootstrap": {},
        "limitations": [
            "agreement with weak labels, not audited strategy",
            "match bootstrap does not capture all repeated-account dependence",
            "new IDs do not imply participant-disjoint or rank-unbiased data",
        ],
    }
    atomic_json(output / "report.json", report)
    saved_predictions = {}
    try:
        for seed in source["optimizer_seeds"]:
            predictions = {}
            for name in methods:
                horizon, repeat = METHODS[name]
                model = HistoryClassifier(
                    test["observations"].shape[1],
                    metadata["feature_config"]["grid_size"],
                    recurrent=horizon > 0,
                    repeat_current=repeat,
                )
                model.load_state_dict(
                    torch.load(
                        (
                            control_from
                            if name == "repeat120" and control_from is not None
                            else experiment
                        )
                        / f"{name}-{seed}.pt",
                        weights_only=True,
                    )
                )
                logits = predict_history(model, test, horizon)
                scores = hybrid_scores(
                    logits,
                    np.zeros_like(logits),
                    source["action_counts"],
                    test["observations"],
                    metadata["feature_names"],
                )
                predicted = scores.argmax(axis=1)
                predictions[name] = predicted
                saved_predictions[f"{name}_{seed}"] = predicted
                metrics = classification_metrics(truth, predicted, 12)
                report["runs"].append({"method": name, "optimizer_seed": seed, "metrics": metrics})
                print(
                    f"Unseen {name} seed {seed}: macro F1 {metrics['macro_f1']:.4f}, balanced accuracy {metrics['balanced_accuracy']:.4f}",
                    flush=True,
                )
                atomic_json(output / "report.json", report)
            report["per_seed_bootstrap"][str(seed)] = match_bootstrap(
                truth,
                predictions,
                test["match_ids"],
                12,
                samples=bootstrap_samples,
                seed=source["split_seed"],
                comparisons=(
                    ("gru60", "snapshot"),
                    ("gru120", "snapshot"),
                    ("gru120", "repeat120"),
                    ("repeat120", "snapshot"),
                ),
            )
        report["summary"] = {}
        for name in methods:
            runs = [row for row in report["runs"] if row["method"] == name]
            report["summary"][name] = {
                metric: {
                    "mean": float(np.mean([r["metrics"][metric] for r in runs])),
                    "seed_std": float(np.std([r["metrics"][metric] for r in runs])),
                }
                for metric in ("heuristic_agreement", "balanced_accuracy", "macro_f1")
            }
        report["majority"] = classification_metrics(
            truth, np.full(len(truth), np.argmax(source["action_counts"])), 12
        )
        if reference_bundle is not None:
            from offline_rift.inference.policy import MacroPolicy
            from rift_common.models.predictor import predictor_logits

            policy = MacroPolicy(reference_bundle)
            for key in ("feature_names", "feature_config", "action_schema"):
                if policy.metadata[key] != metadata[key]:
                    raise ValueError(f"Plugged reference contract mismatch: {key}")
            used = policy.metadata["training_report"]
            if set(report["test_matches"]) & set(
                used["training_matches"] + used["validation_matches"]
            ):
                raise ValueError("Reference bundle has seen test matches")
            if policy.predictor is None:
                raise ValueError("Reference needs a companion predictor")
            logits = predictor_logits(policy.predictor, test["observations"], "cpu")
            values = np.concatenate(
                [
                    np.column_stack(
                        [
                            policy.model.predict_value(batch, np.full(len(batch), a))
                            for a in range(12)
                        ]
                    )
                    for batch in np.array_split(
                        test["observations"], max(1, (len(truth) + 1023) // 1024)
                    )
                ]
            )
            report["plugged_reference"] = {
                "bundle": str(Path(reference_bundle).resolve()),
                "training_matches": len(used["training_matches"]),
                "limitation": "practical checkpoint comparison; training corpus and capacity differ",
                "metrics": {},
            }
            for name, q in (("balanced_bc", np.zeros_like(values)), ("hybrid", values)):
                prediction = hybrid_scores(
                    logits.copy(),
                    q,
                    used["action_counts"],
                    test["observations"],
                    metadata["feature_names"],
                ).argmax(axis=1)
                saved_predictions[f"plugged_{name}"] = prediction
                report["plugged_reference"]["metrics"][name] = classification_metrics(
                    truth, prediction, 12
                )
        np.savez_compressed(
            output / "test_predictions.npz",
            truth=truth,
            match_ids=test["match_ids"],
            **saved_predictions,
        )
        report["status"] = "complete"
        return report
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = str(error)
        raise
    finally:
        atomic_json(output / "report.json", report)
