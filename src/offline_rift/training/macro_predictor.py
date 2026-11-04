"""Confidence-weighted, class-balanced neural weak-label predictor.

The logits model observed macro patterns, not action success or win probability.
Training uses only the training matches; validation never controls weights/stopping.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from offline_rift.pipeline.actions import ACTION_NAMES
from offline_rift.training.encoders import TeamSetEncoderFactory


def create_predictor(size: int, grid_size: int, encoder_type: str) -> nn.Module:
    encoder = (
        TeamSetEncoderFactory(grid_size=grid_size).create((size,))
        if encoder_type == "team_set"
        else nn.Sequential(nn.Linear(size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
    )
    return nn.Sequential(encoder, nn.Linear(128, len(ACTION_NAMES)))


def fit_predictor(
    arrays: dict,
    output: Path,
    *,
    grid_size: int,
    encoder_type: str,
    n_steps: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = create_predictor(arrays["observations"].shape[1], grid_size, encoder_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    counts = np.bincount(arrays["actions"], minlength=len(ACTION_NAMES))
    # Square-root inverse frequency tempers rare/noisy labels; absent classes get no weight.
    weights = np.zeros(len(counts), dtype=np.float32)
    weights[counts > 0] = np.sqrt(counts.sum() / counts[counts > 0])
    weights[counts > 0] /= weights[counts > 0].mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), reduction="none")
    observations = torch.tensor(arrays["observations"], device=device)
    targets = torch.tensor(arrays["actions"], dtype=torch.long, device=device)
    confidence = torch.tensor(arrays["label_confidence"], device=device)
    model.train()
    for _ in range(n_steps):
        index = torch.tensor(rng.integers(len(targets), size=batch_size), device=device)
        loss = (criterion(model(observations[index]), targets[index]) * confidence[index]).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    torch.save(model.state_dict(), output / "macro_predictor.pt")
    return model, {
        "optimizer_steps": n_steps,
        "class_weights": weights.tolist(),
        "loss": "inverse_sqrt_frequency_cross_entropy_x_label_confidence",
        "learning_rate": 3e-4,
    }


def predictor_logits(model: nn.Module, observations: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        # Bound inference memory for large real-data holdouts.
        return np.concatenate(
            [
                model(torch.as_tensor(batch, dtype=torch.float32, device=device)).cpu().numpy()
                for batch in np.array_split(
                    observations, max(1, (len(observations) + 1023) // 1024)
                )
            ]
        )


def hybrid_scores(
    logits: np.ndarray,
    q_values: np.ndarray,
    counts: list[int],
    observations: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Pattern evidence plus a bounded return tie-breaker; no probability claim."""
    if not np.isfinite(logits).all() or not np.isfinite(q_values).all():
        raise ValueError("Non-finite macro prediction")
    advantage = q_values - q_values.mean(axis=1, keepdims=True)
    scale = np.maximum(q_values.std(axis=1, keepdims=True), 1e-6)
    scores = logits + 0.25 * np.clip(advantage / scale, -2, 2)
    scores[:, np.asarray(counts) == 0] = -np.inf
    # Necessary conditions only: never infer positions, vision, safety or success.
    for action, kind in enumerate(("dragon", "baron", "herald")):
        enabled = observations[:, feature_names.index(f"{kind}_enabled")] > 0
        soon = observations[:, feature_names.index(f"{kind}_eta")] <= 0.1 + 1e-7
        scores[~(enabled & soon), action] = -np.inf
    scores[~np.isfinite(scores).any(axis=1), 11] = 0  # Abstain if no supported action remains.
    return scores
