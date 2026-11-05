"""Confidence-weighted classifier optimization; no runtime loading code."""

from pathlib import Path

import numpy as np
import torch
from rift_common.models.predictor import create_predictor
from rift_common.pipeline.actions import ACTION_NAMES
from torch import nn


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
    sampling: str = "uniform",
) -> tuple[nn.Module, dict]:
    if sampling not in {"uniform", "sqrt_frequency"}:
        raise ValueError("Unknown classifier sampling mode")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = create_predictor(arrays["observations"].shape[1], grid_size, encoder_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    counts = np.bincount(arrays["actions"], minlength=len(ACTION_NAMES))
    # Square-root inverse frequency tempers rare/noisy labels; absent classes get no weight.
    weights = np.zeros(len(counts), dtype=np.float32)
    weights[counts > 0] = np.sqrt(counts.sum() / counts[counts > 0])
    weights[counts > 0] /= weights[counts > 0].mean()
    if sampling == "sqrt_frequency":
        weights[counts > 0] = 1
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), reduction="none")
    # Keep the corpus on CPU; accelerator memory scales with a batch, not all matches.
    observations = torch.as_tensor(arrays["observations"])
    targets = torch.as_tensor(arrays["actions"], dtype=torch.long)
    confidence = torch.as_tensor(arrays["label_confidence"])
    supported = np.flatnonzero(counts)
    probabilities = np.sqrt(counts[supported])
    probabilities /= probabilities.sum()
    ordered = np.argsort(arrays["actions"], kind="stable")
    offsets = np.cumsum(counts) - counts
    model.train()
    for _ in range(n_steps):
        if sampling == "uniform":
            index = rng.integers(len(targets), size=batch_size)
        else:
            labels = rng.choice(supported, size=batch_size, p=probabilities)
            index = ordered[offsets[labels] + rng.integers(counts[labels])]
        batch = observations[index].to(device)
        labels = targets[index].to(device)
        reliability = confidence[index].to(device)
        loss = (criterion(model(batch), labels) * reliability).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    torch.save(model.state_dict(), output / "macro_predictor.pt")
    return model, {
        "optimizer_steps": n_steps,
        "class_weights": weights.tolist(),
        "loss": "cross_entropy_x_label_confidence",
        "sampling": sampling,
        "learning_rate": 3e-4,
    }
