"""Train a discrete Conservative Q-Learning macro policy from an NPZ dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/macro_cql"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    from rl_agent import OfflineMacroTrainer

    trainer = OfflineMacroTrainer(
        device=args.device,
        batch_size=args.batch_size,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
    )
    print(
        json.dumps(
            trainer.train(
                args.dataset,
                args.output,
                n_steps=args.steps,
                n_steps_per_epoch=args.steps_per_epoch,
                validation_fraction=args.validation_fraction,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
