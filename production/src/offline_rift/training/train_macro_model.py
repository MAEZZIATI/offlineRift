"""Train CQL or a class-balanced imitation model from a validated NPZ dataset."""

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
    parser.add_argument("--batch-size", type=int, help="Default: 64 for CQL, 128 for imitation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--algorithm", choices=("cql", "imitation"), default="cql")
    parser.add_argument(
        "--encoder-type",
        choices=("team_set", "mlp", "residual"),
        help="Default: team_set for CQL, mlp for imitation",
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--split-from", type=Path, help="Imitation only: reuse a saved bundle's match split"
    )
    parser.add_argument(
        "--prediction-mode", choices=("hybrid", "balanced_bc", "cql"), default="hybrid"
    )
    args = parser.parse_args()
    args.batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (128 if args.algorithm == "imitation" else 64)
    )
    args.encoder_type = args.encoder_type or (
        "mlp" if args.algorithm == "imitation" else "team_set"
    )
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.output.exists():
        parser.error("--output must be a new directory; completed experiments are immutable")
    import torch

    torch.set_num_threads(args.cpu_threads)
    if args.algorithm == "imitation":
        from offline_rift.training.imitation import ImitationTrainer

        report = ImitationTrainer(
            encoder_type=args.encoder_type,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        ).train(
            args.dataset,
            args.output,
            n_steps=args.steps,
            validation_fraction=args.validation_fraction,
            split_from=args.split_from,
        )
        print(json.dumps(report, indent=2))
        return
    if args.split_from or args.encoder_type == "residual":
        parser.error("--split-from and residual encoder require --algorithm imitation")
    from offline_rift.training.rl_agent import OfflineMacroTrainer

    trainer = OfflineMacroTrainer(
        device=args.device,
        batch_size=args.batch_size,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
        encoder_type=args.encoder_type,
        prediction_mode=args.prediction_mode,
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
