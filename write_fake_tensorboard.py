#!/usr/bin/env python3
"""Write synthetic probing metrics to TensorBoard for UI testing."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fake layer-wise probing metrics for TensorBoard."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs/fake_probe"),
        help="TensorBoard log directory (default: runs/fake_probe).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=8,
        help="Number of fake tokenizer layers (default: 8).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of fake training steps (default: 100).",
    )
    args = parser.parse_args()

    if args.layers < 1 or args.steps < 1:
        raise SystemExit("--layers and --steps must be positive integers")

    writer = SummaryWriter(log_dir=str(args.log_dir))
    for step in range(args.steps):
        progress = step / max(args.steps - 1, 1)
        writer.add_scalar("training/loss", 1.2 * math.exp(-4 * progress), step)
        writer.add_scalar("training/accuracy", 1 - math.exp(-4 * progress), step)

        for layer in range(1, args.layers + 1):
            layer_progress = layer / args.layers
            per = 0.42 * math.exp(-2.5 * layer_progress) + 0.02
            per += 0.01 * math.sin(step / 8 + layer)
            eer = 0.20 * math.exp(-1.8 * layer_progress) + 0.01
            eer += 0.005 * math.cos(step / 10 + layer)
            writer.add_scalar(f"probing/PER/layer_{layer:02d}", per, step)
            writer.add_scalar(f"probing/EER/layer_{layer:02d}", eer, step)

            if step % 10 == 0:
                fake_codes = torch.randn(256) + layer_progress
                writer.add_histogram(
                    f"probing/token_distribution/layer_{layer:02d}",
                    fake_codes,
                    step,
                )

    writer.flush()
    writer.close()
    print(f"Wrote fake TensorBoard data to {args.log_dir}")


if __name__ == "__main__":
    main()
