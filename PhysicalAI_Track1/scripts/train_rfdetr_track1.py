#!/usr/bin/env python
import argparse
import os

from rfdetr import RFDETRBase, RFDETRLarge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["base", "large"], default="base")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expanded-scales", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_cls = RFDETRLarge if args.variant == "large" else RFDETRBase
    model = model_cls(resolution=args.resolution)
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        num_workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        tensorboard=True,
        wandb=False,
    )


if __name__ == "__main__":
    main()
