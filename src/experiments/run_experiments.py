import argparse
from pathlib import Path

from src.experiments.cli_args import add_shared_training_args


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run downstream experiment sweeps with optional overrides"
    )
    add_shared_training_args(parser, epochs_default=100, batch_size_default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--simclr-epoch", type=int, default=800)
    parser.add_argument("--label-percents", type=int, nargs="+", default=[10, 25, 50, 75, 100])
    parser.add_argument("--variants", type=str, nargs="+", default=["resnet18", "resnet18_pt"])
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    from src.experiments.set_experiment import run_experiment

    checkpoints_root = Path(__file__).resolve().parents[2] / "checkpoints"

    # SimCLR checkpoint variants: name -> (architecture, out_dim)
    variant_specs = {
        "resnet18": ("resnet18", 128),
        "resnet18_pt": ("resnet18", 128),
        # "resnet50": ("resnet50", 256),
    }

    unknown_variants = [name for name in args.variants if name not in variant_specs]
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}. Available: {list(variant_specs)}")

    for variant_name in args.variants:
        arch, out_dim = variant_specs[variant_name]
        simclr_path = str(
            checkpoints_root
            / f"simclr_pretrain_{variant_name}"
            / f"simclr_epoch_{args.simclr_epoch}.pth"
        )

        for pct in args.label_percents:
            run_experiment(
                experiment_name=f"{variant_name}_{pct}",
                percent_labels=pct,
                simclr_path=simclr_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=args.device,
                base_model=arch,
                out_dim=out_dim,
            )
