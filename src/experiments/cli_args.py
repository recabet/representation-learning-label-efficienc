import argparse
from typing import Optional


def add_shared_training_args(
    parser: argparse.ArgumentParser,
    *,
    epochs_default: Optional[int] = None,
    batch_size_default: Optional[int] = None,
) -> argparse.ArgumentParser:
    """Register shared training flags used by experiment entrypoints."""
    parser.add_argument("--epochs", type=int, default=epochs_default)
    parser.add_argument("--batch-size", type=int, default=batch_size_default)
    return parser

