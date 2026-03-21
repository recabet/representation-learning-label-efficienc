#!/usr/bin/env python
"""
analyze_results.py

Load all metrics.json files produced by run_experiments and print a
comprehensive results summary.

Usage:
    python -m src.analysis.analyze_results
    python -m src.analysis.analyze_results --results-dir ./src/experiments/plots
"""

import argparse
import json
from pathlib import Path

from src.configs.global_config import GLOBAL_CONFIG


def load_all_results(results_dir: Path) -> dict:
    """Scan *results_dir* sub-folders for metrics.json and return
    {experiment_name: dict} sorted by name."""
    results = {}
    if not results_dir.exists():
        return results
    for d in sorted(results_dir.iterdir()):
        metrics_file = d / "metrics.json"
        if d.is_dir() and metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            results[data["experiment_name"]] = data
    return results


def _pct(val):
    """Format a float as a percentage string."""
    return f"{val:.1%}"


def print_summary(results: dict):
    """Print the full results analysis to stdout."""
    if not results:
        print("No results found.")
        return

    # ── Group by variant ─────────────────────────────────────────────────────
    variants = {}
    for name, r in results.items():
        # e.g. "resnet18_pt_10" → variant="resnet18_pt", pct=10
        parts = name.rsplit("_", 1)
        variant, pct = parts[0], int(parts[1])
        variants.setdefault(variant, {})[pct] = r

    # ── Main accuracy table ──────────────────────────────────────────────────
    header = f"{'Experiment':<20} {'Labels':>6}  {'Scratch':>9}  {'ImageNet':>9}  {'SimCLR Probe':>13}"
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("  ACCURACY SUMMARY")
    print(sep)
    print(header)
    print("-" * len(header))

    for name in sorted(results.keys()):
        r = results[name]
        s = r["scratch"]["accuracy"]
        i = r["imagenet"]["accuracy"]
        p = r["simclr_probe"]["accuracy"]
        pct = r["percent_labels"]
        print(f"{name:<20} {pct:>5}%  {_pct(s):>9}  {_pct(i):>9}  {_pct(p):>13}")

    # ── Per-variant comparison (if more than one variant) ────────────────────
    if len(variants) > 1:
        print(f"\n{sep}")
        print("  VARIANT COMPARISON (SimCLR Probe only)")
        print(sep)
        var_names = sorted(variants.keys())
        header2 = f"{'Labels':>6}" + "".join(f"  {v:>16}" for v in var_names) + "      Δ"
        print(header2)
        print("-" * len(header2))

        pcts = sorted(set(p for v in variants.values() for p in v.keys()))
        for pct in pcts:
            vals = [variants[v].get(pct, {}).get("simclr_probe", {}).get("accuracy")
                    for v in var_names]
            row = f"{pct:>5}%"
            for val in vals:
                row += f"  {_pct(val):>16}" if val is not None else f"  {'—':>16}"
            if len(vals) == 2 and all(v is not None for v in vals):
                row += f"  {vals[1]-vals[0]:>+7.1%}"
            print(row)

    # ── F1 macro table ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  F1 MACRO SCORES")
    print(sep)
    print(f"{'Experiment':<20} {'Scratch':>9}  {'ImageNet':>9}  {'SimCLR Probe':>13}")
    print("-" * 56)
    for name in sorted(results.keys()):
        r = results[name]
        s = r["scratch"]["f1_macro"]
        i = r["imagenet"]["f1_macro"]
        p = r["simclr_probe"]["f1_macro"]
        print(f"{name:<20} {_pct(s):>9}  {_pct(i):>9}  {_pct(p):>13}")

    # ── Headline ─────────────────────────────────────────────────────────────
    # Find best SimCLR probe at 10% and best scratch at 100%
    best_probe_10 = max(
        (r["simclr_probe"]["accuracy"]
         for r in results.values() if r["percent_labels"] == 10),
        default=None,
    )
    best_scratch_100 = max(
        (r["scratch"]["accuracy"]
         for r in results.values() if r["percent_labels"] == 100),
        default=None,
    )

    if best_probe_10 is not None and best_scratch_100 is not None:
        print(f"\n{sep}")
        print(f"  HEADLINE")
        print(f"  SimCLR @ 10% labels : {_pct(best_probe_10)}")
        print(f"  Scratch @ 100% labels : {_pct(best_scratch_100)}")
        if best_probe_10 > best_scratch_100:
            print(f"  ✅ SimCLR with 10% labels BEATS scratch with 100% by "
                  f"{best_probe_10 - best_scratch_100:+.1%}")
        else:
            print(f"  Gap: {best_probe_10 - best_scratch_100:+.1%}")
        print(sep)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results from metrics.json files.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing per-experiment sub-folders with metrics.json. "
             "Defaults to <project_root>/src/experiments/plots/",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = GLOBAL_CONFIG.PROJECT_ROOT / "src" / "experiments" / "plots"

    print(f"📂 Scanning: {results_dir.resolve()}")
    results = load_all_results(results_dir)
    print(f"   Found {len(results)} experiment(s).\n")

    print_summary(results)


if __name__ == "__main__":
    main()

