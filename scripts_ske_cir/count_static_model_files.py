#!/usr/bin/env python3
"""Count result files for each model under static_ske_per_neuron."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "static_ske_per_neuron"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count generated result files recursively for every model."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Count every regular file instead of JSON result files only.",
    )
    return parser.parse_args()


def count_files(directory: Path, all_files: bool) -> int:
    files = directory.rglob("*") if all_files else directory.rglob("*.json")
    return sum(path.is_file() for path in files)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"error: results directory does not exist: {results_dir}")

    model_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    if not model_dirs:
        raise SystemExit(f"error: no model directories found under: {results_dir}")

    rows = []
    for model_dir in model_dirs:
        long_count = count_files(model_dir / "long", args.all_files)
        short_count = count_files(model_dir / "short", args.all_files)
        total_count = count_files(model_dir, args.all_files)
        other_count = total_count - long_count - short_count
        rows.append((model_dir.name, long_count, short_count, other_count, total_count))

    model_width = max(len("MODEL"), *(len(row[0]) for row in rows))
    header = (
        f"{'MODEL':<{model_width}}  {'LONG':>8}  {'SHORT':>8}  "
        f"{'OTHER':>8}  {'TOTAL':>8}"
    )
    print(header)
    print("-" * len(header))
    for model, long_count, short_count, other_count, total_count in rows:
        print(
            f"{model:<{model_width}}  {long_count:>8}  {short_count:>8}  "
            f"{other_count:>8}  {total_count:>8}"
        )

    print("-" * len(header))
    print(
        f"{'ALL MODELS':<{model_width}}  "
        f"{sum(row[1] for row in rows):>8}  "
        f"{sum(row[2] for row in rows):>8}  "
        f"{sum(row[3] for row in rows):>8}  "
        f"{sum(row[4] for row in rows):>8}"
    )


if __name__ == "__main__":
    main()
