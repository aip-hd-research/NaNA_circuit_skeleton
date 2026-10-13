#!/usr/bin/env python3
"""Dedicated entry point for static orthogonal-decomposition experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from W_decompose.static_runner import (  # noqa: E402
    DEFAULT_METHODS,
    ORTHOGONAL_METHODS,
    build_runs,
    execute_runs,
)


SCRATCH_ROOT = Path("/pfs/work9/workspace/scratch/hd_nf184-nana-cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply SVD, QR, Haar, or original-coordinate down-projection "
            "factorizations to the existing static per-neuron MLP circuit."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ORTHOGONAL_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--haar-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--decomposition-cache-root",
        default=str(SCRATCH_ROOT / "pre_compute_w_decompose"),
        help="Root containing <method-tag>/<model>/within_ske_cache.",
    )
    parser.add_argument(
        "--svd-cache-root",
        default=str(SCRATCH_ROOT / "pre_compute_skeleton_cache"),
        help="Root containing the existing SVD <model>/within_ske_cache.",
    )
    parser.add_argument(
        "--results-root",
        default=str(SCRATCH_ROOT / "results_compare_decompose_methods" / "static"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-cache-validation",
        action="store_true",
        help="Let the underlying runner diagnose caches instead of checking first.",
    )
    parser.add_argument("forwarded", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = build_runs(
        args.methods,
        args.haar_seeds,
        decomposition_cache_root=args.decomposition_cache_root,
        svd_cache_root=args.svd_cache_root,
        results_root=args.results_root,
    )
    try:
        execute_runs(
            runs,
            args.forwarded,
            repository_root=REPOSITORY_ROOT,
            dry_run=args.dry_run,
            check_caches=not args.skip_cache_validation,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
