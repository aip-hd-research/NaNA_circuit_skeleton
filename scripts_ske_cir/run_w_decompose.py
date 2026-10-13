#!/usr/bin/env python3
"""Precompute and run matched down-projection decomposition experiments.

Examples
--------
Create caches::

  python scripts_ske_cir/run_w_decompose.py precompute \
    --models openai-community/gpt2 \
    --methods qr original_coordinate haar --haar-seeds 0 1 2 --device cuda

Run the existing dynamic experiment once per generated cache::

  python scripts_ske_cir/run_w_decompose.py dynamic \
    --methods qr original_coordinate haar --haar-seeds 0 1 2 -- \
    --models openai-community/gpt2 --dataset_path DATASET.json \
    --k_percents 0.1 1 5 --start_layer 4 --no-rely_ori_acv

For ``static``, arguments after ``--`` are forwarded to
``run_static_ske_per_neuron.py``.  Cache and output paths are supplied by this
wrapper so results from different decompositions cannot overwrite each other.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "model_cache"))

from W_decompose.decompositions import DECOMPOSITION_METHODS  # noqa: E402
from W_decompose.precompute import (  # noqa: E402
    method_tag,
    precompute_model_methods,
)
from W_decompose.static_runner import (  # noqa: E402
    arguments_for_model,
    forwarded_models,
)


DEFAULT_METHODS = ("qr", "original_coordinate", "haar")
DEFAULT_CACHE_ROOT = Path(
    "/pfs/work9/workspace/scratch/hd_nf184-nana-cache/pre_compute_w_decompose"
)


def _method_runs(methods: list[str], haar_seeds: list[int]):
    for method in methods:
        if method == "haar":
            for seed in haar_seeds:
                yield method, seed, method_tag(method, seed)
        else:
            yield method, 0, method_tag(method, 0)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--methods", nargs="+", choices=DECOMPOSITION_METHODS, default=list(DEFAULT_METHODS)
    )
    parser.add_argument("--haar-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--cache-root", default=str(DEFAULT_CACHE_ROOT)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    precompute = commands.add_parser("precompute")
    _common(precompute)
    precompute.add_argument("--models", nargs="+", required=True)
    precompute.add_argument("--device", default="cpu")
    precompute.add_argument("--torch-dtype", choices=("float32", "bfloat16"))
    precompute.add_argument("--start-layer", type=int, default=0)
    precompute.add_argument("--end-layer", type=int)
    precompute.add_argument("--overwrite", action="store_true")
    precompute.add_argument("--store-sorted", action="store_true")

    for name in ("dynamic", "static"):
        run = commands.add_parser(name)
        _common(run)
        run.add_argument(
            "--results-root", default=str(REPO_ROOT / "results" / "w_decompose")
        )
        run.add_argument("forwarded", nargs=argparse.REMAINDER)
    return parser


def _reject_controlled_args(arguments: list[str], names: tuple[str, ...]) -> None:
    for argument in arguments:
        if any(argument == name or argument.startswith(name + "=") for name in names):
            raise SystemExit(
                f"{argument} is controlled by run_w_decompose.py; remove it from arguments after --"
            )


def _run_existing(args) -> None:
    forwarded = list(args.forwarded)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    if not forwarded:
        raise SystemExit(f"{args.command} requires runner arguments after --")
    if args.command == "static" and "neuron" in args.methods:
        raise SystemExit(
            "the neuron decomposition has one atom per neuron and is not valid for "
            "the per-neuron Top-k runner; remove neuron and run it as a separate "
            "global-neuron baseline"
        )

    runner = (
        REPO_ROOT / "scripts_ske_cir" / "run_skeleton_pred.py"
        if args.command == "dynamic"
        else REPO_ROOT / "scripts_ske_cir" / "run_static_ske_per_neuron.py"
    )
    controlled = ("--output_dir",)
    if args.command == "static":
        controlled += ("--cache_base",)
    _reject_controlled_args(forwarded, controlled)
    try:
        models = forwarded_models(forwarded)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    for method, seed, tag in _method_runs(args.methods, args.haar_seeds):
        cache_base = Path(args.cache_root) / tag
        for model_name in models:
            output_dir = (
                Path(args.results_root) / model_name.replace("/", "_") / tag
            )
            model_arguments = arguments_for_model(forwarded, model_name)
            command = [sys.executable, "-u", str(runner), *model_arguments]
            environment = os.environ.copy()
            environment["NANA_MODEL_SCOPED_OUTPUT"] = "1"
            if args.command == "dynamic":
                command.extend(["--output_dir", str(output_dir)])
                environment["SKELETON_CACHE_BASE"] = str(cache_base)
            else:
                command.extend([
                    "--cache_base", str(cache_base),
                    "--output_dir", str(output_dir),
                ])
            print(
                f"[{args.command}] model={model_name} decomposition={method} "
                f"seed={seed} tag={tag} output={output_dir}",
                flush=True,
            )
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "precompute":
        dtype = {
            None: None,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[args.torch_dtype]
        for model_name in args.models:
            runs = list(_method_runs(args.methods, args.haar_seeds))
            for method, seed, tag in runs:
                print(
                    f"[precompute] model={model_name} decomposition={method} "
                    f"seed={seed} tag={tag}", flush=True
                )
            destinations = precompute_model_methods(
                model_name,
                [(method, seed) for method, seed, _ in runs],
                args.cache_root,
                device=args.device,
                torch_dtype=dtype,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                overwrite=args.overwrite,
                store_sorted=args.store_sorted,
            )
            for tag, destination in destinations.items():
                print(f"[saved] tag={tag} path={destination}", flush=True)
    else:
        _run_existing(args)


if __name__ == "__main__":
    main()
