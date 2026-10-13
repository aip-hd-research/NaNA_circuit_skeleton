"""Prepare residual coupling, baseline activation, and derivative caches."""

import argparse

from circuit_skeleton.skeleton_analysis_residual.run import configure_runtime


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--start_layer", type=int, required=True)
    parser.add_argument("--target_layers", required=True)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--test_sample_offset", type=int, default=0)
    parser.add_argument("--coefficient_n_samples", type=int, default=1000)
    parser.add_argument("--coefficient_sample_offset", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--coupling_device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--shared_cache_dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.cache_only = True
    configure_runtime(args)

    from circuit_skeleton.skeleton_analysis_residual.causal_patching_runner_generic import (
        run_causal_patching,
    )

    run_causal_patching(args)


if __name__ == "__main__":
    main()
