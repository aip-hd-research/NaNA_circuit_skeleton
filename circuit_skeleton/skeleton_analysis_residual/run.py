"""CLI for residual-stream cross-MLP causal patching.

The path is source neuron j -> residual-stream subspace i -> downstream neuron k.
It uses the repository's shared model loader, adapters, and coupling builder.

--target_layers and --n_neurons default to "auto":
     target_layers auto -> the next --target_span layers (clipped at model end)
     n_neurons     auto -> d_ff of the start layer's up-projection matrix
                            ("all neurons of the start layer")
   Resolved via model_dims.get_model_dims() (meta-device model construction,
   no real weights loaded, so this is cheap even for 13B models). When
   --n_neurons is an explicit count smaller than d_ff, --neuron_selection
   controls which neurons: "random" (default, fixed-seed sample) or "first"
   (indices 0..n_neurons-1).

--gpu accepts a comma-separated list ("0" or "0,1"). CUDA_VISIBLE_DEVICES
   is set to that list before any CUDA/torch initialization, so a single GPU
   loads normally (device "cuda:0" after remapping) and multiple GPUs trigger
   device_map="auto" (accelerate shards the model across the visible GPUs).
   --dtype defaults to bfloat16 (float32 does not fit 7B+ models on a single
   24GB GPU; see block_interp.model_load.load_model_and_embeddings).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Causal patching (architecture-agnostic): j -> (subspace i ->) target k",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name",    required=True)
    p.add_argument("--start_layer",   type=int, required=True)
    p.add_argument("--target_layers", default="auto",
                   help="e.g. '17-20', '17,18,19,20', or 'auto' (next target_span layers)")
    p.add_argument("--target_span", type=int, default=4,
                   help="Number of following layers used when --target_layers=auto")
    p.add_argument("--n_neurons",     default="auto",
                   help="int, or 'auto' (= d_ff of start_layer's up-projection matrix)")
    p.add_argument("--neuron_selection", default="random", choices=["random", "first"],
                   help="How --n_neurons maps to actual neuron indices: a fixed-seed "
                        "random sample of the layer's d_ff, or indices 0..n_neurons-1")
    p.add_argument("--neuron_seed",   type=int, default=42,
                   help="Seed for --neuron_selection random (reproducible across resume/replot)")
    p.add_argument("--neuron_shard_count", type=int, default=1,
                   help="Split the resolved neuron list into this many deterministic shards")
    p.add_argument("--neuron_shard_index", type=int, default=0,
                   help="Zero-based neuron shard handled by this process")
    p.add_argument("--topk",          type=int, default=30,
                   help="Number of paths selected per source neuron per group (top / rand / bot)")
    p.add_argument("--n_samples",     type=int, default=1000,
                   help="Number of patching-test prompts")
    p.add_argument("--test_sample_offset", type=int, default=0,
                   help="Zero-based start of the patching-test slice")
    p.add_argument("--coefficient_n_samples", type=int, default=1000,
                   help="Number of disjoint prompts used only to estimate Jacobian coefficients")
    p.add_argument("--coefficient_sample_offset", type=int, default=1000,
                   help="Zero-based start of the coefficient-estimation slice")
    p.add_argument("--intervention",  default="zero",
                   choices=["zero", "mean", "scale"])
    p.add_argument("--scale",         type=float, default=10.0)
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--output_base",   required=True)
    p.add_argument("--gpu",           default="0",
                   help="Comma-separated GPU indices, e.g. '0' or '0,1'. "
                        "More than one triggers device_map=auto.")
    p.add_argument("--dtype",         default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--coupling_device", default="gpu", choices=["gpu", "cpu"],
                   help="Device used to build coupling strengths and run the SVD")
    p.add_argument("--batch_size",    type=int, default=32,
                   help="Number of prompts per forward pass (default 32)")
    p.add_argument("--path_chunk_subspaces", type=int, default=16,
                   help="Subspace rows per exact Top/Bottom selection chunk")
    p.add_argument("--intervention_chunk_size", type=int, default=1,
                   help="Independent subspace interventions evaluated per expanded forward batch")
    p.add_argument("--shared_cache_dir", default=None,
                   help="Shared cache for coupling/SVD, baseline activations, and Jacobian coefficients")
    p.add_argument("--cache_only", action="store_true", default=False,
                   help="Build/load shared coupling, baseline, and Jacobian caches, then exit")
    p.add_argument("--resume",        action="store_true", default=True)
    p.add_argument("--no_resume",     dest="resume", action="store_false")
    p.add_argument("--no_finalize", action="store_true", default=False,
                   help="Save per-neuron results but skip aggregate/plots (for concurrent shards)")
    p.add_argument("--no-plots", action="store_true",
                   help="Keep aggregate results but skip statistical plot generation")
    p.add_argument("--skeleton_mode", default="complete_skeleton_global",
                   choices=["complete_skeleton_global", "complete_skeleton_fullinfo"])
    p.add_argument("--print_cpl_delta_abs", action="store_true", default=False)
    return p.parse_args(argv)


def _fmt(seconds: float) -> str:
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    return f"{seconds:.1f}s"


def configure_runtime(args):
    """Set CUDA visibility before importing torch, then attach runtime options."""
    gpu_list = [int(g) for g in str(args.gpu).split(",") if g.strip() != ""]
    if not gpu_list:
        raise ValueError("--gpu must list at least one GPU index (or -1 for CPU)")
    if gpu_list != [-1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    device_map = "auto" if len(gpu_list) > 1 else None
    import torch
    args.device_map = device_map
    args.torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    args.gpu = 0 if device_map is None and gpu_list != [-1] else (-1 if gpu_list == [-1] else 0)


def main(argv=None):
    args = parse_args(argv)
    configure_runtime(args)

    from circuit_skeleton.skeleton_analysis_residual.model_dims import get_model_dims
    from circuit_skeleton.skeleton_analysis_residual.causal_patching_runner_generic import (
        run_causal_patching,
    )

    print(f"[Info] Resolving model dimensions for {args.model_name} "
          f"(start_layer={args.start_layer}) via meta-device inspection …")
    dims = get_model_dims(args.model_name, args.start_layer)
    if dims.get("model_type") in {"gemma2", "gemma3", "gemma3_text"}:
        raise ValueError("Gemma 2/3 are intentionally excluded from this residual experiment")
    print(f"[Info] {args.model_name}: model_type={dims.get('model_type')}  "
          f"n_layers={dims['n_layers']}  d_ff@L{args.start_layer}={dims['d_ff']}  "
          f"total_params={dims['total_params'] / 1e9:.2f}B")

    if args.target_layers == "auto":
        if args.start_layer >= dims["n_layers"] - 1:
            raise ValueError(
                f"start_layer={args.start_layer} is already the last layer "
                f"(n_layers={dims['n_layers']}) — no downstream layers to target."
            )
        if args.target_span <= 0:
            raise ValueError("--target_span must be positive")
        target_end = min(
            args.start_layer + args.target_span,
            dims["n_layers"] - 1,
        )
        args.target_layers = f"{args.start_layer + 1}-{target_end}"
    if args.n_neurons == "auto":
        args.n_neurons = dims["d_ff"]
    else:
        args.n_neurons = int(args.n_neurons)
    print(f"[Info] Resolved target_layers='{args.target_layers}'  n_neurons={args.n_neurons}")

    t_total_start = time.time()
    print(f"[Timer] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("─" * 60)

    should_finalize = run_causal_patching(args)

    elapsed = time.time() - t_total_start
    print("─" * 60)
    print(f"[Timer] Total elapsed: {_fmt(elapsed)}"
          f"  ({elapsed:.1f}s)  "
          f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if should_finalize and not args.no_plots:
        from circuit_skeleton.skeleton_analysis_residual.plot_patching_results_generic import plot_all
        print("\n[Plot] Generating statistical plots …")
        plot_all(args.output_base)
        print(f"[Plot] Plots saved to: {args.output_base}/plots/")
    elif should_finalize:
        print("[Finalize] Aggregate results saved; plot generation skipped.")
    else:
        print("[Finalize] Skipped aggregation and plots for this concurrent shard.")


if __name__ == "__main__":
    main()
