"""
causal_patching_runner.py

Top-level driver for the causal patching experiment:
  - Loads model, tokenizer, dataset
  - Collects baseline activations
  - Loops over source neurons, calling analyse_one_neuron for each
  - Aggregates and prints results
"""

import json
import os

from block_interp.model_load import load_model_and_embeddings
from circuit_skeleton.coupl_strength import build_coupling_strength
from circuit_skeleton.skeleton_analysis_crossmlp.static_dynamic_crosslayer import collect_activations
from circuit_skeleton.skeleton_analysis_crossmlp.coupling_path_selector import _parse_layers
from circuit_skeleton.skeleton_analysis_crossmlp.per_neuron_causal_analyse import analyse_one_neuron
from circuit_skeleton.skeleton_analysis_crossmlp.result_aggregate import (
    aggregate_and_print,
    _load_and_aggregate,
)


def run_causal_patching(args):
    """Entry point called by scripts/run_ske4align_global_ske.py."""
    device        = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    target_layers = _parse_layers(args.target_layers)
    neuron_ids    = list(range(args.n_neurons))

    #  Resume logic 
    if args.resume:
        pending = [
            j for j in neuron_ids
            if not os.path.isfile(
                os.path.join(args.output_base, f"neuron_{j}", "patching_results.json")
            )
        ]
        print(f"Resume: {args.n_neurons - len(pending)} done, {len(pending)} to run.")
    else:
        pending = neuron_ids

    if not pending:
        print("All done. Aggregating...")
        _load_and_aggregate(args)
        return

    #  Load model & data 
    cs                  = build_coupling_strength(args.model_name, args.start_layer,
                                                   target_layers, device)
    model, tokenizer, _ = load_model_and_embeddings(args.model_name, device)

    with open(args.dataset_path) as f:
        raw = json.load(f)
    prompts = [raw[k]["prompt"] for k in list(raw.keys())[:args.n_samples]]

    batch_size    = getattr(args, "batch_size", 32)
    all_layers    = sorted({args.start_layer} | set(target_layers))
    baseline_acts = collect_activations(model, tokenizer, prompts, all_layers, device,
                                        batch_size=batch_size)
    mean_acts_src = baseline_acts[args.start_layer].mean(axis=0)
    baseline_tgt  = {li: baseline_acts[li] for li in target_layers}

    #  Per-neuron loop 
    summaries = []
    for pos, j in enumerate(pending, 1):
        out_dir = os.path.join(args.output_base, f"neuron_{j}")
        print(f"===== Neuron {j:4d}  ({pos}/{len(pending)}) =====")
        try:
            summary = analyse_one_neuron(
                j, cs,
                model, tokenizer, prompts,
                baseline_tgt,
                target_layers, args.start_layer,
                args.topk, args.intervention,
                float(mean_acts_src[j]), args.scale,
                device, out_dir,
                skeleton_mode=getattr(args, "skeleton_mode", "complete_skeleton_global"),
                batch_size=batch_size,
                print_cpl_delta_abs=getattr(args, "print_cpl_delta_abs", False),
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"  [ERROR] neuron {j}: {exc}")

    # merge previously-completed neurons (resume mode)
    for j in set(neuron_ids) - set(pending):
        p = os.path.join(args.output_base, f"neuron_{j}", "patching_results.json")
        if os.path.isfile(p):
            with open(p) as f:
                summaries.append(json.load(f))

    summaries.sort(key=lambda s: s["config"]["src_neuron"])
    aggregate_and_print(summaries, args.output_base, args.intervention)
