"""
Experiment: Static skeleton vs. dynamic behavior alignment.

Hypothesis: when neuron j dynamically drives subspace i*, the static skeleton
should rank i* near the top of j's connections (low rank = strong structural link).

For each (prompt, target_token, layer):
  1. circuit_analysis -> top subspaces I* and, for each i*, top driving neurons J*(i*)
  2. For each (j, i*) pair: find rank of i* in j's static skeleton (delta row)
  3. Control: for the same j, find rank of a randomly sampled subspace i_rand
  4. Compare observed vs. control rank distributions and save plots

The control rules out the mathematical coupling: j was selected because |U[j,i*]| is
large, which also inflates skeleton rank. The control uses a different random subspace
for the same j, exposing whether j's skeleton is uniformly flat or truly concentrated.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
from collections import defaultdict

from block_interp.interp_mlp import load_model_and_embeddings
from block_interp.model_load import parse_layers_arg
from circuit.circuit_analysis import analyze_mlp_subspaces
from circuit.collect_circuit_info import layer_info
from circuit_skeleton.within_layer_ske import WithinLayerSkeleton
from circuit_skeleton.skeleton_analysis_withinmlp.plot_alignment import save_plots


# Main experiment
def run(
    model_name: str,
    dataset_path: str,
    layers,
    topk_subspaces: int,
    topk_neurons: int,
    n_samples: int,
    output_dir: str,
    device,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    with open(dataset_path) as f:
        raw = json.load(f)
    items = [raw[k] for k in list(raw.keys())[:n_samples]]
    print(f"Loaded {len(items)} samples from {dataset_path}")

    # Load model
    model, tokenizer, W_E = load_model_and_embeddings(model_name, device)
    layers_to_use = parse_layers_arg(layers, model_name)
    print(f"Layers to process: {layers_to_use}")

    # Pre-load static skeletons and precompute rank lookup tables
    print("Pre-loading static skeletons...")
    skeletons = {
        layer_idx: WithinLayerSkeleton(model_name, layer_idx, device="cpu")
        for layer_idx in layers_to_use
    }
    n_subspaces = next(iter(skeletons.values())).n_subspaces
    print(f"  n_subspaces = {n_subspaces}")

    print("Precomputing rank lookup tables...")
    rank_tables = {}
    for layer_idx, ske in skeletons.items():
        rank_tables[layer_idx] = {
            info["neuron_idx"]: {conn["subspace_idx"]: conn["rank"] for conn in info["connections"]}
            for info in ske.topk_connections("ALL", n_subspaces)
        }
    print("  Done.")

    rng = np.random.default_rng(seed=42)

    # record fields:
    #   sent_idx, layer_idx, neuron_idx, subspace_idx
    #   rank / rank_pct           <- observed (dynamic i*)
    #   rank_ctrl / rank_pct_ctrl <- control  (random i_rand, same j)
    records = []

    for sent_idx, item in enumerate(items):
        in_seq      = item["prompt"]
        target_word = item["target_token"]
        # Use the sample's own id when the dataset provides one (gen_next_token_labels.py
        # datasets do) instead of letting analyze_mlp_subspaces re-resolve target_word by
        # string — distinct tokens can decode to the same text, so re-resolving risks
        # scoring a different token than the one this specific sample actually has.
        target_id = item.get("target_token_id")
        print(f"[{sent_idx+1}/{len(items)}] {in_seq[:50]!r}  -> {target_word!r}"
              + (f" (id={target_id})" if target_id is not None else ""))

        for layer_idx in layers_to_use:
            try:
                _, U, S, Vh, layer_io, _, _, _, _, _, _ = layer_info(
                    model, tokenizer, W_E, in_seq, layer_idx, device, "c_proj"
                )
                _, circuit_score = analyze_mlp_subspaces(
                    model_name, tokenizer, W_E, U, S, Vh,
                    layer_idx, layer_io, target_word,
                    target_id=target_id,
                    topk_subspaces=topk_subspaces,
                    weight_type="c_proj",
                    mode="DeEf",
                    do_interp=False,
                    return_heatmap=False,
                    topk_neurons=topk_neurons,
                    device=device,
                )
            except Exception as e:
                print(f"  Layer {layer_idx} skipped: {e}")
                continue

            rtable = rank_tables[layer_idx]

            for sub_res in circuit_score["subspace_results"]:
                i_star       = sub_res["subspace_index"] - 1   # 0-indexed
                neurons_1idx = sub_res.get("neuron_indices", [])
                for j_1idx in neurons_1idx:
                    j         = j_1idx - 1
                    rank      = rtable[j][i_star]
                    i_rand    = int(rng.choice([i for i in range(n_subspaces) if i != i_star]))
                    rank_ctrl = rtable[j][i_rand]
                    records.append({
                        "sent_idx":      sent_idx,
                        "layer_idx":     layer_idx,
                        "neuron_idx":    j,
                        "subspace_idx":  i_star,
                        "rank":          rank,
                        "rank_pct":      rank / n_subspaces,
                        "rank_ctrl":     rank_ctrl,
                        "rank_pct_ctrl": rank_ctrl / n_subspaces,
                    })

    if not records:
        print("No records collected. Check dataset fields and layer coverage.")
        return {}

    # Aggregate
    pcts      = np.array([r["rank_pct"]      for r in records])
    pcts_ctrl = np.array([r["rank_pct_ctrl"] for r in records])
    ranks     = np.array([r["rank"]          for r in records])

    by_layer      = defaultdict(list)
    by_layer_ctrl = defaultdict(list)
    for r in records:
        by_layer[r["layer_idx"]].append(r["rank_pct"])
        by_layer_ctrl[r["layer_idx"]].append(r["rank_pct_ctrl"])

    print(f"\n=== Alignment Results ===")
    print(f"  pairs total          : {len(ranks)}")
    print(f"                         top-k     |  baseline")
    print(f"  mean rank pct        : {pcts.mean():8.4f}  | {pcts_ctrl.mean():8.4f}")
    print(f"  median rank          : {np.median(ranks):8.1f}  | {np.median([r['rank_ctrl'] for r in records]):8.1f}")
    print(f"  % in top-10%         : {100*(pcts<0.10).mean():7.1f}%  | {100*(pcts_ctrl<0.10).mean():7.1f}%")
    print(f"  % in top-20%         : {100*(pcts<0.20).mean():7.1f}%  | {100*(pcts_ctrl<0.20).mean():7.1f}%")
    print("\nPer-layer (top-k | baseline):")
    for li in sorted(by_layer):
        print(f"  Layer {li:3d}: {np.mean(by_layer[li]):.4f} | "
              f"{np.mean(by_layer_ctrl[li]):.4f}  (n={len(by_layer[li])})")

    summary = {
        "config": {
            "model_name":     model_name,
            "layers":         layers_to_use,
            "topk_subspaces": topk_subspaces,
            "topk_neurons":   topk_neurons,
            "n_samples":      n_samples,
            "n_subspaces":    int(n_subspaces),
        },
        "global": {
            "n_pairs":             int(len(ranks)),
            "mean_rank_pct":       float(pcts.mean()),
            "mean_rank_pct_ctrl":  float(pcts_ctrl.mean()),
            "median_rank":         float(np.median(ranks)),
            "pct_top10":           float((pcts < 0.10).mean()),
            "pct_top10_ctrl":      float((pcts_ctrl < 0.10).mean()),
            "pct_top20":           float((pcts < 0.20).mean()),
            "pct_top20_ctrl":      float((pcts_ctrl < 0.20).mean()),
        },
        "per_layer": {
            str(k): {
                "mean_rank_pct":      float(np.mean(by_layer[k])),
                "mean_rank_pct_ctrl": float(np.mean(by_layer_ctrl[k])),
                "n_pairs":            len(by_layer[k]),
            }
            for k in by_layer
        },
        "records": records,
    }

    # Save JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON to {json_path}")

    # Save plots
    save_plots(summary, output_dir)

    return summary
