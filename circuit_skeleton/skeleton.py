import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_skeleton.within_layer_ske import WithinLayerSkeleton, parse_neurons_arg
from circuit_skeleton.cross_layer_ske import CrossLayerSkeleton
from circuit_skeleton.coupl_strength import CouplingStrength


def compute_skeleton(
    model_name: str,
    start_layer: int,
    target_layers=None,
    start_neuron=None,
    start_space=None,
    within_mlp_topk: int = 20,
    cross_mlp_topk: int = 50,
    device: str = "cpu",
    mode: str = "both",
) -> dict:

    within_results = []
    cross_results  = []
    resolved_target_layers = target_layers

    if mode in ("both", "within"):
        within_ske = WithinLayerSkeleton(model_name, start_layer, device)
        within_results = within_ske.topk_connections(start_neuron, within_mlp_topk)

    if mode in ("both", "cross"):
        cross_ske = CrossLayerSkeleton(model_name, start_layer, target_layers, device)
        cross_results = cross_ske.topk_connections(start_space, cross_mlp_topk)
        resolved_target_layers = cross_ske.target_layers

    return {
        "config": {
            "model_name":      model_name,
            "start_layer":     start_layer,
            "target_layers":   resolved_target_layers,
            "start_neuron":    str(start_neuron) if start_neuron is not None else None,
            "start_space":     str(start_space)  if start_space  is not None else None,
            "within_mlp_topk": within_mlp_topk,
            "cross_mlp_topk":  cross_mlp_topk,
            "device":          device,
            "mode":            mode,
        },
        "within_layer": within_results,
        "cross_layer":  cross_results,
    }


def compute_complete_skeleton(
    model_name: str,
    start_layer: int,
    target_layers,
    start_neuron,
    fullske_topk_sp: int = 5,
    fullske_topk_neu: int = 20,
    device: str = "cpu",
) -> dict:
  
    within_ske = WithinLayerSkeleton(model_name, start_layer, device)
    within_results = within_ske.topk_connections(start_neuron, fullske_topk_sp)
    cross_ske = CrossLayerSkeleton(model_name, start_layer, target_layers, device)
    cs = CouplingStrength(within_ske, cross_ske)

    complete_results = []
    for neuron_rec in within_results:
        j = neuron_rec["neuron_idx"]

        # Top-sp subspace indices for neuron j
        top_sp_indices = [c["subspace_idx"]
                          for c in neuron_rec["connections"][:fullske_topk_sp]]

        # full_strengths[sp_pos, flat_k] = delta[j, i] * gamma[i, k]
        # shape: [1, sp, total_cross]  →  squeeze to [sp, total_cross]
        full_strengths = cs.w_full_path(j, subspace_indices=top_sp_indices).squeeze(0)

        # delta[j, i] values for the selected subspaces — [sp]
        delta_vals = within_ske.delta[j, top_sp_indices]

        # Global top-k over all (subspace, cross-neuron) combinations
        flat = full_strengths.flatten()
        k    = min(fullske_topk_neu, flat.shape[0])
        topk_abs_vals, topk_flat_idx = torch.topk(flat.abs(), k)

        paths = []
        for rank, (flat_idx, abs_full) in enumerate(
            zip(topk_flat_idx.tolist(), topk_abs_vals.tolist()), 1
        ):
            sp_pos     = flat_idx // cs.total_cross
            cross_flat = flat_idx %  cs.total_cross

            i                   = top_sp_indices[sp_pos]
            tgt_layer, neuron_idx = cs.flat_to_layer_neuron(cross_flat)
            delta_val           = delta_vals[sp_pos].item()
            # gamma_stacked rows correspond to all subspaces; index by actual i
            gamma_val           = cs.gamma_stacked[i, cross_flat].item()
            full_val            = flat[flat_idx].item()

            paths.append({
                "subspace_idx":      i,
                "target_layer":      tgt_layer,
                "neuron_idx":        neuron_idx,
                "within_coupling":   delta_val,
                "cross_coupling":    gamma_val,
                "full_strength":     full_val,
                "abs_full_strength": abs_full,
                "rank":              rank,
            })

        complete_results.append({"neuron_idx": j, "paths": paths})

    # Global whole_rank: flatten all paths across all neurons, rank by abs_full_strength
    whole_rank_list = [
        {"src_neuron_idx": rec["neuron_idx"], **p}
        for rec in complete_results
        for p in rec["paths"]
    ]
    whole_rank_list.sort(key=lambda x: x["abs_full_strength"], reverse=True)
    for wr, entry in enumerate(whole_rank_list, 1):
        entry["whole_rank"] = wr

    # Keep standard topk cross-layer results (for graph / plot), topk = fullske_topk_neu
    all_subspace_indices = sorted({c["subspace_idx"]
                                   for rec in within_results
                                   for c in rec["connections"]})
    cross_results = cross_ske.topk_connections(all_subspace_indices, fullske_topk_neu)

    return {
        "config": {
            "model_name":       model_name,
            "start_layer":      start_layer,
            "target_layers":    cross_ske.target_layers,
            "start_neuron":     str(start_neuron),
            "fullske_topk_sp":  fullske_topk_sp,
            "fullske_topk_neu": fullske_topk_neu,
            "device":           device,
            "mode":             "complete",
        },
        "complete_skeleton": complete_results,
        "whole_rank":        whole_rank_list,
        "within_layer":      within_results,
        "cross_layer":       cross_results,
    }


def compute_complete_skeleton_global(
    model_name: str,
    start_layer: int,
    target_layers,
    start_neuron,
    fullske_topk_node: int = 50,
    device: str = "cpu",
) -> dict:
    
    within_ske = WithinLayerSkeleton(model_name, start_layer, device)
    cross_ske  = CrossLayerSkeleton(model_name, start_layer, target_layers, device)
    cs         = CouplingStrength(within_ske, cross_ske)
    print("--------------cs-------",cs.__dict__)

    neuron_indices = parse_neurons_arg(start_neuron, within_ske.d_ff)  # list[int]
    n_sp           = within_ske.n_subspaces

    # Memory guard is handled inside w_full_path (warn_gb_threshold=4.0).
    full_strengths = cs.w_full_path(neuron_indices)   # [n_sel, n_sp, total_cross]

    # global top-k by |full_strength| 
    flat = full_strengths.reshape(-1)
    k    = min(fullske_topk_node, flat.shape[0])
    topk_abs_vals, topk_flat_idx = torch.topk(flat.abs(), k)

    global_top_paths = []
    for rank, (flat_idx, abs_full) in enumerate(
        zip(topk_flat_idx.tolist(), topk_abs_vals.tolist()), 1
    ):
        flat_idx   = int(flat_idx)
        n_pos      = flat_idx // (n_sp * cs.total_cross)
        rem        = flat_idx %  (n_sp * cs.total_cross)
        sp_pos     = rem // cs.total_cross
        cross_flat = rem %  cs.total_cross

        j                     = neuron_indices[n_pos]
        i                     = sp_pos
        tgt_layer, neuron_idx = cs.flat_to_layer_neuron(cross_flat)
        delta_val             = within_ske.delta[j, i].item()
        gamma_val             = cs.gamma_stacked[i, cross_flat].item()
        full_val              = flat[flat_idx].item()

        global_top_paths.append({
            "rank":              rank,
            "src_neuron_idx":    j,
            "subspace_idx":      i,
            "target_layer":      tgt_layer,
            "neuron_idx":        neuron_idx,
            "within_coupling":   delta_val,
            "cross_coupling":    gamma_val,
            "full_strength":     full_val,
            "abs_full_strength": abs_full,
        })

    # synthesise within_layer / cross_layer for graph building  
    # within_layer: group by src_neuron
    within_by_neuron: dict = defaultdict(list)
    for p in global_top_paths:
        j         = p["src_neuron_idx"]
        i         = p["subspace_idx"]
        delta_val = p["within_coupling"]
        within_by_neuron[j].append({
            "neuron_idx":    j,
            "subspace_idx":  i,
            "coupling_pos":  max(delta_val, 0.0),
            "coupling_neg":  min(delta_val, 0.0),
            "abs_strength":  abs(delta_val),
            "rank":          p["rank"],
        })

    within_layer = [
        {"neuron_idx": j, "connections": conns}
        for j, conns in sorted(within_by_neuron.items())
    ]

    # cross_layer: group by subspace
    cross_by_subspace: dict = defaultdict(list)
    for p in global_top_paths:
        i         = p["subspace_idx"]
        gamma_val = p["cross_coupling"]
        cross_by_subspace[i].append({
            "subspace_idx":  i,
            "source_layer":  start_layer,
            "neuron_idx":    p["neuron_idx"],
            "target_layer":  p["target_layer"],
            "coupling_pos":  max(gamma_val, 0.0),
            "coupling_neg":  min(gamma_val, 0.0),
            "abs_strength":  abs(gamma_val),
            "rank":          p["rank"],
        })

    cross_layer = [
        {"source_layer": start_layer, "subspace_idx": i, "connections": conns}
        for i, conns in sorted(cross_by_subspace.items())
    ]

    return {
        "config": {
            "model_name":        model_name,
            "start_layer":       start_layer,
            "target_layers":     cross_ske.target_layers,
            "start_neuron":      str(start_neuron),
            "fullske_topk_node": fullske_topk_node,
            "device":            device,
            "mode":              "complete_global",
        },
        "global_top_paths": global_top_paths,
        "within_layer":     within_layer,
        "cross_layer":      cross_layer,
    }


def compute_complete_skeleton_fullinfo(
    model_name: str,
    start_layer: int,
    target_layers,
    start_neuron,
    fullske_topk_node: int = 50,
    device: str = "cpu",
) -> dict:
    """
    Full-info skeleton: rank (j, k) pairs by |w_contracted[j, k]|.

    w_contracted[j, k] = Σ_i  delta[j,i] * gamma[i,k]

    Unlike complete_skeleton_global (which ranks individual j→i→k paths),
    this sums over ALL intermediate subspaces i first, giving the NET
    weight-space coupling of neuron j to target neuron k — a structural
    prior / static coupling strength, not a measured causal effect: gamma
    (from CrossLayerSkeleton) is a static weight-direction dot product that
    ignores intermediate attention, residual accumulation, real per-token
    LayerNorm/RMSNorm normalization, and gated-MLP nonlinearity (see
    cross_layer_ske.py's module docstring). No subspace index is retained
    because the contribution is already aggregated.
    """
    within_ske = WithinLayerSkeleton(model_name, start_layer, device)
    cross_ske  = CrossLayerSkeleton(model_name, start_layer, target_layers, device)
    cs         = CouplingStrength(within_ske, cross_ske)

    neuron_indices = parse_neurons_arg(start_neuron, within_ske.d_ff)  # list[int]

    # w_contracted: [n_sel, total_cross]  (summed over all subspaces i)
    contracted = cs.w_contracted(neuron_indices)

    # Global top-k by |w_contracted|
    flat = contracted.reshape(-1)                          # [n_sel * total_cross]
    k    = min(fullske_topk_node, flat.shape[0])
    topk_abs_vals, topk_flat_idx = torch.topk(flat.abs(), k)

    global_top_paths = []
    for rank, (flat_idx, abs_val) in enumerate(
        zip(topk_flat_idx.tolist(), topk_abs_vals.tolist()), 1
    ):
        flat_idx   = int(flat_idx)
        n_pos      = flat_idx // cs.total_cross
        cross_flat = flat_idx %  cs.total_cross

        j                     = neuron_indices[n_pos]
        tgt_layer, neuron_idx = cs.flat_to_layer_neuron(cross_flat)
        contracted_val        = flat[flat_idx].item()

        global_top_paths.append({
            "rank":                    rank,
            "src_neuron_idx":          j,
            "target_layer":            tgt_layer,
            "neuron_idx":              neuron_idx,
            "contracted_strength":     contracted_val,
            "abs_contracted_strength": abs_val,
        })

    # synthesise within_layer (group by src_neuron, no subspace info)
    within_by_neuron: dict = defaultdict(list)
    for p in global_top_paths:
        j = p["src_neuron_idx"]
        v = p["contracted_strength"]
        within_by_neuron[j].append({
            "neuron_idx":   j,
            "target_layer": p["target_layer"],
            "tgt_neuron":   p["neuron_idx"],
            "coupling_pos": max(v, 0.0),
            "coupling_neg": min(v, 0.0),
            "abs_strength": abs(v),
            "rank":         p["rank"],
        })

    within_layer = [
        {"neuron_idx": j, "connections": conns}
        for j, conns in sorted(within_by_neuron.items())
    ]

    return {
        "config": {
            "model_name":        model_name,
            "start_layer":       start_layer,
            "target_layers":     cross_ske.target_layers,
            "start_neuron":      str(start_neuron),
            "fullske_topk_node": fullske_topk_node,
            "device":            device,
            "mode":              "complete_fullinfo",
        },
        "global_top_paths": global_top_paths,
        "within_layer":     within_layer,
        "cross_layer":      [],          # no per-subspace info in this mode
    }


def save_skeleton(skeleton_result: dict, path: str) -> None:
    """Save skeleton result dict to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(skeleton_result, f, indent=2)


def load_skeleton(path: str) -> dict:
    """Load a previously saved skeleton result from JSON."""
    with open(path) as f:
        return json.load(f)
