"""
exp1_semantic_consistency.py

Experiment 1: Semantic Consistency of Static Pathways

Reads skeleton.json and labels every endpoint with vocabulary tokens:

  Neuron j   (within-layer source / cross-layer target)
      -> W_up_fused[j] @ W_U.T   (detection direction, LN-fused)
      -> positive direction: what tokens trigger this neuron
      -> negative direction: what tokens suppress this neuron

  Subspace i  (within-layer target / cross-layer source)
      -> V[i] @ W_U.T             (effector direction, from c_proj SVD)
      -> positive direction: what tokens this subspace writes
      -> negative direction: what tokens it suppresses

Both within-layer (neuron->subspace) and cross-layer (subspace->neuron)
connections are labelled and displayed as grouped tables.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from block_interp.model_load import load_model_and_embeddings, get_mlp_matrices
from block_interp.top_tok import top_tokens
from ske_pre_valid.model_adapter import get_adapter
from circuit_skeleton.skeleton import load_skeleton


# token projection helpers

def _proj_tokens(direction: torch.Tensor, W_U: torch.Tensor,
                 tokenizer, topk: int, only_ascii: bool = True) -> tuple:
    """
    Project a d_model direction onto vocab space.

    Returns
    -------
    pos : list[(token_str, score)]   highest scoring tokens
    neg : list[(token_str, score)]   lowest scoring tokens (scores are negative)
    """
    scores = direction.float() @ W_U.T.float()          # [vocab_size]
    pos = top_tokens(tokenizer, scores,  k=topk, only_ascii=only_ascii)
    neg = top_tokens(tokenizer, -scores, k=topk, only_ascii=only_ascii)
    neg = [(t, -float(v)) for t, v in neg]
    pos = [(t,  float(v)) for t, v in pos]
    return pos, neg


# model loading helpers

def _load_wup_and_v(model, layer_idx: int, device: str):
    """Return (W_up_fused, V) for a given layer."""
    adapter = get_adapter(model)
    W_in, W_out, ln, _ = get_mlp_matrices(model, layer_idx)
    with torch.no_grad():
        W_up_fused = adapter.w_in_fused(W_in, ln.weight.detach()).to(device)
    _, _, V = adapter.svd_w_out(W_out)
    return W_up_fused, V.to(device)


# main analysis

def label_skeleton(
    skeleton_path: str,
    topk_tokens: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Label every (neuron, subspace) connection recorded in skeleton.json
    with vocabulary tokens via logit-lens projection.

    Returns
    -------
    dict:
        "config"        : original skeleton config + exp parameters
        "within_layer"  : labelled within-layer connection records
        "cross_layer"   : labelled cross-layer connection records
    """
    skeleton   = load_skeleton(skeleton_path)
    cfg        = skeleton["config"]
    model_name = cfg["model_name"]
    start_layer = cfg["start_layer"]

    print(f"Loading {model_name} ...")
    model, tokenizer, W_U = load_model_and_embeddings(model_name, device)

    # Source layer: W_up_fused for neuron labels, V for subspace labels
    W_up_src, V_src = _load_wup_and_v(model, start_layer, device)

    # Cache W_up_fused for target layers (cross-layer connections)
    wup_cache = {start_layer: W_up_src}
    def _get_wup(layer):
        if layer not in wup_cache:
            wup, _ = _load_wup_and_v(model, layer, device)
            wup_cache[layer] = wup
        return wup_cache[layer]

    # within-layer
    labelled_within = []
    for neuron_rec in skeleton["within_layer"]:
        j = neuron_rec["neuron_idx"]
        neu_pos, neu_neg = _proj_tokens(W_up_src[j], W_U, tokenizer, topk_tokens)

        conns = []
        for conn in neuron_rec["connections"]:
            i = conn["subspace_idx"]
            sub_pos, sub_neg = _proj_tokens(V_src[i], W_U, tokenizer, topk_tokens)
            conns.append({
                **conn,
                "neuron_pos_tokens":   neu_pos,
                "neuron_neg_tokens":   neu_neg,
                "subspace_pos_tokens": sub_pos,
                "subspace_neg_tokens": sub_neg,
            })

        labelled_within.append({"neuron_idx": j, "connections": conns})

    # cross-layer
    labelled_cross = []
    for space_rec in skeleton["cross_layer"]:
        i = space_rec["subspace_idx"]
        sub_pos, sub_neg = _proj_tokens(V_src[i], W_U, tokenizer, topk_tokens)

        conns = []
        for conn in space_rec["connections"]:
            tgt = conn["target_layer"]
            k   = conn["neuron_idx"]
            neu_pos, neu_neg = _proj_tokens(_get_wup(tgt)[k], W_U, tokenizer, topk_tokens)
            conns.append({
                **conn,
                "subspace_pos_tokens": sub_pos,
                "subspace_neg_tokens": sub_neg,
                "neuron_pos_tokens":   neu_pos,
                "neuron_neg_tokens":   neu_neg,
            })

        labelled_cross.append({
            "source_layer": space_rec["source_layer"],
            "subspace_idx": i,
            "connections":  conns,
        })

    return {
        "config":       {**cfg, "topk_tokens": topk_tokens},
        "within_layer": labelled_within,
        "cross_layer":  labelled_cross,
    }
