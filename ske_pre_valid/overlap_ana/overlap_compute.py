"""
overlap_compute.py — per-neuron Jaccard overlap for static skeleton analysis.

For each pair of target tokens (A, B):
  For each layer l:
    For each neuron j:
      Jaccard( top-k subspaces selected for A,  top-k subspaces selected for B )
    → average over all neurons  → layer Jaccard
  → average over all layers     → final Jaccard

W_emb is loaded from the pre-saved W_U.pt in the within-skeleton cache directory,
so no model weights are loaded at runtime — only the tokenizer is needed.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from transformers import AutoTokenizer

from ske_pre_valid.within_ske_cache import WithinSkeCache
from ske_pre_valid.forward_hook.forward_per_neuron import select_per_neuron_abs_topk
from ske_pre_valid.overlap_ana.overlap_plot_clustered_heatmap import (
    plot_overlap_clustered_heatmap, plot_overlap_clustered_heatmap_normalized,
)


def _load_tokenizer_and_wemb(model_name: str, cache_dir: str):
    """Load tokenizer from HF cache and W_emb from pre-saved W_U.pt in cache_dir.

    No model weights are loaded — W_U.pt is produced by pre_compute.sh.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    w_u_path  = os.path.join(cache_dir, "W_U.pt")
    if not os.path.exists(w_u_path):
        raise FileNotFoundError(
            f"W_U.pt not found at {w_u_path}\n"
            "Run pre_compute.sh to generate the within-skeleton cache first."
        )
    W_emb = torch.load(w_u_path, map_location="cpu", weights_only=True).float()
    print(f"  W_emb loaded from {w_u_path}  shape={tuple(W_emb.shape)}")
    return tokenizer, W_emb


# ── effector helper ────────────────────────────────────────────────────────────

def _effector_scores(V, W_emb, target_id, post_ffn_norm_weight=None):
    V_cpu = V.cpu()
    W_cpu = W_emb.cpu()
    if post_ffn_norm_weight is not None:
        V_cpu = V_cpu * post_ffn_norm_weight.cpu()
    return V_cpu @ W_cpu[target_id]


# ── path-dict builder ──────────────────────────────────────────────────────────

def build_path_dict_per_neuron(cache, W_emb, target_id,
                               start_layer, end_layer, k_per_neuron):
    """
    For each layer, select top-k_per_neuron subspaces per neuron by |delta*eff|.

    Returns
    -------
    dict[layer_idx, dict[neuron_j, frozenset[subspace_idx]]]
    """
    W_emb_target = W_emb[target_id].cpu()
    result = {}
    for li in range(start_layer, end_layer + 1):
        ld  = cache.layer_skeleton(li)
        eff = _effector_scores(
            ld["V"], W_emb, target_id, ld.get("post_ffn_norm_weight"))

        paths  = select_per_neuron_abs_topk(
            ld["delta"], ld["V"], W_emb_target,
            k_per_neuron, precomputed_effector=eff,
        )
        d_ff   = ld["delta"].shape[0]
        k_each = paths["i_idx"].shape[0] // d_ff   # actual k (may be < k_per_neuron)
        i_mat  = paths["i_idx"].reshape(d_ff, k_each)

        result[li] = {j: frozenset(i_mat[j].tolist()) for j in range(d_ff)}
    return result


# ── Jaccard metrics ────────────────────────────────────────────────────────────

def _jaccard_per_neuron_layer(a_layer: dict, b_layer: dict) -> float:
    """Mean Jaccard over all neurons at one layer.

    a_layer[j] = frozenset of selected subspace indices for neuron j.
    """
    neurons = sorted(set(a_layer) & set(b_layer))
    if not neurons:
        return 0.0
    jaccards = []
    for j in neurons:
        a, b  = a_layer[j], b_layer[j]
        union = len(a | b)
        jaccards.append(len(a & b) / union if union > 0 else 0.0)
    return float(np.mean(jaccards))


def mean_jaccard_per_neuron(dict_a: dict, dict_b: dict) -> float:
    """Mean Jaccard: first average over neurons per layer, then over layers."""
    layers = sorted(set(dict_a) & set(dict_b))
    if not layers:
        return 0.0
    return float(np.mean([
        _jaccard_per_neuron_layer(dict_a[l], dict_b[l]) for l in layers
    ]))


def overlap_matrix(path_dicts: list[dict]) -> np.ndarray:
    n   = len(path_dicts)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = mean_jaccard_per_neuron(path_dicts[i], path_dicts[j])
    return mat


# ── experiment runner ──────────────────────────────────────────────────────────

def run_overlap_analysis(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HOME", "/nfs/data/shared/model_cache")

    tokenizer, W_emb = _load_tokenizer_and_wemb(args.model_name, args.cache_dir)
    cache = WithinSkeCache(args.cache_dir, expected_model_name=args.model_name)

    from ske_pre_valid.static_skeleton import _resolve_target_info
    tokens    = args.target_tokens
    token_ids = [_resolve_target_info(tokenizer, t)["target_id"] for t in tokens]
    print(f"\nTarget tokens ({len(tokens)}): {list(zip(tokens, token_ids))}")

    n_layers  = args.end_layer - args.start_layer + 1

    tag = f"k/neuron={args.k_per_neuron}"
    print(f"\n[Per-Neuron]  {tag}  layers {args.start_layer}–{args.end_layer}"
          "  effector=raw-dot")

    path_dicts = []
    for t, tid in zip(tokens, token_ids):
        pd = build_path_dict_per_neuron(
            cache, W_emb, tid,
            args.start_layer, args.end_layer, args.k_per_neuron,
        )
        total = sum(len(s) for layer in pd.values() for s in layer.values())
        path_dicts.append(pd)
        print(f"  {t!r:20s}  id={tid:6d}  total_paths={total}"
              f"  ({total // n_layers}/layer avg)")

    mat = overlap_matrix(path_dicts)

    categories = args.categories

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n  Per-neuron Jaccard (avg over neurons, avg over layers):")
    _print_matrix(mat, tokens)

    plot_overlap_clustered_heatmap(
        mat, tokens,
        title=args.figure_title,
        out_path=os.path.join(args.output_dir, "per_neuron_overlap.png"),
        categories=categories,
        cmap=args.matrix_cmap,
        matrix_colors=args.matrix_colors,
        category_colors=args.category_colors,
        category_palette=args.category_palette,
        dendrogram_ratio=args.dendrogram_ratio,
        annotation_gap_mm=args.annotation_gap_mm,
        dendrogram_gap_mm=args.dendrogram_gap_mm,
        color_target_labels=args.color_target_labels,
        target_font_size=args.target_font_size,
        target_font_color=args.target_font_color,
        target_category_colors=args.target_category_colors,
        target_font_weight=args.target_font_weight,
        cluster_gap=args.cluster_gap,
        cluster_separator_width=args.cluster_separator_width,
        tile_min_size=args.tile_min_size,
        tile_max_size=args.tile_max_size,
        colorbar_gap_mm=args.colorbar_gap_mm,
        colorbar_width_mm=args.colorbar_width_mm,
        colorbar_height_mm=args.colorbar_height_mm,
    )
    plot_overlap_clustered_heatmap_normalized(
        mat, tokens,
        title=args.figure_title,
        out_path=os.path.join(args.output_dir, "per_neuron_overlap_norm.png"),
        categories=categories,
        cmap=args.matrix_cmap,
        matrix_colors=args.matrix_colors,
        category_colors=args.category_colors,
        category_palette=args.category_palette,
        dendrogram_ratio=args.dendrogram_ratio,
        annotation_gap_mm=args.annotation_gap_mm,
        dendrogram_gap_mm=args.dendrogram_gap_mm,
        color_target_labels=args.color_target_labels,
        target_font_size=args.target_font_size,
        target_font_color=args.target_font_color,
        target_category_colors=args.target_category_colors,
        target_font_weight=args.target_font_weight,
        cluster_gap=args.cluster_gap,
        cluster_separator_width=args.cluster_separator_width,
        tile_min_size=args.tile_min_size,
        tile_max_size=args.tile_max_size,
        colorbar_gap_mm=args.colorbar_gap_mm,
        colorbar_width_mm=args.colorbar_width_mm,
        colorbar_height_mm=args.colorbar_height_mm,
    )


def _print_matrix(mat, labels):
    w = max(len(t) for t in labels) + 2
    print(" " * w + "  ".join(f"{t:>{w}}" for t in labels))
    for i, lbl in enumerate(labels):
        row = "  ".join(f"{mat[i, j]:>{w}.3f}" for j in range(len(labels)))
        print(f"{lbl:<{w}}{row}")
