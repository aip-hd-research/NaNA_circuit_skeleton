"""
Pairwise per-neuron skeleton-path Jaccard overlap across target tokens.

For each pair of target tokens, and each layer:
  - Select top-k_per_neuron subspaces per neuron by |delta*eff|
  - Compute Jaccard between selected subspace sets, per neuron
  - Average Jaccard over all neurons (layer score)
  - Average over all layers (final score)

Example:
    python scripts_ske_cir/run_skeleton_overlap.py \
        --model_name openai-community/gpt2-medium \
        --start_layer 5 --end_layer 22 \
        --k_per_neuron 5 \
        --gpu 0
"""

import os
import sys
import argparse
import glob
import re

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = "/pfs/work9/workspace/scratch/hd_nf184-nana-cache"
sys.path.insert(0, CODE_ROOT)

from model_config.paths import within_ske_path
from ske_pre_valid.overlap_ana.overlap_compute import run_overlap_analysis

CACHE_BASE = os.path.join(REPO_ROOT, "pre_compute_skeleton_cache")
MODEL_CACHE_DIR = os.path.join(REPO_ROOT, "model_cache")

DEFAULT_CATEGORIES = {
    "Country": [" France", " Germany", " Italy", " England"],
    "Number": [" two", " three", " four", " five"],
    "Month": [" January", " December", " February", " March"],
    "Direction": [ " north", " south", " east", " west"]
}


def _last_layer(cache_dir: str) -> int:
    files = glob.glob(os.path.join(cache_dir, "layer_*.pt"))
    if not files:
        raise RuntimeError(f"No layer_*.pt files found in {cache_dir}")
    return max(int(re.search(r"layer_(\d+)", f).group(1)) for f in files)


def parse_args():
    p = argparse.ArgumentParser(
        description="Pairwise per-neuron Jaccard overlap of skeleton paths for different target tokens"
    )
    p.add_argument("--model_name",    default="openai-community/gpt2")
    p.add_argument("--cache_base",    default=CACHE_BASE,
                   help=f"Root of pre-computed skeleton cache (default: {CACHE_BASE})")
    p.add_argument("--start_layer",   type=int, default=5,
                   help="First layer (default: 0)")
    p.add_argument("--end_layer",     type=int, default=8,
                   help="Last layer (default: auto-detect from cache)")
    p.add_argument("--k_per_neuron",  type=int, default=3,
                   help="Top-k subspaces per neuron")
    p.add_argument("--matrix_cmap", default="emerald_green",
                   help="Heatmap colormap or preset (emerald_green, soft_coral, reference)")
    p.add_argument("--matrix_colors", nargs="+", default=None,
                   help="Custom low-to-high heatmap colors; overrides --matrix_cmap")
    p.add_argument("--category_colors", nargs="+", default=None,
                   help="Category colors in DEFAULT_CATEGORIES order (for example '#E64B35')")
    p.add_argument("--category_palette", default="neutral_green",
                   choices=("neutral_green", "static_groups", "nature_muted", "npg", "okabe_ito", "tol_muted"),
                   help="Preset category-strip palette; ignored with --category_colors")
    p.add_argument("--dendrogram_ratio", type=float, default=0.08,
                   help="Fractional width/height allocated to row/column dendrograms")
    p.add_argument("--annotation_gap_mm", type=float, default=1.5,
                   help="Gap in millimetres between category strips and heatmap")
    p.add_argument("--dendrogram_gap_mm", type=float, default=0.0,
                   help="Gap in millimetres between dendrograms and category strips")
    p.add_argument("--color_target_labels", action=argparse.BooleanOptionalAction,
                   default=True, help="Color row/column target labels by category")
    p.add_argument("--target_font_size", type=float, default=7.0,
                   help="Row/column target-label font size in points")
    p.add_argument("--target_font_color", default=None,
                   help="Fixed target-label color; overrides category label colors")
    p.add_argument("--target_category_colors", nargs="+",
                   default=("#176b45", "#176b45", "#3a3a3a", "#3a3a3a"),
                   help="Colors assigned in DEFAULT_CATEGORIES order")
    p.add_argument("--target_font_weight", default="bold",
                   choices=("normal", "semibold", "bold"),
                   help="Row/column target-label font weight")
    p.add_argument("--cluster_gap", type=float, default=0.0,
                   help="Optional gap between adjacent category clusters, in cell units")
    p.add_argument("--cluster_separator_width", type=float, default=0.7,
                   help="Dark category-boundary line width in points")
    p.add_argument("--tile_min_size", type=float, default=0.10,
                   help="Inner-square side length at zero overlap, in cell units")
    p.add_argument("--tile_max_size", type=float, default=0.86,
                   help="Inner-square side length at unit overlap, in cell units")
    p.add_argument("--colorbar_gap_mm", type=float, default=3.0,
                   help="Horizontal gap between row dendrogram and colorbar")
    p.add_argument("--colorbar_width_mm", type=float, default=3.0,
                   help="Colorbar width in millimetres")
    p.add_argument("--colorbar_height_mm", type=float, default=30.0,
                   help="Colorbar height in millimetres")
    p.add_argument("--figure_title", default="",
                   help="Optional panel title; empty by default for publication figures")
    p.add_argument("--target_tokens", nargs="+", default=[
        token for tokens in DEFAULT_CATEGORIES.values() for token in tokens
    ],
                   help="Target tokens to compare (use quotes for tokens with spaces)")
    p.add_argument("--output_dir",    default=os.path.join(
        CODE_ROOT, "exp_nature", "all_figures", "overlap_ana"
    ))
    p.add_argument("--gpu",           type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
    args = parse_args()
    args.categories = DEFAULT_CATEGORIES

    cache_dir = within_ske_path(args.model_name, args.cache_base)
    if cache_dir is None:
        raise RuntimeError(
            f"No within-skeleton cache found for {args.model_name!r} "
            f"under {args.cache_base}.\nRun pre_compute.sh first."
        )
    args.cache_dir = cache_dir

    last_layer = _last_layer(cache_dir)
    if args.start_layer is None:
        args.start_layer = 0
    if args.end_layer is None:
        args.end_layer = last_layer
    if args.start_layer < 0 or args.start_layer > args.end_layer:
        raise ValueError(
            f"Invalid layer range {args.start_layer}-{args.end_layer}; "
            "require 0 <= start_layer <= end_layer."
        )
    if args.end_layer > last_layer:
        raise ValueError(
            f"Requested end layer {args.end_layer}, but the cache for "
            f"{args.model_name!r} only goes through layer {last_layer}."
        )

    print(f"Model     : {args.model_name}")
    print(f"Cache     : {cache_dir}")
    print(f"Layers    : {args.start_layer}–{args.end_layer}")

    run_overlap_analysis(args)
