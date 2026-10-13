"""
Plotting utilities for the static-skeleton vs. dynamic-alignment experiment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _plot_cdf(records: list, output_path: str):
    pcts      = np.sort([r["rank_pct"]      for r in records])
    pcts_ctrl = np.sort([r["rank_pct_ctrl"] for r in records])
    y = np.arange(1, len(pcts) + 1) / len(pcts)

    pct_top10      = (np.array([r["rank_pct"]      for r in records]) < 0.10).mean() * 100
    pct_top10_ctrl = (np.array([r["rank_pct_ctrl"] for r in records]) < 0.10).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pcts,      y, color="steelblue", linewidth=2,   label="Top-k")
    ax.plot(pcts_ctrl, y, color="tomato",    linewidth=2,
            linestyle="--", label="Baseline")
    ax.axvline(0.10, color="dimgray", linestyle=":", linewidth=1)
    ax.text(0.11, 0.05,
            f"top-10%:\n  top-k    {pct_top10:.1f}%\n  baseline {pct_top10_ctrl:.1f}%",
            color="dimgray", fontsize=8.5)
    ax.set_xlabel("Rank percentile (lower = better alignment)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Static skeleton rank: top-k vs. baseline", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_layer_mean(per_layer: dict, output_path: str):
    layers = sorted(per_layer.keys(), key=int)
    x      = [int(l) for l in layers]
    obs    = [per_layer[l]["mean_rank_pct"]      for l in layers]
    ctrl   = [per_layer[l]["mean_rank_pct_ctrl"] for l in layers]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, obs,  color="steelblue", marker="o", linewidth=2,
            markersize=5, label="Top-k")
    ax.plot(x, ctrl, color="tomato",    marker="s", linewidth=2,
            markersize=5, linestyle="--", label="Baseline")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean rank percentile", fontsize=12)
    ax.set_title("Per-layer skeleton alignment: top-k vs. baseline", fontsize=13)
    ax.set_xticks(x); ax.xaxis.set_tick_params(labelsize=8)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for xi, l in zip(x, layers):
        ax.text(xi, 0.01, str(per_layer[l]["n_pairs"]),
                ha="center", va="bottom", fontsize=6, color="dimgray", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_layer_violin(records: list, per_layer: dict, output_path: str):
    layers = sorted(per_layer.keys(), key=int)
    x_int  = [int(l) for l in layers]

    obs_by_layer  = {l: [] for l in x_int}
    ctrl_by_layer = {l: [] for l in x_int}
    for r in records:
        li = r["layer_idx"]
        if li in obs_by_layer:
            obs_by_layer[li].append(r["rank_pct"])
            ctrl_by_layer[li].append(r["rank_pct_ctrl"])

    offset = 0.22
    fig, ax = plt.subplots(figsize=(max(9, len(layers) * 0.8), 5))

    vp_obs = ax.violinplot(
        [obs_by_layer[l] for l in x_int],
        positions=[xi - offset for xi in x_int],
        widths=0.38, showmedians=True, showextrema=False,
    )
    vp_ctrl = ax.violinplot(
        [ctrl_by_layer[l] for l in x_int],
        positions=[xi + offset for xi in x_int],
        widths=0.38, showmedians=True, showextrema=False,
    )
    for pc in vp_obs["bodies"]:
        pc.set_facecolor("steelblue"); pc.set_alpha(0.5)
    vp_obs["cmedians"].set_color("navy"); vp_obs["cmedians"].set_linewidth(2)
    for pc in vp_ctrl["bodies"]:
        pc.set_facecolor("tomato"); pc.set_alpha(0.45)
    vp_ctrl["cmedians"].set_color("darkred"); vp_ctrl["cmedians"].set_linewidth(2)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2)
    ax.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.5, label="Top-k"),
        Patch(facecolor="tomato",    alpha=0.45, label="Baseline"),
    ], fontsize=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Rank percentile", fontsize=12)
    ax.set_title("Rank percentile distribution per layer: top-k vs. baseline", fontsize=13)
    ax.set_xticks(x_int); ax.xaxis.set_tick_params(labelsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_plots(summary: dict, output_dir: str):
    records   = summary["records"]
    per_layer = summary["per_layer"]
    os.makedirs(output_dir, exist_ok=True)
    _plot_cdf(records,          os.path.join(output_dir, "cdf.png"))
    _plot_layer_mean(per_layer, os.path.join(output_dir, "layer_mean.png"))
    _plot_layer_violin(records, per_layer,
                                os.path.join(output_dir, "layer_violin.png"))
