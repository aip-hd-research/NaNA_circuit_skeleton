"""
Visualize skeleton-vs-dynamic alignment experiment results.

Three figures, each showing observed vs. empirical random control:
  1. Rank percentile CDF
  2. Per-layer mean rank percentile (line plot)
  3. Per-layer rank percentile distribution (violin plot)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_results(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def plot_rank_cdf(records: list, output_path: str):
    """CDF of rank percentiles: observed vs. empirical random control."""
    pcts      = np.sort([r["rank_pct"]      for r in records])
    pcts_ctrl = np.sort([r["rank_pct_ctrl"] for r in records])
    y = np.arange(1, len(pcts) + 1) / len(pcts)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pcts,      y, color="steelblue", linewidth=2,   label="Observed (dynamic i*)")
    ax.plot(pcts_ctrl, y, color="tomato",    linewidth=2,
            linestyle="--", label="Control (random i, same j)")
    ax.plot([0, 1],    [0, 1], color="gray", linewidth=1.2,
            linestyle=":", label="Theoretical random")

    pct_top10      = (np.array([r["rank_pct"]      for r in records]) < 0.10).mean() * 100
    pct_top10_ctrl = (np.array([r["rank_pct_ctrl"] for r in records]) < 0.10).mean() * 100
    ax.axvline(0.10, color="dimgray", linestyle=":", linewidth=1)
    ax.text(0.11, 0.05,
            f"top-10%:\n  observed {pct_top10:.1f}%\n  control  {pct_top10_ctrl:.1f}%",
            color="dimgray", fontsize=8.5)

    ax.set_xlabel("Rank percentile (lower = better alignment)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Static skeleton rank: observed vs. control", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_layer_mean(per_layer: dict, output_path: str):
    """Line plot: observed vs. control mean rank percentile per layer."""
    layers = sorted(per_layer.keys(), key=int)
    x      = [int(l) for l in layers]
    obs    = [per_layer[l]["mean_rank_pct"]      for l in layers]
    ctrl   = [per_layer[l]["mean_rank_pct_ctrl"] for l in layers]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, obs,  color="steelblue", marker="o", linewidth=2, markersize=5,
            label="Observed (dynamic i*)")
    ax.plot(x, ctrl, color="tomato",    marker="s", linewidth=2, markersize=5,
            linestyle="--", label="Control (random i, same j)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2,
               label="Theoretical random (0.50)")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean rank percentile", fontsize=12)
    ax.set_title("Per-layer skeleton alignment: observed vs. control", fontsize=13)
    ax.set_xticks(x)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # n_pairs annotation
    for xi, l in zip(x, layers):
        ax.text(xi, 0.01, str(per_layer[l]["n_pairs"]),
                ha="center", va="bottom", fontsize=6, color="dimgray", rotation=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_layer_violin(records: list, per_layer: dict, output_path: str):
    """Side-by-side violin: observed (blue) vs. control (red) per layer."""
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

    pos_obs  = [xi - offset for xi in x_int]
    pos_ctrl = [xi + offset for xi in x_int]

    vp_obs = ax.violinplot(
        [obs_by_layer[l] for l in x_int],
        positions=pos_obs, widths=0.38,
        showmedians=True, showextrema=False,
    )
    vp_ctrl = ax.violinplot(
        [ctrl_by_layer[l] for l in x_int],
        positions=pos_ctrl, widths=0.38,
        showmedians=True, showextrema=False,
    )

    for pc in vp_obs["bodies"]:
        pc.set_facecolor("steelblue"); pc.set_alpha(0.5)
    vp_obs["cmedians"].set_color("navy"); vp_obs["cmedians"].set_linewidth(2)

    for pc in vp_ctrl["bodies"]:
        pc.set_facecolor("tomato"); pc.set_alpha(0.45)
    vp_ctrl["cmedians"].set_color("darkred"); vp_ctrl["cmedians"].set_linewidth(2)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Theoretical random")

    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.5, label="Observed (dynamic i*)"),
        Patch(facecolor="tomato",    alpha=0.45, label="Control (random i, same j)"),
    ], fontsize=10)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Rank percentile", fontsize=12)
    ax.set_title("Rank percentile distribution per layer: observed vs. control", fontsize=13)
    ax.set_xticks(x_int)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_alignment(json_path: str, output_dir: str = None):
    """
    Load alignment results and save all three plots.

    Args:
        json_path:  path to result/alignment_exp.json
        output_dir: directory to save figures (defaults to same dir as json_path)
    """
    data = load_results(json_path)

    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)

    records   = data["records"]
    per_layer = data["per_layer"]
    g         = data["global"]

    print(f"Loaded {g['n_pairs']} (neuron, subspace) pairs.")
    print(f"  Observed  mean rank pct: {g['mean_rank_pct']:.4f}")
    print(f"  Control   mean rank pct: {g['mean_rank_pct_ctrl']:.4f}  (theoretical: 0.5000)")
    print(f"  top-10% observed: {g['pct_top10']*100:.1f}%  |  control: {g['pct_top10_ctrl']*100:.1f}%")

    base = os.path.splitext(os.path.basename(json_path))[0]
    plot_rank_cdf(records,
                  os.path.join(output_dir, f"{base}_cdf.png"))
    plot_per_layer_mean(per_layer,
                        os.path.join(output_dir, f"{base}_layer_mean.png"))
    plot_per_layer_violin(records, per_layer,
                          os.path.join(output_dir, f"{base}_layer_violin.png"))
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path",  default="result/alignment_exp.json")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    plot_alignment(args.json_path, args.output_dir)
