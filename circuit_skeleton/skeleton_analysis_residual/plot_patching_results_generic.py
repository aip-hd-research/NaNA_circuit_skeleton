"""
plot_patching_results_generic.py

Architecture-agnostic plots for top-k, middle-k, random-k, and bottom-k paths.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_TOP  = "#2196F3"   # blue
C_MID  = "#8E44AD"   # purple
C_RAND = "#9E9E9E"   # grey
C_BOT  = "#FF7043"   # orange-red
DPI    = 150


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _load_results(output_base: str):
    """Glob neuron_*/patching_results.json rather than assuming a dense
    0..n_neurons-1 range — with neuron_selection="random" the neuron_{j}
    folder names are scattered indices, not contiguous.
    """
    summaries = []
    for p in sorted(glob.glob(os.path.join(output_base, "neuron_*", "patching_results.json"))):
        with open(p) as f:
            summaries.append(json.load(f))
    return summaries


def plot_pooled_violin(summaries, plot_dir: str):
    """Violin + strip: distribution of |Δa_k| across all paths and neurons."""
    top_all  = np.concatenate([s["top_abs"]  for s in summaries])
    mid_all  = np.concatenate([s["middle_abs"] for s in summaries])
    rand_all = np.concatenate([s["rand_abs"] for s in summaries])
    bot_all  = np.concatenate([s["bot_abs"]  for s in summaries])

    floor = 1e-10
    data   = [np.log10(np.maximum(top_all,  floor)),
              np.log10(np.maximum(mid_all,  floor)),
              np.log10(np.maximum(rand_all, floor)),
              np.log10(np.maximum(bot_all,  floor))]
    labels = ["Top-k\n(high |w|)", "Middle-k", "Random-k", "Bot-k\n(low |w|)"]
    colors = [C_TOP, C_MID, C_RAND, C_BOT]

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = [1, 2, 3, 4]
    vp = ax.violinplot(data, positions=positions, showmedians=True, showextrema=True)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.5)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[part].set_color("black")
        vp[part].set_linewidth(1.2)

    rng = np.random.default_rng(0)
    for arr, xi, color in zip(data, positions, colors):
        jitter = rng.uniform(-0.12, 0.12, len(arr))
        ax.scatter(np.full(len(arr), xi) + jitter, arr,
                   s=8, alpha=0.35, color=color, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("log₁₀ |Δa_k|")
    ax.set_title("Pooled distribution of causal effect |Δa_k|")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(plot_dir, "pooled_violin.png"))


def plot_cdf(summaries, plot_dir: str):
    """CDF of |Δa_k| for all four path groups on a log x-axis."""
    top_all  = np.sort(np.concatenate([s["top_abs"]  for s in summaries]))
    mid_all  = np.sort(np.concatenate([s["middle_abs"] for s in summaries]))
    rand_all = np.sort(np.concatenate([s["rand_abs"] for s in summaries]))
    bot_all  = np.sort(np.concatenate([s["bot_abs"]  for s in summaries]))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for arr, label, color in [
        (top_all,  "Top-k",    C_TOP),
        (mid_all,  "Middle-k", C_MID),
        (rand_all, "Random-k", C_RAND),
        (bot_all,  "Bot-k",    C_BOT),
    ]:
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, cdf, color=color, label=label, lw=2)

    ax.set_xscale("log")
    ax.set_xlabel("|Δa_k|  (log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative distribution of causal effect")
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _savefig(fig, os.path.join(plot_dir, "cdf.png"))


def plot_all(output_base: str):
    plot_dir  = os.path.join(output_base, "plots")
    summaries = _load_results(output_base)
    summaries = [s for s in summaries if "middle_abs" in s]
    if not summaries:
        print("[ERROR] No result files found — nothing to plot.")
        return
    print(f"Loaded {len(summaries)} neurons. Generating plots → {plot_dir}/")
    plot_pooled_violin(summaries, plot_dir)
    plot_cdf          (summaries, plot_dir)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot causal patching results (generic)")
    p.add_argument("--output_base", required=True)
    args = p.parse_args()
    plot_all(args.output_base)
