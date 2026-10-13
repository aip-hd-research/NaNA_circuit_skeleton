"""
result_aggregate_generic.py

Architecture-agnostic counterpart to result_aggregate.py. Aggregates the
top-k, middle-k, random-k, and bottom-k path-selection groups.
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


C_TOP  = "#2196F3"   # blue
C_MID  = "#8E44AD"   # purple
C_RAND = "#BDBDBD"   # grey
C_BOT  = "#FF7043"   # orange-red


def _savefig(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_bar(top_abs, middle_abs, rand_abs, bot_abs, j, start_layer, interv, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    data   = [top_abs, middle_abs, rand_abs, bot_abs]
    labels = ["Top-k\n(high |w|)", "Middle-k", "Random-k\n(|w|)", "Bot-k\n(low |w|)"]
    colors = [C_TOP, C_MID, C_RAND, C_BOT]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(0)
    for arr, xi, color in zip(data, [1, 2, 3, 4], colors):
        jitter = rng.uniform(-0.15, 0.15, len(arr))
        ax.scatter(np.full(len(arr), xi) + jitter, arr,
                   s=6, alpha=0.5, color=color, zorder=3)

    ax.set_ylabel(r"mean $|\Delta a_k|$  (global (i,k) pairs)", fontsize=10)
    ax.set_title(
        f"Causal Effect  j={j} @ L{start_layer}  [{interv}]\n"
        f"Means: Top={top_abs.mean():.4f} | Middle={middle_abs.mean():.4f} | "
        f"Rand={rand_abs.mean():.4f} | "
        f"Bot={bot_abs.mean():.4f}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _savefig(fig, path)


def aggregate_and_print(summaries: list, output_dir: str, intervention: str):
    """Compute win-rates and signal multipliers across all per-neuron summaries.

    Parameters
    ----------
    summaries   : list of dicts returned by analyse_one_neuron (generic)
    output_dir  : directory to write aggregate_patching.json
    intervention: label string (zero / mean / scale)
    """
    if not summaries:
        return

    n          = len(summaries)
    top_means  = np.array([s["stats"]["top_mean"]            for s in summaries])
    mid_means  = np.array([s["stats"]["middle_mean"]         for s in summaries])
    rand_means = np.array([s["stats"]["rand_mean"]           for s in summaries])
    bot_means  = np.array([s["stats"].get("bot_mean", 0.0)   for s in summaries])

    pct_top_gt_rand = np.mean([s["stats"]["top_gt_rand"]           for s in summaries]) * 100
    pct_top_gt_mid  = np.mean([s["stats"]["top_gt_middle"]         for s in summaries]) * 100
    pct_top_gt_bot  = np.mean([s["stats"].get("top_gt_bot", False) for s in summaries]) * 100
    pct_rand_gt_bot = float(np.mean(rand_means > bot_means)) * 100

    ratio_t_r = top_means.sum()  / rand_means.sum() if rand_means.sum() > 0 else 0.0
    ratio_t_m = top_means.sum()  / mid_means.sum()  if mid_means.sum()  > 0 else 0.0
    ratio_t_b = top_means.sum()  / bot_means.sum()  if bot_means.sum()  > 0 else 0.0

    # per-target pairwise win counts
    tr_wins = tr_total = tb_wins = tb_total = 0
    for s in summaries:
        if "top_abs" not in s:
            continue
        t = np.array(s["top_abs"])
        r = np.array(s.get("rand_abs", []))
        b = np.array(s.get("bot_abs",  []))

        n_tr = min(len(t), len(r))
        tr_wins  += int((t[:n_tr] > r[:n_tr]).sum());  tr_total += n_tr

        n_tb = min(len(t), len(b))
        tb_wins  += int((t[:n_tb] > b[:n_tb]).sum());  tb_total += n_tb

    pct_target_tr = 100.0 * tr_wins / tr_total if tr_total > 0 else 0.0
    pct_target_tb = 100.0 * tb_wins / tb_total if tb_total > 0 else 0.0

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Causal Patching Aggregate  ({n} neurons, intervention={intervention})")
    print(sep)
    print(f"  {'Metric':<27}  {'Mean |Δa|':>10}  {'Std':>10}")
    print("─" * 72)
    print(f"  {'Top-k (High |w|)':<27}  {top_means.mean():>10.5f}  {top_means.std():>10.5f}")
    print(f"  {'Middle-k (Median |w|)':<27}  {mid_means.mean():>10.5f}  {mid_means.std():>10.5f}")
    print(f"  {'Random-k (|w|)':<27}  {rand_means.mean():>10.5f}  {rand_means.std():>10.5f}")
    print(f"  {'Bot-k (Low |w|)':<27}  {bot_means.mean():>10.5f}  {bot_means.std():>10.5f}")
    print("─" * 72)
    print(f"  Win Rate  per-neuron  Top>Rand : {pct_top_gt_rand:5.1f}%  ({int(round(pct_top_gt_rand*n/100))}/{n} neurons)")
    print(f"  Win Rate  per-neuron  Top>Mid  : {pct_top_gt_mid:5.1f}%  ({int(round(pct_top_gt_mid*n/100))}/{n} neurons)")
    print(f"  Win Rate  per-neuron  Top>Bot  : {pct_top_gt_bot:5.1f}%  ({int(round(pct_top_gt_bot*n/100))}/{n} neurons)")
    print(f"  Win Rate  per-neuron  Rand>Bot : {pct_rand_gt_bot:5.1f}%  ({int(round(pct_rand_gt_bot*n/100))}/{n} neurons)")
    print(f"  Win Rate  per-target  Top>Rand : {pct_target_tr:5.1f}%  ({tr_wins}/{tr_total} targets)")
    print(f"  Win Rate  per-target  Top>Bot  : {pct_target_tb:5.1f}%  ({tb_wins}/{tb_total} targets)")
    print(f"  Signal Multiplier  Top/Rand    : {ratio_t_r:.2f}×")
    print(f"  Signal Multiplier  Top/Middle  : {ratio_t_m:.2f}×")
    print(f"  Signal Multiplier  Top/Bot     : {ratio_t_b:.2f}×")
    print(sep)

    experiment_hashes = {
        s.get("config", {}).get("experiment_hash") for s in summaries
    }
    experiment_hashes.discard(None)
    agg = {
        "config": {
            "n_neurons": n,
            "intervention": intervention,
            "experiment_hash": (
                next(iter(experiment_hashes)) if len(experiment_hashes) == 1 else None
            ),
        },
        "averages": {
            "top_mean":  float(top_means.mean()),
            "middle_mean": float(mid_means.mean()),
            "rand_mean": float(rand_means.mean()),
            "bot_mean":  float(bot_means.mean()),
        },
        "win_rates": {
            "top_gt_rand_per_neuron_pct":  float(pct_top_gt_rand),
            "top_gt_middle_per_neuron_pct": float(pct_top_gt_mid),
            "top_gt_bot_per_neuron_pct":   float(pct_top_gt_bot),
            "rand_gt_bot_per_neuron_pct":  float(pct_rand_gt_bot),
            "top_gt_rand_per_target_pct":  float(pct_target_tr),
            "top_gt_bot_per_target_pct":   float(pct_target_tb),
            "per_target_tr_wins":          int(tr_wins),
            "per_target_tr_total":         int(tr_total),
            "per_target_tb_wins":          int(tb_wins),
            "per_target_tb_total":         int(tb_total),
        },
        "multipliers": {
            "top_over_rand": float(ratio_t_r),
            "top_over_middle": float(ratio_t_m),
            "top_over_bot":  float(ratio_t_b),
        },
    }
    with open(os.path.join(output_dir, "aggregate_patching.json"), "w") as f:
        json.dump(agg, f, indent=2)


def _load_and_aggregate(args):
    """Reload per-neuron JSON results and aggregate (used in resume mode).

    Globs neuron_*/patching_results.json under output_base rather than
    assuming indices 0..n_neurons-1 — with neuron_selection="random" the
    saved neuron_{j} folder names are scattered indices, not a dense range.
    """
    summaries = []
    expected_hash = getattr(args, "experiment_hash", None)
    for p in sorted(glob.glob(os.path.join(args.output_base, "neuron_*", "patching_results.json"))):
        with open(p) as f:
            summary = json.load(f)
        if expected_hash is not None and (
            summary.get("config", {}).get("experiment_hash") != expected_hash
        ):
            continue
        summaries.append(summary)
    aggregate_and_print(summaries, args.output_base, args.intervention)
