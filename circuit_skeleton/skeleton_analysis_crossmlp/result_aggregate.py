import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Plotting helpers
def _savefig(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_bar(top_abs, rand_abs, bot_abs, j, start_layer, interv, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    data   = [top_abs, rand_abs, bot_abs]
    labels = ["Top-k\n(high |w|)", "Random-k\n(|w|)", "Bot-k\n(low |w|)"]
    colors = ["#2196F3", "#BDBDBD", "#FF7043"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(0)
    for arr, xi, color in zip(data, [1, 2, 3], colors):
        jitter = rng.uniform(-0.15, 0.15, len(arr))
        ax.scatter(np.full(len(arr), xi) + jitter, arr,
                   s=6, alpha=0.5, color=color, zorder=3)

    ax.set_ylabel(r"mean $|\Delta a_k|$  (global (i,k) pairs)", fontsize=10)
    ax.set_title(
        f"Causal Effect  j={j} @ L{start_layer}  [{interv}]\n"
        f"Means: Top={top_abs.mean():.4f} | Rand={rand_abs.mean():.4f} | Bot={bot_abs.mean():.4f}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _savefig(fig, path)


# Aggregate across all source neurons
def aggregate_and_print(summaries: list, output_dir: str, intervention: str):
    """Compute win-rates and signal multipliers across all per-neuron summaries.

    Parameters
    ----------
    summaries   : list of dicts returned by analyse_one_neuron
    output_dir  : directory to write aggregate_patching.json
    intervention: label string (zero / mean / scale)
    """
    if not summaries:
        return

    n          = len(summaries)
    top_means  = np.array([s["stats"]["top_mean"]            for s in summaries])
    rand_means = np.array([s["stats"]["rand_mean"]           for s in summaries])
    bot_means  = np.array([s["stats"].get("bot_mean", 0.0)   for s in summaries])

    pct_top_gt_rand = np.mean([s["stats"]["top_gt_rand"]           for s in summaries]) * 100
    pct_top_gt_bot  = np.mean([s["stats"].get("top_gt_bot", False) for s in summaries]) * 100
    pct_rand_gt_bot = float(np.mean(rand_means > bot_means)) * 100

    ratio_t_r = top_means.sum()  / rand_means.sum() if rand_means.sum() > 0 else 0.0
    ratio_t_b = top_means.sum()  / bot_means.sum()  if bot_means.sum()  > 0 else 0.0
    ratio_r_b = rand_means.sum() / bot_means.sum()  if bot_means.sum()  > 0 else 0.0

    # per-target pairwise win counts
    tr_wins = tr_total = tb_wins = tb_total = rb_wins = rb_total = 0
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

        n_rb = min(len(r), len(b))
        rb_wins  += int((r[:n_rb] > b[:n_rb]).sum());  rb_total += n_rb

    pct_target_tr = 100.0 * tr_wins / tr_total if tr_total > 0 else 0.0
    pct_target_tb = 100.0 * tb_wins / tb_total if tb_total > 0 else 0.0
    pct_target_rb = 100.0 * rb_wins / rb_total if rb_total > 0 else 0.0

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Causal Patching Aggregate  ({n} neurons, intervention={intervention})")
    print(sep)
    print(f"  {'Metric':<27}  {'Mean |Δa|':>10}  {'Std':>10}")
    print("─" * 72)
    print(f"  {'Top-k (High |w|)':<27}  {top_means.mean():>10.5f}  {top_means.std():>10.5f}")
    print(f"  {'Random-k (|w|)':<27}  {rand_means.mean():>10.5f}  {rand_means.std():>10.5f}")
    print(f"  {'Bot-k (Low |w|)':<27}  {bot_means.mean():>10.5f}  {bot_means.std():>10.5f}")
    print("─" * 72)
    print(f"  Win Rate  per-neuron  Top>Rand : {pct_top_gt_rand:5.1f}%  ({int(round(pct_top_gt_rand*n/100))}/{n} neurons)")
    print(f"  Win Rate  per-neuron  Top>Bot  : {pct_top_gt_bot:5.1f}%  ({int(round(pct_top_gt_bot*n/100))}/{n} neurons)")
    print(f"  Win Rate  per-neuron  Rand>Bot : {pct_rand_gt_bot:5.1f}%  ({int(round(pct_rand_gt_bot*n/100))}/{n} neurons)")
    print(f"  Win Rate  per-target  Top>Rand : {pct_target_tr:5.1f}%  ({tr_wins}/{tr_total} targets)")
    print(f"  Win Rate  per-target  Top>Bot  : {pct_target_tb:5.1f}%  ({tb_wins}/{tb_total} targets)")
    print(f"  Win Rate  per-target  Rand>Bot : {pct_target_rb:5.1f}%  ({rb_wins}/{rb_total} targets)")
    print(f"  Signal Multiplier  Top/Rand    : {ratio_t_r:.2f}×")
    print(f"  Signal Multiplier  Top/Bot     : {ratio_t_b:.2f}×")
    print(f"  Signal Multiplier  Rand/Bot    : {ratio_r_b:.2f}×")
    print(sep)

    agg = {
        "config": {"n_neurons": n, "intervention": intervention},
        "averages": {
            "top_mean":  float(top_means.mean()),
            "rand_mean": float(rand_means.mean()),
            "bot_mean":  float(bot_means.mean()),
        },
        "win_rates": {
            "top_gt_rand_per_neuron_pct":  float(pct_top_gt_rand),
            "top_gt_bot_per_neuron_pct":   float(pct_top_gt_bot),
            "rand_gt_bot_per_neuron_pct":  float(pct_rand_gt_bot),
            "top_gt_rand_per_target_pct":  float(pct_target_tr),
            "top_gt_bot_per_target_pct":   float(pct_target_tb),
            "rand_gt_bot_per_target_pct":  float(pct_target_rb),
            "per_target_tr_wins":          int(tr_wins),
            "per_target_tr_total":         int(tr_total),
            "per_target_tb_wins":          int(tb_wins),
            "per_target_tb_total":         int(tb_total),
            "per_target_rb_wins":          int(rb_wins),
            "per_target_rb_total":         int(rb_total),
        },
        "multipliers": {
            "top_over_rand": float(ratio_t_r),
            "top_over_bot":  float(ratio_t_b),
            "rand_over_bot": float(ratio_r_b),
        },
    }
    with open(os.path.join(output_dir, "aggregate_patching.json"), "w") as f:
        json.dump(agg, f, indent=2)


def _load_and_aggregate(args):
    """Reload per-neuron JSON results and aggregate (used in resume mode)."""
    summaries = []
    for j in range(args.n_neurons):
        p = os.path.join(args.output_base, f"neuron_{j}", "patching_results.json")
        if os.path.isfile(p):
            with open(p) as f:
                summaries.append(json.load(f))
    aggregate_and_print(summaries, args.output_base, args.intervention)
