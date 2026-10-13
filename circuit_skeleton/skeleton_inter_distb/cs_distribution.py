"""
cs_distribution.py

Given a source neuron j, compute ALL coupling strengths in the skeleton
and plot their sorted absolute-value distribution (descending).

Two modes:
  complete_skeleton_global   : strength_{j,i,k} = delta[j,i] * gamma[i,k]
                               N = n_sp * total_cross  (e.g. 1024 * 16384 = 16.8M)
  complete_skeleton_fullinfo : strength_{j,k}   = Σ_i delta[j,i] * gamma[i,k]
                               N = total_cross          (e.g. 16384)
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from circuit_skeleton.coupl_strength import build_coupling_strength


# ─────────────────────────────────────────────────────────────────────────────
#  Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_sorted_strengths(
    model_name: str,
    start_layer: int,
    target_layers: list,
    neuron_idx: int,
    skeleton_mode: str = "complete_skeleton_global",
    device: str = "cpu",
) -> tuple[np.ndarray, dict]:
    """
    Compute all coupling strengths for source neuron j, sorted descending by |value|.

    Returns
    -------
    sorted_abs : np.ndarray  [N]   — absolute strengths, descending
    meta       : dict              — n_sp, total_cross, raw max/mean/median
    """
    cs = build_coupling_strength(model_name, start_layer, target_layers, device)

    if skeleton_mode == "complete_skeleton_fullinfo":
        raw = cs.w_contracted([neuron_idx]).squeeze(0).cpu()   # [total_cross]
    else:  # complete_skeleton_global
        raw = cs.w_full_path([neuron_idx]).reshape(-1).cpu()   # [n_sp * total_cross]

    abs_vals = raw.abs().detach().numpy()
    sorted_abs = np.sort(abs_vals)[::-1].copy()   # descending

    meta = {
        "n_sp":        cs.within_ske.n_subspaces,
        "total_cross": cs.total_cross,
        "n_paths":     int(len(sorted_abs)),
        "max":         float(sorted_abs[0]),
        "mean":        float(sorted_abs.mean()),
        "median":      float(np.median(sorted_abs)),
    }
    return sorted_abs, meta


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(sorted_abs: np.ndarray, top_pcts=(0.001, 0.01, 0.05, 0.10)) -> dict:
    """
    Compute cumulative-strength statistics at several top-% thresholds.

    Returns dict keyed by percentage string, e.g. "1.0%".
    """
    N     = len(sorted_abs)
    total = sorted_abs.sum()
    stats = {}
    for pct in top_pcts:
        cutoff  = max(1, int(N * pct))
        cum_pct = float(sorted_abs[:cutoff].sum() / total * 100)
        stats[f"{pct*100:.1f}%"] = {
            "n_paths":  cutoff,
            "cum_pct_strength": cum_pct,
        }
    return stats


def print_stats(sorted_abs: np.ndarray, meta: dict, neuron_idx: int,
                start_layer: int, target_layers: list, skeleton_mode: str) -> None:
    N = meta["n_paths"]
    print(f"\n{'='*55}")
    print(f"  Neuron {neuron_idx}  @  Layer {start_layer}  →  Layers {target_layers}")
    print(f"  Mode : {skeleton_mode}")
    print(f"{'='*55}")
    print(f"  Total paths N     : {N:>12,}")
    print(f"  n_subspaces       : {meta['n_sp']:>12,}")
    print(f"  total_cross       : {meta['total_cross']:>12,}")
    print(f"  Max  |strength|   : {meta['max']:>12.6f}")
    print(f"  Mean |strength|   : {meta['mean']:>12.6f}")
    print(f"  Median|strength|  : {meta['median']:>12.6f}")
    print()
    total = sorted_abs.sum()
    print(f"  {'Top %':>8}  {'# paths':>10}  {'Cum. strength %':>16}")
    print(f"  {'-'*37}")
    for pct in [0.001, 0.01, 0.05, 0.10, 0.20]:
        cutoff  = max(1, int(N * pct))
        cum_pct = sorted_abs[:cutoff].sum() / total * 100
        print(f"  {pct*100:>7.1f}%  {cutoff:>10,}  {cum_pct:>15.1f}%")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _subsample(arr: np.ndarray, max_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x_indices, y_values) subsampled to at most max_pts points.
    Always keeps the first and last point; samples the rest uniformly.
    """
    N = len(arr)
    if N <= max_pts:
        return np.arange(1, N + 1, dtype=np.float64), arr

    idx = np.unique(
        np.concatenate([
            [0],
            np.linspace(0, N - 1, max_pts - 2, dtype=int),
            [N - 1],
        ])
    )
    return (idx + 1).astype(np.float64), arr[idx]


def plot_cs_distribution(
    sorted_abs: np.ndarray,
    meta: dict,
    neuron_idx: int,
    start_layer: int,
    target_layers: list,
    skeleton_mode: str,
    output_dir: str,
    log_scale_y: bool = False,
    max_plot_points: int = 50_000,
    top_pct_lines: tuple = (0.01, 0.05, 0.10),
    zoom_xlim: int = 4_000_000,
) -> str:
    """
    Draw two panels:
      Top — full sorted |strength| curve (all N paths)
      Bottom — zoomed view of the first `zoom_xlim` paths

    Returns the saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    N = meta["n_paths"]

    x_sub, y_sub = _subsample(sorted_abs, max_plot_points)

    # zoomed panel: subsample only within [0, zoom_xlim]
    zoom_data  = sorted_abs[:zoom_xlim]
    x_zoom, y_zoom = _subsample(zoom_data, max_plot_points)

    # ── color palette ──────────────────────────────────────────────────────
    COLORS   = ["#ef4444", "#7c3aed", "#16a34a"]   # red / purple / green
    LINE_COL = "#2563eb"

    fig, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"hspace": 0.45},
    )
    fig.suptitle(
        f"Coupling Strength Distribution  —  Neuron {neuron_idx} @ Layer {start_layer} "
        f"→ Layers {target_layers}\nMode: {skeleton_mode}  |  N = {N:,}",
        fontsize=10,
    )

    def _draw_ax(ax, x, y, xlim, title_suffix):
        ax.plot(x, y, linewidth=0.8, color=LINE_COL)
        for pct, color in zip(top_pct_lines, COLORS):
            cutoff    = max(1, int(N * pct))
            threshold = sorted_abs[cutoff - 1]
            if cutoff <= xlim:
                ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.0,
                           label=f"top {pct*100:.0f}%  ({cutoff:,} paths, ≥{threshold:.4f})")
                ax.axhline(threshold, color=color, linestyle=":", linewidth=0.7, alpha=0.5)
        if log_scale_y:
            ax.set_yscale("log")
        ax.set_xlabel("Rank  (sorted by |strength|, descending)", fontsize=9)
        ax.set_ylabel("|Coupling Strength|", fontsize=9)
        ax.set_title(title_suffix, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlim(0, xlim)

    _draw_ax(ax_full, x_sub, y_sub, N,
             f"Full view  (0 – {N:,})")
    _draw_ax(ax_zoom, x_zoom, y_zoom, zoom_xlim,
             f"Zoomed view  (0 – {zoom_xlim:,})")

    fname = (f"cs_distrib_L{start_layer}_N{neuron_idx}"
             f"_{'global' if 'global' in skeleton_mode else 'fullinfo'}.png")
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {fpath}")
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level entry
# ─────────────────────────────────────────────────────────────────────────────

def run_cs_distribution(
    model_name: str,
    start_layer: int,
    target_layers: list,
    neuron_idx: int,
    skeleton_mode: str = "complete_skeleton_global",
    output_dir: str = "result/cs_distribution",
    device: str = "cpu",
    log_scale_y: bool = False,
    max_plot_points: int = 50_000,
    top_pct_lines: tuple = (0.01, 0.05, 0.10),
    zoom_xlim: int = 4_000_000,
    save_json: bool = True,
) -> dict:
    """
    Full pipeline: compute → print stats → plot → (optionally) save JSON.

    Returns the summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compute
    print(f"[CS-Distribution] Loading model and computing coupling strengths …")
    sorted_abs, meta = compute_sorted_strengths(
        model_name, start_layer, target_layers, neuron_idx, skeleton_mode, device
    )

    # 2. Stats
    print_stats(sorted_abs, meta, neuron_idx, start_layer, target_layers, skeleton_mode)
    stats = compute_stats(sorted_abs)

    # 3. Plot
    plot_path = plot_cs_distribution(
        sorted_abs, meta,
        neuron_idx, start_layer, target_layers, skeleton_mode,
        output_dir, log_scale_y, max_plot_points, top_pct_lines,
        zoom_xlim=zoom_xlim,
    )

    # 4. Save JSON summary (without the full array — too large)
    summary = {
        "config": {
            "model_name":    model_name,
            "start_layer":   start_layer,
            "target_layers": target_layers,
            "neuron_idx":    neuron_idx,
            "skeleton_mode": skeleton_mode,
        },
        "meta":  meta,
        "stats": stats,
        "plot":  plot_path,
    }
    if save_json:
        jpath = os.path.join(
            output_dir,
            f"cs_distrib_L{start_layer}_N{neuron_idx}.json"
        )
        with open(jpath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[JSON] Saved → {jpath}")

    return summary
