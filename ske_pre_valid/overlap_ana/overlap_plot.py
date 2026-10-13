"""
overlap_plot.py — heatmap visualization for skeleton path overlap.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _draw_heatmap(mat: np.ndarray, labels: list[str],
                  title: str, out_path: str,
                  vmin: float = 0, vmax: float | None = None,
                  annot_mat: np.ndarray | None = None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n    = len(labels)
    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(
        mat, annot=annot_mat if annot_mat is not None else True,
        fmt="" if annot_mat is not None else ".2f",
        vmin=vmin, vmax=vmax,
        cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        mask=mask,
        linewidths=0.5, linecolor="white", ax=ax,
    )
    for k in range(n):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True, color="#dddddd", lw=0))
        ax.text(k + 0.5, k + 0.5, "—",
                ha="center", va="center", fontsize=9, color="#888888")

    ax.set_title(title, fontsize=12, pad=12)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_overlap_heatmap(mat: np.ndarray, labels: list[str],
                         title: str, out_path: str):
    _draw_heatmap(mat, labels, title, out_path, vmin=0, vmax=None)


def plot_overlap_heatmap_normalized(mat: np.ndarray, labels: list[str],
                                    title: str, out_path: str):
    """Min-max normalise off-diagonal; annotate with original Jaccard values."""
    n    = len(labels)
    mask = np.eye(n, dtype=bool)

    offdiag = mat[~mask]
    lo, hi  = offdiag.min(), offdiag.max()
    norm    = (mat - lo) / (hi - lo) if hi > lo else np.zeros_like(mat)
    norm[mask] = 0.0

    annot = np.empty_like(mat, dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{mat[i,j]:.2f}" if i != j else "—"

    _draw_heatmap(norm, labels, title, out_path, vmin=0, vmax=1, annot_mat=annot)
