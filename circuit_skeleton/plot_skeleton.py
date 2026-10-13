"""
plot_skeleton.py

Horizontal bipartite visualisation of the MLP circuit skeleton.

Layout
------
- Each (layer, node-type) group occupies one horizontal ROW.
- Nodes within a row are circles spread left-to-right; index shown below.
- Row label (type + layer) is printed on the left margin.
- Rows are stacked top-to-bottom; edges connect rows vertically.

plot_intra_layer  : 2 rows  —  Neuron L{n}  (top)  |  Subspace L{n}  (bottom)
plot_inter_layer  : N+1 rows — Subspace L{src} (top) | Neuron L{t1} | Neuron L{t2} …
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

# style 
ROW_GAP    = 1.5    # vertical distance between rows
NODE_GAP   = 0.9    # horizontal distance between nodes in one row
CIRCLE_R   = 6      # marker size (pt)
LW_MIN, LW_MAX       = 0.5, 3.0
ALPHA_MIN, ALPHA_MAX = 0.20, 0.85
COLOR_POS  = "#d62728"
COLOR_NEG  = "#1f77b4"
COLOR_NODE = "#333333"
LABEL_X    = -0.8   # x for row headers (right-aligned)


# helpers  

def _xs(n: int) -> np.ndarray:
    """Horizontally centred x-coordinates for n nodes."""
    return (np.arange(n) - (n - 1) / 2.0) * NODE_GAP


def _scale(values, lo, hi):
    arr = np.abs(np.array(values, dtype=float))
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-12:
        return np.full(len(arr), (lo + hi) / 2.0)
    return lo + (hi - lo) * (arr - vmin) / (vmax - vmin)


def _draw_row(ax, y, indices, header) -> dict:
    """
    Draw one horizontal row.

    Returns
    -------
    pos : dict  {node_index -> x}
    """
    xs  = _xs(len(indices))
    pos = {}
    for idx, x in zip(indices, xs):
        ax.plot(x, y, "o", markersize=CIRCLE_R, color=COLOR_NODE,
                zorder=3, markeredgewidth=0)
        ax.text(x, y - 0.22, str(idx),
                fontsize=10, ha="center", va="top", color="#222222", zorder=4)
        pos[idx] = x

    # Row label on the left
    ax.text(LABEL_X + min(xs) - 0.1, y, header,
            ha="right", va="center", fontsize=11,
            fontweight="bold", color="#111111")
    return pos


def _draw_edges_between_rows(ax, y0, y1, pos0, pos1, edges):
    """
    Draw edges between two rows.

    edges : list of (src_idx, dst_idx, abs_strength, sign)
    """
    if not edges:
        return
    lws    = _scale([e[2] for e in edges], LW_MIN, LW_MAX)
    alphas = _scale([e[2] for e in edges], ALPHA_MIN, ALPHA_MAX)
    for (si, di, _, sign), lw, al in zip(edges, lws, alphas):
        color = COLOR_POS if sign == "+" else COLOR_NEG
        ax.plot([pos0[si], pos1[di]], [y0, y1],
                color=color, lw=lw, alpha=al, zorder=2, solid_capstyle="round")


def _finalise(ax, fig, output_path, title, x_min, x_max, y_min, y_max):
    pad_x = NODE_GAP
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - 0.8, y_max + 0.8)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=10)
    handles = [
        Line2D([], [], color=COLOR_POS, lw=1.8, label="Positive coupling"),
        Line2D([], [], color=COLOR_NEG, lw=1.8, label="Negative coupling"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=8, frameon=False)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {output_path}")


# plot_intra_layer

def plot_intra_layer(
    G: nx.DiGraph,
    skeleton_result: dict,
    output_path: str = "intra_layer.png",
    title: str = None,
):
    """
    Row 0 (top) : Neuron   L{start_layer}
    Row 1 (bot) : Subspace L{start_layer}
    """
    start_layer = skeleton_result["config"]["start_layer"]

    within_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d["edge_type"] == "within_layer"
    ]
    if not within_edges:
        print("[plot_intra_layer] No within-layer edges; skipping.")
        return

    neu_indices = sorted({G.nodes[u]["idx"] for u, _, _ in within_edges})
    sub_indices = sorted({G.nodes[v]["idx"] for _, v, _ in within_edges})

    max_n  = max(len(neu_indices), len(sub_indices))
    fig_w  = max(5.0, max_n * NODE_GAP + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))

    y_neu, y_sub = 0.0, -ROW_GAP
    pos_neu = _draw_row(ax, y_neu, neu_indices, f"Neuron / L{start_layer}")
    pos_sub = _draw_row(ax, y_sub, sub_indices, f"Subspace / L{start_layer}")

    edge_tuples = [
        (G.nodes[u]["idx"], G.nodes[v]["idx"], d["abs_strength"], d["sign"])
        for u, v, d in within_edges
    ]
    _draw_edges_between_rows(ax, y_neu, y_sub, pos_neu, pos_sub, edge_tuples)

    all_xs = list(pos_neu.values()) + list(pos_sub.values())
    _finalise(ax, fig, output_path,
              title=title or f"Intra-layer skeleton  (L{start_layer})",
              x_min=min(all_xs), x_max=max(all_xs),
              y_min=y_sub, y_max=y_neu)


# plot_complete_skeleton  

def plot_complete_skeleton(
    G: nx.DiGraph,
    skeleton_result: dict,
    output_path: str = "complete_skeleton.png",
    title: str = None,
):
    """
    Combined figure for complete-skeleton mode:
      Row 0 (top) : Neuron   L{start_layer}   — input neurons
      Row 1       : Subspace L{start_layer}   — intermediate subspaces
      Row 2, 3, … : Neuron   L{t1}, L{t2}, … — target neurons per layer
    """
    start_layer   = skeleton_result["config"]["start_layer"]
    target_layers = sorted(skeleton_result["config"]["target_layers"])

    within_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                    if d["edge_type"] == "within_layer"]
    cross_edges  = [(u, v, d) for u, v, d in G.edges(data=True)
                    if d["edge_type"] == "cross_layer"]

    if not within_edges and not cross_edges:
        print("[plot_complete_skeleton] No edges; skipping.")
        return

    # Collect node indices per row
    neu_indices = sorted({G.nodes[u]["idx"] for u, _, _ in within_edges})
    sub_indices = sorted(
        {G.nodes[v]["idx"] for _, v, _ in within_edges} |
        {G.nodes[u]["idx"] for u, _, _ in cross_edges}
    )
    tgt_indices = {
        tgt: sorted({G.nodes[v]["idx"]
                     for _, v, d in cross_edges if d["target_layer"] == tgt})
        for tgt in target_layers
    }
    tgt_indices   = {tgt: idxs for tgt, idxs in tgt_indices.items() if idxs}
    target_layers = sorted(tgt_indices.keys())

    n_rows = 2 + len(target_layers)
    max_n  = max(len(neu_indices), len(sub_indices),
                 max((len(v) for v in tgt_indices.values()), default=1))
    fig_w  = max(6.0, max_n * NODE_GAP + 2.5)
    fig_h  = max(6.0, n_rows * ROW_GAP + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y_neu = 0.0
    y_sub = -ROW_GAP
    pos_neu = _draw_row(ax, y_neu, neu_indices, f"Neuron / L{start_layer}")
    pos_sub = _draw_row(ax, y_sub, sub_indices, f"Subspace / L{start_layer}")

    pos_tgt = {}
    y_rows  = {}
    for row_i, tgt in enumerate(target_layers):
        y = -(row_i + 2) * ROW_GAP
        pos_tgt[tgt] = _draw_row(ax, y, tgt_indices[tgt], f"Neuron / L{tgt}")
        y_rows[tgt]  = y

    # Within-layer edges: neuron row -> subspace row
    within_tuples = [
        (G.nodes[u]["idx"], G.nodes[v]["idx"], d["abs_strength"], d["sign"])
        for u, v, d in within_edges
    ]
    _draw_edges_between_rows(ax, y_neu, y_sub, pos_neu, pos_sub, within_tuples)

    # Cross-layer edges: subspace row -> each target-neuron row
    if cross_edges:
        lws    = _scale([d["abs_strength"] for _, _, d in cross_edges], LW_MIN, LW_MAX)
        alphas = _scale([d["abs_strength"] for _, _, d in cross_edges], ALPHA_MIN, ALPHA_MAX)
        for (u, v, d), lw, al in zip(cross_edges, lws, alphas):
            tgt    = d["target_layer"]
            si_sub = G.nodes[u]["idx"]
            ni_tgt = G.nodes[v]["idx"]
            if tgt not in pos_tgt or si_sub not in pos_sub or ni_tgt not in pos_tgt[tgt]:
                continue
            color = COLOR_POS if d["sign"] == "+" else COLOR_NEG
            ax.plot([pos_sub[si_sub], pos_tgt[tgt][ni_tgt]],
                    [y_sub, y_rows[tgt]],
                    color=color, lw=lw, alpha=al, zorder=2, solid_capstyle="round")

    all_xs = (list(pos_neu.values()) + list(pos_sub.values()) +
              [x for pm in pos_tgt.values() for x in pm.values()])
    y_bot  = -(1 + len(target_layers)) * ROW_GAP
    _finalise(ax, fig, output_path,
              title=title or f"Complete skeleton  (L{start_layer} → {target_layers})",
              x_min=min(all_xs), x_max=max(all_xs),
              y_min=y_bot, y_max=y_neu)


# plot_inter_layer

def plot_inter_layer(
    G: nx.DiGraph,
    skeleton_result: dict,
    output_path: str = "inter_layer.png",
    title: str = None,
):
    """
    Row 0       : Subspace L{start_layer}
    Row 1, 2, … : Neuron   L{tgt_1}, L{tgt_2}, …
    """
    start_layer   = skeleton_result["config"]["start_layer"]
    target_layers = sorted(skeleton_result["config"]["target_layers"])

    cross_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d["edge_type"] == "cross_layer"
    ]
    if not cross_edges:
        print("[plot_inter_layer] No cross-layer edges; skipping.")
        return

    src_indices = sorted({G.nodes[u]["idx"] for u, _, _ in cross_edges})
    tgt_indices = {
        tgt: sorted({
            G.nodes[v]["idx"]
            for _, v, d in cross_edges if d["target_layer"] == tgt
        })
        for tgt in target_layers
    }
    # drop target layers that have no neurons in the graph
    tgt_indices     = {tgt: idxs for tgt, idxs in tgt_indices.items() if idxs}
    target_layers   = sorted(tgt_indices.keys())
    if not target_layers:
        print("[plot_inter_layer] No cross-layer target neurons; skipping.")
        return

    n_rows = 1 + len(target_layers)
    max_n  = max(len(src_indices),
                 max((len(v) for v in tgt_indices.values()), default=1))
    fig_w  = max(5.0, max_n * NODE_GAP + 2.5)
    fig_h  = max(4.0, n_rows * ROW_GAP + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw source row
    y_src = 0.0
    pos_src = _draw_row(ax, y_src, src_indices, f"Subspace / L{start_layer}")

    # Draw target rows
    pos_tgt = {}
    y_rows  = {start_layer: y_src}
    for row_i, tgt in enumerate(target_layers):
        y = -(row_i + 1) * ROW_GAP
        pos_tgt[tgt] = _draw_row(ax, y, tgt_indices[tgt], f"Neuron / L{tgt}")
        y_rows[tgt] = y

    # Draw edges (source row -> each target row)
    lws    = _scale([d["abs_strength"] for _, _, d in cross_edges], LW_MIN, LW_MAX)
    alphas = _scale([d["abs_strength"] for _, _, d in cross_edges], ALPHA_MIN, ALPHA_MAX)
    for (u, v, d), lw, al in zip(cross_edges, lws, alphas):
        tgt   = d["target_layer"]
        xi    = pos_src[G.nodes[u]["idx"]]
        xj    = pos_tgt[tgt][G.nodes[v]["idx"]]
        color = COLOR_POS if d["sign"] == "+" else COLOR_NEG
        ax.plot([xi, xj], [y_src, y_rows[tgt]],
                color=color, lw=lw, alpha=al, zorder=2, solid_capstyle="round")

    all_xs = (list(pos_src.values()) +
              [x for pm in pos_tgt.values() for x in pm.values()])
    y_bot  = -(len(target_layers)) * ROW_GAP
    _finalise(ax, fig, output_path,
              title=title or f"Inter-layer skeleton  (L{start_layer} -> {target_layers})",
              x_min=min(all_xs), x_max=max(all_xs),
              y_min=y_bot, y_max=y_src)
