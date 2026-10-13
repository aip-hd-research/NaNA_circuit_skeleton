"""Full-matrix and category-network visualization for skeleton-path overlap."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


MATRIX_CMAP = colors.LinearSegmentedColormap.from_list(
    "overlap_blues", ["#f7fbff", "#9ecae1", "#2171b5", "#08306b"]
)
EDGE_COLORS = ("#6699CC", "#009E73", "#D55E00")
BASE_FONT_SIZE = 7
FIGURE_WIDTH_IN = 180 / 25.4
FIGURE_HEIGHT_IN = 125 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": BASE_FONT_SIZE,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
})


def _display_label(label: str) -> str:
    """Hide tokenizer-significant leading whitespace in display labels."""
    return label.lstrip()


def _off_diagonal_range(matrix: np.ndarray) -> tuple[float, float]:
    mask = ~np.eye(len(matrix), dtype=bool)
    values = matrix[mask]
    return (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)


def _edge_style(value: float, low: float, high: float) -> tuple[str, float]:
    span = high - low
    scaled = (value - low) / span if span > 0 else 0.5
    color = EDGE_COLORS[0 if scaled < 1 / 3 else 1 if scaled < 2 / 3 else 2]
    return color, 0.35 + 1.0 * max(0.0, min(1.0, scaled))


def _overlap_breaks(low: float, high: float) -> tuple[float, float]:
    span = high - low
    return low + span / 3, low + 2 * span / 3


def _category_members(labels: list[str], categories: dict[str, list[str]]) -> dict[str, list[int]]:
    token_to_category = {}
    for category, tokens in categories.items():
        for token in tokens:
            token_to_category.setdefault(token.strip(), category)

    members = {}
    for index, label in enumerate(labels):
        category = token_to_category.get(label.strip(), "uncategorized")
        members.setdefault(category, []).append(index)
    return members


def _resolve_cmap(name: str):
    if name == "overlap_blues":
        return MATRIX_CMAP
    try:
        return matplotlib.colormaps[name]
    except KeyError as error:
        available = ", ".join(sorted(matplotlib.colormaps)[:12])
        raise ValueError(
            f"Unknown matrix colormap {name!r}; examples: {available}"
        ) from error


def _draw_triangle_network(matrix: np.ndarray, labels: list[str], title: str,
                           out_path: str, categories: dict[str, list[str]],
                           normalized: bool = False,
                           cmap: str = "overlap_blues") -> None:
    matrix = np.asarray(matrix, dtype=float)
    n = len(labels)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix shape {matrix.shape} does not match {n} labels")

    raw_low, raw_high = _off_diagonal_range(matrix)
    shown = matrix.copy()
    if normalized:
        shown = ((shown - raw_low) / (raw_high - raw_low)
                 if raw_high > raw_low else np.zeros_like(shown))
        np.fill_diagonal(shown, 1.0)
    display_low, display_high = _off_diagonal_range(shown)
    norm = colors.Normalize(vmin=min(0.0, display_low), vmax=max(1.0, display_high))
    matrix_cmap = _resolve_cmap(cmap)

    fig, axis = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    axis.set_facecolor("white")

    # Full matrix: outlined cells containing value-scaled colored squares.
    for row in range(n):
        for col in range(n):
            value = float(shown[row, col])
            axis.add_patch(Rectangle(
                (col, row), 1, 1, facecolor="#ffffff", edgecolor="#c8cdd1", lw=0.45,
            ))
            side = 0.88 if row == col else 0.12 + 0.76 * np.sqrt(max(0.0, value))
            axis.add_patch(Rectangle(
                (col + (1 - side) / 2, row + (1 - side) / 2), side, side,
                facecolor=matrix_cmap(norm(value)), edgecolor="#000000", lw=0.28,
            ))

    axis.add_patch(Rectangle(
        (0, 0), n, n, facecolor="none", edgecolor="#000000", lw=0.65, zorder=5,
    ))

    # Token links start at the matrix boundary and converge at category nodes.
    category_members = _category_members(labels, categories)
    category_positions = {}
    ordered_categories = sorted(
        category_members, key=lambda category: np.mean(category_members[category])
    )
    for rank, category in enumerate(ordered_categories):
        members = category_members[category]
        category_positions[category] = (
            n + 0.85 + rank * 0.55,
            float(np.mean(members)) + 0.5,
        )

    for category, members in category_members.items():
        category_position = category_positions[category]
        for member in members:
            peers = [peer for peer in members if peer != member]
            value = (float(np.mean(shown[member, peers])) if peers
                     else float(shown[member, member]))
            color, width = _edge_style(value, display_low, display_high)
            start = (n + 0.12, member + 0.5)
            bend = -0.08 if category_position[1] <= start[1] else 0.08
            axis.add_patch(FancyArrowPatch(
                start, category_position, arrowstyle="-",
                connectionstyle=f"arc3,rad={bend}",
                color=color, linewidth=width, alpha=0.78, zorder=2,
            ))

    centers = np.arange(n) + 0.5
    axis.scatter(
        np.full(n, n + 0.12), centers, s=16, facecolor="#9c2f62",
        edgecolor="white", linewidth=0.45, zorder=4,
    )
    for category, (x_pos, y_pos) in category_positions.items():
        axis.scatter(
            [x_pos], [y_pos], s=18, facecolor="#9c2f62", edgecolor="white",
            linewidth=0.45, zorder=4,
        )
        axis.text(
            x_pos + 0.11, y_pos, category, va="center", ha="left",
            fontsize=BASE_FONT_SIZE,
        )

    shown_labels = [_display_label(label) for label in labels]
    axis.set_xticks(centers, shown_labels, rotation=50, ha="right", fontsize=BASE_FONT_SIZE)
    axis.set_yticks(centers, shown_labels, fontsize=BASE_FONT_SIZE)
    axis.tick_params(length=0, pad=3)
    right_edge = max(x for x, _ in category_positions.values()) + 2.0
    axis.set_xlim(-0.1, right_edge)
    axis.set_ylim(n + 0.2, -0.55)
    axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_visible(False)
    if title:
        axis.set_title(title, fontsize=7.5, pad=6, loc="left")

    color_axis = inset_axes(
        axis, width="2.4%", height="24%", loc="lower left",
        bbox_to_anchor=(-0.31, 0.03, 1, 1), bbox_transform=axis.transAxes,
        borderpad=0,
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=matrix_cmap), cax=color_axis,
    )
    colorbar.ax.tick_params(labelsize=6, width=0.45, length=2)
    colorbar.set_label(
        "Normalized Jaccard" if normalized else "Jaccard overlap",
        fontsize=BASE_FONT_SIZE, labelpad=2,
    )
    colorbar.outline.set_linewidth(0.45)

    break_1, break_2 = _overlap_breaks(display_low, display_high)
    color_labels = (
        f"<{break_1:.2f}",
        f"{break_1:.2f}-{break_2:.2f}",
        f">={break_2:.2f}",
    )
    color_legend = [
        Line2D([0], [0], color=color, lw=0.9, label=label)
        for color, label in zip(EDGE_COLORS, color_labels)
    ]
    legend_color = axis.legend(
        handles=color_legend, title="Mean Jaccard", frameon=False,
        loc="upper left", bbox_to_anchor=(-0.32, 0.80), borderaxespad=0,
        fontsize=6, title_fontsize=BASE_FONT_SIZE, handlelength=2.3,
    )
    axis.add_artist(legend_color)
    width_legend = [
        Line2D([0], [0], color="#666666", lw=width, label=label)
        for width, label in zip((0.4, 0.85, 1.35), color_labels)
    ]
    axis.legend(
        handles=width_legend, title="Line width", frameon=False,
        loc="upper left", bbox_to_anchor=(-0.32, 0.58), borderaxespad=0,
        fontsize=6, title_fontsize=BASE_FONT_SIZE, handlelength=2.3,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.19, top=0.94)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {pdf_path}")


def plot_overlap_triangle_network(matrix: np.ndarray, labels: list[str],
                                  title: str, out_path: str,
                                  categories: dict[str, list[str]],
                                  cmap: str = "overlap_blues") -> None:
    _draw_triangle_network(
        matrix, labels, title, out_path, categories, normalized=False, cmap=cmap
    )


def plot_overlap_triangle_network_normalized(matrix: np.ndarray, labels: list[str],
                                             title: str, out_path: str,
                                             categories: dict[str, list[str]],
                                             cmap: str = "overlap_blues") -> None:
    _draw_triangle_network(
        matrix, labels, title, out_path, categories, normalized=True, cmap=cmap
    )
