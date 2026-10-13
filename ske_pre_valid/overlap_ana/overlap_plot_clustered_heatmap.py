"""Publication-style clustered heatmaps for skeleton-path overlap."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


BASE_FONT_SIZE = 7
FIGURE_SIZE = (180 / 25.4, 145 / 25.4)
DEFAULT_CMAP = sns.blend_palette(
    ["#f7f7f7", "#d9f0ee", "#91d1c2", "#4dbbd5", "#3c5488"], as_cmap=True
)
CATEGORY_PALETTES = {
    "neutral_green": ("#dfe9e4",) * 10,
    "static_groups": (
        "#d9eaf4", "#f6e8c3", "#ddefe8", "#f4dddc", "#e9e0f1",
        "#e2e7ea", "#eadfd5", "#dce8dd", "#e8dfe8", "#eee7d8",
    ),
    "nature_muted": (
        "#6f8faf", "#c9897b", "#7fa38a", "#9b8bb4", "#d2ae6d",
        "#7895a4", "#b58b7b", "#879a75", "#8d829b", "#b99b68",
    ),
    "npg": (
        "#e64b35", "#4dbbd5", "#00a087", "#3c5488", "#f39b7f",
        "#8491b4", "#91d1c2", "#dc0000", "#7e6148", "#b09c85",
    ),
    "okabe_ito": (
        "#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9",
        "#d55e00", "#f0e442", "#000000", "#999999", "#882255",
    ),
    "tol_muted": (
        "#332288", "#88ccee", "#44aa99", "#117733", "#999933",
        "#ddcc77", "#cc6677", "#882255", "#aa4499", "#dddddd",
    ),
}
HEATMAP_PRESETS = {
    "emerald_green": ("#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#005a32"),
    "soft_coral": ("#3b8bc2", "#b9ddf0", "#fffdf5", "#f8c0a8", "#e56b5d"),
    "reference": ("#2166ac", "#d1e5f0", "#f7f7f7", "#fddbc7", "#b2182b"),
    "deep_blue_red": ("#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"),
    "red_blue": ("#d6604d", "#efb19d", "#f7f7f7", "#8fbed3", "#2f6f9f"),
    "npg": ("#f7f7f7", "#d9f0ee", "#91d1c2", "#4dbbd5", "#3c5488"),
    "batlow": ("#011959", "#23496c", "#3f756d", "#7c9d59", "#c5b957", "#facc80"),
    "teal": ("#f7faf9", "#d9ece7", "#9bcabc", "#4b9b8e", "#245b5a"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": BASE_FONT_SIZE,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _display_label(label: str) -> str:
    return label.lstrip()


def _resolve_cmap(name: str, custom_colors: list[str] | None = None):
    if custom_colors:
        if len(custom_colors) < 2:
            raise ValueError("matrix_colors requires at least two colors")
        return colors.LinearSegmentedColormap.from_list("custom_overlap", custom_colors)
    if name == "overlap_blues":
        return DEFAULT_CMAP
    if name in HEATMAP_PRESETS:
        return colors.LinearSegmentedColormap.from_list(name, HEATMAP_PRESETS[name])
    try:
        return matplotlib.colormaps[name]
    except KeyError as error:
        raise ValueError(f"Unknown matrix colormap: {name!r}") from error


def _token_categories(labels: list[str], categories: dict[str, list[str]],
                      palette: list[str] | None = None,
                      palette_name: str = "nature_muted"):
    lookup = {}
    for category, tokens in categories.items():
        for token in tokens:
            lookup.setdefault(token.strip(), category)

    category_names = list(categories)
    palette = palette or list(CATEGORY_PALETTES[palette_name])
    if len(palette) < len(category_names):
        raise ValueError(
            f"category_colors has {len(palette)} colors but {len(category_names)} are required"
        )
    category_colors = {
        category: palette[index]
        for index, category in enumerate(category_names)
    }
    assigned = [lookup.get(label.strip(), "uncategorized") for label in labels]
    if "uncategorized" in assigned:
        category_colors["uncategorized"] = "#999999"
    return assigned, category_colors


def _category_runs(values: list[str]):
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            yield start, index, values[start]
            start = index


def _label_color(hex_color: str) -> str:
    red, green, blue = colors.to_rgb(hex_color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#ffffff" if luminance < 0.48 else "#111111"


def _target_label_color(hex_color: str) -> tuple[float, float, float]:
    """Darken category colors enough for small text on white."""
    red, green, blue = colors.to_rgb(hex_color)
    return red * 0.68, green * 0.68, blue * 0.68


def _draw_category_strips(grid, assigned: list[str], category_color_map: dict,
                          row_order: list[int], col_order: list[int],
                          cluster_gap: float = 0.0,
                          separator_width: float = 0.7) -> None:
    row_values = [assigned[index] for index in row_order]
    col_values = [assigned[index] for index in col_order]

    row_axis = grid.ax_row_colors
    row_axis.clear()
    row_axis.set_xlim(0, 1)
    row_axis.set_ylim(len(row_values), 0)
    for start, stop, category in _category_runs(row_values):
        color = category_color_map[category]
        pad = min(cluster_gap / 2, (stop - start) / 4)
        row_axis.add_patch(Rectangle(
            (0, start + pad), 1, stop - start - 2 * pad, color=color, ec="none"
        ))
        row_axis.text(
            0.5, (start + stop) / 2, category, rotation=90,
            ha="center", va="center", fontsize=5.5, fontweight="bold",
            color=_label_color(color),
        )
        if stop < len(row_values):
            row_axis.axhline(stop, color="#303030", linewidth=separator_width)

    col_axis = grid.ax_col_colors
    col_axis.clear()
    col_axis.set_xlim(0, len(col_values))
    col_axis.set_ylim(0, 1)
    for start, stop, category in _category_runs(col_values):
        color = category_color_map[category]
        pad = min(cluster_gap / 2, (stop - start) / 4)
        col_axis.add_patch(Rectangle(
            (start + pad, 0), stop - start - 2 * pad, 1, color=color, ec="none"
        ))
        col_axis.text(
            (start + stop) / 2, 0.5, category,
            ha="center", va="center", fontsize=5.5, fontweight="bold",
            color=_label_color(color),
        )
        if stop < len(col_values):
            col_axis.axvline(stop, color="#303030", linewidth=separator_width)

    for axis in (row_axis, col_axis):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#000000")
            spine.set_linewidth(0.5)


def _cluster_linkage(matrix: np.ndarray) -> np.ndarray:
    distance = 1.0 - np.clip((matrix + matrix.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method="average")


def _draw_clustered_heatmap(matrix: np.ndarray, labels: list[str], title: str,
                            out_path: str, categories: dict[str, list[str]],
                            normalized: bool = False,
                            cmap: str = "emerald_green",
                            matrix_colors: list[str] | None = None,
                            category_colors: list[str] | None = None,
                            category_palette: str = "neutral_green",
                            dendrogram_ratio: float = 0.08,
                            annotation_gap_mm: float = 1.5,
                            dendrogram_gap_mm: float = 0.0,
                            color_target_labels: bool = True,
                            target_font_size: float = 7.0,
                            target_font_color: str | None = None,
                            target_category_colors: list[str] | tuple[str, ...] = (
                                "#176b45", "#176b45", "#3a3a3a", "#3a3a3a"
                            ),
                            target_font_weight: str = "bold",
                            cluster_gap: float = 0.0,
                            cluster_separator_width: float = 0.7,
                            tile_min_size: float = 0.10,
                            tile_max_size: float = 0.86,
                            colorbar_gap_mm: float = 3.0,
                            colorbar_width_mm: float = 3.0,
                            colorbar_height_mm: float = 30.0) -> None:
    matrix = np.asarray(matrix, dtype=float)
    n = len(labels)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix shape {matrix.shape} does not match {n} labels")
    if n < 2:
        raise ValueError("clustered heatmap requires at least two tokens")
    if not 0.0 <= tile_min_size <= tile_max_size <= 1.0:
        raise ValueError("tile sizes must satisfy 0 <= min <= max <= 1")

    linkage_matrix = _cluster_linkage(matrix)
    shown = matrix.copy()
    colorbar_label = "Jaccard overlap"
    if normalized:
        mask = ~np.eye(n, dtype=bool)
        low, high = shown[mask].min(), shown[mask].max()
        shown = ((shown - low) / (high - low)
                 if high > low else np.zeros_like(shown))
        np.fill_diagonal(shown, 1.0)
        colorbar_label = "Normalized Jaccard"

    assigned, category_color_map = _token_categories(
        labels, categories, category_colors, category_palette
    )
    annotation_colors = [category_color_map[category] for category in assigned]
    display_labels = [_display_label(label) for label in labels]

    grid = sns.clustermap(
        shown,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        row_colors=annotation_colors,
        col_colors=annotation_colors,
        cmap=_resolve_cmap(cmap, matrix_colors),
        vmin=0.0,
        vmax=1.0,
        linewidths=0.25,
        linecolor="#202020",
        xticklabels=display_labels,
        yticklabels=display_labels,
        figsize=FIGURE_SIZE,
        dendrogram_ratio=(dendrogram_ratio, dendrogram_ratio),
        colors_ratio=(0.025, 0.025),
        cbar_pos=(0.01, 0.01, 0.02, 0.2),
        tree_kws={"colors": "#202020", "linewidths": 0.6},
    )

    heat_axis = grid.ax_heatmap
    row_order = grid.dendrogram_row.reordered_ind
    col_order = grid.dendrogram_col.reordered_ind
    _draw_category_strips(
        grid, assigned, category_color_map, row_order, col_order,
        cluster_gap, cluster_separator_width,
    )
    ordered_matrix = shown[np.ix_(row_order, col_order)]
    ordered_xlabels = [display_labels[index] for index in col_order]
    ordered_ylabels = [display_labels[index] for index in row_order]

    # Square area and color both encode overlap, following correlation-matrix conventions.
    heat_axis.clear()
    matrix_cmap = _resolve_cmap(cmap, matrix_colors)
    matrix_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    for row in range(n):
        for col in range(n):
            value = float(ordered_matrix[row, col])
            heat_axis.add_patch(Rectangle(
                (col, row), 1, 1,
                facecolor="#ffffff", edgecolor="#202020", linewidth=0.28,
            ))
            side = tile_min_size + (tile_max_size - tile_min_size) * np.sqrt(
                np.clip(value, 0.0, 1.0)
            )
            inset = (1.0 - side) / 2.0
            heat_axis.add_patch(Rectangle(
                (col + inset, row + inset), side, side,
                facecolor=matrix_cmap(matrix_norm(value)),
                edgecolor="#202020", linewidth=0.24,
            ))
    heat_axis.set_xlim(0, n)
    heat_axis.set_ylim(n, 0)
    heat_axis.set_aspect("equal")
    heat_axis.set_xticks(np.arange(n) + 0.5, ordered_xlabels)
    heat_axis.set_yticks(np.arange(n) + 0.5, ordered_ylabels)
    heat_axis.tick_params(axis="x", length=0, labelsize=target_font_size, pad=2)
    heat_axis.tick_params(
        axis="y", length=0, labelsize=target_font_size, pad=2,
        labelleft=False, labelright=True,
    )
    heat_axis.yaxis.tick_right()
    plt.setp(heat_axis.get_xticklabels(), rotation=50, ha="right", rotation_mode="anchor")
    plt.setp(heat_axis.get_yticklabels(), rotation=0)
    plt.setp(
        heat_axis.get_xticklabels() + heat_axis.get_yticklabels(),
        fontweight=target_font_weight,
    )
    if target_font_color:
        plt.setp(
            heat_axis.get_xticklabels() + heat_axis.get_yticklabels(),
            color=target_font_color,
        )
    elif color_target_labels:
        category_label_colors = {
            category: target_category_colors[index % len(target_category_colors)]
            for index, category in enumerate(categories)
        }
        for tick, index in zip(heat_axis.get_xticklabels(), col_order):
            tick.set_color(category_label_colors.get(assigned[index], "#3a3a3a"))
        for tick, index in zip(heat_axis.get_yticklabels(), row_order):
            tick.set_color(category_label_colors.get(assigned[index], "#3a3a3a"))

    row_categories = [assigned[index] for index in row_order]
    col_categories = [assigned[index] for index in col_order]
    row_boundaries = [i for i in range(1, n) if row_categories[i] != row_categories[i - 1]]
    col_boundaries = [i for i in range(1, n) if col_categories[i] != col_categories[i - 1]]
    gap = max(0.0, min(float(cluster_gap), 0.8))
    if gap > 0:
        for boundary in row_boundaries:
            heat_axis.axhspan(boundary - gap / 2, boundary + gap / 2,
                              color="white", ec="none", zorder=10)
        for boundary in col_boundaries:
            heat_axis.axvspan(boundary - gap / 2, boundary + gap / 2,
                              color="white", ec="none", zorder=10)
    for boundary in row_boundaries:
        heat_axis.axhline(
            boundary, color="#303030", linewidth=cluster_separator_width, zorder=11
        )
    for boundary in col_boundaries:
        heat_axis.axvline(
            boundary, color="#303030", linewidth=cluster_separator_width, zorder=11
        )
    for spine in heat_axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.65)

    # Re-anchor all marginal axes after the equal-aspect heatmap has its final size.
    grid.fig.canvas.draw()
    heat_position = heat_axis.get_position()
    row_color_position = grid.ax_row_colors.get_position()
    col_color_position = grid.ax_col_colors.get_position()
    row_gap = annotation_gap_mm / 25.4 / FIGURE_SIZE[0]
    col_gap = annotation_gap_mm / 25.4 / FIGURE_SIZE[1]
    row_tree_gap = dendrogram_gap_mm / 25.4 / FIGURE_SIZE[0]
    col_tree_gap = dendrogram_gap_mm / 25.4 / FIGURE_SIZE[1]
    grid.ax_row_colors.set_position([
        heat_position.x0 - row_gap - row_color_position.width,
        heat_position.y0,
        row_color_position.width,
        heat_position.height,
    ])
    grid.ax_col_colors.set_position([
        heat_position.x0,
        heat_position.y1 + col_gap,
        heat_position.width,
        col_color_position.height,
    ])
    row_dendrogram_position = grid.ax_row_dendrogram.get_position()
    grid.ax_row_dendrogram.set_visible(True)
    grid.ax_row_dendrogram.set_position([
        heat_position.x0 - row_gap - row_color_position.width
        - row_tree_gap - row_dendrogram_position.width,
        heat_position.y0,
        row_dendrogram_position.width,
        heat_position.height,
    ])
    col_dendrogram_position = grid.ax_col_dendrogram.get_position()
    grid.ax_col_dendrogram.set_position([
        heat_position.x0,
        heat_position.y1 + col_gap + col_color_position.height + col_tree_gap,
        heat_position.width,
        col_dendrogram_position.height,
    ])

    # Place the colorbar after the actual right-hand token-label extent.
    renderer = grid.fig.canvas.get_renderer()
    label_right = max(
        grid.fig.transFigure.inverted().transform_bbox(
            tick.get_window_extent(renderer=renderer)
        ).x1
        for tick in heat_axis.get_yticklabels()
    )
    colorbar_width = colorbar_width_mm / 25.4 / FIGURE_SIZE[0]
    colorbar_height = min(
        colorbar_height_mm / 25.4 / FIGURE_SIZE[1], heat_position.height
    )
    colorbar_gap = colorbar_gap_mm / 25.4 / FIGURE_SIZE[0]
    grid.cax.set_position([
        label_right + colorbar_gap,
        heat_position.y0,
        colorbar_width,
        colorbar_height,
    ])

    for dendrogram_axis in (grid.ax_row_dendrogram, grid.ax_col_dendrogram):
        dendrogram_axis.set_facecolor("white")
        dendrogram_axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    grid.cax.tick_params(labelsize=6, width=0.45, length=2)
    grid.cax.yaxis.set_ticks_position("right")
    grid.cax.tick_params(axis="y", labelleft=False, labelright=True)
    grid.cax.set_yticks(np.linspace(0.0, 1.0, 5))
    grid.cax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
    grid.cax.set_ylabel(colorbar_label, fontsize=BASE_FONT_SIZE, labelpad=2)
    grid.cax.yaxis.set_label_position("right")

    grid.cax.patch.set_facecolor("white")
    grid.cax.patch.set_edgecolor("#000000")
    grid.cax.patch.set_linewidth(0.6)
    if title:
        title_y = min(0.995, grid.ax_col_dendrogram.get_position().y1 + 0.012)
        grid.fig.suptitle(
            title, x=heat_position.x0, y=title_y,
            ha="left", va="bottom", fontsize=7.5, fontweight="bold",
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.fig.savefig(
        out_path, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white"
    )
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    grid.fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(grid.fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {pdf_path}")


def plot_overlap_clustered_heatmap(matrix: np.ndarray, labels: list[str],
                                   title: str, out_path: str,
                                   categories: dict[str, list[str]],
                                   cmap: str = "emerald_green",
                                   matrix_colors: list[str] | None = None,
                                   category_colors: list[str] | None = None,
                                   category_palette: str = "neutral_green",
                                   dendrogram_ratio: float = 0.08,
                                   annotation_gap_mm: float = 1.5,
                                   dendrogram_gap_mm: float = 0.0,
                                   color_target_labels: bool = True,
                                   target_font_size: float = 7.0,
                                   target_font_color: str | None = None,
                                   target_category_colors: list[str] | tuple[str, ...] = (
                                       "#176b45", "#176b45", "#3a3a3a", "#3a3a3a"
                                   ),
                                   target_font_weight: str = "bold",
                                   cluster_gap: float = 0.0,
                                   cluster_separator_width: float = 0.7,
                                   tile_min_size: float = 0.10,
                                   tile_max_size: float = 0.86,
                                   colorbar_gap_mm: float = 3.0,
                                   colorbar_width_mm: float = 3.0,
                                   colorbar_height_mm: float = 30.0) -> None:
    _draw_clustered_heatmap(
        matrix, labels, title, out_path, categories, normalized=False, cmap=cmap,
        matrix_colors=matrix_colors,
        category_colors=category_colors, category_palette=category_palette,
        dendrogram_ratio=dendrogram_ratio,
        annotation_gap_mm=annotation_gap_mm,
        dendrogram_gap_mm=dendrogram_gap_mm,
        color_target_labels=color_target_labels,
        target_font_size=target_font_size, target_font_color=target_font_color,
        target_category_colors=target_category_colors,
        target_font_weight=target_font_weight, cluster_gap=cluster_gap,
        cluster_separator_width=cluster_separator_width,
        tile_min_size=tile_min_size, tile_max_size=tile_max_size,
        colorbar_gap_mm=colorbar_gap_mm, colorbar_width_mm=colorbar_width_mm,
        colorbar_height_mm=colorbar_height_mm,
    )


def plot_overlap_clustered_heatmap_normalized(
    matrix: np.ndarray, labels: list[str], title: str, out_path: str,
    categories: dict[str, list[str]], cmap: str = "emerald_green",
    matrix_colors: list[str] | None = None,
    category_colors: list[str] | None = None,
    category_palette: str = "neutral_green",
    dendrogram_ratio: float = 0.08,
    annotation_gap_mm: float = 1.5,
    dendrogram_gap_mm: float = 0.0,
    color_target_labels: bool = True,
    target_font_size: float = 7.0,
    target_font_color: str | None = None,
    target_category_colors: list[str] | tuple[str, ...] = (
        "#176b45", "#176b45", "#3a3a3a", "#3a3a3a"
    ),
    target_font_weight: str = "bold",
    cluster_gap: float = 0.0,
    cluster_separator_width: float = 0.7,
    tile_min_size: float = 0.10,
    tile_max_size: float = 0.86,
    colorbar_gap_mm: float = 3.0,
    colorbar_width_mm: float = 3.0,
    colorbar_height_mm: float = 30.0,
) -> None:
    _draw_clustered_heatmap(
        matrix, labels, title, out_path, categories, normalized=True, cmap=cmap,
        matrix_colors=matrix_colors,
        category_colors=category_colors, category_palette=category_palette,
        dendrogram_ratio=dendrogram_ratio,
        annotation_gap_mm=annotation_gap_mm,
        dendrogram_gap_mm=dendrogram_gap_mm,
        color_target_labels=color_target_labels,
        target_font_size=target_font_size, target_font_color=target_font_color,
        target_category_colors=target_category_colors,
        target_font_weight=target_font_weight, cluster_gap=cluster_gap,
        cluster_separator_width=cluster_separator_width,
        tile_min_size=tile_min_size, tile_max_size=tile_max_size,
        colorbar_gap_mm=colorbar_gap_mm, colorbar_width_mm=colorbar_width_mm,
        colorbar_height_mm=colorbar_height_mm,
    )
