"""Create paper figures for frozen-static prospective recruitment results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


MODEL_ORDER = (
    "openai-community/gpt2",
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B",
    "mistralai/Mistral-7B-v0.3",
)
MODEL_LABELS = {
    "openai-community/gpt2": "GPT-2",
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B-v0.3",
}
STATIC_COLORS = {
    1.0: "#0072B2",
    5.0: "#D55E00",
    10.0: "#009E73",
}
# Validated categorical order (dataviz skill reference palette, light mode):
# blue, orange, aqua, yellow -- adjacent-pair CVD/contrast checks pass for
# line charts with up to 8 series; only the first 4 are used here.
MODEL_COLORS = {
    "openai-community/gpt2": "#2a78d6",
    "Qwen/Qwen3-0.6B": "#eb6834",
    "meta-llama/Llama-3.2-1B": "#1baf7a",
    "mistralai/Mistral-7B-v0.3": "#eda100",
}
RANK_DISTRIBUTION_COLOR = "#2a78d6"
RANK_DISTRIBUTION_BASELINE_COLOR = "#52514e"


def _close(value: float, target: float) -> bool:
    return bool(np.isclose(value, target, rtol=1e-9, atol=1e-12))


def _static_colors(static_percents):
    extras = [value for value in static_percents if value not in STATIC_COLORS]
    extra_colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(extras), 1)))
    colors = dict(STATIC_COLORS)
    colors.update(zip(extras, extra_colors))
    return colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", required=False, default=None)
    parser.add_argument("--rank_distribution_csv", required=False, default=None)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float | str]]:
    numeric = {
        "static_percent_requested",
        "static_k_per_sensor",
        "static_percent_actual",
        "dynamic_fraction",
        "dynamic_percent",
        "n_observations",
        "mean_precision",
        "ci95_low_precision",
        "ci95_high_precision",
        "mean_recall",
        "ci95_low_recall",
        "ci95_high_recall",
        "mean_enrichment",
        "ci95_low_enrichment",
        "ci95_high_enrichment",
        "mean_matched_random_precision",
        "mean_matched_random_recall",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def model_rows(rows, model: str, static_percent: float):
    return sorted(
        (
            row
            for row in rows
            if row["model_name"] == model
            and _close(row["static_percent_requested"], static_percent)
        ),
        key=lambda row: row["dynamic_percent"],
    )


def save_line_figure(rows, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        2,
        len(MODEL_ORDER),
        figsize=(15.8, 6.6),
        sharex=True,
        constrained_layout=True,
    )
    static_percents = sorted({row["static_percent_requested"] for row in rows})
    colors = _static_colors(static_percents)
    for column, model in enumerate(MODEL_ORDER):
        precision_ax = axes[0, column]
        enrichment_ax = axes[1, column]
        for static_percent in static_percents:
            selected = model_rows(rows, model, static_percent)
            x = np.asarray([row["dynamic_percent"] for row in selected])
            precision = 100.0 * np.asarray([row["mean_precision"] for row in selected])
            precision_low = 100.0 * np.asarray(
                [row["ci95_low_precision"] for row in selected]
            )
            precision_high = 100.0 * np.asarray(
                [row["ci95_high_precision"] for row in selected]
            )
            enrichment = np.asarray([row["mean_enrichment"] for row in selected])
            enrichment_low = np.asarray(
                [row["ci95_low_enrichment"] for row in selected]
            )
            enrichment_high = np.asarray(
                [row["ci95_high_enrichment"] for row in selected]
            )
            color = colors[static_percent]
            precision_ax.plot(x, precision, marker="o", ms=3.5, lw=1.8, color=color)
            precision_ax.fill_between(
                x, precision_low, precision_high, color=color, alpha=0.18, linewidth=0
            )
            enrichment_ax.plot(
                x, enrichment, marker="o", ms=3.5, lw=1.8, color=color
            )
            enrichment_ax.fill_between(
                x,
                enrichment_low,
                enrichment_high,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

        reference = model_rows(rows, model, static_percents[0])
        x = np.asarray([row["dynamic_percent"] for row in reference])
        random_precision = 100.0 * np.asarray(
            [row["mean_matched_random_precision"] for row in reference]
        )
        precision_ax.plot(x, random_precision, color="0.2", ls="--", lw=1.3)
        enrichment_ax.axhline(1.0, color="0.2", ls="--", lw=1.3)

        precision_ax.set_title(MODEL_LABELS[model], fontsize=11, pad=7)
        precision_ax.set_xscale("log")
        enrichment_ax.set_xscale("log")
        enrichment_ax.set_yscale("log")
        precision_ax.grid(alpha=0.2, which="both")
        enrichment_ax.grid(alpha=0.2, which="both")
        enrichment_ax.set_xlabel("Global Dynamic Top (%)")
        enrichment_ax.set_ylim(0.85, 120)
        enrichment_ax.set_yticks([1, 3, 10, 30, 100])
        enrichment_ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        for axis in (precision_ax, enrichment_ax):
            axis.set_xticks([0.001, 0.01, 0.1, 1, 5])
            axis.set_xticklabels(["0.001", "0.01", "0.1", "1", "5"])

    axes[0, 0].set_ylabel("Frozen Static pathways recruited\n(precision, %)")
    axes[1, 0].set_ylabel("Enrichment over matched random")
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[percent],
            marker="o",
            lw=1.8,
            label=f"Static Top {percent:g}% per sensor",
        )
        for percent in static_percents
    ]
    handles.append(
        Line2D([0], [0], color="0.2", ls="--", lw=1.3, label="Matched random")
    )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"prospective_recruitment_precision_enrichment.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_heatmap(rows, output_dir: Path) -> None:
    static_percents = sorted({row["static_percent_requested"] for row in rows})
    dynamic_percents = sorted({row["dynamic_percent"] for row in rows})
    values = np.asarray([float(row["mean_enrichment"]) for row in rows])
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("mean_enrichment values must be finite and strictly positive")
    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        vmax = vmin * (1.0 + 1e-6)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(
        1, len(MODEL_ORDER), figsize=(16.0, 3.0), constrained_layout=True
    )
    image = None
    for axis, model in zip(axes, MODEL_ORDER):
        lookup = {
            (row["static_percent_requested"], row["dynamic_percent"]): row
            for row in rows
            if row["model_name"] == model
        }
        missing = [
            (static, dynamic) for static in static_percents for dynamic in dynamic_percents
            if (static, dynamic) not in lookup
        ]
        if missing:
            raise ValueError(f"{model}: incomplete static/dynamic grid; missing {missing}")
        matrix = np.asarray([[lookup[(static, dynamic)]["mean_enrichment"]
                              for dynamic in dynamic_percents]
                             for static in static_percents])
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", norm=norm)
        axis.set_title(MODEL_LABELS[model], fontsize=11)
        axis.set_xticks(range(len(dynamic_percents)))
        axis.set_xticklabels([f"{value:g}" for value in dynamic_percents], rotation=55)
        axis.set_yticks(range(len(static_percents)))
        axis.set_yticklabels([f"{value:g}" for value in static_percents])
        axis.set_xlabel("Dynamic Top (%)")
        for row_index in range(len(static_percents)):
            for column_index in range(len(dynamic_percents)):
                value = matrix[row_index, column_index]
                color = "white" if norm(value) < 0.52 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=color,
                )
    axes[0].set_ylabel("Static Top per sensor (%)")
    colorbar = fig.colorbar(image, ax=axes, pad=0.012, shrink=0.88)
    colorbar.set_label("Enrichment over matched random")
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"prospective_recruitment_enrichment_heatmap.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_primary_table(rows, output_dir: Path) -> None:
    selected = [
        row
        for row in rows
        if _close(row["static_percent_requested"], 1.0)
        and _close(row["dynamic_percent"], 0.1)
    ]
    selected.sort(key=lambda row: MODEL_ORDER.index(row["model_name"]))
    fields = (
        "model",
        "n_inputs",
        "static_percent_requested",
        "static_percent_actual",
        "dynamic_percent",
        "precision_percent",
        "ci95_low_precision_percent",
        "ci95_high_precision_percent",
        "recall_percent",
        "enrichment_x",
        "ci95_low_enrichment_x",
        "ci95_high_enrichment_x",
    )
    with (output_dir / "primary_static1_dynamic0p1_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "model": MODEL_LABELS[row["model_name"]],
                    "n_inputs": int(row["n_observations"]),
                    "static_percent_requested": row["static_percent_requested"],
                    "static_percent_actual": row["static_percent_actual"],
                    "dynamic_percent": row["dynamic_percent"],
                    "precision_percent": 100.0 * row["mean_precision"],
                    "ci95_low_precision_percent": 100.0
                    * row["ci95_low_precision"],
                    "ci95_high_precision_percent": 100.0
                    * row["ci95_high_precision"],
                    "recall_percent": 100.0 * row["mean_recall"],
                    "enrichment_x": row["mean_enrichment"],
                    "ci95_low_enrichment_x": row["ci95_low_enrichment"],
                    "ci95_high_enrichment_x": row["ci95_high_enrichment"],
                }
            )


def save_primary_table_figure(rows, output_dir: Path) -> None:
    selected = [
        row
        for row in rows
        if _close(row["static_percent_requested"], 1.0)
        and _close(row["dynamic_percent"], 0.1)
    ]
    selected.sort(key=lambda row: MODEL_ORDER.index(row["model_name"]))
    columns = (
        "Model",
        "Realized Static circuit",
        "Static pathways recruited\n(Intersection / Static)",
        "Random\nexpectation",
        "Enrichment\nover random",
        "Dynamic Top covered\n(Intersection / Dynamic)",
    )
    cells = []
    for row in selected:
        cells.append(
            [
                MODEL_LABELS[row["model_name"]],
                f'{row["static_percent_actual"]:.3f}%  (K={int(row["static_k_per_sensor"])})',
                (
                    f'{100.0 * row["mean_precision"]:.2f}%\n'
                    f'[{100.0 * row["ci95_low_precision"]:.2f}, '
                    f'{100.0 * row["ci95_high_precision"]:.2f}]'
                ),
                f'{100.0 * row["mean_matched_random_precision"]:.2f}%',
                (
                    f'{row["mean_enrichment"]:.2f}x\n'
                    f'[{row["ci95_low_enrichment"]:.2f}, '
                    f'{row["ci95_high_enrichment"]:.2f}]'
                ),
                f'{100.0 * row["mean_recall"]:.2f}%',
            ]
        )
    fig, axis = plt.subplots(figsize=(14.2, 3.0))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.14, 0.18, 0.22, 0.12, 0.18, 0.20],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.9)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row_index == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.25)
        else:
            cell.set_facecolor("#EDF4F5" if row_index % 2 else "#DCEAEC")
            if cell.get_text().get_text().startswith(tuple(MODEL_LABELS.values())):
                cell.set_text_props(weight="bold")
    axis.set_title(
        "Frozen Static Top 1% per sensor → exact global Dynamic Top 0.1%  "
        f"({int(selected[0]['n_observations'])} held-out inputs/model)" if selected else "",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    axis.text(
        0.5,
        -0.03,
        "Brackets denote 95% input-bootstrap confidence intervals. "
        "Random expectation for Static recruitment is 0.1%.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.3",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"primary_static1_dynamic0p1_table.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_primary_latex_table(rows, output_dir: Path) -> None:
    selected = [
        row
        for row in rows
        if _close(row["static_percent_requested"], 1.0)
        and _close(row["dynamic_percent"], 0.1)
    ]
    selected.sort(key=lambda row: MODEL_ORDER.index(row["model_name"]))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        (
            r"Model & Realized static & Static recruited (\%) & Random (\%) "
            r"& Enrichment ($\times$) & Dynamic covered (\%) \\"
        ),
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f'{MODEL_LABELS[row["model_name"]]} & '
            f'{row["static_percent_actual"]:.3f}\% ($K={int(row["static_k_per_sensor"])}$) & '
            f'{100.0 * row["mean_precision"]:.2f} '
            f'[{100.0 * row["ci95_low_precision"]:.2f}, '
            f'{100.0 * row["ci95_high_precision"]:.2f}] & '
            f'{100.0 * row["mean_matched_random_precision"]:.2f} & '
            f'{row["mean_enrichment"]:.2f} '
            f'[{row["ci95_low_enrichment"]:.2f}, '
            f'{row["ci95_high_enrichment"]:.2f}] & '
            f'{100.0 * row["mean_recall"]:.2f} \\\\'
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Prospective recruitment for frozen Static Top 1\% "
                r"per sensor and exact global Dynamic Top 0.1\%. Static recruited "
                r"uses $|S\cap D|/|S|$; Dynamic covered uses $|S\cap D|/|D|$. "
                r"Brackets denote 95\% input-bootstrap confidence intervals "
                rf"($N={int(selected[0]['n_observations']) if selected else 0}$ inputs per model).}}"
            ),
            r"\label{tab:prospective-recruitment}",
            r"\end{table}",
        ]
    )
    (output_dir / "primary_static1_dynamic0p1_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def save_dynamic_conditioned_static_rank_cdf(rows, output_dir: Path) -> None:
    """Plot P(static within-sensor rank <= q | pathway is Dynamic Top)."""
    shown_dynamic = tuple(sorted({row["dynamic_percent"] for row in rows}))
    colors = plt.get_cmap("plasma")(np.linspace(0.08, 0.88, len(shown_dynamic)))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    for axis, model in zip(axes.flat, MODEL_ORDER):
        for dynamic_percent, color in zip(shown_dynamic, colors):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["model_name"] == model
                    and _close(row["dynamic_percent"], dynamic_percent)
                ),
                key=lambda row: row["static_percent_requested"],
            )
            x = np.asarray([row["static_percent_actual"] for row in selected])
            y = 100.0 * np.asarray([row["mean_recall"] for row in selected])
            low = 100.0 * np.asarray([row["ci95_low_recall"] for row in selected])
            high = 100.0 * np.asarray([row["ci95_high_recall"] for row in selected])
            axis.plot(x, y, color=color, marker="o", lw=1.8, ms=4)
            axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        reference = sorted(
            (
                row
                for row in rows
                if row["model_name"] == model
                and _close(row["dynamic_percent"], shown_dynamic[0])
            ),
            key=lambda row: row["static_percent_requested"],
        )
        random_x = np.asarray([row["static_percent_actual"] for row in reference])
        axis.plot(random_x, random_x, color="0.15", ls="--", lw=1.4)
        axis.set_title(MODEL_LABELS[model], fontsize=11)
        axis.set_xscale("log")
        all_x = np.asarray([row["static_percent_actual"] for row in reference])
        if all_x.size and np.all(all_x > 0):
            axis.set_xlim(float(all_x.min()) * 0.8, float(all_x.max()) * 1.2)
        axis.set_ylim(0, 103)
        axis.set_xticks(all_x)
        axis.set_xticklabels([f"{value:g}" for value in all_x])
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.grid(alpha=0.22, which="both")
        axis.set_xlabel("Within-sensor Static rank cutoff, Top q%")
    axes[0, 0].set_ylabel("Dynamic Top pathways within cutoff (%)")
    axes[1, 0].set_ylabel("Dynamic Top pathways within cutoff (%)")
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            lw=1.8,
            label=f"Dynamic Top {dynamic:g}%",
        )
        for dynamic, color in zip(shown_dynamic, colors)
    ]
    handles.append(
        Line2D([0], [0], color="0.15", ls="--", lw=1.4, label="Sensor-matched random")
    )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"dynamic_conditioned_static_rank_cdf.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_dynamic_conditioned_rank_table(rows, output_dir: Path) -> None:
    """Make the direct Dynamic Top 0.1% -> within-sensor Static rank table."""
    dynamic_percent = 0.1
    static_percents = sorted({row["static_percent_requested"] for row in rows})
    if not static_percents:
        raise ValueError("no Static cutoff rows were provided")
    cells = []
    latex_rows = []
    for model in MODEL_ORDER:
        selected = {
            row["static_percent_requested"]: row
            for row in rows
            if row["model_name"] == model
            and _close(row["dynamic_percent"], dynamic_percent)
        }
        cells.append(
            [MODEL_LABELS[model]]
            + [
                (
                    f'{100.0 * selected[percent]["mean_recall"]:.2f}%\n'
                    f'({selected[percent]["mean_enrichment"]:.1f}x random)'
                )
                for percent in static_percents
            ]
        )
        latex_rows.append(
            f'{MODEL_LABELS[model]} & '
            + " & ".join(
                f'{100.0 * selected[percent]["mean_recall"]:.2f}\% '
                f'({selected[percent]["mean_enrichment"]:.1f}$\\times$)'
                for percent in static_percents
            )
            + r" \\"
        )

    columns = ["Model"] + [f"Static rank in Top {percent:g}%" for percent in static_percents]
    fig, axis = plt.subplots(figsize=(10.6, 2.8))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.20] + [0.80 / len(static_percents)] * len(static_percents),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.85)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row_index == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.22)
        else:
            cell.set_facecolor("#EDF4F5" if row_index % 2 else "#DCEAEC")
            if column_index == 0:
                cell.set_text_props(weight="bold")
    axis.set_title(
        "For global Dynamic Top 0.1% pathways, where does the effector rank "
        "within the same sensor's frozen Static ranking?",
        fontsize=11.5,
        weight="bold",
        pad=11,
    )
    axis.text(
        0.5,
        -0.03,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.3",
        s=(
            "Cells show the fraction of Dynamic Top pathways at or above each "
            "Static rank cutoff; parentheses show enrichment over sensor-matched random."
        ),
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"dynamic_top0p1_within_sensor_static_rank_table.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(fig)

    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{l{'c' * len(static_percents)}}}",
        r"\toprule",
        "Model & " + " & ".join(f"Static Top {percent:g}\\%" for percent in static_percents) + r" \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Within-sensor static ranks of pathways entering the exact "
            r"global Dynamic Top 0.1\%. Each entry reports the percentage of Dynamic "
            r"Top pathways whose effector lies within the indicated frozen static "
            r"rank cutoff; parentheses report enrichment over a sensor-matched "
            r"uniform-effector baseline.}"
        ),
        r"\label{tab:dynamic-conditioned-static-rank}",
        r"\end{table}",
    ]
    (output_dir / "dynamic_top0p1_within_sensor_static_rank_table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def save_dynamic_conditioned_rank_cdf_csv(rows, output_dir: Path) -> None:
    fields = (
        "model",
        "static_rank_cutoff_requested_percent",
        "static_rank_cutoff_actual_percent",
        "dynamic_top_percent",
        "n_inputs",
        "dynamic_conditioned_static_rank_coverage_percent",
        "ci95_low_coverage_percent",
        "ci95_high_coverage_percent",
        "sensor_matched_random_coverage_percent",
        "rank_enrichment_x",
        "ci95_low_rank_enrichment_x",
        "ci95_high_rank_enrichment_x",
    )
    with (output_dir / "dynamic_conditioned_static_rank_cdf.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                MODEL_ORDER.index(item["model_name"]),
                item["dynamic_percent"],
                item["static_percent_requested"],
            ),
        ):
            writer.writerow(
                {
                    "model": MODEL_LABELS[row["model_name"]],
                    "static_rank_cutoff_requested_percent": row[
                        "static_percent_requested"
                    ],
                    "static_rank_cutoff_actual_percent": row[
                        "static_percent_actual"
                    ],
                    "dynamic_top_percent": row["dynamic_percent"],
                    "n_inputs": int(row["n_observations"]),
                    "dynamic_conditioned_static_rank_coverage_percent": 100.0
                    * row["mean_recall"],
                    "ci95_low_coverage_percent": 100.0 * row["ci95_low_recall"],
                    "ci95_high_coverage_percent": 100.0 * row[
                        "ci95_high_recall"
                    ],
                    "sensor_matched_random_coverage_percent": 100.0
                    * row["mean_matched_random_recall"],
                    "rank_enrichment_x": row["mean_enrichment"],
                    "ci95_low_rank_enrichment_x": row["ci95_low_enrichment"],
                    "ci95_high_rank_enrichment_x": row["ci95_high_enrichment"],
                }
            )


def load_rank_distribution_rows(path: Path) -> list[dict[str, float | str]]:
    """Load the actual within-sensor Static-rank distribution of recruited pathways.

    This is the ``dynamic_conditioned_static_rank_histograms`` /
    ``static_rank_distribution_metrics`` output (median/quartile Static rank
    of pathways that enter the global Dynamic Top set), not the fixed-band
    coverage numbers in ``summary_dynamic_conditioned_static_rank_cdf.csv``.
    """
    numeric = {
        "dynamic_fraction",
        "dynamic_percent",
        "n_observations",
        "mean_n_dynamic_observed",
        "mean_mean_static_rank",
        "mean_q25_static_rank",
        "ci95_low_q25_static_rank",
        "ci95_high_q25_static_rank",
        "mean_median_static_rank",
        "ci95_low_median_static_rank",
        "ci95_high_median_static_rank",
        "mean_q75_static_rank",
        "ci95_low_q75_static_rank",
        "ci95_high_q75_static_rank",
        "mean_mean_static_rank_top_percent",
        "mean_q25_static_rank_top_percent",
        "ci95_low_q25_static_rank_top_percent",
        "ci95_high_q25_static_rank_top_percent",
        "mean_median_static_rank_top_percent",
        "ci95_low_median_static_rank_top_percent",
        "ci95_high_median_static_rank_top_percent",
        "mean_q75_static_rank_top_percent",
        "ci95_low_q75_static_rank_top_percent",
        "ci95_high_q75_static_rank_top_percent",
        "mean_rank_cdf_advantage_over_uniform",
        "ci95_low_rank_cdf_advantage_over_uniform",
        "ci95_high_rank_cdf_advantage_over_uniform",
        "mean_sensor_matched_random_mean_rank_top_percent",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def save_rank_distribution_grid(rows, output_dir: Path) -> None:
    """Median (+ IQR band) within-sensor Static rank of Dynamic Top pathways.

    One panel per model. X axis is the global Dynamic Top cutoff used to
    decide recruitment; Y axis is the recruited effector's **absolute**
    position in its own sensor's frozen Static ranking (1 = the effector
    that sensor's weights favor most). The dashed line is the sensor-matched
    random baseline -- derived per row from the reported random top-percent
    and the sensor's implied subspace count, so it reflects each model's own
    (much larger) candidate pool rather than a shared percentage.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), constrained_layout=True)
    for axis, model in zip(axes.flat, MODEL_ORDER):
        selected = sorted(
            (row for row in rows if row["model_name"] == model),
            key=lambda row: row["dynamic_percent"],
        )
        x = np.asarray([row["dynamic_percent"] for row in selected])
        median = np.asarray([row["mean_median_static_rank"] for row in selected])
        q25 = np.asarray([row["mean_q25_static_rank"] for row in selected])
        q75 = np.asarray([row["mean_q75_static_rank"] for row in selected])
        # n_subspaces is implied by rank / rank_top_percent (both reported per
        # row); the random baseline is sensor_matched_random_top_percent of
        # that same implied pool, so it lands on the correct absolute scale
        # per model instead of a shared ~50% line.
        n_subspaces_implied = np.asarray(
            [
                100.0 * row["mean_median_static_rank"] / row["mean_median_static_rank_top_percent"]
                for row in selected
            ]
        )
        baseline = np.asarray(
            [row["mean_sensor_matched_random_mean_rank_top_percent"] for row in selected]
        ) / 100.0 * n_subspaces_implied
        axis.fill_between(x, q25, q75, color=RANK_DISTRIBUTION_COLOR, alpha=0.18, linewidth=0)
        axis.plot(
            x, median, color=RANK_DISTRIBUTION_COLOR, marker="o", lw=2.0, ms=4,
            label="Median Static rank (IQR band)",
        )
        axis.plot(
            x, baseline, color=RANK_DISTRIBUTION_BASELINE_COLOR, ls="--", lw=1.4,
            label="Sensor-matched random",
        )
        axis.set_title(MODEL_LABELS[model], fontsize=11)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(0.5, float(baseline.max()) * 2.0)
        axis.grid(alpha=0.22, which="both")
        axis.set_xlabel("Global Dynamic Top (%)")
    axes[0, 0].set_ylabel("Within-sensor Static rank\nof recruited effector (#, 1 = best)")
    axes[1, 0].set_ylabel("Within-sensor Static rank\nof recruited effector (#, 1 = best)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=2, frameon=False,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"dynamic_conditioned_static_rank_distribution.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_rank_distribution_cross_model(rows, output_dir: Path) -> None:
    """Cross-model comparison of how non-random the recruited rank is."""
    fig, axis = plt.subplots(figsize=(6.6, 5.0))
    for model in MODEL_ORDER:
        selected = sorted(
            (row for row in rows if row["model_name"] == model),
            key=lambda row: row["dynamic_percent"],
        )
        x = np.asarray([row["dynamic_percent"] for row in selected])
        y = np.asarray([row["mean_rank_cdf_advantage_over_uniform"] for row in selected])
        axis.plot(
            x, y, color=MODEL_COLORS[model], marker="o", lw=2.0, ms=4,
            label=MODEL_LABELS[model],
        )
    axis.axhline(
        0.0, color=RANK_DISTRIBUTION_BASELINE_COLOR, ls="--", lw=1.4,
        label="Random baseline",
    )
    axis.set_xscale("log")
    # 0 is drawn as a real line, not the axis spine: give it headroom below,
    # otherwise the baseline sits invisibly flush against the bottom border.
    axis.set_ylim(-0.05, 1.02)
    axis.set_xlabel("Global Dynamic Top (%)")
    axis.set_ylabel("CDF advantage over uniform\n(1 = fully at Static rank 1; 0 = random)")
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=False, fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"dynamic_conditioned_static_rank_cdf_advantage_cross_model.{suffix}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_csv is not None:
        rows = load_rows(Path(args.summary_csv))
        missing = set(MODEL_ORDER) - {row["model_name"] for row in rows}
        if missing:
            raise ValueError(f"missing models: {sorted(missing)}")
        save_line_figure(rows, output_dir)
        save_heatmap(rows, output_dir)
        save_primary_table(rows, output_dir)
        save_primary_table_figure(rows, output_dir)
        save_primary_latex_table(rows, output_dir)
        save_dynamic_conditioned_static_rank_cdf(rows, output_dir)
        save_dynamic_conditioned_rank_table(rows, output_dir)
        save_dynamic_conditioned_rank_cdf_csv(rows, output_dir)
    if args.rank_distribution_csv is not None:
        rank_rows = load_rank_distribution_rows(Path(args.rank_distribution_csv))
        missing = set(MODEL_ORDER) - {row["model_name"] for row in rank_rows}
        if missing:
            raise ValueError(f"missing models in rank distribution CSV: {sorted(missing)}")
        save_rank_distribution_grid(rank_rows, output_dir)
        save_rank_distribution_cross_model(rank_rows, output_dir)
    if args.summary_csv is None and args.rank_distribution_csv is None:
        raise ValueError("pass --summary_csv and/or --rank_distribution_csv")
    print(f"[done] wrote paper figures and primary table to {output_dir}")


if __name__ == "__main__":
    main()
