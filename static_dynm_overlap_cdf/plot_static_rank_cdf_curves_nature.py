"""Plot empirical Static-rank CDFs produced by the companion compute script.

Each panel is one model. The x axis is the within-sensor Static rank
percentile of pathways recruited into a signed global Dynamic Top set; the y
axis is their empirical cumulative fraction. The dashed diagonal is the
uniform-rank baseline.

Example
-------
python static_dynm_overlap_cdf/plot_static_rank_cdf_curves_nature.py \
  --curves_json result/static_dynamic_overlap_cdf/static_rank_cdf_curves.json \
  --output_dir result/static_dynamic_overlap_cdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREFERRED_MODEL_ORDER = (
    "openai-community/gpt2",
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B",
    "mistralai/Mistral-7B-v0.3",
)
MODEL_LABELS = {
    "openai-community/gpt2": "GPT-2",
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B": "LLaMA3-1B",
    "mistralai/Mistral-7B-v0.3": "Mistral v0.3",
    "mistralai/Mistral-7B-v0.1": "Mistral v0.1",
    "jonghyunlee/MolGPT_pretrained-by-ZINC15": "MolGPT",
    "jonghyunlee/MolGPT_long_context_pretrained-by-ZINC15": "MolGPT-Long",
}
FRACTION_COLORS = (
    "#3C5488",
    "#4DBBD5",
    "#00A087",
    "#E64B35",
    "#F39B7F",
)
DIAGONAL_COLOR = "#767671"
INK = "#0b0b0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nature-style panels for empirical Static-rank CDF curves."
    )
    parser.add_argument("--curves_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--combine",
        action="store_true",
        help="draw the selected models as one row with a shared y axis",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="optional model subset/order; by default plot every model in the JSON",
    )
    parser.add_argument(
        "--dynamic_percents",
        nargs="+",
        type=float,
        help="DC Top percentages to plot, in the requested display order",
    )
    parser.add_argument(
        "--fig_width_mm",
        type=float,
        default=89.0,
        help="width of each model figure in millimetres (default: 89)",
    )
    parser.add_argument(
        "--fig_height_mm",
        type=float,
        default=70.0,
        help="height of each model figure in millimetres (default: 70)",
    )
    parser.add_argument(
        "--xlabel_fontsize",
        type=float,
        default=7.5,
        help="x-axis label font size in points (default: 7.5)",
    )
    parser.add_argument(
        "--ylabel_fontsize",
        type=float,
        default=7.5,
        help="y-axis label font size in points (default: 7.5)",
    )
    parser.add_argument(
        "--xtick_fontsize",
        type=float,
        default=6.5,
        help="x-axis tick-label font size in points (default: 6.5)",
    )
    parser.add_argument(
        "--ytick_fontsize",
        type=float,
        default=6.5,
        help="y-axis tick-label font size in points (default: 6.5)",
    )
    parser.add_argument(
        "--legend_fontsize",
        type=float,
        default=6.5,
        help="legend font size in points (default: 6.5)",
    )
    parser.add_argument(
        "--title_fontsize",
        type=float,
        default=7.5,
        help="model title font size in points (default: 7.5)",
    )
    parser.add_argument(
        "--combined_legend_height_mm",
        type=float,
        default=12.0,
        help="extra height reserved below combined panels for the legend",
    )
    return parser.parse_args()


def _configure_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _model_order(models: dict, requested: list[str] | None) -> list[str]:
    if not models:
        raise ValueError("curves JSON contains no models")
    if requested:
        missing = [model for model in requested if model not in models]
        if missing:
            raise ValueError(f"requested models missing from curves JSON: {missing}")
        return list(dict.fromkeys(requested))
    preferred = [model for model in PREFERRED_MODEL_ORDER if model in models]
    remaining = [model for model in models if model not in preferred]
    return preferred + remaining


def _curve_colors(count: int):
    if count <= len(FRACTION_COLORS):
        return FRACTION_COLORS[:count]
    return plt.get_cmap("viridis")(np.linspace(0.08, 0.9, count))


def _validated_cdf(
    curve: dict, n_subspaces: int, model: str, fraction_key: str
) -> np.ndarray:
    observed = np.asarray(curve.get("observed_cdf"), dtype=float)
    if observed.shape != (n_subspaces,):
        raise ValueError(
            f"{model}, Dynamic fraction {fraction_key}: observed_cdf has "
            f"length {observed.size}, expected {n_subspaces}"
        )
    if not np.all(np.isfinite(observed)):
        raise ValueError(f"{model}, Dynamic fraction {fraction_key}: non-finite CDF")
    if np.any(observed < 0.0) or np.any(observed > 1.0):
        raise ValueError(f"{model}, Dynamic fraction {fraction_key}: CDF outside [0, 1]")
    if np.any(np.diff(observed) < -1e-12):
        raise ValueError(f"{model}, Dynamic fraction {fraction_key}: CDF is not monotone")
    if not np.isclose(observed[-1], 1.0):
        raise ValueError(f"{model}, Dynamic fraction {fraction_key}: CDF must end at 1")
    return observed


def _dynamic_percent(curve: dict, fraction_key: str) -> float:
    return float(curve.get("dynamic_k_percent", 100.0 * float(fraction_key)))


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.rsplit("/", 1)[-1])


def _safe_model_slug(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def main() -> None:
    args = parse_args()
    if any(
        value <= 0
        for value in (
            args.fig_width_mm,
            args.fig_height_mm,
            args.xlabel_fontsize,
            args.ylabel_fontsize,
            args.xtick_fontsize,
            args.ytick_fontsize,
            args.legend_fontsize,
            args.title_fontsize,
            args.combined_legend_height_mm,
        )
    ):
        raise ValueError("figure dimensions and axis-label font sizes must be positive")
    _configure_nature_style()

    curves_path = Path(args.curves_json)
    with curves_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"{curves_path}: expected a top-level 'models' object")
    model_order = _model_order(models, args.models)
    all_fraction_keys = sorted(
        {
            fraction
            for model in model_order
            for fraction in models[model].get("curves", {})
        },
        key=float,
    )
    if not all_fraction_keys:
        raise ValueError(f"{curves_path}: selected models contain no curves")

    if args.dynamic_percents:
        fraction_keys = []
        missing_percents = []
        for requested_percent in args.dynamic_percents:
            matched_key = next(
                (
                    key
                    for key in all_fraction_keys
                    if any(
                        key in models[model].get("curves", {})
                        and np.isclose(
                            _dynamic_percent(models[model]["curves"][key], key),
                            requested_percent,
                        )
                        for model in model_order
                    )
                ),
                None,
            )
            if matched_key is None:
                missing_percents.append(requested_percent)
            elif matched_key not in fraction_keys:
                fraction_keys.append(matched_key)
        if missing_percents:
            raise ValueError(
                f"requested DC Top percentages missing from curves JSON: "
                f"{missing_percents}"
            )
    else:
        fraction_keys = all_fraction_keys

    colors = _curve_colors(len(fraction_keys))
    mm = 1.0 / 25.4
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.combine:
        fig, axes = plt.subplots(
            1,
            len(model_order),
            figsize=(
                args.fig_width_mm * len(model_order) * mm,
                (args.fig_height_mm + args.combined_legend_height_mm) * mm,
            ),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes.ravel()
        legend_handles = []
        for axis, model in zip(axes, model_order):
            entry = models[model]
            n_subspaces = int(entry.get("n_subspaces", 0))
            if n_subspaces <= 0:
                raise ValueError(f"{model}: n_subspaces must be positive")
            rank_percent = np.concatenate(
                [[0.0], 100.0 * np.arange(1, n_subspaces + 1) / n_subspaces]
            )
            model_handles = []
            for fraction_key, color in zip(fraction_keys, colors):
                curve = entry.get("curves", {}).get(fraction_key)
                if curve is None:
                    continue
                observed = _validated_cdf(curve, n_subspaces, model, fraction_key)
                dynamic_percent = _dynamic_percent(curve, fraction_key)
                (handle,) = axis.step(
                    rank_percent,
                    np.concatenate([[0.0], observed]),
                    where="post",
                    color=color,
                    lw=1.0,
                    solid_capstyle="round",
                    label=f"DC Top {dynamic_percent:g}%",
                )
                model_handles.append(handle)
            (uniform_handle,) = axis.plot(
                [0.0, 100.0],
                [0.0, 1.0],
                color=DIAGONAL_COLOR,
                lw=0.9,
                ls=(0, (3, 1.5)),
                label="Uniform (random)",
            )
            model_handles.append(uniform_handle)
            if not legend_handles:
                legend_handles = model_handles
            axis.set_xlim(0.0, 100.0)
            axis.set_ylim(0.0, 1.03)
            axis.set_xticks([0, 20, 40, 60, 80, 100])
            axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            axis.tick_params(axis="x", labelsize=args.xtick_fontsize)
            axis.tick_params(
                axis="y",
                labelsize=args.ytick_fontsize,
                labelleft=True,
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(_model_label(model), pad=3, fontsize=args.title_fontsize)
            axis.set_xlabel(
                "Static Top-k threshold (%)",
                fontsize=args.xlabel_fontsize,
                labelpad=2,
            )
        fig.supylabel(
            "Empirical CDF",
            fontsize=args.ylabel_fontsize,
        )
        fig.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncol=len(legend_handles),
            handlelength=1.7,
            columnspacing=1.2,
            fontsize=args.legend_fontsize,
            frameon=False,
        )
        output_stem = output_dir / "static_rank_cdf_four_models"
        for suffix in ("pdf", "png"):
            fig.savefig(
                output_stem.with_suffix(f".{suffix}"),
                dpi=600,
            )
        plt.close(fig)
        print(f"[done] wrote {output_stem.with_suffix('.pdf')}")
        return

    for model in model_order:
        fig, axis = plt.subplots(
            figsize=(args.fig_width_mm * mm, args.fig_height_mm * mm),
            constrained_layout=True,
        )
        entry = models[model]
        n_subspaces = int(entry.get("n_subspaces", 0))
        if n_subspaces <= 0:
            raise ValueError(f"{model}: n_subspaces must be positive")
        rank_percent = np.concatenate(
            [[0.0], 100.0 * np.arange(1, n_subspaces + 1) / n_subspaces]
        )

        handles = []
        for fraction_key, color in zip(fraction_keys, colors):
            curve = entry.get("curves", {}).get(fraction_key)
            if curve is None:
                continue
            observed = _validated_cdf(curve, n_subspaces, model, fraction_key)
            dynamic_percent = _dynamic_percent(curve, fraction_key)
            (handle,) = axis.step(
                rank_percent,
                np.concatenate([[0.0], observed]),
                where="post",
                color=color,
                lw=1.0,
                solid_capstyle="round",
                label=f"DC Top {dynamic_percent:g}%",
            )
            handles.append(handle)

        (uniform_handle,) = axis.plot(
            [0.0, 100.0],
            [0.0, 1.0],
            color=DIAGONAL_COLOR,
            lw=0.9,
            ls=(0, (3, 1.5)),
            label="Uniform (random)",
        )
        handles.append(uniform_handle)
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.0, 1.03)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.tick_params(axis="x", labelsize=args.xtick_fontsize)
        axis.tick_params(axis="y", labelsize=args.ytick_fontsize)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(_model_label(model), pad=3, fontsize=args.title_fontsize)
        axis.set_xlabel(
            "Static Top-k threshold (%)",
            fontsize=args.xlabel_fontsize,
            labelpad=2,
        )
        axis.set_ylabel(
            "Empirical CDF",
            fontsize=args.ylabel_fontsize,
            labelpad=3,
        )
        axis.legend(
            handles=handles,
            loc="lower right",
            handlelength=1.7,
            fontsize=args.legend_fontsize,
            frameon=False,
        )

        output_stem = output_dir / f"{_safe_model_slug(model)}_static_rank_cdf"
        for suffix in ("pdf", "png"):
            fig.savefig(
                Path(f"{output_stem}.{suffix}"),
                dpi=600,
                bbox_inches="tight",
            )
        plt.close(fig)
        print(f"[done] wrote {output_stem}.pdf")


if __name__ == "__main__":
    main()
