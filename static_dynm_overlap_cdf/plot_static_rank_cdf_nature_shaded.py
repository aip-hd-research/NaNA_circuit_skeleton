"""Refined publication figure for the original Static-rank ECDF.

Uses a balanced 2x2 layout and subtle line shadows instead of filled areas.
The shadows are only a visual separation aid, not confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


DEFAULT_MODELS = (
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
}
# Colorblind-safe, print-friendly palette (dark blue -> vermillion).
COLORS = ("#332288", "#44AA99", "#E69F00", "#CC3311", "#AA4499")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nature-style shaded Static-rank ECDF")
    parser.add_argument("--curves_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--dynamic_percents", nargs="+", type=float,
                        default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--shadow_alpha", type=float, default=0.16,
                        help="subtle line-shadow opacity (not a confidence band)")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.65,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def fraction_key(curves: dict, percent: float) -> str:
    for key, curve in curves.items():
        actual = float(curve.get("dynamic_k_percent", 100 * float(key)))
        if np.isclose(actual, percent):
            return key
    raise ValueError(f"Dynamic Top {percent:g}% is missing")


def validated_cdf(curve: dict, n: int, model: str) -> np.ndarray:
    y = np.asarray(curve.get("observed_cdf"), dtype=float)
    if y.shape != (n,) or not np.all(np.isfinite(y)):
        raise ValueError(f"{model}: malformed observed_cdf")
    if np.any(np.diff(y) < -1e-12) or np.any((y < 0) | (y > 1)):
        raise ValueError(f"{model}: observed_cdf is not a valid CDF")
    if not np.isclose(y[-1], 1):
        raise ValueError(f"{model}: observed_cdf does not end at 1")
    return y


def main() -> None:
    args = parse_args()
    if not 0 <= args.shadow_alpha <= 1:
        raise ValueError("--shadow_alpha must be between 0 and 1")
    configure_style()
    with Path(args.curves_json).open(encoding="utf-8") as handle:
        models = json.load(handle).get("models", {})
    missing = [model for model in args.models if model not in models]
    if missing:
        raise ValueError(f"models missing from JSON: {missing}")
    if len(args.models) != 4:
        raise ValueError("this 2x2 figure requires exactly four models")

    # A 2x2 layout gives each panel a readable aspect ratio at 180 mm width.
    mm = 1 / 25.4
    fig, axes = plt.subplots(
        2, 2, figsize=(180 * mm, 126 * mm), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    axes = axes.ravel()
    legend_handles = []

    for panel, (axis, model) in enumerate(zip(axes, args.models)):
        entry = models[model]
        n = int(entry.get("n_subspaces", 0))
        if n <= 0:
            raise ValueError(f"{model}: n_subspaces must be positive")
        x = np.arange(1, n + 1, dtype=float) / n
        xs = np.concatenate(([0.0], x))
        for index, percent in enumerate(args.dynamic_percents):
            key = fraction_key(entry.get("curves", {}), percent)
            y = validated_cdf(entry["curves"][key], n, model)
            ys = np.concatenate(([0.0], y))
            color = COLORS[index] if index < len(COLORS) else plt.cm.viridis(index / len(args.dynamic_percents))
            (line,) = axis.step(
                100 * xs, ys, where="post", color=color, lw=1.45,
                label=f"DC Top {percent:g}%", zorder=3,
            )
            # A small print-safe drop shadow separates overlapping curves.  It
            # is a graphical effect, not an uncertainty interval.
            line.set_path_effects([
                pe.SimpleLineShadow(offset=(0.55, -0.55),
                                    alpha=args.shadow_alpha, rho=0.75),
                pe.Normal(),
            ])
            if panel == 0:
                legend_handles.append(line)

        (null_line,) = axis.plot(
            [0, 100], [0, 1], color="#707070", lw=0.85,
            ls=(0, (3.5, 2.2)), label="Uniform null", zorder=2,
        )
        if panel == 0:
            legend_handles.append(null_line)
        axis.set_title(MODEL_LABELS.get(model, model.rsplit("/", 1)[-1]), pad=5,
                       fontweight="semibold")
        axis.text(-0.075, 1.025, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=10, fontweight="bold", va="bottom", ha="right")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 1.015)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", lw=0.45, zorder=0)

    for axis in axes[-2:]:
        axis.set_xlabel("Static Top-k threshold (%)", labelpad=3)
    axes[0].set_ylabel("Cumulative fraction", labelpad=4)
    axes[2].set_ylabel("Cumulative fraction", labelpad=4)
    fig.legend(
        handles=legend_handles, loc="outside lower center",
        ncol=len(legend_handles), handlelength=2.2, columnspacing=1.4,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "static_rank_cdf_nature_clean"
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {stem}.pdf/.png/.svg")


if __name__ == "__main__":
    main()
