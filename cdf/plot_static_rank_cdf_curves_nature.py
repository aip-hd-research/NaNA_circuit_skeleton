"""Nature-style figure: the actual empirical CDF of within-sensor frozen
Static rank among pathways recruited into the global Dynamic Top set.

Consumes the JSON written by ``compute_static_rank_cdf_curves.py``. One panel
per model, x = Static rank percentile within the sensor (linear scale, 0% = the
sensor's most weight-favored effector), y = cumulative fraction of Dynamic
Top pathways at or below that rank. The dashed diagonal is the uniform/random
expectation (a pathway landing at rank r with no relationship to Static order
would put a fraction r/n_subspaces of the mass below r).

Example
-------
python cdf/plot_static_rank_cdf_curves_nature.py \
  --curves_json result/future_recruitment_percent_no_verify/figures/static_rank_cdf_curves.json \
  --output_dir result/future_recruitment_percent_no_verify/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
# Validated categorical order (dataviz skill reference palette, light mode,
# slots 1-3): blue, orange, aqua. These are retained for the common three-curve
# case; larger inputs receive additional colors from a perceptually uniform map.
FRACTION_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
DIAGONAL_COLOR = "#767671"
INK = "#0b0b0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves_json", required=True)
    parser.add_argument("--output_dir", required=True)
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
            "pdf.fonttype": 42,  # embed as editable text, not curves
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _curve_colors(count: int):
    if count <= len(FRACTION_COLORS):
        return FRACTION_COLORS[:count]
    return plt.get_cmap("viridis")(np.linspace(0.08, 0.9, count))


def _validated_cdf(curve, n_subspaces: int, model: str, fraction: str) -> np.ndarray:
    observed = np.asarray(curve["observed_cdf"], dtype=float)
    if observed.shape != (n_subspaces,):
        raise ValueError(
            f"{model} fraction {fraction}: observed_cdf has length "
            f"{observed.size}, expected {n_subspaces}"
        )
    if not np.all(np.isfinite(observed)):
        raise ValueError(f"{model} fraction {fraction}: observed_cdf is not finite")
    if np.any((observed < 0.0) | (observed > 1.0)) or np.any(np.diff(observed) < -1e-12):
        raise ValueError(
            f"{model} fraction {fraction}: observed_cdf must be monotone in [0, 1]"
        )
    if not np.isclose(observed[-1], 1.0):
        raise ValueError(f"{model} fraction {fraction}: observed_cdf must end at 1")
    return observed


def main() -> None:
    args = parse_args()
    _configure_nature_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.curves_json).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    models = payload["models"]
    missing = set(MODEL_ORDER) - set(models)
    if missing:
        raise ValueError(f"missing models in {args.curves_json}: {sorted(missing)}")
    fraction_keys = sorted(
        {frac for model in models.values() for frac in model["curves"]}, key=float
    )
    colors = _curve_colors(len(fraction_keys))

    # 183 mm (double-column) wide, ~50 mm tall: one compact row of panels.
    mm = 1.0 / 25.4
    fig, axes = plt.subplots(
        1, 4, figsize=(183 * mm, 50 * mm), constrained_layout=True, sharey=True
    )
    for panel_index, (axis, model) in enumerate(zip(axes, MODEL_ORDER)):
        entry = models[model]
        n_subspaces = int(entry["n_subspaces"])
        if n_subspaces <= 0:
            raise ValueError(f"{model}: n_subspaces must be positive")
        # Prepend the (0, 0) origin so every curve -- observed and the
        # diagonal -- visibly starts at the same point; without it, the
        # observed curves start partway up the y axis (their first real
        # point is already at rank 100/n_subspaces %) and the "diagonal"
        # baseline has nothing to anchor against on a linear axis.
        rank_percent = np.concatenate(
            [[0.0], 100.0 * (np.arange(1, n_subspaces + 1) / n_subspaces)]
        )
        for fraction_key, color in zip(fraction_keys, colors):
            curve = entry["curves"].get(fraction_key)
            if curve is None:
                continue
            observed = _validated_cdf(curve, n_subspaces, model, fraction_key)
            observed_cdf = np.concatenate([[0.0], observed])
            dynamic_percent = 100.0 * float(fraction_key)
            axis.plot(
                rank_percent, observed_cdf, color=color, lw=1.0, solid_capstyle="round",
                label=f"Dynamic Top {dynamic_percent:g}%",
            )
        # True 45 deg diagonal: on a linear axis this is exactly what "no
        # relationship between Static rank and recruitment" looks like.
        axis.plot(
            [0.0, 100.0], [0.0, 1.0], color=DIAGONAL_COLOR, lw=0.9, ls=(0, (3, 1.5)),
            label="Uniform (random)" if panel_index == 0 else None,
        )
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.0, 1.03)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(MODEL_LABELS[model], fontsize=7.5, pad=3)
        axis.text(
            -0.02, 1.05, "abcd"[panel_index], transform=axis.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="right",
        )
        axis.set_xlabel("Static rank (top %)", labelpad=2)
    axes[0].set_ylabel("Cumulative fraction of\nDynamic Top pathways", labelpad=3)
    handles = [
        plt.Line2D([0], [0], color=color, lw=1.0,
                   label=f"Dynamic Top {100.0 * float(key):g}%")
        for key, color in zip(fraction_keys, colors)
    ]
    handles.append(
        plt.Line2D([0], [0], color=DIAGONAL_COLOR, lw=0.9, ls=(0, (3, 1.5)),
                   label="Uniform (random)")
    )
    labels = [handle.get_label() for handle in handles]
    fig.legend(
        handles, labels, loc="outside upper center", bbox_to_anchor=(0.5, 1.14),
        ncol=4, handlelength=1.6, columnspacing=1.1, frameon=False,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"static_rank_cdf_curves_nature.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"[done] wrote {output_dir / 'static_rank_cdf_curves_nature.pdf'}")


if __name__ == "__main__":
    main()
