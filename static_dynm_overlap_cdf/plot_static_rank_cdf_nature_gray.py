"""Original ECDF design with a single neutral-grey enrichment region.

Writes one four-model overview and one standalone figure per model.  The grey
area is the region between the uniform null and the pointwise lowest displayed
CDF (the enrichment shared by all displayed Dynamic Top settings).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = (
    "openai-community/gpt2", "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B", "mistralai/Mistral-7B-v0.3",
)
LABELS = {
    "openai-community/gpt2": "GPT-2", "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B": "LLaMA3-1B",
    "mistralai/Mistral-7B-v0.3": "Mistral v0.3",
}
COLORS = ("#3C5488", "#4DBBD5", "#00A087", "#E64B35", "#8491B4")


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nature-style ECDF with grey shading")
    p.add_argument("--curves_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--dynamic_percents", nargs="+", type=float,
                   default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument("--shade_alpha", type=float, default=0.32)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "axes.linewidth": 0.65, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "legend.fontsize": 7, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def get_curves(entry: dict, model: str, percents: list[float]):
    n = int(entry["n_subspaces"])
    curves = entry.get("curves", {})
    result = []
    for percent in percents:
        key = next((k for k, v in curves.items() if np.isclose(
            float(v.get("dynamic_k_percent", 100 * float(k))), percent)), None)
        if key is None:
            raise ValueError(f"{model}: Dynamic Top {percent:g}% is missing")
        y = np.asarray(curves[key]["observed_cdf"], dtype=float)
        if y.shape != (n,) or np.any(np.diff(y) < -1e-12):
            raise ValueError(f"{model}: invalid CDF for Dynamic Top {percent:g}%")
        result.append((percent, y))
    x = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    return x, result


def draw(axis, model: str, entry: dict, percents: list[float], shade_alpha: float):
    x, curves = get_curves(entry, model, percents)
    ys = [np.concatenate(([0.0], y)) for _, y in curves]
    common_lower = np.maximum(x, np.min(np.vstack(ys), axis=0))
    axis.fill_between(
        100 * x, x, common_lower, step="post", color="#D6D6D6",
        alpha=shade_alpha, linewidth=0, zorder=1,
    )
    handles = []
    for index, ((percent, _), y) in enumerate(zip(curves, ys)):
        color = COLORS[index] if index < len(COLORS) else plt.cm.viridis(index / len(ys))
        (line,) = axis.step(100 * x, y, where="post", color=color, lw=1.25,
                            label=f"DC Top {percent:g}%", zorder=3)
        handles.append(line)
    (null,) = axis.plot([0, 100], [0, 1], color="#777777", lw=0.85,
                        ls=(0, (3.2, 2)), label="Uniform null", zorder=2)
    handles.append(null)
    axis.set(xlim=(0, 100), ylim=(0, 1.02),
             xticks=[0, 25, 50, 75, 100], yticks=[0, .25, .5, .75, 1])
    axis.set_title(LABELS.get(model, model.rsplit("/", 1)[-1]), pad=4)
    axis.set_xlabel("Static Top-k threshold (%)", labelpad=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return handles


def save(fig, stem: Path, dpi: int) -> None:
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(Path(f"{stem}.{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    if not 0 <= args.shade_alpha <= 1:
        raise ValueError("--shade_alpha must be between 0 and 1")
    style()
    with Path(args.curves_json).open(encoding="utf-8") as f:
        models = json.load(f).get("models", {})
    missing = [m for m in args.models if m not in models]
    if missing:
        raise ValueError(f"models missing from JSON: {missing}")
    out = Path(args.output_dir)
    individual = out / "individual"
    individual.mkdir(parents=True, exist_ok=True)
    mm = 1 / 25.4

    # Standalone, single-column figures without panel letters.
    for model in args.models:
        fig, ax = plt.subplots(figsize=(89 * mm, 72 * mm), constrained_layout=True)
        handles = draw(ax, model, models[model], args.dynamic_percents, args.shade_alpha)
        ax.set_ylabel("Cumulative fraction", labelpad=4)
        ax.legend(handles=handles, loc="lower right", handlelength=2.0)
        slug = model.replace("/", "_").replace(" ", "_")
        save(fig, individual / f"{slug}_static_rank_cdf", args.dpi)

    # Compact full-width overview, also without panel letters.
    fig, axes = plt.subplots(1, len(args.models), figsize=(180 * mm, 57 * mm),
                             sharex=True, sharey=True, constrained_layout=True,
                             squeeze=False)
    handles = None
    for ax, model in zip(axes.ravel(), args.models):
        handles = draw(ax, model, models[model], args.dynamic_percents, args.shade_alpha)
    axes.ravel()[0].set_ylabel("Cumulative fraction", labelpad=4)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               handlelength=2.0, columnspacing=1.25)
    save(fig, out / "static_rank_cdf_four_models_gray", args.dpi)
    print(f"[done] wrote overview to {out}")
    print(f"[done] wrote standalone figures to {individual}")


if __name__ == "__main__":
    main()
