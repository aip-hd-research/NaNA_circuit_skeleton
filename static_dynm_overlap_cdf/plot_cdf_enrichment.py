"""Publication-style ECDF and enrichment plots for Static/Dynamic overlap.

The left column shows the conventional empirical CDF.  The right column
shows the excess over the uniform null, F(x) - x.  Positive values mean that
Dynamic-Top pathways are enriched near the best Static ranks.

Example:
    python static_dynm_overlap_cdf/plot_cdf_enrichment.py \
      --curves_json result_CDF/static_rank_cdf_curves.json \
      --output_dir result_CDF/figures_enrichment \
      --models openai-community/gpt2 Qwen/Qwen3-0.6B \
               meta-llama/Llama-3.2-1B mistralai/Mistral-7B-v0.3 \
      --dynamic_percents 0.001 0.01 0.1 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
COLORS = ("#3C5488", "#4DBBD5", "#00A087", "#E64B35", "#7E6148")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ECDFs together with CDF enrichment over a uniform null."
    )
    parser.add_argument("--curves_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", help="model subset and display order")
    parser.add_argument("--dynamic_percents", nargs="+", type=float)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.rsplit("/", 1)[-1])


def select_fraction_keys(
    entry: dict, requested_percents: list[float] | None
) -> list[str]:
    curves = entry.get("curves", {})
    keys = sorted(curves, key=float)
    if not requested_percents:
        return keys
    selected = []
    for percent in requested_percents:
        match = next(
            (
                key
                for key in keys
                if np.isclose(
                    float(curves[key].get("dynamic_k_percent", 100 * float(key))),
                    percent,
                )
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Dynamic Top {percent:g}% is missing")
        selected.append(match)
    return list(dict.fromkeys(selected))


def validate_curve(model: str, key: str, curve: dict, n: int) -> np.ndarray:
    cdf = np.asarray(curve.get("observed_cdf"), dtype=float)
    if cdf.shape != (n,):
        raise ValueError(f"{model}, fraction {key}: expected {n} CDF points")
    if not np.all(np.isfinite(cdf)) or np.any(np.diff(cdf) < -1e-12):
        raise ValueError(f"{model}, fraction {key}: invalid empirical CDF")
    if np.any((cdf < 0) | (cdf > 1)) or not np.isclose(cdf[-1], 1):
        raise ValueError(f"{model}, fraction {key}: CDF must be in [0,1] and end at 1")
    return cdf


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    configure_style()

    with Path(args.curves_json).open(encoding="utf-8") as handle:
        models = json.load(handle).get("models", {})
    if not models:
        raise ValueError("curves JSON contains no models")

    requested_models = args.models or [m for m in DEFAULT_MODELS if m in models]
    if not requested_models:
        requested_models = list(models)
    missing = [m for m in requested_models if m not in models]
    if missing:
        raise ValueError(f"models missing from JSON: {missing}")

    nrows = len(requested_models)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(7.2, max(2.0, 1.72 * nrows)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    metric_rows = []
    legend_handles = None

    for row, model in enumerate(requested_models):
        entry = models[model]
        n = int(entry.get("n_subspaces", 0))
        if n <= 0:
            raise ValueError(f"{model}: n_subspaces must be positive")
        x = np.arange(1, n + 1, dtype=float) / n
        x_step = np.concatenate(([0.0], x))
        keys = select_fraction_keys(entry, args.dynamic_percents)

        for index, key in enumerate(keys):
            curve = entry["curves"][key]
            cdf = validate_curve(model, key, curve, n)
            percent = float(curve.get("dynamic_k_percent", 100 * float(key)))
            gain = cdf - x
            color = COLORS[index] if index < len(COLORS) else plt.cm.viridis(index / len(keys))
            label = f"DC Top {percent:g}%"
            axes[row, 0].step(
                100 * x_step, np.concatenate(([0.0], cdf)), where="post",
                color=color, lw=1.25, label=label,
            )
            axes[row, 1].step(
                100 * x_step, np.concatenate(([0.0], gain)), where="post",
                color=color, lw=1.25, label=label,
            )
            ks_index = int(np.argmax(gain))
            # Integral gain is normalized by its theoretical maximum, 0.5.
            normalized_auc = 2.0 * float(np.trapz(gain, x))
            metric_rows.append(
                {
                    "model": model,
                    "dynamic_top_percent": percent,
                    "ks_gain": float(gain[ks_index]),
                    "ks_rank_percent": float(100 * x[ks_index]),
                    "normalized_auc_gain": normalized_auc,
                }
            )

        left, right = axes[row]
        left.plot([0, 100], [0, 1], color="#777777", lw=0.9,
                  ls=(0, (3, 2)), label="Uniform null")
        right.axhline(0, color="#777777", lw=0.9, ls=(0, (3, 2)))
        left.set_ylim(0, 1.03)
        max_gain = max((line.get_ydata().max() for line in right.lines[:-1]), default=0.1)
        right.set_ylim(min(-0.02, -0.06 * max_gain), max(0.05, 1.08 * max_gain))
        left.set_ylabel(f"{model_label(model)}\nEmpirical CDF")
        right.set_ylabel(r"CDF gain  $F(x)-x$")
        for axis in (left, right):
            axis.set_xlim(0, 100)
            axis.set_xticks([0, 20, 40, 60, 80, 100])
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.65)
        if legend_handles is None:
            legend_handles = left.get_legend_handles_labels()

    axes[0, 0].set_title("A  Empirical cumulative distribution", loc="left", weight="bold")
    axes[0, 1].set_title("B  Enrichment over uniform ranks", loc="left", weight="bold")
    axes[-1, 0].set_xlabel("Static Top-k threshold (%)")
    axes[-1, 1].set_xlabel("Static Top-k threshold (%)")
    if legend_handles:
        fig.legend(*legend_handles, loc="outside lower center", ncol=len(legend_handles[0]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "static_rank_ecdf_enrichment"
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    metrics_path = output_dir / "static_rank_enrichment_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"[done] wrote {stem}.pdf/.png/.svg")
    print(f"[done] wrote {metrics_path}")


if __name__ == "__main__":
    main()
