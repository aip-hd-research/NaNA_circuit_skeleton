#!/usr/bin/env python3
"""Create the three-panel FP32 GPU SVD scaling figure from benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "openai-community/gpt2": "GPT-2",
    "Qwen/Qwen3-0.6B": "Qwen3",
    "AI4PD/ZymCTRL": "ZymCTRL",
    "google/gemma-3-1b-it": "Gemma-3",
    "meta-llama/Llama-3.2-1B": "Llama-3.2",
    "microsoft/BioGPT-Large": "BioGPT-Large",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5",
    "bigcode/starcoder2-3b": "StarCoder2",
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3 Mini",
    "meta-llama/Llama-2-7b-hf": "Llama-2",
    "mistralai/Mistral-7B-v0.1": "Mistral-v0.1",
    "google/gemma-2-9b-it": "Gemma-2",
}

PARAMETER_LABELS = {
    "openai-community/gpt2": "124M",
    "Qwen/Qwen3-0.6B": "0.6B",
    "AI4PD/ZymCTRL": "1.2B",
    "google/gemma-3-1b-it": "1B",
    "meta-llama/Llama-3.2-1B": "1B",
    "microsoft/BioGPT-Large": "1.5B",
    "Qwen/Qwen2.5-1.5B": "1.5B",
    "bigcode/starcoder2-3b": "3B",
    "microsoft/Phi-3-mini-4k-instruct": "3.8B",
    "meta-llama/Llama-2-7b-hf": "7B",
    "mistralai/Mistral-7B-v0.1": "7B",
    "google/gemma-2-9b-it": "9B",
}

# Original muted teal pairing, with thin dark bar outlines added below.
BLUE =  "#4198AC"
ORANGE = "#BFDFD2"
GREY = "#4D4D4D"


def load_data(root: Path) -> list[dict]:
    models: list[dict] = []
    for path_str in glob.glob(str(root / "*" / "svd_layers.csv")):
        path = Path(path_str)
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        model = rows[0]["model_name"]
        entry = {"model": model, "name": DISPLAY_NAMES.get(model, model.split("/")[-1])}
        for component in ("mlp", "attention"):
            comp = [r for r in rows if r["component"] == component and int(r["repeat"]) == 1]
            if not comp:
                raise ValueError(f"No repeat-1 {component} rows in {path}")
            entry[f"{component}_time"] = sum(float(r["time_s"]) for r in comp)
            entry[f"{component}_work"] = sum(float(r["svd_work_units"]) for r in comp)
            entry[f"{component}_layer_params"] = np.array([
                int(r["batch_count"]) * int(r["m"]) * int(r["n"]) for r in comp
            ], dtype=float)
            entry[f"{component}_layer_times"] = np.array(
                [float(r["time_s"]) for r in comp], dtype=float
            )
            # Layers run sequentially, so the model-level requirement is the maximum.
            # Source values are MiB (bytes / 2**20); convert to decimal GB.
            entry[f"{component}_memory_gb"] = (
                max(float(r["peak_memory_mib"]) for r in comp) * 2**20 / 1e9
            )
        entry["total_time"] = entry["mlp_time"] + entry["attention_time"]
        entry["total_work"] = entry["mlp_work"] + entry["attention_work"]
        models.append(entry)
    if not models:
        raise FileNotFoundError(f"No */svd_layers.csv files found below {root}")
    # Workload gives a reproducible architecture-size ordering without relying
    # on rounded parameter counts embedded in model names.
    return sorted(models, key=lambda d: d["total_work"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="energy_time/results_a100short/a100_fp32_svd")
    parser.add_argument("--output-dir", default="energy_time/figures")
    parser.add_argument("--basename", default="svd_scaling_ab")
    parser.add_argument("--panel-width", type=float, default=5.7)
    parser.add_argument("--panel-height", type=float, default=4)
    parser.add_argument("--axis-label-size", type=float, default=12.0)
    parser.add_argument("--x-tick-size", type=float, default=12)
    parser.add_argument("--y-tick-size", type=float, default=12)
    parser.add_argument("--legend-size", type=float, default=12)
    parser.add_argument("--panel-label-size", type=float, default=13.0)
    parser.add_argument("--bar-annotation-size", type=float, default=10.0,
                        help="Font size of parameter counts above bars")
    parser.add_argument("--bar-width", type=float, default=0.72)
    parser.add_argument("--bar-edge-width", type=float, default=0.45)
    parser.add_argument("--model-label-rotation", type=float, default=55.0)
    parser.add_argument("--mlp-color", default=BLUE)
    parser.add_argument("--attention-color", default=ORANGE)
    args = parser.parse_args()

    data = load_data(Path(args.input_dir))
    names = [d["name"] for d in data]
    parameter_labels = [PARAMETER_LABELS.get(d["model"], "") for d in data]
    x = np.arange(len(data))
    mlp_t = np.array([d["mlp_time"] for d in data])
    att_t = np.array([d["attention_time"] for d in data])
    mlp_mem = np.array([d["mlp_memory_gb"] for d in data])
    att_mem = np.array([d["attention_memory_gb"] for d in data])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": args.axis_label_size,
        "axes.linewidth": 0.8,
        "xtick.labelsize": args.x_tick_size,
        "ytick.labelsize": args.y_tick_size,
        "legend.fontsize": args.legend_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * args.panel_width, args.panel_height), constrained_layout=True
    )

    # a: component times sum to the full model SVD runtime.
    bar_edge = "#222222"
    axes[0].bar(x, mlp_t, width=args.bar_width, color=args.mlp_color,
                edgecolor=bar_edge, linewidth=args.bar_edge_width, label="MLP")
    axes[0].bar(x, att_t, width=args.bar_width, bottom=mlp_t,
                color=args.attention_color, edgecolor=bar_edge,
                linewidth=args.bar_edge_width, label="Attention")
    axes[0].set_ylabel("Cumulative runtime (s)")
    axes[0].set_xticks(x, names, rotation=args.model_label_rotation, ha="right")
    axes[0].legend(frameon=False, loc="upper left")
    total_t = mlp_t + att_t
    time_pad = max(total_t) * 0.012
    for xpos, height, text in zip(x, total_t, parameter_labels):
        axes[0].text(xpos, height + time_pad, text, ha="center", va="bottom",
                     fontsize=args.bar_annotation_size, color="#222222")
    axes[0].set_ylim(0, max(total_t) * 1.09)

    # b: grouped, not stacked—these are sequential peak memory requirements.
    width = args.bar_width / 2
    axes[1].bar(x - width / 2, mlp_mem, width=width,
                color=args.mlp_color, edgecolor=bar_edge,
                linewidth=args.bar_edge_width, label="MLP")
    axes[1].bar(x + width / 2, att_mem, width=width,
                color=args.attention_color, edgecolor=bar_edge,
                linewidth=args.bar_edge_width, label="Attention")
    axes[1].set_ylabel("Peak GPU memory (GB)")
    axes[1].set_xticks(x, names, rotation=args.model_label_rotation, ha="right")
    axes[1].legend(frameon=False, loc="upper left")
    memory_top = np.maximum(mlp_mem, att_mem)
    memory_pad = max(memory_top) * 0.012
    for xpos, height, text in zip(x - width / 2, mlp_mem, parameter_labels):
        axes[1].text(xpos, height + memory_pad, text, ha="center", va="bottom",
                     fontsize=args.bar_annotation_size, color="#222222")
    axes[1].set_ylim(0, max(memory_top) * 1.09)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
        ax.grid(False)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        target = out / f"{args.basename}.{suffix}"
        fig.savefig(target, bbox_inches="tight", facecolor="white", **kwargs)
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
