"""
Generate per-model per-target plots from static skeleton experiment results.

For each (model, target) pair:
  - One figure with 3 subplots: mean_rank | mean_prob | mean_logit
  - X-axis: k_percent
  - Each subplot has 5 lines, one per evaluation condition

Usage:
  python scripts_ske_cir/plot_static_ske.py
  python scripts_ske_cir/plot_static_ske.py --results_dir results/static_ske --output_dir figures/static_ske
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt

from model_config import PROMPT_FORMAT_VERSION, safe_slug
from result_summary import load_per_neuron_means_for_plot, load_static_means_for_plot

# static_skeleton.py's run_static_experiment output.
CONDITIONS = ["Original", "Ske.preserve", "Ske.ablation", "Random", "Anti-align.pre"]

# run_static_ske_per_neuron.py's output (COND_DISPLAY there) — a completely different
# label vocabulary sharing the same mean_rank/mean_prob/mean_logit summary-row schema.
# Selected by x_key ("k_per_neuron" only ever comes from this pipeline) in plot_one.
PER_NEURON_CONDITIONS = [
    "Original",
    "PerNeuron.top(gate)", "PerNeuron.rnd(gate)", "PerNeuron.mid(gate)",
    "PerNeuron.bot(gate)",
    "PerNeuron.top_eff(gate)", "PerNeuron.rnd_eff(gate)", "PerNeuron.bot_eff(gate)",
    "PerNeuron.top(nogate)", "PerNeuron.rnd(nogate)", "PerNeuron.bot(nogate)",
    "PerNeuron.top_abla(gate)", "PerNeuron.top_abla(nogate)",
]

METRICS = [
    ("mean_rank",  "Mean Rank (lower = better)"),
    ("mean_prob",  "Mean Prob"),
    ("mean_logit", "Mean Logit"),
]


def _eff_mode_from(d: dict) -> str:
    """Extract the effector label; magnitude restoration is model-internal policy."""
    return d.get("effector_mode", "raw_dot")


def load_all(results_dir: str) -> dict:
    grouped: dict[tuple, list] = {}
    covered_runs: set[tuple[str, str]] = set()
    result_family = os.path.basename(os.path.normpath(results_dir))
    results_root = os.path.dirname(os.path.normpath(results_dir))
    if result_family == "static_ske":
        mean_data, mean_stats = load_static_means_for_plot(
            results_root, PROMPT_FORMAT_VERSION
        )
        if mean_data:
            for target, model_dict in mean_data.items():
                for model, mode_dict in model_dict.items():
                    covered_runs.add((model, ""))
                    for eff_mode, entries in mode_dict.items():
                        grouped[(model, "", target, eff_mode, "k_percent")] = [
                            {"x_val": x, "x_key": "k_percent", "by_label": row}
                            for x, row in entries
                        ]
            print(f"Using {len(mean_stats)} complete model mean summaries.")
    elif result_family == "static_ske_per_neuron":
        mean_data, mean_stats = load_per_neuron_means_for_plot(
            results_root, PROMPT_FORMAT_VERSION
        )
        if mean_data:
            for target, model_dict in mean_data.items():
                for model, seq_dict in model_dict.items():
                    for seq_type, mode_dict in seq_dict.items():
                        covered_runs.add((model, seq_type))
                        for eff_mode, entries in mode_dict.items():
                            grouped[(model, seq_type, target, eff_mode, "k_per_neuron")] = [
                                {"x_val": x, "x_key": "k_per_neuron", "by_label": row}
                                for x, row in entries
                            ]
            print(f"Using {len(mean_stats)} complete model mean summaries.")

    # Backward-compatible raw-file fallback.
    loaded_versions: dict[tuple, int] = {}
    # static_skeleton.py saves to <results_dir>/<safe_model>/*.json (2 levels) but
    # run_static_ske_per_neuron.py saves to <results_dir>/<safe_model>/<seq_type>/*.json
    # (3 levels) — recurse instead of assuming a fixed depth so both are found.
    for fpath in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        with open(fpath) as f:
            d = json.load(f)
        if d.get("prompt_format_version") != PROMPT_FORMAT_VERSION:
            continue
        # static_skeleton.py stores "k_percent"; run_static_ske_per_neuron.py stores
        # "k_per_neuron" (an absolute count, not a percentage) under a different key.
        x_val, x_key = d.get("k_percent"), "k_percent"
        if x_val is None:
            x_val, x_key = d.get("k_per_neuron"), "k_per_neuron"
        if x_val is None:
            continue
        eff_mode = _eff_mode_from(d)
        # run_static_ske_per_neuron.py doesn't store seq_type (short/long/random) in the
        # JSON payload at all — it's only encoded in the output directory name. Without it
        # in the key, two different seq_types sharing model/target/eff_mode/k_per_neuron
        # would be merged into one curve. static_skeleton.py has no seq_type subdirectory
        # (always 2 levels), so it gets "" — harmless since its x_key never collides with
        # per-neuron's anyway.
        rel_parts = os.path.relpath(fpath, results_dir).split(os.sep)
        seq_type = rel_parts[-2] if len(rel_parts) == 3 else ""
        if (d.get("model_name"), seq_type) in covered_runs:
            continue
        # x_key is part of the grouping key so a static_skeleton.py run (k_percent, a
        # percentage) never gets merged onto the same x-axis as a
        # run_static_ske_per_neuron.py run (k_per_neuron, an absolute count) even if
        # they happen to share model/target/eff_mode under the same --results_dir.
        key = (d["model_name"], seq_type, d["target_word"].strip(), eff_mode, x_key)
        version_key = key + (x_val,)
        result_version = int(d.get("result_format_version", 1))
        previous_version = loaded_versions.get(version_key, 0)
        if result_version < previous_version:
            continue
        if result_version > previous_version:
            grouped[key] = [
                run for run in grouped.get(key, []) if run["x_val"] != x_val
            ]
            loaded_versions[version_key] = result_version
        by_label = {row["label"]: row for row in d["summary"]}
        grouped.setdefault(key, []).append({"x_val": x_val, "x_key": x_key, "by_label": by_label})
    return grouped


def plot_one(model: str, seq_type: str, target: str, eff_mode: str, runs: list, output_dir: str) -> None:
    runs = sorted(runs, key=lambda r: r["x_val"])
    xs = [r["x_val"] for r in runs]
    is_per_neuron = runs[0]["x_key"] == "k_per_neuron"
    x_label = "k_per_neuron" if is_per_neuron else "k_percent (%)"
    preferred = PER_NEURON_CONDITIONS if is_per_neuron else CONDITIONS
    present = list(dict.fromkeys(
        label for run in runs for label in run["by_label"]
    ))
    # Keep the familiar order when possible, but never drop newly added static
    # conditions merely because this plotting file has an older label registry.
    conditions = [c for c in preferred if c in present]
    conditions += [c for c in present if c not in conditions]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    safe_model = model.split("/")[-1]
    seq_tag = f"[{seq_type}]  " if seq_type else ""
    fig.suptitle(f"{safe_model}  |  {seq_tag}target: '{target}'  |  eff={eff_mode}", fontsize=11)

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for cond in conditions:
            points = [
                (r["x_val"], r["by_label"][cond][metric_key])
                for r in runs if cond in r["by_label"]
            ]
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker="o", markersize=4, label=cond)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label.split(" (")[0])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if metric_key == "mean_rank":
            ax.invert_yaxis()

    fig.tight_layout()

    safe_model_dir = model.replace("/", "_")
    safe_target = safe_slug(target)
    x_key = runs[0]["x_key"]
    # seq_type as a subdirectory keeps different seq_types' figures from overwriting
    # each other now that they're never merged into the same group (see load_all).
    out_dir = os.path.join(output_dir, safe_model_dir, safe_slug(seq_type) if seq_type else "")
    os.makedirs(out_dir, exist_ok=True)
    # x_key suffix keeps a k_percent group and a k_per_neuron group for the same
    # (model, target, eff_mode) — a real possibility now that they're never merged
    # in load_all — from overwriting each other's PNG.
    out_path = os.path.join(out_dir, f"{safe_target}_{eff_mode}_{x_key}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/static_ske")
    p.add_argument("--output_dir",  default="figures/static_ske")
    args = p.parse_args()

    grouped = load_all(args.results_dir)
    if not grouped:
        print("No results with k_percent/k_per_neuron found under --results_dir.")
        return

    print(f"Found {len(grouped)} (model, seq_type, target, eff_mode, x_key) groups.")
    for (model, seq_type, target, eff_mode, x_key), runs in sorted(grouped.items()):
        print(f"  {model} | [{seq_type}] | '{target}' | {eff_mode} | {x_key} — {len(runs)} points")
        plot_one(model, seq_type, target, eff_mode, runs, args.output_dir)


if __name__ == "__main__":
    main()
