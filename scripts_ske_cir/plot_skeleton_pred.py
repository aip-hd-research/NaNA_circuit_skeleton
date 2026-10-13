"""
Generate per-model per-sample per-condition plots from dynamic skeleton prediction results.

For each (model, dataset, sample_idx, target, vocab_tag, ori_acv_tag) group:
  - One figure with 3 subplots: mean_rank | mean_prob | mean_logit
  - X-axis: k_percent
  - Each subplot has 5 lines, one per evaluation condition

Usage:
  python scripts_ske_cir/plot_skeleton_pred.py
  python scripts_ske_cir/plot_skeleton_pred.py --results_dir results/dynamic_ske --output_dir figures/dynamic_ske
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt

from model_config import PROMPT_FORMAT_VERSION, safe_slug

CONDITIONS_KEYS = ["ori", "ske_pre", "ske_abl", "rnd_pre", "rnd_abl"]

METRICS = [
    ("rank",  "Rank (lower = better)"),
    ("prob",  "Prob"),
    ("logit", "Logit"),
]


def load_all(results_dir: str) -> dict:
    grouped: dict[tuple, list] = {}
    loaded_versions: dict[tuple, int] = {}
    # run_skeleton_pred.py saves to <results_dir>/<safe_model>/<dataset_name>/*.json —
    # a fixed "*", "*.json" glob misses that extra dataset_name level entirely. Recurse
    # instead so this keeps working regardless of how many levels of nesting are used.
    for fpath in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        with open(fpath) as f:
            d = json.load(f)
        if d.get("prompt_format_version") != PROMPT_FORMAT_VERSION:
            continue
        k_percent = d.get("k_percent")
        if k_percent is None:
            continue
        vocab_tag   = d.get("effector_mode", "raw_dot")
        ori_acv_tag = "ori_acv_true" if d.get("rely_ori_acv")     else "ori_acv_false"
        # Single-sample mode saves "target_word"/"in_seq"; dataset mode (sequential and
        # batched — what run_dynm_ske.sh actually drives) saves "target_token"/"prompt".
        target = d.get("target_word", d.get("target_token"))
        in_seq = d.get("in_seq", d.get("prompt"))
        if target is None or in_seq is None:
            continue
        # dataset_name (e.g. "next_token_short_200samples" vs "..._long_...") is only
        # present in dataset mode; sample_idx restarts from 1 per dataset, so without
        # this in the key, two different datasets sharing a sample_idx + target would
        # be merged into one curve — even plotting two points at the same k_percent.
        dataset = d.get("dataset", "")
        key = (d["model_name"], dataset, d["sample_idx"], target.strip(),
               vocab_tag, ori_acv_tag)
        version_key = key + (k_percent,)
        result_version = int(d.get("result_format_version", 1))
        previous_version = loaded_versions.get(version_key, 0)
        if result_version < previous_version:
            continue
        if result_version > previous_version:
            grouped[key] = [
                run for run in grouped.get(key, [])
                if run["k_percent"] != k_percent
            ]
            loaded_versions[version_key] = result_version
        by_label = {row["label"]: row for row in d["summary"]}
        grouped.setdefault(key, []).append({
            "k_percent": k_percent,
            "in_seq":    in_seq,
            "by_label":  by_label,
        })
    return grouped


def plot_one(model: str, dataset: str, sample_idx: int, target: str,
             vocab_tag: str, ori_acv_tag: str,
             runs: list, output_dir: str) -> None:
    runs = sorted(runs, key=lambda r: r["k_percent"])
    xs   = [r["k_percent"] for r in runs]
    in_seq = runs[0]["in_seq"]

    conditions = {row["label"]: row["label"] for r in runs for row in r["by_label"].values()}
    condition_names = list(dict.fromkeys(
        row["label"] for r in runs for row in r["by_label"].values()
    ))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    safe_model = model.split("/")[-1]
    dataset_tag = f"[{dataset}]  " if dataset else ""
    fig.suptitle(
        f"{safe_model}  |  {dataset_tag}[{sample_idx}] {in_seq!r} -> {target!r}\n"
        f"{vocab_tag}  |  {ori_acv_tag}",
        fontsize=10,
    )

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for cond_name in condition_names:
            points = [
                (r["k_percent"], r["by_label"][cond_name][metric_key])
                for r in runs if cond_name in r["by_label"]
            ]
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker="o", markersize=4, label=cond_name)
        ax.set_xlabel("k_percent (%)")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label.split(" (")[0])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if metric_key == "rank":
            ax.invert_yaxis()

    fig.tight_layout()

    safe_model_dir = model.replace("/", "_")
    safe_target    = safe_slug(target)
    # dataset as a subdirectory keeps different datasets' figures from overwriting each
    # other now that they're never merged into the same group (see load_all).
    out_dir = os.path.join(output_dir, safe_model_dir, safe_slug(dataset) if dataset else "")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"{sample_idx}_{safe_target}_{vocab_tag}_{ori_acv_tag}.png",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/dynamic_ske")
    p.add_argument("--output_dir",  default="figures/dynamic_ske")
    args = p.parse_args()

    grouped = load_all(args.results_dir)
    if not grouped:
        print("No results with k_percent found.")
        return

    print(f"Found {len(grouped)} groups.")
    for (model, dataset, sample_idx, target, vocab_tag, ori_acv_tag), runs in sorted(grouped.items()):
        print(f"  {model} | [{dataset}] [{sample_idx}] {target!r} | {vocab_tag} | "
              f"{ori_acv_tag} — {len(runs)} k_percent points")
        plot_one(model, dataset, sample_idx, target, vocab_tag, ori_acv_tag,
                 runs, args.output_dir)


if __name__ == "__main__":
    main()
