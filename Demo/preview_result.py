"""Print a compact preview of circuit result JSON files after a shell run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _files(directory: Path):
    return sorted(
        directory.rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if directory.is_dir() else []


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _score(item: dict, *, mean: bool = False) -> str:
    rank_key = "mean_rank" if mean else "rank"
    prob_key = "mean_prob" if mean else "prob"
    rank = item.get(rank_key)
    prob = item.get(prob_key)
    if rank is None or prob is None:
        return "unavailable"
    return f"rank={rank:.3g}, p={prob:.3g}"


def _preview_static(args):
    results = {}
    seq_type = args.dataset_path.name.split("_")[0] if args.dataset_path else None
    for path in _files(args.output_dir):
        data = _read(path)
        if not isinstance(data, dict) or data.get("model_name") != args.model:
            continue
        if seq_type is not None and path.parent.name != seq_type:
            continue
        if data.get("target_word") != args.target or not data.get("per_prompt"):
            continue
        key = data.get("k_per_neuron")
        results.setdefault(key, (path, data))
    for key in sorted(results):
        path, data = results[key]
        print(f"[RESULT] static k={key}, prompts={len(data['per_prompt'])}, target={args.target!r}")
        for item in data.get("summary", []):
            print(f"  {item.get('label', '?')}: {_score(item, mean=True)}")
        print(f"  JSON: {path}")
    return len(results)


def _preview_dynamic_mlp(args):
    results = {}
    for path in _files(args.output_dir):
        data = _read(path)
        if not isinstance(data, dict) or data.get("model_name") != args.model:
            continue
        if data.get("target_word") != args.target or data.get("in_seq") != args.prompt:
            continue
        key = data.get("k_percent")
        results.setdefault(key, (path, data))
    for key in sorted(results):
        path, data = results[key]
        print(f"[RESULT] dynamic MLP k={key:g}%, target={args.target!r}")
        labels = ["original", *data.get("conditions", [])]
        for label, item in zip(labels, data.get("summary", [])):
            print(f"  {label}: {_score(item)}")
        print(f"  JSON: {path}")
    return len(results)


def _preview_attention(args):
    results = {}
    for path in _files(args.output_dir):
        data = _read(path)
        if not isinstance(data, dict) or data.get("model_name") != args.model:
            continue
        if data.get("target_token") != args.target or data.get("prompt") != args.prompt:
            continue
        key = (data.get("type"), data.get("k_percent"))
        results.setdefault(key, (path, data))
    for circuit_type, percent in sorted(results):
        path, data = results[(circuit_type, percent)]
        print(f"[RESULT] {circuit_type} k={percent:g}%, target={args.target!r}")
        for label, item in data.get("summary", {}).items():
            print(f"  {label}: {_score(item)}")
        print(f"  JSON: {path}")
    return len(results)


def _preview_residual(args):
    count = 0
    for intervention in ("zero", "scale"):
        path = args.output_dir / intervention / "aggregate_patching.json"
        data = _read(path)
        if not isinstance(data, dict):
            continue
        config = data.get("config", {})
        averages = data.get("averages", {})
        win_rates = data.get("win_rates", {})
        print(f"[RESULT] residual {intervention}, neurons={config.get('n_neurons', '?')}")
        print(f"  top mean={averages.get('top_mean', float('nan')):.3g}, "
              f"random mean={averages.get('rand_mean', float('nan')):.3g}, "
              f"top > random={win_rates.get('top_gt_rand_per_neuron_pct', float('nan')):.3g}%")
        print(f"  JSON: {path}")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("static", "dynamic_mlp", "attention", "residual"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--target")
    args = parser.parse_args()
    preview = {
        "static": _preview_static,
        "dynamic_mlp": _preview_dynamic_mlp,
        "attention": _preview_attention,
        "residual": _preview_residual,
    }[args.kind]
    if preview(args) == 0:
        parser.exit(1, "[ERROR] No matching result JSON files found.\n")


if __name__ == "__main__":
    main()
