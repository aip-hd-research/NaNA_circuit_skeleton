"""Evaluate whether static anatomy predicts recruitment on unseen inputs.

The leakage boundary in this program is explicit: static rankings are computed
and frozen before ``--heldout_inputs`` is opened.  Each pathway is a
``(neuron, subspace)`` pair within one layer.

Example
-------
python scripts_ske_cir/run_future_dynamic_recruitment.py \
  --model_name gpt2-medium \
  --cache_dir result/within_ske_cache \
  --target_word " happy" \
  --heldout_inputs heldout.json \
  --input_field prompt \
  --start_layer 5 --end_layer 22 \
  --static_k_per_sensor 1 2 5 10 \
  --dynamic_k_values 50 100 200 500 \
  --output_dir result/future_recruitment/happy \
  --norm_eff --plot

JSON input may be a list, a single object, or a dict containing a list under
``data``, ``inputs``, ``samples`` or ``examples``. JSONL and one-input-per-line
text files are also accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from scipy.stats import hypergeom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_interp.model_load import load_model_and_embeddings
from model_config.templates import apply_template, needs_true_add_special_tokens
from scripts_ske_cir.run_static_dynamic_compare import build_effectors, collect_h_values
from ske_pre_valid.static_skeleton import _resolve_target_info
from ske_pre_valid.within_ske_cache import WithinSkeCache


DEFAULT_INPUT_FIELDS = ("prompt", "input", "text", "in_seq")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test static Top pathways against dynamic Top pathways on held-out inputs."
    )
    p.add_argument("--model_name", default="gpt2-medium")
    p.add_argument("--cache_dir", default="result/within_ske_cache")
    p.add_argument("--target_word", required=True)
    p.add_argument("--heldout_inputs", required=True)
    p.add_argument("--input_field", default=None)
    p.add_argument("--start_layer", type=int, default=5)
    p.add_argument("--end_layer", type=int, default=22)
    p.add_argument(
        "--static_k_per_sensor", type=int, nargs="+", default=[1, 2, 5, 10],
        help="Static subspaces retained independently under each sensor/neuron.",
    )
    p.add_argument(
        "--dynamic_k_values", type=int, nargs="+", default=[50, 100, 200, 500],
        help="Global Dynamic Top-K values.",
    )
    p.add_argument("--seq_reduce", choices=["last", "mean"], default="last")
    p.add_argument("--norm_eff", action="store_true")
    p.add_argument("--max_inputs", type=int, default=None)
    p.add_argument("--bootstrap_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--write_sensor_details",
        action="store_true",
        help="Write one row per active sensor/input/K combination (can be very large).",
    )
    return p.parse_args()


def _records_from_json(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "inputs", "samples", "examples"):
            if isinstance(obj.get(key), list):
                return obj[key]
        # Repository datasets commonly use sample IDs as top-level keys:
        # {"1": {"prompt": ...}, "2": {"prompt": ...}, ...}.
        if obj and all(isinstance(value, (dict, str)) for value in obj.values()):
            return list(obj.values())
        return [obj]
    raise ValueError("JSON root must be an object or a list")


def _record_text(record: Any, input_field: str | None) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise ValueError(f"Input record must be a string or object, got {type(record).__name__}")
    fields = (input_field,) if input_field else DEFAULT_INPUT_FIELDS
    for field in fields:
        if field and isinstance(record.get(field), str):
            return record[field]
    raise ValueError(f"No text field found; tried {fields}. Pass --input_field explicitly.")


def load_heldout_inputs(path: str, input_field: str | None, max_inputs: int | None) -> list[str]:
    """Open held-out data. This function must only be called after static freeze."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        with file_path.open(encoding="utf-8") as f:
            records = _records_from_json(json.load(f))
    elif suffix in (".jsonl", ".ndjson"):
        with file_path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        with file_path.open(encoding="utf-8") as f:
            records = [line.rstrip("\n") for line in f if line.strip()]
    texts = [_record_text(record, input_field) for record in records]
    return texts[:max_inputs] if max_inputs is not None else texts


def freeze_static_rankings(
    cache: WithinSkeCache,
    effectors: dict[int, torch.Tensor],
    start_layer: int,
    end_layer: int,
) -> dict[int, dict[str, torch.Tensor | int]]:
    """Compute all input-independent objects before held-out inputs are opened.

    ``effectors[layer][i]`` is already the scalar target alignment
    ``V[i] @ W_emb[target_id]``.  Static ranking therefore uses
    ``abs(delta) * abs(target_alignment)``; it must not multiply alignment twice.
    """
    frozen = {}
    for layer in range(start_layer, end_layer + 1):
        delta = cache.layer_skeleton(layer)["delta"].cpu().float()
        target_alignment = effectors[layer].cpu().float()
        static_scores = delta.abs() * target_alignment.abs().unsqueeze(0)
        signed_target_base = delta * target_alignment.unsqueeze(0)
        frozen[layer] = {
            # Each row is one sensor/neuron; it has its own independent ranking.
            "order_per_sensor": torch.argsort(static_scores, dim=1, descending=True),
            "signed_target_base": signed_target_base,
            "n_total": static_scores.numel(),
            "n_sensors": static_scores.shape[0],
            "n_subspaces": static_scores.shape[1],
        }
    return frozen


def ranking_metrics(
    static_order_per_sensor: torch.Tensor,
    dynamic_scores: torch.Tensor,
    static_k_per_sensor: int,
    dynamic_k: int,
    dynamic_order: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Compare per-sensor Static Top-k against global Dynamic Top-K."""
    dynamic_scores = dynamic_scores.flatten().cpu().float()
    n_total = dynamic_scores.numel()
    n_sensors, n_subspaces = static_order_per_sensor.shape
    static_k_per_sensor = min(static_k_per_sensor, n_subspaces)
    dynamic_k = min(dynamic_k, n_total)
    if static_k_per_sensor <= 0 or dynamic_k <= 0:
        raise ValueError("Static and dynamic K must be positive")

    sensor_offsets = torch.arange(n_sensors).unsqueeze(1) * n_subspaces
    static_top = (static_order_per_sensor[:, :static_k_per_sensor] + sensor_offsets).flatten()
    n_static = static_top.numel()
    if dynamic_order is None:
        dynamic_order = torch.argsort(dynamic_scores, descending=True)
    dynamic_top = dynamic_order[:dynamic_k]
    dynamic_mask = torch.zeros(n_total, dtype=torch.bool)
    dynamic_mask[dynamic_top] = True
    hits = dynamic_mask[static_top]
    overlap = int(hits.sum().item())
    static_matrix = static_top.reshape(n_sensors, static_k_per_sensor)
    overlap_per_sensor = dynamic_mask[static_matrix].sum(dim=1)
    dynamic_sensor_ids = torch.div(dynamic_top, n_subspaces, rounding_mode="floor")
    dynamic_count_per_sensor = torch.bincount(dynamic_sensor_ids, minlength=n_sensors)
    active_sensor_mask = dynamic_count_per_sensor > 0
    active_sensor_count = int(active_sensor_mask.sum().item())
    hit_sensor_count = int(((overlap_per_sensor > 0) & active_sensor_mask).sum().item())
    active_sensor_recall = (
        overlap_per_sensor[active_sensor_mask].float()
        / dynamic_count_per_sensor[active_sensor_mask].float()
    )
    expected_overlap = n_static * dynamic_k / n_total
    union = n_static + dynamic_k - overlap

    return {
        "static_k_per_sensor": static_k_per_sensor,
        "dynamic_k": dynamic_k,
        "n_sensors": n_sensors,
        "n_static": n_static,
        "n_total": n_total,
        "overlap": overlap,
        # Static set is the prediction; global Dynamic Top-K is the truth set.
        "precision": overlap / n_static,
        "recall": overlap / dynamic_k,
        "jaccard": overlap / union if union else 1.0,
        "mean_static_coverage_per_sensor": float(
            (overlap_per_sensor.float() / static_k_per_sensor).mean().item()
        ),
        "active_sensor_hit_rate": hit_sensor_count / active_sensor_count,
        "mean_active_sensor_recall": float(active_sensor_recall.mean().item()),
        "n_active_dynamic_sensors": active_sensor_count,
        "enrichment": overlap / expected_overlap if expected_overlap else 0.0,
        "random_precision": dynamic_k / n_total,
        "p_value_hypergeom": float(
            hypergeom.sf(overlap - 1, n_total, dynamic_k, n_static)
        ),
    }


def per_sensor_metrics(
    input_index: int,
    layer: int,
    static_order_per_sensor: torch.Tensor,
    dynamic_scores: torch.Tensor,
    static_k_per_sensor: int,
    dynamic_k: int,
) -> list[dict[str, int | float | bool]]:
    """Assign global Dynamic Top-K pathways to sensors, then compare locally."""
    n_sensors, n_subspaces = static_order_per_sensor.shape
    static_k_per_sensor = min(static_k_per_sensor, n_subspaces)
    dynamic_flat = dynamic_scores.flatten().cpu().float()
    dynamic_k = min(dynamic_k, dynamic_flat.numel())
    dynamic_top = torch.topk(dynamic_flat, dynamic_k).indices
    dynamic_mask = torch.zeros(dynamic_flat.numel(), dtype=torch.bool)
    dynamic_mask[dynamic_top] = True
    dynamic_counts = torch.bincount(
        torch.div(dynamic_top, n_subspaces, rounding_mode="floor"), minlength=n_sensors
    )
    offsets = torch.arange(n_sensors).unsqueeze(1) * n_subspaces
    static_top = static_order_per_sensor[:, :static_k_per_sensor] + offsets
    overlaps = dynamic_mask[static_top].sum(dim=1)

    rows = []
    for sensor in range(n_sensors):
        overlap = int(overlaps[sensor].item())
        dynamic_count = int(dynamic_counts[sensor].item())
        rows.append(
            {
                "input_index": input_index,
                "layer": layer,
                "sensor": sensor,
                "static_k_per_sensor": static_k_per_sensor,
                "dynamic_k_global": dynamic_k,
                "dynamic_count_in_sensor": dynamic_count,
                "overlap": overlap,
                "static_coverage": overlap / static_k_per_sensor,
                "dynamic_recall_in_sensor": overlap / dynamic_count if dynamic_count else 0.0,
                "dynamic_sensor_active": dynamic_count > 0,
                "sensor_hit": overlap > 0,
            }
        )
    return rows


def bootstrap_mean_ci(values: Iterable[float], n_boot: int, seed: int) -> tuple[float, float]:
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1 or n_boot <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lo = means[max(0, math.floor(0.025 * n_boot))]
    hi = means[min(n_boot - 1, math.ceil(0.975 * n_boot) - 1)]
    return lo, hi


def summarize(rows: list[dict[str, Any]], n_boot: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["layer"], row["static_k_per_sensor"], row["dynamic_k"])].append(row)

    metrics = (
        "precision",
        "recall",
        "jaccard",
        "mean_static_coverage_per_sensor",
        "active_sensor_hit_rate",
        "mean_active_sensor_recall",
        "enrichment",
    )
    summary = []
    for (layer, static_k, dynamic_k), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "layer": layer,
            "static_k_per_sensor": static_k,
            "dynamic_k": dynamic_k,
            "n_static": group[0]["n_static"],
            "n_inputs": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            out[f"mean_{metric}"] = sum(values) / len(values)
            lo, hi = bootstrap_mean_ci(
                values, n_boot, seed + layer * 100003 + static_k * 1009 + dynamic_k
            )
            out[f"ci95_low_{metric}"] = lo
            out[f"ci95_high_{metric}"] = hi
        # Fisher's method combines evidence across held-out inputs.
        log_sum = sum(math.log(max(float(row["p_value_hypergeom"]), 1e-300)) for row in group)
        from scipy.stats import chi2

        out["combined_p_fisher"] = float(chi2.sf(-2.0 * log_sum, 2 * len(group)))
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_enrichment(summary: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    by_curve: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_curve[(row["layer"], row["static_k_per_sensor"])].append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (layer, static_k), rows in sorted(by_curve.items()):
        rows.sort(key=lambda row: row["dynamic_k"])
        x = [row["dynamic_k"] for row in rows]
        y = [row["mean_enrichment"] for row in rows]
        lo = [row["ci95_low_enrichment"] for row in rows]
        hi = [row["ci95_high_enrichment"] for row in rows]
        ax.plot(x, y, marker="o", linewidth=1.3, label=f"L{layer}, static k/sensor={static_k}")
        ax.fill_between(x, lo, hi, alpha=0.10)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="random")
    ax.set_xscale("log")
    ax.set_xlabel("Global dynamic K")
    ax.set_ylabel("Per-sensor static enrichment in global Dynamic Top-K")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.start_layer > args.end_layer:
        raise ValueError("--start_layer must not exceed --end_layer")
    if any(k <= 0 for k in args.static_k_per_sensor + args.dynamic_k_values):
        raise ValueError("All static and dynamic K values must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, w_emb = load_model_and_embeddings(args.model_name, device)
    model.eval()
    cache = WithinSkeCache(args.cache_dir, expected_model_name=args.model_name)
    target_info = _resolve_target_info(tokenizer, args.target_word)
    target_id = target_info["target_id"]

    # LEAKAGE BOUNDARY: everything above/below until the freeze message is
    # input-independent. Do not move held-out loading above this block.
    effectors = build_effectors(
        cache, w_emb.cpu(), target_id, args.start_layer, args.end_layer, args.norm_eff
    )
    frozen = freeze_static_rankings(cache, effectors, args.start_layer, args.end_layer)
    del effectors
    print(f"[freeze] Static rankings frozen for target {args.target_word!r}; opening held-out inputs now.")

    inputs = load_heldout_inputs(args.heldout_inputs, args.input_field, args.max_inputs)
    if not inputs:
        raise ValueError("Held-out input file contains no usable inputs")
    print(f"[held-out] Loaded {len(inputs)} inputs from {args.heldout_inputs}")

    rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    for input_index, raw_text in enumerate(inputs):
        text = apply_template(raw_text, args.model_name, tokenizer)
        input_ids = tokenizer.encode(
            text,
            return_tensors="pt",
            add_special_tokens=needs_true_add_special_tokens(args.model_name, tokenizer),
        )
        h_by_layer = collect_h_values(
            model, input_ids, args.start_layer, args.end_layer, str(device), args.seq_reduce
        )
        for layer in range(args.start_layer, args.end_layer + 1):
            layer_frozen = frozen[layer]
            # Dynamic = activation * delta * (V @ target unembedding).
            dynamic_scores = (
                h_by_layer[layer].cpu().float().unsqueeze(1)
                * layer_frozen["signed_target_base"]
            )
            # Reuse one partial global ordering for every K combination. We only
            # need up to max(dynamic_k), so a full multimillion-path argsort is
            # unnecessary.
            max_dynamic_k = min(max(args.dynamic_k_values), dynamic_scores.numel())
            dynamic_order = torch.topk(
                dynamic_scores.flatten(), max_dynamic_k, largest=True, sorted=True
            ).indices
            for static_k in sorted(set(args.static_k_per_sensor)):
                for dynamic_k in sorted(set(args.dynamic_k_values)):
                    metrics = ranking_metrics(
                        layer_frozen["order_per_sensor"],
                        dynamic_scores,
                        static_k,
                        dynamic_k,
                        dynamic_order=dynamic_order,
                    )
                    rows.append({"input_index": input_index, "layer": layer, **metrics})
                    if args.write_sensor_details:
                        sensor_rows.extend(
                            per_sensor_metrics(
                                input_index,
                                layer,
                                layer_frozen["order_per_sensor"],
                                dynamic_scores,
                                static_k,
                                dynamic_k,
                            )
                        )
        print(f"[evaluate] {input_index + 1}/{len(inputs)}", end="\r", flush=True)
    print()

    summary = summarize(rows, args.bootstrap_samples, args.seed)
    write_csv(output_dir / "per_input_metrics.csv", rows)
    if args.write_sensor_details:
        write_csv(output_dir / "per_input_sensor_metrics.csv", sensor_rows)
    write_csv(output_dir / "summary_by_layer_k.csv", summary)
    metadata = {
        "model_name": args.model_name,
        "cache_dir": args.cache_dir,
        "target_word": args.target_word,
        "target_id": target_id,
        "target_decoded": target_info["target_decoded"],
        "heldout_inputs": args.heldout_inputs,
        "n_inputs": len(inputs),
        "layers": [args.start_layer, args.end_layer],
        "static_k_per_sensor": sorted(set(args.static_k_per_sensor)),
        "dynamic_k_values": sorted(set(args.dynamic_k_values)),
        "seq_reduce": args.seq_reduce,
        "norm_eff": args.norm_eff,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "write_sensor_details": args.write_sensor_details,
        "leakage_rule": "held-out file opened only after static rankings were frozen",
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2, ensure_ascii=False)
    if args.plot:
        plot_enrichment(summary, output_dir / "enrichment_by_k.png")

    print(f"Results written to {output_dir}")
    for row in summary:
        print(
            f"L{row['layer']:02d} static-k/sensor={row['static_k_per_sensor']:>3} "
            f"dynamic-K={row['dynamic_k']:>6}: "
            f"precision={row['mean_precision']:.4f}  "
            f"recall={row['mean_recall']:.4f}  "
            f"enrichment={row['mean_enrichment']:.2f}x  "
            f"p(Fisher)={row['combined_p_fisher']:.3g}"
        )


if __name__ == "__main__":
    main()
