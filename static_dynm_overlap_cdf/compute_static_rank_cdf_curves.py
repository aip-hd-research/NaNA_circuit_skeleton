"""Empirical CDF of Static rank among pathways in the Dynamic Top set.

The pathway selectors are not reimplemented here. Dynamic pathways come from
the trace returned by the repository's production dynamic-circuit runner, and
Static within-sensor ordering comes from the production static selector. This
keeps the comparison on exactly the same pathway definition and ranking rules
as the two circuit experiments.

For a recruited pathway ``(sensor/neuron j, effector/subspace i)``, its Static
rank is its 1-based position among all subspaces for sensor ``j``. Static rank
uses weights and the target direction only; Dynamic Top additionally uses the
actual activation produced by the input.

Example
-------
python static_dynm_overlap_cdf/compute_static_rank_cdf_curves.py \
  --models openai-community/gpt2 Qwen/Qwen3-0.6B \
  --n_inputs 500 --batch_size 20 --layer 4 \
  --dynamic_k_percents 0.001 0.1 5.0
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CACHE_ROOT = "/pfs/work9/workspace/scratch/hd_nf184-nana-cache"
DEFAULT_MODEL_CACHE = f"{DEFAULT_CACHE_ROOT}/model_cache"
DEFAULT_CACHE_BASE = f"{DEFAULT_CACHE_ROOT}/pre_compute_skeleton_cache"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result_CDF"
os.environ.setdefault("HF_HOME", DEFAULT_MODEL_CACHE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Reuse the real dynamic and static circuit implementations. In particular,
# do not duplicate their score formulas in this analysis script.
from model_config import NEXT_TOKEN_MAX_LENGTH, apply_template
from model_config.paths import within_ske_path
from scripts_ske_cir.run_skeleton_pred import (
    _load_dataset_items,
    _resolve_item_target_id,
)
from ske_pre_valid.forward_hook.forward_per_neuron import select_per_neuron_abs_topk
from ske_pre_valid.forward_hook.skeleton_mlp_forward_ori_acv import _collect_original_h
from ske_pre_valid.skeleton_pred import (
    _compute_per_sample_effectors,
    setup_skeleton_pred,
)
from ske_pre_valid.within_ske_selector import select_and_reconstruct_skeleton


# Domain below dataset/skeleton_dymn. The dataset directory itself is the
# model name with '/' replaced by '_'.
MODEL_DOMAINS = {
    "openai-community/gpt2": "natural_language",
    "microsoft/Phi-3-mini-4k-instruct": "natural_language",
    "google/gemma-3-1b-it": "natural_language",
    "meta-llama/Llama-3.2-1B": "natural_language",
    "Qwen/Qwen3-0.6B": "natural_language",
    "Qwen/Qwen2.5-1.5B": "natural_language",
    "meta-llama/Llama-2-7b-hf": "natural_language",
    "mistralai/Mistral-7B-v0.3": "natural_language",
    "mistralai/Mistral-7B-v0.1": "natural_language",
    "deepseek-ai/deepseek-llm-7b-base": "natural_language",
    "google/gemma-2-9b-it": "natural_language",
    "bigcode/starcoder2-3b": "programming_language",
    "codellama/CodeLlama-7b-hf": "programming_language",
    "deepseek-ai/deepseek-coder-6.7b-base": "programming_language",
    "microsoft/biogpt": "biomedical_nl",
    "microsoft/BioGPT-Large": "biomedical_nl",
    "liyinghong/BioQwen-0.5B": "biomedical_nl",
    "liyinghong/BioQwen-1.8B": "biomedical_nl",
    "nferruz/ProtGPT2": "protein_sequence",
    "AI4PD/ZymCTRL": "protein_sequence",
    "hugohrban/progen2-small": "protein_sequence",
    "hugohrban/progen2-medium": "protein_sequence",
    "ncfrey/ChemGPT-4.7M": "chemical_selfies",
    "ncfrey/ChemGPT-19M": "chemical_selfies",
    "ncfrey/ChemGPT-1.2B": "chemical_selfies",
    "entropy/gpt2_zinc_87m": "chemical_smiles",
    "ChemFM/ChemFM-1B": "chemical_smiles",
    "jonghyunlee/MolGPT_pretrained-by-ZINC15": "chemical_smiles",
    "jonghyunlee/MolGPT_long_context_pretrained-by-ZINC15": "chemical_smiles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CDF of within-sensor Static rank for pathways selected by the "
            "production Dynamic Top circuit."
        )
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_DOMAINS))
    parser.add_argument("--subset", choices=("short", "long"), default="long")
    parser.add_argument("--cache_base", default=DEFAULT_CACHE_BASE)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--n_inputs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--dynamic_k_percents",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0, 5.0],
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--checkpoint_dir",
        help=(
            "Directory for resumable partial state. Defaults to a checkpoints "
            "subdirectory next to output_dir."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> list[float]:
    if args.n_inputs <= 0:
        raise ValueError("n_inputs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.layer < 0:
        raise ValueError("layer must be non-negative")
    unknown = sorted(set(args.models) - set(MODEL_DOMAINS))
    if unknown:
        raise ValueError(f"models missing from MODEL_DOMAINS: {unknown}")
    percents = sorted(set(float(value) for value in args.dynamic_k_percents))
    if not percents or any(not 0.0 < value <= 100.0 for value in percents):
        raise ValueError("dynamic_k_percents must contain values in (0, 100]")
    return percents


def _dataset_path(model_name: str, subset: str) -> Path:
    slug = model_name.replace("/", "_")
    return (
        REPO_ROOT
        / "dataset"
        / "skeleton_dymn"
        / MODEL_DOMAINS[model_name]
        / slug
        / f"next_token_{subset}.json"
    )


def _dynamic_k_values(
    n_total: int, dynamic_k_percents: list[float]
) -> dict[float, int]:
    """Match the dynamic experiment's percentage-to-k ceil policy."""
    return {
        percent: min(n_total, max(1, math.ceil(n_total * percent / 100.0)))
        for percent in dynamic_k_percents
    }


def _atomic_json_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _safe_model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def _checkpoint_path(checkpoint_dir: Path, model_name: str) -> Path:
    return checkpoint_dir / f"{_safe_model_slug(model_name)}.static_rank_cdf_checkpoint.json"


def _percent_key(percent: float) -> str:
    return f"{percent:g}"


def _result_curves(
    totals: dict[float, torch.Tensor],
    k_by_percent: dict[float, int],
    dynamic_k_percents: list[float],
) -> dict:
    curves = {}
    for percent in dynamic_k_percents:
        histogram = totals[percent]
        n_recruited = int(histogram.sum().item())
        if n_recruited == 0:
            raise RuntimeError(f"no Dynamic Top pathways collected at {percent:g}%")
        # Keep fraction-valued keys for compatibility with the existing plotter.
        fraction_key = f"{percent / 100.0:g}"
        curves[fraction_key] = {
            "dynamic_k_percent": percent,
            "dynamic_k_per_input": k_by_percent[percent],
            "n_recruited": n_recruited,
            "observed_cdf": (histogram.cumsum(0) / n_recruited).tolist(),
        }
    return curves


def _load_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    model_name: str,
    dynamic_k_percents: list[float],
    n_subspaces: int,
) -> tuple[int, int, int, dict[float, torch.Tensor]]:
    totals = {
        percent: torch.zeros(n_subspaces, dtype=torch.float64)
        for percent in dynamic_k_percents
    }
    if not checkpoint_path.is_file():
        return 0, 0, 0, totals

    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[resume] ignoring unreadable checkpoint {checkpoint_path}: {exc}")
        return 0, 0, 0, totals

    expected = {
        "model_name": model_name,
        "subset": args.subset,
        "layer": args.layer,
        "n_inputs_requested": args.n_inputs,
        "dynamic_k_percents": dynamic_k_percents,
        "n_subspaces": n_subspaces,
    }
    metadata = payload.get("metadata", {})
    mismatches = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if mismatches:
        print(
            f"[resume] ignoring checkpoint with incompatible fields: {mismatches}"
        )
        return 0, 0, 0, totals

    raw_totals = payload.get("totals", {})
    try:
        for percent in dynamic_k_percents:
            values = raw_totals[_percent_key(percent)]
            if len(values) != n_subspaces:
                raise ValueError(
                    f"{percent:g}% histogram length {len(values)} != {n_subspaces}"
                )
            totals[percent] = torch.tensor(values, dtype=torch.float64)
        next_start = int(payload.get("next_batch_start", 0))
        processed = int(payload.get("processed", 0))
        skipped = int(payload.get("skipped", 0))
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[resume] ignoring invalid checkpoint {checkpoint_path}: {exc}")
        return 0, 0, 0, {
            percent: torch.zeros(n_subspaces, dtype=torch.float64)
            for percent in dynamic_k_percents
        }

    print(
        f"[resume] {model_name}: continuing at sample {next_start + 1}; "
        f"processed={processed}, skipped={skipped}"
    )
    return next_start, processed, skipped, totals


def _save_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    model_name: str,
    dynamic_k_percents: list[float],
    n_subspaces: int,
    next_batch_start: int,
    processed: int,
    skipped: int,
    totals: dict[float, torch.Tensor],
) -> None:
    payload = {
        "metadata": {
            "model_name": model_name,
            "subset": args.subset,
            "layer": args.layer,
            "n_inputs_requested": args.n_inputs,
            "dynamic_k_percents": dynamic_k_percents,
            "n_subspaces": n_subspaces,
        },
        "next_batch_start": next_batch_start,
        "processed": processed,
        "skipped": skipped,
        "totals": {
            _percent_key(percent): totals[percent].tolist()
            for percent in dynamic_k_percents
        },
    }
    _atomic_json_write(checkpoint_path, payload)


def _static_rank_lookup(
    layer_data: dict,
    target_direction: torch.Tensor,
    effector: torch.Tensor,
) -> torch.Tensor:
    """Return ``rank[j, i]`` using the production Static selector.

    Asking for every subspace makes ``i_idx`` the complete Static order within
    each sensor. The returned lookup is 1-based, so rank 1 is the most
    Static-favoured pathway for that sensor.
    """
    delta = layer_data["delta"]
    n_sensors, n_subspaces = delta.shape
    selected = select_per_neuron_abs_topk(
        delta,
        layer_data["V"],
        target_direction,
        k_per_neuron=n_subspaces,
        precomputed_effector=effector,
    )
    static_order = selected["i_idx"].reshape(n_sensors, n_subspaces)
    ranks = torch.empty_like(static_order)
    one_based = torch.arange(1, n_subspaces + 1, dtype=static_order.dtype)
    ranks.scatter_(1, static_order, one_based.unsqueeze(0).expand(n_sensors, -1))
    return ranks


def _dynamic_trace_from_activation(
    activation: torch.Tensor,
    layer_tensors: dict,
    effector: torch.Tensor,
    largest_k: int,
) -> dict[str, torch.Tensor]:
    """Call the production Dynamic selector for one sample's captured activation."""
    device = layer_tensors["delta"].device
    dtype = layer_tensors["delta"].dtype
    _, trace = select_and_reconstruct_skeleton(
        activation.to(device=device, dtype=dtype),
        layer_tensors["delta"],
        layer_tensors["V"],
        largest_k,
        seq_reduce="last",
        effector_scores=effector,
        bias=None,
        token_mode="last",
        return_trace=True,
    )
    required = {
        "sensor_indices",
        "effector_indices",
        "selection_scores",
    }
    missing = required - set(trace)
    if missing:
        raise RuntimeError(f"dynamic path trace is missing fields: {sorted(missing)}")
    return trace


def _accumulate_dynamic_prefixes(
    trace: dict[str, torch.Tensor],
    static_ranks: torch.Tensor,
    k_by_percent: dict[float, int],
    totals: dict[float, torch.Tensor],
) -> None:
    """Add Static-rank histograms for nested Dynamic Top-k prefixes."""
    dynamic_order = torch.argsort(trace["selection_scores"], descending=True)
    sensors = trace["sensor_indices"].long()[dynamic_order]
    effectors = trace["effector_indices"].long()[dynamic_order]
    largest_k = max(k_by_percent.values())
    if sensors.numel() != largest_k:
        raise RuntimeError(
            f"dynamic trace has {sensors.numel()} paths, expected {largest_k}"
        )
    recruited_static_ranks = static_ranks[sensors, effectors]
    n_subspaces = static_ranks.shape[1]
    for percent, dynamic_k in k_by_percent.items():
        totals[percent] += torch.bincount(
            recruited_static_ranks[:dynamic_k] - 1, minlength=n_subspaces
        ).to(torch.float64)


def _runner_args(
    args: argparse.Namespace,
    model_name: str,
    cache_dir: str,
    first_item: dict,
) -> types.SimpleNamespace:
    """Arguments required by the existing dynamic setup/evaluation API."""
    return types.SimpleNamespace(
        model_name=model_name,
        cache_dir=cache_dir,
        in_seq=first_item["prompt"],
        target_word=first_item["target_token"],
        target_id=first_item.get("target_token_id"),
        target_resolution=(
            "strict" if first_item.get("target_token_id") is not None else "encode_only"
        ),
        start_layer=args.layer,
        end_layer=args.layer,
        seq_reduce="last",
        random_seed=42,
        use_bias=True,
        rely_ori_acv=False,
        gpu=args.gpu,
        attention_k1=None,
        attention_selection=None,
        attention_intervention=None,
    )


@torch.no_grad()
def _compute_model(
    args: argparse.Namespace,
    model_name: str,
    dynamic_k_percents: list[float],
    checkpoint_dir: Path,
) -> dict:
    dataset = _dataset_path(model_name, args.subset)
    cache_dir = within_ske_path(model_name, args.cache_base)
    if cache_dir is None:
        raise FileNotFoundError(
            f"no within-skeleton cache for {model_name} below {args.cache_base}"
        )
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset}")

    items = _load_dataset_items(str(dataset), model_name)[: args.n_inputs]
    if not items:
        raise ValueError(f"dataset has no samples: {dataset}")
    print(f"[load] {model_name}: dataset={dataset}, cache={cache_dir}")

    run_args = _runner_args(args, model_name, cache_dir, items[0])
    setup = setup_skeleton_pred(run_args)
    tokenizer = setup["tokenizer"]
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    layer_data = setup["cache"].layer_skeleton(args.layer)
    layer_tensors = setup["layer_gpu_tensors"][args.layer]
    n_sensors, n_subspaces = layer_data["delta"].shape
    n_total = n_sensors * n_subspaces
    k_by_percent = _dynamic_k_values(n_total, dynamic_k_percents)
    largest_k = max(k_by_percent.values())
    checkpoint = _checkpoint_path(checkpoint_dir, model_name)
    resume_start, processed, skipped, totals = _load_checkpoint(
        checkpoint,
        args,
        model_name,
        dynamic_k_percents,
        n_subspaces,
    )
    resume_start = min(resume_start, len(items))

    for batch_start in range(resume_start, len(items), args.batch_size):
        raw_batch = items[batch_start : batch_start + args.batch_size]
        kept_items = []
        kept_indices = []
        target_ids = []
        for offset, item in enumerate(raw_batch):
            input_index = batch_start + offset + 1
            target_id = _resolve_item_target_id(tokenizer, item)
            if target_id is None:
                skipped += 1
                print(f"  [skip] sample {input_index}: target is not one valid token")
                continue
            kept_items.append(item)
            kept_indices.append(input_index)
            target_ids.append(target_id)
        if not kept_items:
            _save_checkpoint(
                checkpoint,
                args,
                model_name,
                dynamic_k_percents,
                n_subspaces,
                min(batch_start + args.batch_size, len(items)),
                processed,
                skipped,
                totals,
            )
            continue

        wrapped_prompts = [
            apply_template(item["prompt"], model_name, tokenizer)
            for item in kept_items
        ]
        tokens = tokenizer(
            wrapped_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=NEXT_TOKEN_MAX_LENGTH,
            add_special_tokens=setup.get("add_special_tokens", False),
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_type_ids = tokens.get("token_type_ids")
        last_real_pos = attention_mask.sum(dim=1) - 1

        # One model forward captures actual MLP activations for this sample batch.
        activations = _collect_original_h(
            setup["model"],
            input_ids,
            [args.layer],
            setup["device"],
            setup["adapter"],
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[args.layer]
        batch_effectors = _compute_per_sample_effectors(
            setup["layer_gpu_tensors"],
            target_ids,
            setup["W_emb"],
            args.layer,
            args.layer,
        )[args.layer]

        for batch_index, (input_index, target_id) in enumerate(
            zip(kept_indices, target_ids)
        ):
            # Both selectors receive the same production effector, including
            # model-specific post-FFN gamma correction.
            effector = batch_effectors[batch_index]
            static_ranks = _static_rank_lookup(
                layer_data,
                setup["W_emb"][target_id],
                effector,
            )
            last_position = int(last_real_pos[batch_index].item())
            activation = activations[
                batch_index : batch_index + 1,
                last_position : last_position + 1,
                :,
            ]
            trace = _dynamic_trace_from_activation(
                activation,
                layer_tensors,
                effector,
                largest_k,
            )
            _accumulate_dynamic_prefixes(trace, static_ranks, k_by_percent, totals)
            processed += 1

        print(
            f"  [samples] {kept_indices[-1]}/{len(items)} "
            f"(batch={len(kept_items)})",
            end="\r",
            flush=True,
        )
        _save_checkpoint(
            checkpoint,
            args,
            model_name,
            dynamic_k_percents,
            n_subspaces,
            min(batch_start + args.batch_size, len(items)),
            processed,
            skipped,
            totals,
        )
        del activations, batch_effectors, tokens
    print()
    if processed == 0:
        raise RuntimeError(f"no valid samples were processed for {model_name}")

    return {
        "n_sensors": n_sensors,
        "n_subspaces": n_subspaces,
        "n_inputs": processed,
        "n_skipped_inputs": skipped,
        "layer": args.layer,
        "curves": _result_curves(totals, k_by_percent, dynamic_k_percents),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dynamic_k_percents = _validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_dir = Path(args.output_dir)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else output_dir.parent / "checkpoints"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for model_name in args.models:
        results[model_name] = _compute_model(
            args, model_name, dynamic_k_percents, checkpoint_dir
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = output_dir / "static_rank_cdf_curves.json"
    payload = {
        "metadata": {
            "layer": args.layer,
            "n_inputs_requested": args.n_inputs,
            "batch_size": args.batch_size,
            "dynamic_k_percents": dynamic_k_percents,
            "static_selector": (
                "ske_pre_valid.forward_hook.forward_per_neuron."
                "select_per_neuron_abs_topk"
            ),
            "dynamic_selector": (
                "ske_pre_valid.within_ske_selector."
                "select_and_reconstruct_skeleton(return_trace=True)"
            ),
            "note": (
                "Empirical CDF of within-sensor Static rank among pathways "
                "in the actual signed Dynamic Top set. Static rank 1 is "
                "best; the uniform baseline is rank / n_subspaces."
            ),
        },
        "models": results,
    }
    _atomic_json_write(output_path, payload)
    for model_name in args.models:
        _checkpoint_path(checkpoint_dir, model_name).unlink(missing_ok=True)
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
