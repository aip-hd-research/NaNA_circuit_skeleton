"""Dataset CLI for the new source-resolved Attention+MLP circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from block_interp.model_load import load_model_and_embeddings
from model_config import NEXT_TOKEN_MAX_LENGTH
from model_config import within_ske_path
from model_config.templates import needs_true_add_special_tokens
from scripts_ske_cir.run_skeleton_pred import _load_dataset_items, apply_template

from .forward import CircuitForwardConfig, DynamicCircuitForward
from .cache import AttentionOVCache
from .types import CircuitMode, PathSelection
from ske_pre_valid.within_ske_cache import WithinSkeCache
from scripts_ske_cir.run_skeleton_pred import CACHE_BASE


ATTENTION_CACHE_BASE = os.environ.get(
    "ATTENTION_OV_CACHE_BASE",
    "/pfs/work9/workspace/scratch/hd_nf184-nana-cache/pre_compute_skeleton_attn_cache",
)


CONDITIONS = {
    "ske_pre": ("topk", "preserve"),
    "ske_abl": ("topk", "ablation"),
    "rnd_pre": ("random", "preserve"),
    "rnd_abl": ("random", "ablation"),
}


def _configuration_tag(args) -> str:
    """Return a filename-safe tag for every setting that changes results."""
    attention_k = "auto" if args.attention_k is None else str(args.attention_k)
    conditions = "-".join(args.conditions)
    topk_tag = f"_top-{args.topk_tokens}" if args.topk_tokens > 0 else ""
    return (
        f"budget-{args.attention_budget}_ak-{attention_k}_"
        f"norm-{args.post_norm_mode}_rely-{int(args.rely_ori_acv)}_"
        f"cond-{conditions}{topk_tag}"
    )


def _completed_result(path: Path) -> bool:
    """Treat only a parseable final JSON object as resumable output."""
    if not path.is_file():
        return False
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, json.JSONDecodeError):
        return False


def _atomic_json_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _args():
    p = argparse.ArgumentParser(description="Dynamic per-head coarse OV + MLP circuits")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset_path", default=None,
                   help="Dataset JSON; alternatively provide --prompt and --target_token")
    p.add_argument("--prompt", default=None,
                   help="One prompt to analyze directly")
    p.add_argument("--target_token", default=None,
                   help="One next-token target for --prompt")
    p.add_argument("--output_dir", default="results/dynamic_ske_mlp_attn")
    p.add_argument("--start_layer", type=int, default=4)
    p.add_argument("--end_layer", type=int)
    p.add_argument("--k_percents", type=float, nargs="+", required=True)
    p.add_argument("--attention_k", type=int)
    p.add_argument("--attention_budget", choices=["global", "per_head"], default="global")
    p.add_argument("--type", choices=["attn", "mlp", "attn_and_mlp"], default="attn_and_mlp")
    p.add_argument(
        "--post_norm_mode", choices=["native", "magnitude_preserving"],
        default="magnitude_preserving",
        help="Gemma 2/3 post-branch RMSNorm policy; no effect on models without post norms",
    )
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=["ske_pre", "ske_abl"])
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument(
        "--sample_start", type=int, default=1,
        help="1-based first dataset sample to process",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--topk_tokens", type=int, default=0,
        help="Save this many highest-probability predicted tokens per condition; "
             "0 disables token-list output.",
    )
    p.add_argument("--gpu", default="0")
    p.add_argument("--rely_ori_acv", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _target_id(item, tokenizer):
    if item.get("target_token_id") is not None:
        return int(item["target_token_id"])
    ids = tokenizer.encode(item["target_token"], add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"target is not one token: {item['target_token']!r} -> {ids}")
    return ids[0]


def _stats(logits, target_id, batch_index=0, tokenizer=None, topk_tokens=0):
    values = logits[batch_index, -1].float()
    target = values[target_id]
    rank = int((values > target).sum().item()) + 1
    probabilities = values.softmax(-1)
    result = {"rank": rank, "logit": float(target.item()),
              "prob": float(probabilities[target_id].item()),
              "top1_id": int(values.argmax().item())}
    if topk_tokens > 0:
        if tokenizer is None:
            raise ValueError("tokenizer is required when topk_tokens is positive")
        count = min(topk_tokens, probabilities.numel())
        top_probabilities, top_ids = torch.topk(probabilities, count)
        result["top_tokens"] = [
            {
                "token": tokenizer.decode([int(token_id)]),
                "token_id": int(token_id),
                "probability": float(probability),
            }
            for token_id, probability in zip(top_ids.tolist(), top_probabilities.tolist())
        ]
    return result


def main():
    args = _args()
    if args.dataset_path is None:
        if args.prompt is None or args.target_token is None:
            raise ValueError("provide --dataset_path or both --prompt and --target_token")
        if args.sample_start != 1:
            raise ValueError("--sample_start must be 1 in direct-prompt mode")
    elif args.prompt is not None or args.target_token is not None:
        raise ValueError("--dataset_path cannot be combined with --prompt or --target_token")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if args.sample_start <= 0:
        raise ValueError("sample_start must be positive")
    if args.topk_tokens < 0:
        raise ValueError("topk_tokens must be non-negative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, output_embedding = load_model_and_embeddings(args.model, device=device)
    model.eval()
    # Source-resolved OV paths require the actual QK softmax matrix. Flash/SDPA
    # kernels commonly return no weights, so use the native eager implementation.
    # This changes only the kernel, not Q/K projections, RoPE, masks or softmax.
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
    cache_path = within_ske_path(args.model, CACHE_BASE)
    if cache_path is None:
        raise RuntimeError(f"no precomputed MLP skeleton cache for {args.model}")
    mlp_cache = WithinSkeCache(cache_path, expected_model_name=args.model)
    attention_cache_path = os.path.join(
        ATTENTION_CACHE_BASE, args.model, "attention_ov_cache"
    )
    attention_cache = AttentionOVCache(
        os.path.realpath(attention_cache_path), expected_model_name=args.model
    )
    runner = DynamicCircuitForward(
        model, mlp_cache=mlp_cache, attention_cache=attention_cache
    )
    end_layer = len(runner.blocks) - 1 if args.end_layer is None else args.end_layer
    if not (0 <= args.start_layer <= end_layer < len(runner.blocks)):
        raise ValueError("invalid layer interval")

    # Preserve the original 1-based dataset index so disjoint workers write
    # disjoint result filenames.
    if args.dataset_path is None:
        all_items = [{"prompt": args.prompt, "target_token": args.target_token}]
        case_id = hashlib.sha256(
            f"{args.prompt}\0{args.target_token}".encode("utf-8")
        ).hexdigest()[:12]
        dataset_name = f"inline_{case_id}"
    else:
        all_items = _load_dataset_items(args.dataset_path, args.model)
        dataset_name = Path(args.dataset_path).stem
    sample_offset = args.sample_start - 1
    items = all_items[sample_offset:sample_offset + args.n_samples]
    out_dir = Path(args.output_dir) / args.model.replace("/", "_") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    add_special = needs_true_add_special_tokens(args.model, tokenizer)
    if args.rely_ori_acv:
        raise ValueError(
            "rely_ori_acv=True is not implemented for the new coarse OV circuit; "
            "the default false uses activations propagated through each intervention"
        )
    mode = {
        "attn": CircuitMode.ATTENTION,
        "mlp": CircuitMode.MLP,
        "attn_and_mlp": CircuitMode.ATTENTION_MLP,
    }[args.type]
    budget = PathSelection(args.attention_budget)
    configuration_tag = _configuration_tag(args)

    # Dimensions are constant across the audited dense model suite. Derive
    # budgets from the first intervened layer and verify lazily per execution.
    attn_skeleton = runner._attention_skeleton(args.start_layer)
    mlp_skeleton = runner._mlp_skeleton(args.start_layer)
    mlp_total = mlp_skeleton.delta.numel()

    # Keep dataset/sample order while batching consecutive equal-length prompts.
    # Mixing lengths would require padding, which can change the native QK
    # probabilities used by the source-resolved attention intervention.  The old
    # implementation globally bucketed by length; although numerically correct,
    # that made result files appear in an arbitrary sample-index order (for
    # example sample 284 could be written before sample 1).
    batches = []
    current_batch = []
    current_length = None
    for sample_index, item in enumerate(items, args.sample_start):
        prompt = apply_template(item["prompt"], args.model, tokenizer)
        encoded = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=NEXT_TOKEN_MAX_LENGTH, add_special_tokens=add_special,
        )
        target_id = _target_id(item, tokenizer)
        length = int(encoded["input_ids"].shape[1])
        if current_batch and (
            length != current_length or len(current_batch) >= args.batch_size
        ):
            batches.append(current_batch)
            current_batch = []
        current_batch.append((sample_index, item, encoded, target_id))
        current_length = length
    if current_batch:
        batches.append(current_batch)

    for records in batches:
        input_ids = torch.cat([r[2]["input_ids"] for r in records]).to(device)
        token_parts = [r[2].get("token_type_ids") for r in records]
        token_type_ids = None
        if all(part is not None for part in token_parts):
            token_type_ids = torch.cat(token_parts).to(device)
        target_ids = torch.tensor([r[3] for r in records], device=output_embedding.device)
        target_direction = output_embedding.index_select(0, target_ids).detach()
        native_inputs = {"input_ids": input_ids, "use_cache": False}
        if token_type_ids is not None:
            native_inputs["token_type_ids"] = token_type_ids
        with torch.no_grad():
            original = model(**native_inputs).logits

        for percent in args.k_percents:
            percent_tag = str(percent).replace(".", "p")
            output_paths = [
                out_dir / (
                    f"{sample_index}_kp{percent_tag}_L{args.start_layer}-{end_layer}_"
                    f"{args.type}_{configuration_tag}.json"
                )
                for sample_index, _, _, _ in records
            ]
            if all(_completed_result(path) for path in output_paths):
                print(
                    f"[batch {records[0][0]}..{records[-1][0]}] kp={percent:g} "
                    "[SKIP complete]"
                )
                continue
            mlp_k = min(mlp_total, max(1, round(mlp_total * percent / 100)))
            attention_k = args.attention_k
            if attention_k is None:
                per_head_paths = attn_skeleton.rank
                attention_total = (
                    per_head_paths if budget == PathSelection.PER_HEAD
                    else per_head_paths * attn_skeleton.num_heads
                )
                attention_k = min(
                    attention_total, max(1, round(attention_total * percent / 100))
                )
            summaries = [{"ori": _stats(
                original, r[3], index, tokenizer, args.topk_tokens
            )}
                         for index, r in enumerate(records)]
            for condition in args.conditions:
                selection, intervention = CONDITIONS[condition]
                config = CircuitForwardConfig(
                    start_layer=args.start_layer, end_layer=end_layer,
                    mlp_k=mlp_k, attention_k=attention_k, mode=mode,
                    selection=selection, intervention=intervention,
                    attention_budget=budget,
                    post_norm_mode=args.post_norm_mode,
                )
                forward_result = runner(
                    input_ids, target_direction, config, token_type_ids=token_type_ids,
                    return_traces=True,
                )
                traces = forward_result["attention_paths"]
                if not isinstance(traces, list):
                    traces = [traces]
                for index, record in enumerate(records):
                    summaries[index][condition] = _stats(
                        forward_result["logits"], record[3], index,
                        tokenizer, args.topk_tokens,
                    )
                    summaries[index][condition]["attention_paths"] = traces[index]

            common = {
                "model_name": args.model,
                "dataset": dataset_name,
                "type": args.type,
                "mode": mode.value,
                "information_flow": (
                    "attention_intervention_then_modified_residual_enters_mlp"
                    if args.type == "attn_and_mlp" and runner.audit.residual == "sequential"
                    else "attention_and_mlp_share_pre_layer_input"
                    if args.type == "attn_and_mlp" and runner.audit.residual == "parallel"
                    else "single_branch_intervention_with_native_downstream_flow"
                ),
                "rely_ori_acv": args.rely_ori_acv,
                "start_layer": args.start_layer,
                "end_layer": end_layer,
                "k_percent": percent,
                "mlp_k": mlp_k,
                "attention_k": attention_k,
                "attention_budget": budget.value,
                "post_norm_mode": args.post_norm_mode,
                "conditions": list(args.conditions),
                "topk_tokens": args.topk_tokens,
                "batch_size_requested": args.batch_size,
                "batch_size_effective": len(records),
                "norm_audit": {
                    "model_type": runner.audit.model_type,
                    "residual": runner.audit.residual,
                    "attention_input": runner.audit.layers[0].attention_input.__dict__,
                    "attention_output": None if runner.audit.layers[0].attention_output is None else runner.audit.layers[0].attention_output.__dict__,
                    "mlp_input": runner.audit.layers[0].mlp_input.__dict__,
                    "mlp_output": None if runner.audit.layers[0].mlp_output is None else runner.audit.layers[0].mlp_output.__dict__,
                },
            }
            for index, ((sample_index, item, _, target_id), output_path) in enumerate(
                zip(records, output_paths)
            ):
                if _completed_result(output_path):
                    continue
                result = dict(common)
                result.update({
                    "sample_idx": sample_index,
                    "prompt": item["prompt"],
                    "target_token": item["target_token"],
                    "target_id": target_id,
                    "summary": summaries[index],
                })
                _atomic_json_write(output_path, result)
                sample_end = args.sample_start + len(items) - 1
                print(f"[{sample_index}/{sample_end}] kp={percent:g} -> {output_path}")


if __name__ == "__main__":
    main()
