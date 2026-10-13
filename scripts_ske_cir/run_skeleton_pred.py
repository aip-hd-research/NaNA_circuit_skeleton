"""
run_skeleton_pred.py  —  Orchestrator for skeleton prediction experiments.

Mirrors run_static_ske.py: resolves end_layer (last cached layer), k1 and cache_dir;
all loop variables come from run_dynm_ske.sh.

Usage (invoked by run_dynm_ske.sh, or run directly):

  python scripts_ske_cir/run_skeleton_pred.py \
      --models "openai-community/gpt2-medium" \
      --target " happy" \
      --in_seq "The cat looks very" \
      --k_percents 0.1 0.5 1.0 \
      --start_layer 5 \
      --sample_idx 0

Pass all k_percent values at once so the model is loaded only once per run.
Use --k_percent (singular) for a single value (backward compatible).
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model_config import (
    NEXT_TOKEN_MAX_LENGTH,
    PROMPT_FORMAT_TAG,
    PROMPT_FORMAT_VERSION,
    apply_template,
    safe_slug,
    within_ske_path,
)
from ske_pre_valid.model_adapter import is_gemma_style, GEMMA_APPROX_INTERVENTION_NOTE
from ske_pre_valid.skeleton_pred import (
    setup_skeleton_pred, eval_one_k, CONDITIONS,
    update_effectors_for_target,
    _compute_per_sample_effectors, eval_batch_k,
)
from result_summary import write_dynamic_model_mean


def _model_result_dir(output_dir: str, safe_model: str) -> str:
    """Return the model directory, supporting model-scoped decomposition runs."""
    if os.environ.get("NANA_MODEL_SCOPED_OUTPUT") == "1":
        return output_dir
    return os.path.join(output_dir, safe_model)
import torch

CACHE_BASE          = os.environ.get(
    "SKELETON_CACHE_BASE", os.path.join(_ROOT, "pre_compute_skeleton_cache")
)
DEFAULT_OUTPUT_DIR  = "results/dynamic_ske"

# v2 used direct raw down_proj replacement followed by the native post-FFN RMSNorm.
# v3 restores magnitude-preserving Gemma interventions with an unclamped raw-norm ratio.
# v4 preserves tokenizer-provided token_type_ids (required by ChemGPT's BERT-style
# tokenizer) in both Original and intervened forwards.
RESULT_FORMAT_VERSION = 4
RESULT_FORMAT_TAG = f"rv{RESULT_FORMAT_VERSION}"


def _report_mean_summary(path: str, complete: bool) -> None:
    state = "complete" if complete else "incomplete"
    print(f"  model mean summary ({state}) -> {path}")


def last_layer(within: str) -> int:
    files = sorted(glob.glob(os.path.join(within, "layer_*.pt")))
    if not files:
        raise RuntimeError(f"no layer_*.pt files found in: {within}")
    return max(int(re.search(r"layer_(\d+)", f).group(1)) for f in files)


def compute_k(within: str, k_percent: float) -> int:
    if not 0 <= k_percent <= 100:
        raise ValueError(f"k_percent must be in [0, 100], got {k_percent}")
    layer_files = sorted(glob.glob(os.path.join(within, "layer_*.pt")))
    if not layer_files:
        raise RuntimeError(f"no layer_*.pt files found in: {within}")
    d = torch.load(layer_files[0], weights_only=True, map_location="cpu")
    # Alternative decompositions may contain structural zeros.  In particular,
    # the neuron-level factorization W = I @ W has d_ff real neuron atoms rather
    # than d_ff**2 candidate matrix entries.  Ordinary dense decompositions omit
    # this field and retain the historical d_ff*n_subspaces budget.
    path_count = int(d.get(
        "natural_path_count",
        int(d["n_neurons"]) * int(d["n_subspaces"]),
    ))
    return math.ceil(path_count * k_percent / 100)


def _k_percent_tag(k_percent: float) -> str:
    """Stable filesystem spelling that distinguishes requested sweep points."""
    return format(float(k_percent), ".12g").replace("-", "m").replace(".", "p")


def _resolve_item_target_id(tokenizer, item: dict) -> int | None:
    """Target id for one dataset item, or None if it can't be resolved cleanly.

    gen_next_token_labels.py output carries the model's own argmax id as
    target_token_id. Datasets that only provide target_token (e.g. the
    factual/counterfactual sets, where the
    target word is shared across models and can't be precomputed per-model)
    fall back to tokenizer.encode() here — callers must skip the item when that
    does not produce exactly one non-empty token rather than silently scoring
    against the unk id.
    """
    tid = item.get("target_token_id")
    if tid is not None:
        from ske_pre_valid.static_skeleton import validate_precomputed_target_id
        try:
            return validate_precomputed_target_id(
                tokenizer, item["target_token"], tid
            )
        except ValueError:
            return None
    # Text-only factual/counterfactual datasets have no model-generated id.
    # For these, the tokenizer's real encode path is authoritative: a direct
    # vocabulary lookup may find a second, same-looking token (for example the
    # word-initial versus space-prefixed form), but that is not evidence that
    # encode selected the wrong next-token event.
    encoded = tokenizer.encode(item["target_token"], add_special_tokens=False)
    valid = [tid for tid in encoded if tokenizer.decode([tid]).strip()]
    if len(valid) != 1:
        return None
    return valid[0]


def _load_dataset_items(
    dataset_path: str, model_name: str, *, preserve_indices: bool = False,
) -> list[dict]:
    """Load samples and reject stale model-generated next-token labels.

    Flat factual/counterfactual datasets have no metadata envelope and remain
    supported.  Files emitted by gen_next_token_labels.py contain ``samples``;
    those labels are model- and prompt-format-specific and must match exactly.
    """
    with open(dataset_path) as f:
        data = json.load(f)
    if "samples" not in data:
        samples = data
        if preserve_indices:
            return [
                {**item, "_sample_idx": int(index)}
                for index, item in sorted(samples.items(), key=lambda pair: int(pair[0]))
            ]
        return list(samples.values())

    version = data.get("prompt_format_version")
    if version != PROMPT_FORMAT_VERSION:
        raise RuntimeError(
            f"stale next-token dataset {dataset_path}: prompt_format_version="
            f"{version!r}, expected {PROMPT_FORMAT_VERSION}; rerun "
            "dataset_gene/dymn_dst_gene/gen_next_token_labels.py"
        )
    dataset_model = data.get("model")
    if dataset_model is not None and dataset_model != model_name:
        raise RuntimeError(
            f"next-token dataset model mismatch: file has {dataset_model!r}, "
            f"runner requested {model_name!r}"
        )
    samples = data["samples"]
    if preserve_indices:
        return [
            {**item, "_sample_idx": int(index)}
            for index, item in sorted(samples.items(), key=lambda pair: int(pair[0]))
        ]
    return list(samples.values())


def _slice_dataset_items(
    dataset_path: str,
    model_name: str,
    sample_start: int,
    n_samples: int | None,
) -> list[dict]:
    """Return a positional dataset slice while retaining global 1-based indices."""
    if sample_start <= 0:
        raise ValueError(f"sample_start must be positive, got {sample_start}")
    if n_samples is not None and n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")

    items = [
        {**item, "_sample_idx": index}
        for index, item in enumerate(_load_dataset_items(dataset_path, model_name), 1)
    ]
    sample_offset = sample_start - 1
    sample_stop = None if n_samples is None else sample_offset + n_samples
    return items[sample_offset:sample_stop]


def run_model(model_name: str, target: str, in_seq: str, k_percents: list,
              start_layer: int, rely_ori_acv: bool,
              gpu: str, topk_tokens: int,
              sample_idx: int, output_dir: str,
              gemma_restore_magnit: bool = False,
              conditions: list | None = None) -> None:
    within = within_ske_path(model_name, CACHE_BASE)
    if within is None:
        print(f"[SKIP] no within_ske_cache for: {model_name}")
        return

    try:
        end = last_layer(within)
    except RuntimeError as e:
        print(f"[SKIP] {model_name}: {e}")
        return
    safe_model   = model_name.replace("/", "_")
    safe_target  = safe_slug(target)
    vocab_tag    = "raw_dot"
    ori_acv_tag  = "ori_acv_true" if rely_ori_acv     else "ori_acv_false"
    out_dir      = _model_result_dir(output_dir, safe_model)
    os.makedirs(out_dir, exist_ok=True)

    # Determine active conditions (None → all four)
    _requested   = set(conditions) if conditions else set()
    active_conds = tuple(c for c in CONDITIONS if (not _requested or c in _requested))
    conds_tag    = "-".join(active_conds) if _requested else "all"

    def _write_mean_summary() -> None:
        result = write_dynamic_model_mean(
            output_dir=output_dir,
            model_name=model_name,
            dataset_name="default",
            expected_items=[{"prompt": in_seq, "target_token": target}],
            sample_indices=[sample_idx],
            k_percents=k_percents,
            start_layer=start_layer,
            end_layer=end,
            rely_ori_acv=rely_ori_acv,
            conditions=active_conds,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            result_format_version=RESULT_FORMAT_VERSION,
        )
        _report_mean_summary(*result)

    # Check if all k values are already done before loading the model
    pending = []
    for k_percent in k_percents:
        k1 = compute_k(within, k_percent)
        out_file = os.path.join(
            out_dir,
            f"{sample_idx}_{safe_target}_{vocab_tag}_{ori_acv_tag}"
            f"_kp{_k_percent_tag(k_percent)}_k{k1}"
            f"_L{start_layer}-{end}_{conds_tag}_{PROMPT_FORMAT_TAG}_{RESULT_FORMAT_TAG}.json",
        )
        if os.path.exists(out_file):
            print(f"  [SKIP] exists: {out_file}")
        else:
            pending.append((k_percent, k1, out_file))

    if not pending:
        print(f"[SKIP] all k values done for {model_name}")
        _write_mean_summary()
        return

    print(f"====== {model_name} | target={target!r} | layers {start_layer}-{end} | "
          f"{len(pending)}/{len(k_percents)} k values to run ======")

    # Load model once for all pending k values
    args = types.SimpleNamespace(
        model_name           = model_name,
        cache_dir            = within,
        in_seq               = in_seq,   # wrapped below, once tokenizer is available
        target_word          = target,
        start_layer          = start_layer,
        end_layer            = end,
        k1                   = None,
        topk_tokens          = topk_tokens,
        seq_reduce           = "last",
        random_seed          = 42,
        use_bias             = True,
        rely_ori_acv         = rely_ori_acv,
        gpu                  = gpu,
        gemma_restore_magnit = gemma_restore_magnit,
        conditions           = list(active_conds),
    )
    model_setup = setup_skeleton_pred(args)
    tokenizer   = model_setup["tokenizer"]
    gemma_note  = (GEMMA_APPROX_INTERVENTION_NOTE
                   if is_gemma_style(model_setup["adapter"].get_blocks(model_setup["model"]))
                   else None)
    args.in_seq = apply_template(in_seq, model_name, tokenizer)
    print(f"  wrapped in_seq: {args.in_seq[:80]!r}")

    ske_label = "Skeleton preserve (ori_acv)" if rely_ori_acv else "Skeleton preserve"
    abl_label = "Skeleton ablation (ori_acv)" if rely_ori_acv else "Skeleton ablation"
    condition_labels = {
        "ori":     "Original",
        "ske_pre": ske_label,
        "ske_abl": abl_label,
        "rnd_pre": "Random preserve",
        "rnd_abl": "Random ablation",
    }

    def _as_py(v):
        """Convert tensors / numpy scalars to plain Python types for JSON."""
        return v.item() if hasattr(v, "item") else v

    for k_percent, k1, out_file in pending:
        print(f"  k_percent={k_percent}%  k1={k1}")
        args.k1 = k1
        stats, logits = eval_one_k(k1, model_setup, args, conditions=list(active_conds))

        summary = []
        for key in ("ori",) + active_conds:
            top1_id = int(logits[key].cpu().argmax())
            entry = {
                "label":         condition_labels[key],
                "top1_token":    tokenizer.decode([top1_id]),
                "top1_token_id": top1_id,
            }
            for k, v in stats[key].items():
                entry[k] = _as_py(v)
            summary.append(entry)

        torch.cuda.empty_cache()
        result = {
            "model_name":       model_name,
            "decomposition":    model_setup.get("decomposition", {"method": "svd"}),
            "prompt_format_version": PROMPT_FORMAT_VERSION,
            "result_format_version": RESULT_FORMAT_VERSION,
            "in_seq":           in_seq,
            "sample_idx":       sample_idx,
            "target_word":      target,
            "target_id":        model_setup["target_id"],
            "k_percent":        k_percent,
            "k1":               k1,
            "start_layer":      start_layer,
            "end_layer":        end,
            "effector_mode":     "raw_dot",
            "rely_ori_acv":     rely_ori_acv,
            "gemma_restore_magnit": gemma_note is not None,
            "gemma_restore_mode": (
                "unclamped_raw_norm_ratio" if gemma_note is not None else "disabled"
            ),
            "gemma_intervention_mode": (
                "norm_match_then_native_post_ffn_norm_then_raw_norm_restore"
                if gemma_note is not None else "raw_down_proj_replacement"
            ),
            "conditions":       list(active_conds),
            "gemma_approx_intervention_note": gemma_note,
            "summary":          summary,
        }
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  saved -> {out_file}")

    torch.cuda.empty_cache()
    _write_mean_summary()


def run_model_dataset(
    model_name: str, dataset_path: str, n_samples: int | None,
    k_percents: list, start_layer: int, rely_ori_acv: bool,
    gpu: str, topk_tokens: int, output_dir: str,
    sample_start: int = 1,
    gemma_restore_magnit: bool = False,
    conditions: list | None = None,
    attention_k: int | None = None,
) -> None:
    """Run dynamic skeleton prediction over a JSON dataset.

    The model is loaded ONCE per (dataset, model) call.  For each sample the
    effector scores are recomputed from the pre-GPU V tensors (fast GPU matmul)
    so that changing target tokens does not require reloading model weights.

    Dataset format (dict keyed by string integers):
        {"1": {"prompt": "...", "target_token": " X", ...}, "2": {...}, ...}

    Output goes to:
        {output_dir}/{safe_model}/{dataset_name}/{idx}_{safe_target}_..._{conds_tag}.json
    """
    within = within_ske_path(model_name, CACHE_BASE)
    if within is None:
        print(f"[SKIP] no within_ske_cache for: {model_name}")
        return

    try:
        end = last_layer(within)
    except RuntimeError as e:
        print(f"[SKIP] {model_name}: {e}")
        return
    items = _slice_dataset_items(dataset_path, model_name, sample_start, n_samples)
    if not items:
        print(f"[SKIP] empty dataset slice for {model_name}: "
              f"start={sample_start}, n_samples={n_samples}")
        return

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    safe_model   = model_name.replace("/", "_")
    vocab_tag    = "raw_dot"
    ori_acv_tag  = "ori_acv_true" if rely_ori_acv     else "ori_acv_false"
    out_dir      = os.path.join(_model_result_dir(output_dir, safe_model), dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    _requested   = set(conditions) if conditions else set()
    active_conds = tuple(c for c in CONDITIONS if (not _requested or c in _requested))
    conds_tag    = "-".join(active_conds) if _requested else "all"
    attention_tag = f"_attnk{attention_k}" if attention_k is not None else ""

    def _write_mean_summary() -> None:
        result = write_dynamic_model_mean(
            output_dir=output_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            expected_items=items,
            k_percents=k_percents,
            start_layer=start_layer,
            end_layer=end,
            rely_ori_acv=rely_ori_acv,
            conditions=active_conds,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            result_format_version=RESULT_FORMAT_VERSION,
            sample_indices=[int(item["_sample_idx"]) for item in items],
        )
        _report_mean_summary(*result)

    # Check if there's any pending work before loading model
    pending_any = False
    for item in items:
        idx = int(item["_sample_idx"])
        safe_target = safe_slug(item["target_token"])
        for k_percent in k_percents:
            k1 = compute_k(within, k_percent)
            out_file = os.path.join(
                out_dir,
                f"{idx}_{safe_target}_{vocab_tag}_{ori_acv_tag}"
                f"_kp{_k_percent_tag(k_percent)}_k{k1}"
                f"{attention_tag}_L{start_layer}-{end}_{conds_tag}_{PROMPT_FORMAT_TAG}_{RESULT_FORMAT_TAG}.json",
            )
            if not os.path.exists(out_file):
                pending_any = True
                break
        if pending_any:
            break
    if not pending_any:
        print(f"[SKIP] all samples done for {model_name} / {dataset_name}")
        _write_mean_summary()
        return

    print(f"====== {model_name} | dataset={dataset_name} | {len(items)} samples "
          f"| layers {start_layer}-{end} ======")

    # Setup once with the first sample's target
    first_item = items[0]

    args = types.SimpleNamespace(
        model_name           = model_name,
        cache_dir            = within,
        in_seq               = first_item["prompt"],   # wrapped per-item below, once tokenizer is available
        target_word          = first_item["target_token"],
        target_id            = first_item.get("target_token_id"),  # None → setup_skeleton_pred resolves via tokenizer
        target_resolution    = "encode_only" if first_item.get("target_token_id") is None else "strict",
        start_layer          = start_layer,
        end_layer            = end,
        k1                   = None,
        topk_tokens          = topk_tokens,
        seq_reduce           = "last",
        random_seed          = 42,
        use_bias             = True,
        rely_ori_acv         = rely_ori_acv,
        gpu                  = gpu,
        gemma_restore_magnit = gemma_restore_magnit,
        conditions           = list(active_conds),
        attention_k1         = attention_k,
        attention_selection  = None,
        attention_intervention = None,
    )
    model_setup = setup_skeleton_pred(args)
    tokenizer   = model_setup["tokenizer"]
    gemma_note  = (GEMMA_APPROX_INTERVENTION_NOTE
                   if is_gemma_style(model_setup["adapter"].get_blocks(model_setup["model"]))
                   else None)

    condition_labels = {
        "ori":     "Original",
        "ske_pre": "Skeleton preserve (ori_acv)" if rely_ori_acv else "Skeleton preserve",
        "ske_abl": "Skeleton ablation (ori_acv)"  if rely_ori_acv else "Skeleton ablation",
        "rnd_pre": "Random preserve",
        "rnd_abl": "Random ablation",
    }

    def _as_py(v):
        return v.item() if hasattr(v, "item") else v

    # Keyed by (decoded target string, precomputed target_token_id) — the decoded
    # string alone is not a reliable identity: distinct subword tokens (e.g. CodeLlama
    # long/short variants) can decode to the same text but carry different ids, and
    # comparing strings only would silently reuse the wrong target_id/effectors.
    current_key = (first_item["target_token"], first_item.get("target_token_id"))

    for position, item in enumerate(items, 1):
        idx = int(item["_sample_idx"])
        prompt       = item["prompt"]
        target_token = item["target_token"]
        safe_target  = safe_slug(target_token)
        wrapped      = apply_template(prompt, model_name, tokenizer)

        # Update effectors if target changed (fast GPU matmul)
        item_key = (target_token, item.get("target_token_id"))
        if item_key != current_key:
            # target_token_id is the model's own argmax id from gen_next_token_labels.py —
            # use it directly rather than re-resolving target_token (the decode()'d string)
            # through _resolve_target_info, which is redundant and, for SentencePiece models,
            # relies on decode()/encode() staying complementary for single-token round-trips.
            # Datasets without a precomputed id (factual/counterfactual) resolve here instead.
            from ske_pre_valid.static_skeleton import _token_family_ids
            new_target_id = _resolve_item_target_id(tokenizer, item)
            if new_target_id is None:
                print(f"  [SKIP] {idx}/{len(items)}: target {target_token!r} "
                      f"not resolvable for {model_name}")
                continue
            update_effectors_for_target(
                model_setup["layer_gpu_tensors"], new_target_id,
                model_setup["W_emb"], start_layer, end,
            )
            model_setup["target_id"]   = new_target_id
            model_setup["family_ids"]  = _token_family_ids(tokenizer, target_token, target_id=new_target_id)
            current_key = item_key

        args.in_seq     = wrapped
        args.target_word = target_token

        # Determine pending k values for this sample
        pending = []
        for k_percent in k_percents:
            k1 = compute_k(within, k_percent)
            out_file = os.path.join(
                out_dir,
                f"{idx}_{safe_target}_{vocab_tag}_{ori_acv_tag}"
                f"_kp{_k_percent_tag(k_percent)}_k{k1}"
                f"{attention_tag}_L{start_layer}-{end}_{conds_tag}_{PROMPT_FORMAT_TAG}_{RESULT_FORMAT_TAG}.json",
            )
            if os.path.exists(out_file):
                pass
            else:
                pending.append((k_percent, k1, out_file))

        if not pending:
            continue

        print(f"  [{position}/{len(items)}; sample={idx}] prompt={prompt[:60]!r}  target={target_token!r}"
              f"  ({len(pending)} k values)")

        for k_percent, k1, out_file in pending:
            args.k1 = k1
            stats, logits = eval_one_k(k1, model_setup, args, conditions=list(active_conds))
            summary = []
            for key in ("ori",) + active_conds:
                top1_id = int(logits[key].cpu().argmax())
                entry = {
                    "label":         condition_labels[key],
                    "top1_token":    tokenizer.decode([top1_id]),
                    "top1_token_id": top1_id,
                }
                for k, v in stats[key].items():
                    entry[k] = _as_py(v)
                summary.append(entry)

            result = {
                "model_name":       model_name,
                "decomposition":    model_setup.get("decomposition", {"method": "svd"}),
                "prompt_format_version": PROMPT_FORMAT_VERSION,
                "result_format_version": RESULT_FORMAT_VERSION,
                "dataset":          dataset_name,
                "sample_idx":       idx,
                "prompt":           prompt,
                "target_token":     target_token,
                "target_id":        model_setup["target_id"],
                "k_percent":        k_percent,
                "k1":               k1,
                "attention_k1":     attention_k,
                "circuit_mode":     "attention_mlp" if attention_k is not None else "mlp",
                "start_layer":      start_layer,
                "end_layer":        end,
                "effector_mode":     "raw_dot",
                "rely_ori_acv":     rely_ori_acv,
                "gemma_restore_magnit": gemma_note is not None,
                "gemma_restore_mode": (
                    "unclamped_raw_norm_ratio" if gemma_note is not None else "disabled"
                ),
                "gemma_intervention_mode": (
                    "norm_match_then_native_post_ffn_norm_then_raw_norm_restore"
                    if gemma_note is not None else "raw_down_proj_replacement"
                ),
                "conditions":       list(active_conds),
                "gemma_approx_intervention_note": gemma_note,
                "summary":          summary,
            }
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    _write_mean_summary()


def run_model_dataset_batched(
    model_name: str, dataset_path: str, n_samples: int | None,
    k_percents: list, start_layer: int, rely_ori_acv: bool,
    gpu: str, topk_tokens: int, output_dir: str, batch_size: int,
    sample_start: int = 1,
    gemma_restore_magnit: bool = False,
    conditions: list | None = None,
) -> None:
    """Batched version of run_model_dataset — processes batch_size samples per forward pass.

    Uses right-padded tokenisation so causal attention is exact (no attention-mask changes
    needed: real tokens cannot attend to future padding positions under causal masking).

    Falls back to sequential run_model_dataset when rely_ori_acv=True (not yet supported
    in batch mode) or when batch_size <= 1.
    """
    if rely_ori_acv or batch_size <= 1:
        return run_model_dataset(
            model_name, dataset_path, n_samples, k_percents,
            start_layer, rely_ori_acv,
            gpu, topk_tokens, output_dir,
            sample_start=sample_start,
            gemma_restore_magnit=gemma_restore_magnit,
            conditions=conditions,
        )

    within = within_ske_path(model_name, CACHE_BASE)
    if within is None:
        print(f"[SKIP] no within_ske_cache for: {model_name}")
        return

    try:
        end = last_layer(within)
    except RuntimeError as e:
        print(f"[SKIP] {model_name}: {e}")
        return
    items = _slice_dataset_items(dataset_path, model_name, sample_start, n_samples)
    if not items:
        print(f"[SKIP] empty dataset slice for {model_name}: "
              f"start={sample_start}, n_samples={n_samples}")
        return

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    safe_model   = model_name.replace("/", "_")
    vocab_tag    = "raw_dot"
    ori_acv_tag  = "ori_acv_true" if rely_ori_acv     else "ori_acv_false"
    out_dir      = os.path.join(_model_result_dir(output_dir, safe_model), dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    _requested   = set(conditions) if conditions else set()
    active_conds = tuple(c for c in CONDITIONS if (not _requested or c in _requested))
    conds_tag    = "-".join(active_conds) if _requested else "all"

    def _write_mean_summary() -> None:
        result = write_dynamic_model_mean(
            output_dir=output_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            expected_items=items,
            k_percents=k_percents,
            start_layer=start_layer,
            end_layer=end,
            rely_ori_acv=rely_ori_acv,
            conditions=active_conds,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            result_format_version=RESULT_FORMAT_VERSION,
            sample_indices=[int(item["_sample_idx"]) for item in items],
        )
        _report_mean_summary(*result)

    k1_map = {kp: compute_k(within, kp) for kp in k_percents}

    def _out_file(idx, item, k_percent):
        safe_target = safe_slug(item["target_token"])
        k1 = k1_map[k_percent]
        return os.path.join(
            out_dir,
            f"{idx}_{safe_target}_{vocab_tag}_{ori_acv_tag}"
            f"_kp{_k_percent_tag(k_percent)}_k{k1}"
            f"_L{start_layer}-{end}_{conds_tag}"
            f"_{PROMPT_FORMAT_TAG}_{RESULT_FORMAT_TAG}.json",
        )

    # Check if any work remains
    pending_any = any(
        not os.path.exists(_out_file(int(item["_sample_idx"]), item, kp))
        for item in items
        for kp in k_percents
    )
    if not pending_any:
        print(f"[SKIP] all samples done for {model_name} / {dataset_name}")
        _write_mean_summary()
        return

    print(f"====== {model_name} | dataset={dataset_name} | {len(items)} samples"
          f" | layers {start_layer}-{end} | batch_size={batch_size} ======")

    first_item = items[0]

    args0 = types.SimpleNamespace(
        model_name           = model_name,
        cache_dir            = within,
        in_seq               = first_item["prompt"],   # wrapped per-batch below, once tokenizer is available
        target_word          = first_item["target_token"],
        target_id            = first_item.get("target_token_id"),  # None → setup_skeleton_pred resolves via tokenizer
        target_resolution    = "encode_only" if first_item.get("target_token_id") is None else "strict",
        start_layer          = start_layer,
        end_layer            = end,
        k1                   = None,
        topk_tokens          = topk_tokens,
        seq_reduce           = "last",
        random_seed          = 42,
        use_bias             = True,
        rely_ori_acv         = rely_ori_acv,
        gpu                  = gpu,
        gemma_restore_magnit = gemma_restore_magnit,
        conditions           = list(active_conds),
    )
    model_setup = setup_skeleton_pred(args0)
    tokenizer   = model_setup["tokenizer"]
    gemma_note  = (GEMMA_APPROX_INTERVENTION_NOTE
                   if is_gemma_style(model_setup["adapter"].get_blocks(model_setup["model"]))
                   else None)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    condition_labels = {
        "ori":     "Original",
        "ske_pre": "Skeleton preserve",
        "ske_abl": "Skeleton ablation",
        "rnd_pre": "Random preserve",
        "rnd_abl": "Random ablation",
    }

    def _as_py(v):
        return v.item() if hasattr(v, "item") else v

    # Keyed by (target_word, target_token_id) — the decoded word alone is not a
    # reliable identity: distinct subword tokens (e.g. CodeLlama long/short variants)
    # can decode to the same text but carry different ids, and keying on the string
    # only would silently reuse the wrong cached target_id/family_ids for such items.
    _family_cache: dict = {}   # (target_word, target_token_id) -> (target_id, family_ids)
    _RESOLVE_FAILED = object()  # sentinel: must be a single instance shared across all batches

    for batch_start in range(0, len(items), batch_size):
        batch_items   = items[batch_start : batch_start + batch_size]
        batch_indices = [int(item["_sample_idx"]) for item in batch_items]

        # Skip batch if all output files already exist
        if all(
            os.path.exists(_out_file(idx, item, kp))
            for idx, item in zip(batch_indices, batch_items)
            for kp in k_percents
        ):
            continue

        # Token families per sample, memoised across batches since the same target word
        # is often reused across many samples. When the dataset carries a precomputed id
        # (gen_next_token_labels.py's own argmax id) it's used directly; datasets that only
        # provide target_token (factual/counterfactual) resolve it here via the tokenizer.
        # Text-only items whose target does not encode to exactly one non-empty token
        # are dropped rather than scored against the unk id.
        from ske_pre_valid.static_skeleton import _token_family_ids
        for item in batch_items:
            tgt = item["target_token"]
            tid = item.get("target_token_id")
            cache_key = (tgt, tid)
            if cache_key not in _family_cache:
                resolved_tid = _resolve_item_target_id(tokenizer, item)
                if resolved_tid is None:
                    resolved_tid = _RESOLVE_FAILED
                _family_cache[cache_key] = (
                    _RESOLVE_FAILED if resolved_tid is _RESOLVE_FAILED else resolved_tid,
                    None if resolved_tid is _RESOLVE_FAILED
                    else _token_family_ids(tokenizer, tgt, target_id=resolved_tid),
                )

        kept_items, kept_indices = [], []
        batch_target_ids, batch_family_ids = [], []
        for idx, item in zip(batch_indices, batch_items):
            cached_tid, fam_ids = _family_cache[(item["target_token"], item.get("target_token_id"))]
            if cached_tid is _RESOLVE_FAILED:
                print(f"  [SKIP] {idx}/{len(items)}: target {item['target_token']!r} "
                      f"not resolvable for {model_name}")
                continue
            kept_items.append(item)
            kept_indices.append(idx)
            batch_target_ids.append(cached_tid)
            batch_family_ids.append(fam_ids)
        batch_items, batch_indices = kept_items, kept_indices
        if not batch_items:
            continue

        batch_prompts = [item["prompt"] for item in batch_items]
        batch_targets = [item["target_token"] for item in batch_items]
        batch_wrapped = [apply_template(p, model_name, tokenizer) for p in batch_prompts]

        # Tokenize with right-padding. add_special_tokens defaults to False, matching the
        # static-prompt dataset builder and avoiding tokenizers (e.g. ChemGPT) silently
        # inserting [CLS]/[SEP]. model_setup["add_special_tokens"] is True for base LMs
        # that need their real BOS restored at the token-id level (see model_config.templates).
        tokens = tokenizer(
            batch_wrapped,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=NEXT_TOKEN_MAX_LENGTH,
            add_special_tokens=model_setup.get("add_special_tokens", False),
        )
        input_ids     = tokens["input_ids"]           # [b, max_seq]
        token_type_ids = tokens.get("token_type_ids")
        attention_mask = tokens["attention_mask"]
        last_real_pos = tokens["attention_mask"].sum(dim=1) - 1  # [b]

        print(f"  batch [{batch_start+1}-{batch_start+len(batch_items)}]/{len(items)}"
              f" | lens={last_real_pos.tolist()} | targets={batch_targets}")

        # Pre-compute per-sample effectors once per batch (shared across all k values)
        per_sample_effectors = _compute_per_sample_effectors(
            model_setup["layer_gpu_tensors"],
            batch_target_ids,
            model_setup["W_emb"],
            start_layer, end,
        )

        # "Original" (unmodified) forward also depends only on (model, input_ids) — not
        # on k_percent — so it's computed once per batch here rather than once per
        # k_percent inside eval_batch_k (which redid this full forward pass on every one
        # of the k_percents.sweep values otherwise, e.g. 11x for a typical sweep).
        device_ = model_setup["device"]
        b_range = torch.arange(input_ids.shape[0], device=device_)
        last_dev = last_real_pos.to(device_)
        model_inputs = {"input_ids": input_ids.to(device_)}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids.to(device_)
        model_inputs["attention_mask"] = attention_mask.to(device_)
        with torch.no_grad():
            orig_all = model_setup["model"](**model_inputs).logits  # [b, max_seq, vocab]
        # device_map="auto" can place the lm_head output on a different GPU than device_
        # (multi-GPU models, e.g. Llama-2-13b-hf) — index on orig_all's actual device.
        _orig_dev = orig_all.device
        batch_orig_logits = orig_all[b_range.to(_orig_dev), last_dev.to(_orig_dev), :]  # [b, vocab]
        for k_percent in k_percents:
            k1 = k1_map[k_percent]
            args0.k1 = k1

            batch_results = eval_batch_k(
                k1, model_setup, input_ids, batch_target_ids,
                last_real_pos, per_sample_effectors, args0,
                conditions=list(active_conds),
                batch_family_ids=batch_family_ids,
                orig_logits=batch_orig_logits,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

            for i, (idx, item) in enumerate(zip(batch_indices, batch_items)):
                out_file = _out_file(idx, item, k_percent)
                if os.path.exists(out_file):
                    continue

                stats_i, logits_i = batch_results[i]
                summary = []
                for key in ("ori",) + active_conds:
                    top1_id = int(logits_i[key].cpu().argmax())
                    entry = {
                        "label":         condition_labels[key],
                        "top1_token":    tokenizer.decode([top1_id]),
                        "top1_token_id": top1_id,
                    }
                    for k_name, v in stats_i[key].items():
                        entry[k_name] = _as_py(v)
                    summary.append(entry)

                result = {
                    "model_name":       model_name,
                    "decomposition":    model_setup.get("decomposition", {"method": "svd"}),
                    "prompt_format_version": PROMPT_FORMAT_VERSION,
                    "result_format_version": RESULT_FORMAT_VERSION,
                    "dataset":          dataset_name,
                    "sample_idx":       idx,
                    "prompt":           item["prompt"],
                    "target_token":     item["target_token"],
                    "target_id":        batch_target_ids[i],
                    "k_percent":        k_percent,
                    "k1":               k1,
                    "start_layer":      start_layer,
                    "end_layer":        end,
                    "effector_mode":     "raw_dot",
                    "rely_ori_acv":     rely_ori_acv,
                    "gemma_restore_magnit": gemma_note is not None,
                    "gemma_restore_mode": (
                        "unclamped_raw_norm_ratio" if gemma_note is not None else "disabled"
                    ),
                    "gemma_intervention_mode": (
                        "norm_match_then_native_post_ffn_norm_then_raw_norm_restore"
                        if gemma_note is not None else "raw_down_proj_replacement"
                    ),
                    "conditions":       list(active_conds),
                    "gemma_approx_intervention_note": gemma_note,
                    "summary":          summary,
                }
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    _write_mean_summary()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",           nargs="+", required=True)
    p.add_argument("--target",           default=None,
                   help="Target word (single-sample mode; mutually exclusive with --dataset_path)")
    p.add_argument("--in_seq",           default=None,
                   help="Input sequence (single-sample mode)")
    p.add_argument("--dataset_path",     default=None,
                   help="Path to dataset JSON (dataset mode; mutually exclusive with --in_seq/--target)")
    p.add_argument("--n_samples",        type=int, default=None,
                   help="Max samples to process from dataset (default: all)")
    p.add_argument("--sample_start",     type=int, default=1,
                   help="1-based first dataset sample used with --n_samples (default: 1)")
    p.add_argument("--k_percents",       nargs="+", type=float, default=None,
                   help="One or more k%% values swept in a single model load (preferred)")
    p.add_argument("--k_percent",        type=float, default=None,
                   help="Single k%% value (backward compat; prefer --k_percents)")
    p.add_argument("--start_layer",      type=int, required=True,
                   help="First MLP layer to include (end_layer is always the last cached layer)")
    p.add_argument("--sample_idx",       type=int, default=0,
                   help="Index of the (in_seq, target) pair in the shell arrays (single-sample mode)")
    p.add_argument("--output_dir",       default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--rely_ori_acv",          action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze skeleton path selection on clean (unintervened) activations so "
                        "preserve/ablation compare the same paths at every layer — required for a "
                        "paired causal reading (default: on). Pass --no-rely_ori_acv to select paths "
                        "from the in-progress intervened residual stream instead (faster, but preserve "
                        "and ablation may diverge onto different skeletons at deeper layers). "
                        "Batch mode always falls back to sequential processing when this is enabled.")
    p.add_argument("--gemma_restore_magnit",  action="store_true",
                   help="Deprecated compatibility flag; ignored. Gemma-2/3 always use "
                        "norm matching plus unclamped post-norm magnitude restoration.")
    p.add_argument("--gpu",              type=str, default="0",
                   help="GPU index or comma-separated indices (e.g. '0,1')")
    p.add_argument("--topk_tokens",      type=int, default=20)
    p.add_argument("--batch_size",        type=int, default=1,
                   help="Number of dataset samples processed per forward pass (default: 1 = sequential). "
                        "Batch mode uses right-padding; rely_ori_acv falls back to sequential.")
    p.add_argument("--attention_k", type=int, default=None,
                   help="Enable joint Attention+MLP intervention with this many per-head OV paths per layer. "
                        "Currently supported in sequential dataset mode and the direct forward API.")
    p.add_argument("--conditions",       nargs="+", default=None,
                   choices=list(CONDITIONS),
                   help="Subset of conditions to run (default: all). "
                        f"Available: {', '.join(CONDITIONS)}")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.k_percents is None and args.k_percent is None:
        raise SystemExit("error: provide --k_percents or --k_percent")
    k_percents = args.k_percents if args.k_percents is not None else [args.k_percent]
    os.environ.setdefault("HF_HOME", os.path.join(_ROOT, "model_cache"))

    if args.dataset_path is not None:
        # Dataset mode: one model load per (dataset, model)
        for m in args.models:
            if args.batch_size > 1:
                run_model_dataset_batched(
                    m, args.dataset_path, args.n_samples, k_percents,
                    args.start_layer, args.rely_ori_acv,
                    args.gpu, args.topk_tokens, args.output_dir, args.batch_size,
                    sample_start=args.sample_start,
                    gemma_restore_magnit=args.gemma_restore_magnit,
                    conditions=args.conditions,
                )
            else:
                run_model_dataset(
                    m, args.dataset_path, args.n_samples, k_percents,
                    args.start_layer, args.rely_ori_acv,
                    args.gpu, args.topk_tokens, args.output_dir,
                    sample_start=args.sample_start,
                    gemma_restore_magnit=args.gemma_restore_magnit,
                    conditions=args.conditions,
                    attention_k=args.attention_k,
                )
    else:
        # Single-sample mode (original behaviour)
        if args.target is None or args.in_seq is None:
            raise SystemExit("error: provide --target and --in_seq, or --dataset_path")
        for m in args.models:
            run_model(m, args.target, args.in_seq, k_percents,
                      args.start_layer, args.rely_ori_acv,
                      args.gpu, args.topk_tokens,
                      args.sample_idx, args.output_dir,
                      gemma_restore_magnit=args.gemma_restore_magnit,
                      conditions=args.conditions)
