"""
skeleton_pred.py

Core logic for the "Structure Predicts Computation" experiment:
  - Original forward
  - Skeleton preserve / ablation  (topk or ori_acv)
  - Random preserve / ablation

Public API
----------
setup_skeleton_pred(args) → setup dict
    Load model and precompute everything independent of k1.
    Reusable across multiple k1 values (e.g. in a sweep).

eval_one_k(k1, setup, args) → (stats_dict, logits_dict)
    Run 5 conditions for one k1. Returns:
        stats_dict  — {condition: {rank, logit, prob}}
        logits_dict — {condition: Tensor[vocab]}  (for top-token display)

run_skeleton_pred(args)
    Single-k1 entry point (calls setup + eval_one_k once).
"""

import os
import torch
from tqdm import tqdm

from block_interp.model_load import load_model_and_embeddings
from model_config import NEXT_TOKEN_MAX_LENGTH
from ske_pre_valid.within_ske_cache import WithinSkeCache
from ske_pre_valid.forward_hook.skeleton_mlp_forward import skeleton_forward
from ske_pre_valid.forward_hook.skeleton_mlp_forward_ori_acv import skeleton_forward_ori_acv
from ske_pre_valid.static_skeleton import (
    get_token_stats, print_top_tokens, top_tokens, _token_family_ids, family_rank_prob,
    validate_precomputed_target_id,
)
from ske_pre_valid.model_adapter import gemma_post_ffn_gamma

# Ordered list of all non-ori conditions (ori is always computed)
CONDITIONS = ("ske_pre", "ske_abl", "rnd_pre", "rnd_abl")

COND_DISPLAY = {
    "ori":     "Original",
    "ske_pre": "Skeleton preserve",
    "ske_abl": "Skeleton ablation",
    "rnd_pre": "Random preserve",
    "rnd_abl": "Random ablation",
}


def _precompute_layer_tensors(
    cache, adapter, model, start_layer, end_layer,
    target_id, W_emb, use_bias,
) -> dict:
    """Pre-move delta, V, effector_scores to GPU for every skeleton layer.

    Called once after model load; the returned dict is passed as
    layer_gpu_tensors to skeleton_forward / skeleton_forward_ori_acv so that
    the 283 MB delta + 105 MB V CPU→GPU transfer happens once instead of
    once per forward pass (avoids hundreds of GB of PCIe traffic per model run).

    Heavy per-layer tensors are computed here once and shared across all k values
    and conditions.
    """
    blocks = adapter.get_blocks(model)
    layer_tensors = {}

    # Pre-extract W_emb target row once — avoids re-materialising the full
    # vocab matrix (e.g. 32000×5120×4 B = 655 MB) inside the per-layer loop.
    W_emb_target_cpu = W_emb[target_id].cpu().float() if (
        target_id is not None and W_emb is not None
    ) else None

    n_layers = end_layer - start_layer + 1
    for li in tqdm(range(start_layer, end_layer + 1),
                   total=n_layers, desc="Precompute GPU tensors", unit="layer"):
        ld    = cache.layer_skeleton(li)
        block = blocks[li]
        param = next(iter(block.parameters()), None)
        bd    = param.device if param is not None else torch.device("cpu")

        delta = ld["delta"].to(bd)
        V     = ld["V"].to(bd)
        bias  = ld["bias"].to(bd) if use_bias and "bias" in ld else None

        # Precompute effector scores (topk selection criterion)
        post_ffn_gamma = gemma_post_ffn_gamma(block)

        effector = None
        if target_id is not None and W_emb is not None:
            V_cpu = ld["V"].cpu().float()
            if post_ffn_gamma is not None:
                effector = ((V_cpu * post_ffn_gamma) @ W_emb_target_cpu).to(bd)
            else:
                effector = (V_cpu @ W_emb_target_cpu).to(bd)

        layer_tensors[li] = {
            "delta":          delta,
            "V":              V,
            "bias":           bias,
            "effector":       effector,
            "post_ffn_gamma": post_ffn_gamma,   # None for most models; CPU float32 for Gemma
            "decomposition_method": ld.get("decomposition_method", "svd"),
            "delta_representation": ld.get("delta_representation", "dense"),
        }

    return layer_tensors


def update_effectors_for_target(
    layer_gpu_tensors: dict,
    target_id: int,
    W_emb,
    start_layer: int,
    end_layer: int,
) -> None:
    """In-place update of effector scores for a new target token.

    Called when iterating over dataset samples with different target tokens.
    Reuses V already on GPU — avoids re-loading delta/V from CPU/NFS.

    Uses the raw-dot alignment ``V @ w_target``.
    """
    W_target_cpu = W_emb[target_id].cpu().float()      # [d_model]
    for li in range(start_layer, end_layer + 1):
        lgt  = layer_gpu_tensors[li]
        bd   = lgt["V"].device
        gamma = lgt.get("post_ffn_gamma")               # CPU float32 or None
        if gamma is not None:
            V_eff = lgt["V"].cpu().float() * gamma      # CPU, gamma-corrected
            lgt["effector"] = (V_eff @ W_target_cpu).to(bd)
        else:
            lgt["effector"] = (lgt["V"] @ W_target_cpu.to(bd))  # GPU matmul


def setup_skeleton_pred(args):
    """Load model and precompute everything independent of k1.

    Expensive operations done once here: model load and per-layer tensor preparation.
    The returned dict can be passed to eval_one_k for any number of k1 values.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    multi_gpu  = "," in str(args.gpu)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if multi_gpu else None
    use_bf16   = multi_gpu
    print(f"Device: {device}  multi_gpu={multi_gpu}")

    from ske_pre_valid.model_adapter import get_adapter
    model, tokenizer, W_emb = load_model_and_embeddings(
        args.model_name,
        device      = None if multi_gpu else torch.device(device),
        torch_dtype = torch.bfloat16 if use_bf16 else None,
        device_map  = device_map,
    )
    model.eval()
    adapter = get_adapter(model)

    precomputed_id = getattr(args, "target_id", None)
    if precomputed_id is not None:
        target_id = validate_precomputed_target_id(
            tokenizer, args.target_word, precomputed_id
        )
        print(f"\nTarget : {args.target_word!r}  (id={target_id}, resolve=precomputed)")
    elif getattr(args, "target_resolution", "strict") == "encode_only":
        encoded = tokenizer.encode(args.target_word, add_special_tokens=False)
        valid = [tid for tid in encoded if tokenizer.decode([tid]).strip()]
        if len(valid) != 1:
            reason = "empty_encode" if not encoded else "multi_token"
            raise ValueError(
                f"target {args.target_word!r} not resolvable for {args.model_name!r} "
                f"(resolve={reason}); expected exactly one token from tokenizer.encode()."
            )
        target_id = valid[0]
        print(
            f"\nTarget : {args.target_word!r}  (id={target_id}, "
            f"decoded={tokenizer.decode([target_id])!r}, resolve='encode')"
        )
    else:
        from ske_pre_valid.static_skeleton import _resolve_target_info
        _tgt_info = _resolve_target_info(tokenizer, args.target_word)
        if not _tgt_info["target_is_single"]:
            raise ValueError(
                f"target {args.target_word!r} not resolvable for {args.model_name!r} "
                f"(resolve={_tgt_info['target_resolve']}); refusing to score against "
                f"the unk id instead of a real target token."
            )
        target_id = _tgt_info["target_id"]
        print(f"\nTarget : {args.target_word!r}  (id={target_id}, decoded={_tgt_info['target_decoded']!r}, resolve={_tgt_info['target_resolve']!r})")
    print(f"Layers : [{args.start_layer}, {args.end_layer}]")

    family_ids = _token_family_ids(tokenizer, args.target_word, target_id=target_id)

    cache            = WithinSkeCache(args.cache_dir, expected_model_name=args.model_name)
    # slim_keys: drop sorted_subspace_idx + sorted_delta (static-skeleton only)
    # to free ~566 MB × n_layers of CPU RAM after NFS load.
    cache.preload_layers(
        args.start_layer, args.end_layer,
        slim_keys={"delta", "V", "bias", "n_neurons", "n_subspaces",
                   "layer_idx", "model_name", "decomposition_method",
                   "decomposition_tag", "decomposition_seed",
                   "delta_representation",
                   "natural_path_count"},
    )
    use_bias = getattr(args, "use_bias", True)
    print(f"  Precomputing delta/V/effector on GPU …")
    layer_gpu_tensors = _precompute_layer_tensors(
        cache, adapter, model, args.start_layer, args.end_layer,
        target_id, W_emb, use_bias,
    )
    print(f"  [GPU tensors ready for {len(layer_gpu_tensors)} layers]")

    from model_config.templates import needs_true_add_special_tokens
    add_special_tokens = needs_true_add_special_tokens(args.model_name, tokenizer)
    first_layer_data = cache.layer_skeleton(args.start_layer)
    decomposition = {
        "method": first_layer_data.get("decomposition_method", "svd"),
        "tag": first_layer_data.get("decomposition_tag", "svd"),
        "seed": first_layer_data.get("decomposition_seed"),
        "delta_representation": first_layer_data.get(
            "delta_representation", "dense"
        ),
    }

    return {
        "model":              model,
        "tokenizer":          tokenizer,
        "W_emb":              W_emb,
        "cache":              cache,
        "target_id":          target_id,
        "family_ids":         family_ids,
        "device":             str(device),
        "adapter":            adapter,
        "layer_gpu_tensors":  layer_gpu_tensors,
        "decomposition":      decomposition,
        # See model_config.templates.needs_true_add_special_tokens: True for base
        # LMs (Llama/Mistral/Gemma/DeepSeek) whose leading BOS must be restored via
        # add_special_tokens=True rather than string-splicing it into args.in_seq
        # (wrong for some SentencePiece tokenizers — confirmed on Mistral).
        "add_special_tokens": add_special_tokens,
    }


def eval_one_k(
    k1, setup, args, conditions=None, return_traces=False,
    trace_conditions=None,
):
    """Run conditions for one k1 value using a pre-loaded setup.

    Parameters
    ----------
    conditions : list[str] | None
        Subset of CONDITIONS to run. None means all four are run.
        "ori" (original forward) is always computed regardless.

    Returns
    -------
    stats  : dict[str, dict]   — {"ori"} ∪ active_conds → {rank, logit, prob}
    logits : dict[str, Tensor] — {"ori"} ∪ active_conds → last-token logit vector
    """
    model              = setup["model"]
    tokenizer          = setup["tokenizer"]
    W_emb              = setup["W_emb"]
    cache              = setup["cache"]
    target_id          = setup["target_id"]
    family_ids         = setup.get("family_ids") or [target_id]
    device             = setup["device"]
    adapter            = setup["adapter"]
    layer_gpu_tensors  = setup.get("layer_gpu_tensors")
    add_special_tokens = setup.get("add_special_tokens", False)

    # add_special_tokens defaults to False, matching the static-prompt dataset builder
    # and avoiding tokenizers (e.g. ChemGPT) silently inserting [CLS]/[SEP] around the
    # sequence. setup["add_special_tokens"] is True for base LMs that need their real
    # BOS restored at the token-id level (see model_config.templates).
    encoded = tokenizer(
        args.in_seq,
        return_tensors="pt",
        truncation=True,
        max_length=NEXT_TOKEN_MAX_LENGTH,
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"]
    token_type_ids = encoded.get("token_type_ids")
    attention_mask = encoded.get("attention_mask")
    _requested   = set(conditions) if conditions else set()
    active_conds = tuple(c for c in CONDITIONS if (not _requested or c in _requested))
    trace_condition_set = (
        set(active_conds) if trace_conditions is None else set(trace_conditions)
    )

    _COND_PARAMS = {
        "ske_pre": ("topk",   "preserve", {}),
        "ske_abl": ("topk",   "ablation", {}),
        "rnd_pre": ("random", "preserve", {"seed": args.random_seed}),
        "rnd_abl": ("random", "ablation", {"seed": args.random_seed}),
    }

    def _run(selection, intervention, capture_trace=False, **kw):
        attention_k1 = getattr(args, "attention_k1", None)
        if (getattr(args, "rely_ori_acv", False) and selection == "topk"
                and attention_k1 is None):
            if capture_trace:
                raise ValueError("trace capture requires --no-rely_ori_acv")
            return skeleton_forward_ori_acv(
                model, input_ids, cache,
                args.start_layer, args.end_layer,
                k1, args.seq_reduce, device,
                intervention=intervention, use_bias=args.use_bias,
                target_id=target_id, W_emb=W_emb,
                adapter=adapter,
                layer_gpu_tensors=layer_gpu_tensors,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )["logits"][0, -1, :], None
        output = skeleton_forward(
            model, input_ids, cache,
            args.start_layer, args.end_layer,
            k1, args.seq_reduce, device,
            selection=selection, intervention=intervention,
            use_bias=args.use_bias,
            target_id=target_id, W_emb=W_emb,
            adapter=adapter,
            layer_gpu_tensors=layer_gpu_tensors,
            token_type_ids=token_type_ids,
            capture_path_trace=capture_trace,
            attention_k1=attention_k1,
            attention_selection=getattr(args, "attention_selection", None),
            attention_intervention=getattr(args, "attention_intervention", None),
            **kw,
        )
        paths = output["layer_paths"]
        if attention_k1 is not None:
            paths = {
                "mlp": paths,
                "attention": output["attention_layer_paths"],
            }
        return output["logits"][0, -1, :], paths

    with torch.no_grad():
        model_inputs = {"input_ids": input_ids.to(device)}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids.to(device)
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(device)
        orig_last = model(**model_inputs).logits[0, -1, :]

    cond_logits = {}
    cond_traces = {}
    for cond in tqdm(active_conds, desc=f"k1={k1:,}", unit="cond", leave=False):
        sel, itv, kw = _COND_PARAMS[cond]
        cond_logits[cond], cond_traces[cond] = _run(
            sel, itv,
            capture_trace=return_traces and cond in trace_condition_set,
            **kw,
        )

    def _s(logits):
        r, l, p = get_token_stats(logits, target_id)
        fr, fl, fp = family_rank_prob(logits, family_ids)
        return {"rank": r, "logit": l, "prob": p,
                "family_rank": fr, "family_logit": fl, "family_prob": fp}

    stats  = {"ori": _s(orig_last)}
    logits = {"ori": orig_last}
    for cond in active_conds:
        stats[cond]  = _s(cond_logits[cond])
        logits[cond] = cond_logits[cond]

    if return_traces:
        return stats, logits, cond_traces
    return stats, logits


def run_skeleton_pred(args):
    setup     = setup_skeleton_pred(args)
    tokenizer = setup["tokenizer"]

    print(f"\nInput  : {args.in_seq!r}")
    print(f"k1     : {args.k1}")

    conditions = getattr(args, "conditions", None)
    stats, logits = eval_one_k(args.k1, setup, args, conditions=conditions)

    ske_label = "Skeleton preserve (ori_acv)" if args.rely_ori_acv else "Skeleton preserve"
    abl_label = "Skeleton ablation (ori_acv)" if args.rely_ori_acv else "Skeleton ablation"
    cond_display = {
        **COND_DISPLAY,
        "ske_pre": ske_label,
        "ske_abl": abl_label,
    }

    print_top_tokens("Original model",
                     top_tokens(logits["ori"], tokenizer, args.topk_tokens),
                     stats["ori"]["rank"])
    for cond in CONDITIONS:
        if cond not in logits:
            continue
        print_top_tokens(f"{cond_display[cond]} (k1={args.k1})",
                         top_tokens(logits[cond], tokenizer, args.topk_tokens),
                         stats[cond]["rank"])

    w = 28
    print(f"\n{'─'*60}")
    print(f"  Summary  (target: {args.target_word!r})")
    print(f"{'─'*60}")
    print(f"  {'Condition':<{w}}  {'Rank':>6}  {'Logit':>10}  {'Prob':>10}")
    print(f"  {'─'*w}  {'─'*6}  {'─'*10}  {'─'*10}")
    for key in ("ori",) + CONDITIONS:
        if key not in stats:
            continue
        s = stats[key]
        print(f"  {cond_display[key]:<{w}}  {s['rank']:>6}  {s['logit']:>10.4f}  {s['prob']:>10.6f}")

    print(f"\n{'─'*60}")
    print(f"  Sequence prediction")
    print(f"{'─'*60}")
    for key in ("ori",) + CONDITIONS:
        if key not in logits:
            continue
        top1_tok = top_tokens(logits[key], tokenizer, 1)[0][0]
        print(f"  [{cond_display[key]:<{w}}]  {args.in_seq!r} -> {top1_tok!r}  (rank={stats[key]['rank']})")

    return stats, cond_display


# ── Batch helpers ─────────────────────────────────────────────────────────────

def _compute_per_sample_effectors(
    layer_gpu_tensors: dict,
    target_ids: list,
    W_emb: torch.Tensor,
    start_layer: int,
    end_layer: int,
) -> dict:
    """Compute per-sample effector scores for a batch of target token IDs.

    Returns {layer_idx: Tensor[b, n_sp]} on the same device as V for each layer.

    Uses raw-dot alignment: ``V @ W_emb[target_ids].T`` → ``[b, n_sp]``.
    """
    result = {}
    for li in range(start_layer, end_layer + 1):
        lgt   = layer_gpu_tensors[li]
        V     = lgt["V"]            # [n_sp, d_model]
        bd    = V.device
        gamma = lgt.get("post_ffn_gamma")   # [d_model] CPU float32 or None

        W_tgt = W_emb[target_ids]          # [b, d_model] on CPU
        if gamma is not None:
            V_eff = V.cpu().float() * gamma             # [n_sp, d_model]
            effs  = (V_eff @ W_tgt.cpu().float().T).T  # [b, n_sp]
            result[li] = effs.to(device=bd, dtype=V.dtype)
        else:
            W_tgt_gpu = W_tgt.to(device=bd, dtype=V.dtype)
            result[li] = (V @ W_tgt_gpu.T).T           # [b, n_sp]

    return result


def eval_batch_k(
    k1: int,
    model_setup: dict,
    input_ids: torch.Tensor,         # [b, max_seq] right-padded
    batch_target_ids: list,          # list of int, len b
    last_real_pos: torch.Tensor,     # [b] 0-indexed last real token per sample
    per_sample_effectors: dict,      # {layer_idx: Tensor[b, n_sp]}
    args0,                           # reference namespace (start_layer, end_layer, …)
    conditions: list | None = None,
    batch_family_ids: list | None = None,  # list[list[int]], len b; falls back to [target_id]
    orig_logits: torch.Tensor | None = None,  # [b, vocab], precomputed — see below
    token_type_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> list:
    """Run skeleton prediction for one k1 over a batch of right-padded samples.

    orig_logits: the unmodified model's last-token logits for this batch, precomputed
        by the caller. "Original" depends only on (model, input_ids), not on k1 or the
        target word, so calling this once per k_percent in a sweep (the old behaviour,
        when this arg was omitted) redid the same full forward pass on every iteration —
        e.g. 11x per batch for a typical 11-value K_PERCENTS sweep. Pass it in once,
        computed outside the k_percent loop, and it's reused here instead of recomputed.
        None still works (computed fresh) for any other/older caller.

    Returns list of (stats_dict, logits_dict) — one entry per sample in the batch.
    Each stats_dict has keys "ori" + active_conds → {rank, logit, prob}.
    Each logits_dict has the same keys → Tensor[vocab].
    """
    model    = model_setup["model"]
    device   = model_setup["device"]
    adapter  = model_setup["adapter"]
    lgt_map  = model_setup["layer_gpu_tensors"]
    cache    = model_setup["cache"]

    b       = input_ids.shape[0]
    b_range = torch.arange(b, device=device)
    last_dev = last_real_pos.to(device)

    _requested   = set(conditions) if conditions else set()
    active_conds = tuple(c for c in CONDITIONS if (not _requested or c in _requested))

    _COND_PARAMS = {
        "ske_pre": ("topk",   "preserve", {}),
        "ske_abl": ("topk",   "ablation", {}),
        "rnd_pre": ("random", "preserve", {"seed": args0.random_seed}),
        "rnd_abl": ("random", "ablation", {"seed": args0.random_seed}),
    }

    # Original forward (batched) — right-padding is correct for causal attention.
    # Skipped when the caller already precomputed it (see orig_logits docstring above).
    if orig_logits is None:
        input_dev = input_ids.to(device)
        model_inputs = {"input_ids": input_dev}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids.to(device)
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(device)
        with torch.no_grad():
            orig_all  = model(**model_inputs).logits         # [b, max_seq, vocab]
        _orig_dev = orig_all.device
        orig_logits = orig_all[b_range.to(_orig_dev), last_dev.to(_orig_dev), :]  # [b, vocab]

    # Skeleton conditions (batched)
    cond_logits_batch: dict[str, torch.Tensor] = {}
    for cond in active_conds:
        sel, itv, kw = _COND_PARAMS[cond]
        pse = per_sample_effectors if sel == "topk" else None
        out = skeleton_forward(
            model, input_ids, cache,
            args0.start_layer, args0.end_layer,
            k1, args0.seq_reduce, device,
            selection=sel, intervention=itv,
            use_bias=args0.use_bias,
            target_id=None,
            W_emb=None,
            adapter=adapter,
            layer_gpu_tensors=lgt_map,
            per_sample_effectors=pse,
            last_real_pos=last_dev,
            token_type_ids=token_type_ids,
            **kw,
        )
        all_logits = out["logits"]                        # [b, max_seq, vocab]
        # skeleton_forward's manual per-layer loop leaves the final hidden state (and
        # hence logits) on whichever device the last block/lm_head landed on under
        # device_map="auto" — not necessarily `device` (multi-GPU models can split
        # across GPUs, e.g. Llama-2-13b-hf). Re-derive the index tensors on
        # all_logits' actual device rather than assuming it matches b_range/last_dev.
        _out_dev = all_logits.device
        cond_logits_batch[cond] = all_logits[b_range.to(_out_dev), last_dev.to(_out_dev), :]  # [b, vocab]

    def _s(logits_1d, target_id, family_ids):
        r, l, p = get_token_stats(logits_1d, target_id)
        fr, fl, fp = family_rank_prob(logits_1d, family_ids)
        return {"rank": r, "logit": l, "prob": p,
                "family_rank": fr, "family_logit": fl, "family_prob": fp}

    results = []
    for i in range(b):
        tid       = batch_target_ids[i]
        fam_ids_i = batch_family_ids[i] if batch_family_ids is not None else [tid]
        stats_i  = {"ori": _s(orig_logits[i], tid, fam_ids_i)}
        logits_i = {"ori": orig_logits[i]}
        for cond in active_conds:
            stats_i[cond]  = _s(cond_logits_batch[cond][i], tid, fam_ids_i)
            logits_i[cond] = cond_logits_batch[cond][i]
        results.append((stats_i, logits_i))

    return results
