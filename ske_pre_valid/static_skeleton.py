"""
static_skeleton.py

Experiment runner and shared utilities for static skeleton experiments.
Path selection and forward passes live in ske_pre_valid/forward_hook/.
"""

import json
import os
import sys
import statistics
import torch

from ske_pre_valid.forward_hook.base import forward_static_preserve, forward_static_ablation
from ske_pre_valid.forward_hook.forward_global_neuron import select_global_abs_topk
from ske_pre_valid.forward_hook.forward_per_neuron import select_per_neuron_abs_topk
from ske_pre_valid.forward_hook.eff_select import (
    select_eff_delta_rank, select_eff_delta_weight_rank,
)
from ske_pre_valid.model_adapter import (
    gemma_post_ffn_gamma, is_gemma_style, GEMMA_APPROX_INTERVENTION_NOTE,
)
from model_config import PROMPT_FORMAT_TAG, PROMPT_FORMAT_VERSION, safe_slug
from result_summary import write_static_model_mean


# v2 used direct raw down_proj replacement followed by the native post-MLP norm.
# v3 restores magnitude-preserving Gemma interventions with an unclamped raw-norm ratio.
# v4 preserves tokenizer-provided token_type_ids in Original and intervention forwards.
RESULT_FORMAT_VERSION = 4
RESULT_FORMAT_TAG = f"rv{RESULT_FORMAT_VERSION}"


# ── random baseline selection ─────────────────────────────────────────────────

def select_random_paths(delta, n_paths, seed=42, precomputed_effector=None):
    """Uniformly sample n_paths (j,i) pairs — random baseline.

    precomputed_effector : optional [n_subspaces] tensor (from cache or V@W_emb_target).
        When provided, static_scores = delta[j,i] * effector[i], so the sign-gate
        in forward_static_preserve uses h·delta·eff > 0 (same criterion as topk).
        When None, static_scores = delta[j,i] and the gate is h·delta > 0 only.
    """
    delta      = delta.cpu()
    d_ff, n_sp = delta.shape
    rng        = torch.Generator()
    rng.manual_seed(seed)
    flat_idx = torch.randperm(d_ff * n_sp, generator=rng)[:min(n_paths, d_ff * n_sp)]
    j_idx    = (flat_idx // n_sp).long()
    i_idx    = (flat_idx %  n_sp).long()
    dv       = delta[j_idx, i_idx]
    if precomputed_effector is not None:
        eff_vals     = precomputed_effector.cpu()[i_idx]
        static_scores = dv * eff_vals   # h·delta·eff gate, same as topk
    else:
        static_scores = dv              # h·delta gate only
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    dv,
        "static_scores": static_scores,
        "effector":      precomputed_effector,
    }


# ── inference helpers ─────────────────────────────────────────────────────────

def _decode_single_token(tokenizer, token_id: int) -> str:
    """Decode a single token id to text.

    Byte-level BPE vocabularies (GPT-2, Llama, Mistral, ...) can split a
    multi-byte UTF-8 character (Cyrillic, CJK, accented Latin, ...) across
    several ids; decoding one of those ids in isolation yields the
    tokenizer's replacement character (U+FFFD). Fall back to the raw token
    piece (with byte-level/SentencePiece space markers resolved) so the
    label still reflects the actual predicted token instead of losing it.
    """
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    if "�" not in text:
        return text
    piece = tokenizer.convert_ids_to_tokens(token_id)
    if not piece:
        return text
    return piece.replace("Ġ", " ").replace("▁", " ")


def top_tokens(logits_last, tokenizer, k):
    probs = torch.softmax(logits_last, dim=-1)
    vals, idx = torch.topk(probs, k)
    return [(_decode_single_token(tokenizer, i.item()), v.item()) for i, v in zip(idx, vals)]


def target_rank_prob(logits_last, target_id):
    probs = torch.softmax(logits_last, dim=-1)
    # Competition rank: all tokens tied at the target logit receive the same rank.
    # This matters for Gemma-2, whose final logit soft-cap plus bf16 can make
    # hundreds of vocabulary entries exactly 30.0. argsort assigns those ties an
    # arbitrary order and can report rank 100 for a token tied for the maximum.
    target_logit = logits_last[target_id]
    rank = int((logits_last > target_logit).sum().item()) + 1
    return rank, probs[target_id].item()


def get_token_stats(logits_last, target_id):
    rank, prob = target_rank_prob(logits_last, target_id)
    return rank, logits_last[target_id].item(), prob


def target_tie_count(logits_last, target_id):
    """Number of vocabulary entries with exactly the target token's logit.

    One is the normal, untied case. Large values expose final-softcap saturation:
    a target can have competition rank 1 while still sharing that rank with many
    tokens and therefore receiving very little probability mass.
    """
    return int((logits_last == logits_last[target_id]).sum().item())


def _token_family_ids(tokenizer, target_word: str, target_id: int | None = None) -> list:
    """All vocab ids that decode (whitespace-stripped) to the same surface form as target_word.

    BPE/SentencePiece tokenizers split what is semantically one word across distinct vocab
    entries depending on whether it's preceded by a space (e.g. GPT-2's ' France' id=4881 vs
    the space-less 'France' id=28572). This groups them into one lenient "family" so a rank
    computed against the family (family_rank_prob) can serve as a semantically-tolerant
    companion metric to the single-id rank from target_rank_prob.
    """
    surface    = target_word.strip()
    vocab_size = len(tokenizer)
    decoded    = tokenizer.batch_decode([[i] for i in range(vocab_size)])
    ids = [i for i, s in enumerate(decoded) if s.strip() == surface]
    if not ids and target_id is not None:
        ids = [target_id]   # multi-token/empty-encode fallback: family = the anchor id itself
    return ids


def family_rank_prob(logits_last, family_ids):
    """Best-case rank/logit/prob among a token family — lenient companion to get_token_stats."""
    fam_logits = logits_last[family_ids]
    best_idx   = int(fam_logits.argmax())
    best_logit = fam_logits[best_idx]
    rank = int((logits_last > best_logit).sum().item()) + 1
    prob = torch.softmax(logits_last, dim=-1)[family_ids][best_idx].item()
    return rank, best_logit.item(), prob


def print_top_tokens(label, tokens, target_rank=None):
    print(f"\n{'─'*50}\n  {label}\n{'─'*50}")
    for rank, (tok, val) in enumerate(tokens, 1):
        marker = " ◄" if rank == 1 else ""
        print(f"  {rank:3d}.  {tok!r:20s}  {val:+.4f}{marker}")
    if target_rank is not None:
        print(f"  [target token rank: {target_rank}]")


# ── experiment runner ─────────────────────────────────────────────────────────

def _effector_scores(V, W_emb, target_id, post_ffn_norm_weight=None):
    """Return the raw-dot target alignment ``V[i] @ W_emb[target_id]``."""
    V_cpu = V.cpu().float()
    if post_ffn_norm_weight is not None:
        # Gemma2/3 RMSNorm: output = x_norm * (1 + weight); weight is initialized to 0.
        V_cpu = V_cpu * post_ffn_norm_weight.cpu().float()
    W_cpu = W_emb.cpu().float()
    return V_cpu @ W_cpu[target_id]


def _resolve_target_id(tokenizer, target_word: str) -> int:
    """Return the single token-id for target_word. See _resolve_target_info for details."""
    return _resolve_target_info(tokenizer, target_word)["target_id"]


def validate_precomputed_target_id(tokenizer, target_word: str, target_id: int) -> int:
    """Validate an externally stored target id against this tokenizer.

    Precomputed ids are deliberately not re-resolved from text: two vocabulary
    entries may decode to the same surface form while representing different
    next-token events.  We do still reject stale/out-of-range ids and ids whose
    decoded surface no longer matches the stored target string.
    """
    if isinstance(target_id, bool) or not isinstance(target_id, int):
        raise ValueError(f"precomputed target_id must be an integer, got {target_id!r}")
    if target_id < 0 or target_id >= len(tokenizer):
        raise ValueError(
            f"precomputed target_id={target_id} is outside tokenizer vocabulary "
            f"[0, {len(tokenizer)})"
        )

    decoded = tokenizer.decode([target_id])
    if decoded.strip() != target_word.strip():
        raise ValueError(
            f"precomputed target_id={target_id} decodes to {decoded!r}, "
            f"not stored target {target_word!r}; dataset/target JSON is stale "
            f"or belongs to another tokenizer"
        )
    return target_id


def _resolve_target_info(tokenizer, target_word: str) -> dict:
    """Resolve target_word to a token ID and return full resolution metadata.

    Strategy (encode-first to avoid shadow-token duplicates in BioGPT-style vocabs):
      1. encode(target_word) → single non-empty token.
         Handles space-prefixed tokens (' happy' → ▁happy), normal words, and
         BioGPT where convert_tokens_to_ids returns a stale "shadow" duplicate entry.
      2. convert_tokens_to_ids([target_word]) cross-check when encode already gives
         exactly one token — only used to detect a genuine vocab duplicate (two
         distinct ids for the same string), not as a recovery path.
      3. encode() giving 0 or >1 tokens (e.g. SentencePiece subword-only strings
         like 'ktop'/'atory' that only appear mid-word in training text, so encode()
         splits them) is rejected outright as empty_encode/multi_token — there is
         no convert_tokens_to_ids fallback to recover a single id for these; the
         target is simply not usable for this tokenizer.

    Returns dict with keys:
      target_id          int   — resolved token ID
      target_decoded     str   — tokenizer.decode([target_id]); a display string only —
                                 SentencePiece strips the leading '▁'→space when decoding
                                 a token in isolation, so don't infer tokenization from it
      target_is_single   bool  — True if target_word maps to exactly one token
      target_resolve     str   — "encode" | "empty_encode" | "multi_token" | "encode_vocab_conflict"

    target_is_single is False for every rejection case (empty_encode, multi_token,
    encode_vocab_conflict) — callers should treat target_is_single=False as "skip
    this target for this model" rather than scoring against target_id (which is
    the unk id in all three cases).
    """
    unk = tokenizer.unk_token_id

    # Step 1: encode — handles BPE normalization (' '→'Ġ', SentencePiece '▁', BioGPT '</w>')
    enc_ids = tokenizer.encode(target_word, add_special_tokens=False)
    valid   = [tid for tid in enc_ids if tokenizer.decode([tid]).strip()]

    if len(valid) == 1:
        enc_id    = valid[0]
        # Step 2: vocab cross-check — if vocab gives a *different* non-unk id, flag conflict.
        # BPE tokenizers store 'Ġword' not ' word', so vocab lookup often returns unk for
        # space-prefixed inputs; that is fine (no conflict).  The conflict case is BioGPT ','
        # where vocab(',')=24540 but encode(',')=5 — two distinct vocab entries.
        direct_id = tokenizer.convert_tokens_to_ids([target_word])[0]
        conflict  = (direct_id is not None
                     and (unk is None or direct_id != unk)
                     and direct_id != enc_id)
        if conflict:
            print(f"  [WARN] target {target_word!r}: "
                  f"encode→{enc_id} ({tokenizer.decode([enc_id])!r}), "
                  f"vocab→{direct_id} ({tokenizer.decode([direct_id])!r}); "
                  f"conflict — rejecting (see target_resolve)")
            # Reject rather than silently trusting either id: a genuine vocab
            # duplicate means neither candidate is reliably "the" token this
            # model would produce, so callers should skip this target instead
            # of scoring against a possibly-wrong id (see target_is_single).
            return {
                "target_id":        unk if unk is not None else -1,
                "target_decoded":   target_word,
                "target_is_single": False,
                "target_resolve":   "encode_vocab_conflict",
            }
        return {
            "target_id":        enc_id,
            "target_decoded":   tokenizer.decode([enc_id]),
            "target_is_single": True,
            "target_resolve":   "encode",
        }

    # encode returned empty (e.g. pure whitespace)
    if not enc_ids:
        return {
            "target_id":        unk if unk is not None else -1,
            "target_decoded":   target_word,
            "target_is_single": False,
            "target_resolve":   "empty_encode",
        }

    # Multi-token: not a single vocab entry for this model — no sub-token fallback
    sub_tokens = [tokenizer.decode([i]) for i in enc_ids]
    print(f"  [WARN] target {target_word!r} → {sub_tokens}; multi-token, not universal")
    return {
        "target_id":        unk if unk is not None else -1,
        "target_decoded":   target_word,
        "target_is_single": False,
        "target_resolve":   "multi_token",
    }


def _setup_static_experiment(args):
    """Load model, tokenizer, embeddings, cache, and prompts — shared across all k values.

    Returns a dict consumed by _run_static_k. Call this once per (model, target, dataset)
    combination, then call _run_static_k for each k to avoid redundant model reloads.
    """
    from block_interp.model_load import load_model_and_embeddings
    from ske_pre_valid.within_ske_cache import WithinSkeCache
    from ske_pre_valid.model_adapter import get_adapter

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    multi_gpu  = "," in str(args.gpu)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if multi_gpu else None

    use_bf16 = multi_gpu or getattr(args, "bfloat16", False)
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
        tgt_info = {
            "target_id":        target_id,
            "target_decoded":   tokenizer.decode([target_id]),
            "target_is_single": True,
            "target_resolve":   "precomputed",
        }
    else:
        tgt_info = _resolve_target_info(tokenizer, args.target_word)
        if not tgt_info["target_is_single"]:
            raise ValueError(
                f"target {args.target_word!r} not resolvable for {args.model_name!r} "
                f"(resolve={tgt_info['target_resolve']}); refusing to score against "
                f"the unk id instead of a real target token."
            )
        target_id = tgt_info["target_id"]
    print(f"  target {args.target_word!r} → id={target_id} "
          f"({tgt_info['target_decoded']!r})  resolve={tgt_info['target_resolve']}"
          f"  single={tgt_info['target_is_single']}")

    family_ids = _token_family_ids(tokenizer, args.target_word, target_id=target_id)
    print(f"  target family: {len(family_ids)} vocab id(s) decode to {args.target_word.strip()!r}")

    W_emb_target = W_emb[target_id].cpu().float()
    all_blocks   = list(adapter.get_blocks(model))

    cache = WithinSkeCache(args.cache_dir, expected_model_name=args.model_name)
    cache.preload_layers(args.start_layer, args.end_layer)
    first_layer_data = cache.layer_skeleton(args.start_layer)
    decomposition = {
        "method": first_layer_data.get("decomposition_method", "svd"),
        "tag": first_layer_data.get("decomposition_tag", "svd"),
        "seed": first_layer_data.get("decomposition_seed"),
        "delta_representation": first_layer_data.get(
            "delta_representation", "dense"
        ),
    }

    if getattr(args, "prompt", None) is not None:
        prompts = [args.prompt]
    else:
        with open(args.dataset_path) as f:
            raw = json.load(f)
        prompts = [raw[k]["prompt"] for k in sorted(raw.keys(), key=int)[:args.n_samples]]

    # Prompt construction (domain sequence prefix, then chat template if any) —
    # shared with dataset_gene/dymn_dst_gene/gen_next_token_labels.py and
    # scripts_ske_cir/run_skeleton_pred.py via the same apply_template() call, so
    # all three pipelines build prompts identically. See model_config/templates.py.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from model_config.templates import apply_template, MODEL_TEMPLATE, needs_true_add_special_tokens
    prompts = [apply_template(p, args.model_name, tokenizer) for p in prompts]
    tmpl_key = MODEL_TEMPLATE.get(args.model_name)
    if tmpl_key:
        print(f"  [chat_template={tmpl_key}] first prompt preview: {prompts[0][:80]!r}")
    add_special_tokens = needs_true_add_special_tokens(args.model_name, tokenizer)

    return {
        "model":          model,
        "tokenizer":      tokenizer,
        "W_emb":          W_emb,
        "W_emb_target":   W_emb_target,
        "target_id":      target_id,
        "target_info":    tgt_info,        # {target_id, target_decoded, target_is_single, target_resolve}
        "family_ids":     family_ids,      # vocab ids sharing target_word's stripped surface form
        "cache":          cache,
        "decomposition":  decomposition,
        "all_blocks":     all_blocks,
        "adapter":        adapter,
        "device":         str(device),
        "prompts":        prompts,
        # See model_config.templates.needs_true_add_special_tokens: True for base
        # LMs (Llama/Mistral/Gemma/DeepSeek) whose tokenizer needs add_special_tokens=True
        # to restore the leading BOS these models were trained to always see —
        # restoring it via string-splicing the BOS into the prompt text instead is
        # wrong for some SentencePiece tokenizers (confirmed on Mistral).
        "add_special_tokens": add_special_tokens,
    }


ALL_BASELINES = (
    "ske_preserve_gate",
    "ske_preserve_gate_pos_eff",       # same as ske_preserve_gate but filter V[i]@W[target]<=0
    "ske_preserve_no_gate",
    "ske_ablation_no_gate",
    "ske_ablation_gate",
    "rnd_preserve_gate",
    "rnd_preserve_no_gate",
    "rnd_ablation_no_gate",
    "rnd_ablation_gate",
    "anti_align",
    "ske_preserve_gate_weight_eff",   # |eff|-proportional path allocation (hard quota)
    "ske_preserve_gate_eff",          # uniform path allocation (k // n_sp per subspace)
    "ske_preserve_gate_sqrteff",      # sqrt(|eff|)-proportional allocation (intermediate)
    "ske_preserve_gate_softgate",     # global top-k by sqrt(|delta|)*|eff|
    "ske_preserve_gate_weight_delta", # quota ∝ max_j|delta[j,i]| per subspace
    "ske_preserve_gate_sqrtdelta",    # quota ∝ sqrt(max_j|delta[j,i]|)
    "ske_preserve_gate_eff_dominate",        # quota ∝ raw_dot eff, within-sp by |delta|
    "ske_preserve_gate_eff_dominate_threh",  # same + global top-p% |delta| threshold
    # ── effector-alignment-aware global-pool methods (eff_select.py) ───────────
    "eff_delta_rank_pre",         # harmonic within-sp discounting × |eff_raw|
    "eff_delta_rank_abl",
    "eff_delta_weight_rank_pre",  # weight_eff quota with rank_pct(|eff_raw|)
    "eff_delta_weight_rank_abl",
)


def _run_static_k(setup, args, selection_fn, selection_kwargs, label, baselines=None):
    """Run one k value using an already-loaded setup from _setup_static_experiment.

    args.k and args.k_percent must be set to the current k value before calling.
    Saves results to JSON and returns summary_rows.

    baselines : list/tuple of condition names to compute, or None for all.
                Available names: see ALL_BASELINES above.
    """
    model          = setup["model"]
    tokenizer      = setup["tokenizer"]
    W_emb          = setup["W_emb"]
    W_emb_target   = setup["W_emb_target"]
    target_id      = setup["target_id"]
    family_ids     = setup.get("family_ids") or [target_id]
    cache          = setup["cache"]
    adapter        = setup["adapter"]
    device         = setup["device"]
    prompts        = setup["prompts"]
    add_special_tokens = setup.get("add_special_tokens", False)
    # all_blocks already fetched in setup; fall back to re-fetching if old setup dict
    all_blocks     = setup.get("all_blocks") or list(adapter.get_blocks(model))

    use_bias = getattr(args, "use_bias", True)

    active = set(ALL_BASELINES) if baselines is None else set(baselines)
    unknown = active - set(ALL_BASELINES)
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}. Valid: {ALL_BASELINES}")

    needs_random       = any(c.startswith("rnd_") for c in active)
    needs_pos_eff      = "ske_preserve_gate_pos_eff" in active
    needs_eff_weight   = "ske_preserve_gate_weight_eff" in active
    needs_eff_uniform  = "ske_preserve_gate_eff" in active
    needs_eff_sqrt     = "ske_preserve_gate_sqrteff" in active
    needs_soft_gate    = "ske_preserve_gate_softgate" in active
    needs_weight_delta  = "ske_preserve_gate_weight_delta" in active
    needs_sqrt_delta    = "ske_preserve_gate_sqrtdelta" in active
    needs_eff_dominate       = "ske_preserve_gate_eff_dominate" in active
    needs_eff_dominate_threh = "ske_preserve_gate_eff_dominate_threh" in active
    delta_topk_pct           = getattr(args, "delta_topk_pct", 50.0)
    needs_delta_rank         = any(c.startswith("eff_delta_rank")        for c in active)
    needs_delta_weight_rank  = any(c.startswith("eff_delta_weight_rank") for c in active)

    selection_kwargs = dict(selection_kwargs)

    print(f"\nBuilding {label}  [effector=raw_dot]  (target={args.target_word!r}) ...")
    print(f"  Active baselines: {sorted(active)}")

    pre_selected          = {}
    pos_eff_selected      = {} if needs_pos_eff      else None
    rnd_selected          = {} if needs_random       else None
    eff_weight_selected   = {} if needs_eff_weight   else None
    eff_uniform_selected  = {} if needs_eff_uniform  else None
    eff_sqrt_selected     = {} if needs_eff_sqrt     else None
    soft_gate_selected    = {} if needs_soft_gate    else None
    weight_delta_selected  = {} if needs_weight_delta  else None
    sqrt_delta_selected    = {} if needs_sqrt_delta    else None
    eff_dominate_selected       = {} if needs_eff_dominate       else None
    eff_dominate_threh_selected = {} if needs_eff_dominate_threh else None
    delta_rank_selected         = {} if needs_delta_rank        else None
    delta_weight_rank_selected  = {} if needs_delta_weight_rank else None
    for li in range(args.start_layer, args.end_layer + 1):
        ld    = cache.layer_skeleton(li)
        block = all_blocks[li]

        # γ correction for models with post_feedforward_layernorm (Gemma-2/3).
        # post_ffn_w carries the full (1 + weight) scale factor, not just weight.
        post_ffn_w = gemma_post_ffn_gamma(block)

        eff_scores = _effector_scores(
            ld["V"], W_emb, target_id, post_ffn_norm_weight=post_ffn_w)
        pre_selected[li] = selection_fn(
            ld["delta"], ld["V"], W_emb_target,
            precomputed_effector=eff_scores, **selection_kwargs,
        )
        n_paths = pre_selected[li]["j_idx"].shape[0]
        if needs_pos_eff:
            # Pass the gamma-corrected raw-dot scores through rather than letting
            # select_global_abs_topk recompute an uncorrected V@W_emb_target.
            pos_eff_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                filter_pos_eff=True,
            )
        if needs_random:
            rnd_selected[li] = select_random_paths(
                ld["delta"], n_paths, seed=args.random_seed + li,
                precomputed_effector=eff_scores)
        if needs_eff_weight:
            eff_weight_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                weight_eff=True,
            )
        if needs_eff_uniform:
            eff_uniform_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                uniform_eff=True,
            )
        if needs_eff_sqrt:
            eff_sqrt_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                sqrt_eff=True,
            )
        if needs_soft_gate:
            soft_gate_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                soft_gate=True,
            )
        if needs_weight_delta:
            weight_delta_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                weight_delta=True,
            )
        if needs_sqrt_delta:
            sqrt_delta_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                sqrt_delta=True,
            )
        if needs_eff_dominate:
            eff_dominate_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                eff_dominate=True,
            )
        if needs_eff_dominate_threh:
            eff_dominate_threh_selected[li] = select_global_abs_topk(
                ld["delta"], ld["V"], W_emb_target, args.k,
                precomputed_effector=eff_scores,
                eff_dominate_threh=True,
                delta_topk_pct=delta_topk_pct,
            )
        # eff_select methods use the same gamma-corrected raw-dot effector.
        if needs_delta_rank or needs_delta_weight_rank:
            if needs_delta_rank:
                delta_rank_selected[li] = select_eff_delta_rank(
                    ld["delta"], eff_scores, args.k)
            if needs_delta_weight_rank:
                delta_weight_rank_selected[li] = select_eff_delta_weight_rank(
                    ld["delta"], eff_scores, args.k)
        print(f"  Layer {li:3d}: {n_paths} paths")

    sep = "═" * 70
    print(f"\n{sep}\n  {label}  |  target={args.target_word!r}"
          f"  |  layers {args.start_layer}–{args.end_layer}\n{sep}")

    def _top1(logit_vec):
        return tokenizer.decode([int(logit_vec.cpu().argmax())])

    def _top1_id(logit_vec):
        return int(logit_vec.cpu().argmax())

    # Table: condition name → (display label, forward-pass thunk)
    # Thunks are built per-prompt inside the loop (capture ids/pre_selected/rnd_selected).
    COND_DISPLAY = {
        "ske_preserve_gate":         "Ske.pre(gate)",
        "ske_preserve_gate_pos_eff": "Ske.pre(gate,posEff)",
        "ske_preserve_no_gate": "Ske.pre(no-gate)",
        "ske_ablation_no_gate": "Ske.abl(no-gate)",
        "ske_ablation_gate":    "Ske.abl(gate)",
        "rnd_preserve_gate":    "Rnd.pre(gate)",
        "rnd_preserve_no_gate": "Rnd.pre(no-gate)",
        "rnd_ablation_no_gate": "Rnd.abl(no-gate)",
        "rnd_ablation_gate":    "Rnd.abl(gate)",
        "anti_align":                    "Anti-align.pre",
        "ske_preserve_gate_weight_eff":  "Ske.pre(gate,wEff)",
        "ske_preserve_gate_eff":         "Ske.pre(gate,Eff)",
        "ske_preserve_gate_sqrteff":     "Ske.pre(gate,sqrtEff)",
        "ske_preserve_gate_softgate":    "Ske.pre(softgate)",
        "ske_preserve_gate_weight_delta":  "Ske.pre(gate,wDelta)",
        "ske_preserve_gate_sqrtdelta":     "Ske.pre(gate,sqrtDelta)",
        "ske_preserve_gate_eff_dominate":        "Ske.pre(effDom)",
        "ske_preserve_gate_eff_dominate_threh":  "Ske.pre(effDom,thr)",
        "eff_delta_rank_pre":         "Eff.dRank.pre(gate)",
        "eff_delta_rank_abl":         "Eff.dRank.abl(gate)",
        "eff_delta_weight_rank_pre":  "Eff.dwRank.pre(gate)",
        "eff_delta_weight_rank_abl":  "Eff.dwRank.abl(gate)",
    }

    # Per-prompt accumulators: {cond: {r, l, p, t, ti, fr, fl, fp}}
    # fr/fl/fp = family_rank/family_logit/family_prob, the lenient token-family companion metric
    acc = {cond: {"r": [], "l": [], "p": [], "t": [], "ti": [], "fr": [], "fl": [], "fp": []} for cond in active}
    ori_acc = {"r": [], "l": [], "p": [], "t": [], "ti": [], "fr": [], "fl": [], "fp": []}

    batch_size = getattr(args, "batch_size", 1)

    # Ensure a pad token exists for batched tokenization (right-padding)
    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        B = len(batch_prompts)

        # Tokenize with right-padding.  With causal attention, the last real token
        # at position last_pos[b] cannot attend to right-padding (it comes after),
        # so no explicit attention_mask is needed for the skeleton forward passes.
        _max_len  = (getattr(model.config, "n_positions", None)
                     or getattr(model.config, "max_position_embeddings", None)
                     or 2048)
        # add_special_tokens defaults to False, matching gen_domain_static_datasets.py's
        # encode() helper used to build/measure these prompts — some tokenizers (e.g.
        # ncfrey/ChemGPT's BERT-style post-processor) otherwise silently insert
        # [CLS]/[SEP] and corrupt the sequence. setup["add_special_tokens"] is True for
        # base LMs that need their real BOS restored at the token-id level instead
        # (see model_config.templates.needs_true_add_special_tokens).
        enc       = tokenizer(batch_prompts, return_tensors="pt",
                              padding=True, padding_side="right",
                              max_length=_max_len, truncation=True,
                              add_special_tokens=add_special_tokens)
        batch_ids = enc["input_ids"]                         # [B, max_len]
        batch_token_type_ids = enc.get("token_type_ids")
        last_pos  = enc["attention_mask"].sum(dim=1) - 1    # [B] last real token idx

        model_inputs = {
            "input_ids": batch_ids.to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        if batch_token_type_ids is not None:
            model_inputs["token_type_ids"] = batch_token_type_ids.to(device)
        with torch.no_grad():
            ori_full = model(**model_inputs).logits         # [B, max_len, vocab]
        lp_dev   = last_pos.to(ori_full.device)
        b_range  = torch.arange(B, device=ori_full.device)
        ori_logs = ori_full[b_range, lp_dev, :]             # [B, vocab]

        def _pre_batch(sel, **kw):
            full = forward_static_preserve(
                model, batch_ids, cache, sel,
                args.start_layer, args.end_layer, str(device),
                use_bias=use_bias, adapter=adapter, last_pos=last_pos,
                token_type_ids=batch_token_type_ids,
                **kw,
            )  # [B, max_len, vocab]
            lp2 = last_pos.to(full.device)
            return full[torch.arange(B, device=full.device), lp2, :]  # [B, vocab]

        def _abl_seq(sel, **kw):
            # Ablation not yet batched — trim padding and run single-sample per prompt
            results = []
            for b in range(B):
                seq_len = int(last_pos[b].item()) + 1
                ids_b   = batch_ids[b:b+1, :seq_len]
                type_ids_b = (
                    None if batch_token_type_ids is None
                    else batch_token_type_ids[b:b+1, :seq_len]
                )
                log_b   = forward_static_ablation(
                    model, ids_b, cache, sel,
                    args.start_layer, args.end_layer, str(device),
                    use_bias=use_bias, adapter=adapter,
                    token_type_ids=type_ids_b, **kw,
                )[0, -1, :]
                results.append(log_b)
            return torch.stack(results, dim=0)               # [B, vocab]

        COND_FWD = {
            "ske_preserve_gate":         lambda: _pre_batch(pre_selected),
            "ske_preserve_gate_pos_eff": lambda: _pre_batch(pos_eff_selected),
            "ske_preserve_no_gate": lambda: _pre_batch(pre_selected, use_gate=False),
            "ske_ablation_no_gate": lambda: _abl_seq(pre_selected),
            "ske_ablation_gate":    lambda: _abl_seq(pre_selected, use_gate=True),
            "rnd_preserve_gate":    lambda: _pre_batch(rnd_selected),
            "rnd_preserve_no_gate": lambda: _pre_batch(rnd_selected, use_gate=False),
            "rnd_ablation_no_gate": lambda: _abl_seq(rnd_selected),
            "rnd_ablation_gate":    lambda: _abl_seq(rnd_selected, use_gate=True),
            "anti_align":                    lambda: _pre_batch(pre_selected, anti_align=True),
            "ske_preserve_gate_weight_eff":  lambda: _pre_batch(eff_weight_selected),
            "ske_preserve_gate_eff":         lambda: _pre_batch(eff_uniform_selected),
            "ske_preserve_gate_sqrteff":     lambda: _pre_batch(eff_sqrt_selected),
            "ske_preserve_gate_softgate":    lambda: _pre_batch(soft_gate_selected),
            "ske_preserve_gate_weight_delta": lambda: _pre_batch(weight_delta_selected),
            "ske_preserve_gate_sqrtdelta":     lambda: _pre_batch(sqrt_delta_selected),
            "ske_preserve_gate_eff_dominate":        lambda: _pre_batch(eff_dominate_selected),
            "ske_preserve_gate_eff_dominate_threh":  lambda: _pre_batch(eff_dominate_threh_selected),
            "eff_delta_rank_pre":        lambda: _pre_batch(delta_rank_selected),
            "eff_delta_rank_abl":        lambda: _abl_seq(delta_rank_selected, use_gate=True),
            "eff_delta_weight_rank_pre": lambda: _pre_batch(delta_weight_rank_selected),
            "eff_delta_weight_rank_abl": lambda: _abl_seq(delta_weight_rank_selected, use_gate=True),
        }

        # Run all active conditions for the whole batch
        cond_logs = {}
        for cond in ALL_BASELINES:
            if cond in active:
                cond_logs[cond] = COND_FWD[cond]()          # [B, vocab]

        # Per-sample output and accumulation
        for b in range(B):
            idx    = batch_start + b + 1
            prompt = batch_prompts[b]
            o_r, o_l, o_p = get_token_stats(ori_logs[b], target_id)
            o_fr, o_fl, o_fp = family_rank_prob(ori_logs[b], family_ids)
            ori_acc["r"].append(o_r); ori_acc["l"].append(o_l)
            ori_acc["p"].append(o_p); ori_acc["t"].append(_top1(ori_logs[b]))
            ori_acc["ti"].append(_top1_id(ori_logs[b]))
            ori_acc["fr"].append(o_fr); ori_acc["fl"].append(o_fl); ori_acc["fp"].append(o_fp)

            print(f"\n[{idx:2d}] {prompt!r}")
            print(f"     Original           rank={o_r:5d}  logit={o_l:+.4f}  prob={o_p:.5f}")

            ske_pre_log = None
            for cond in ALL_BASELINES:   # fixed order for consistent printing
                if cond not in active:
                    continue
                log = cond_logs[cond][b]
                r, l, p = get_token_stats(log, target_id)
                fr, fl, fp = family_rank_prob(log, family_ids)
                acc[cond]["r"].append(r); acc[cond]["l"].append(l)
                acc[cond]["p"].append(p); acc[cond]["t"].append(_top1(log))
                acc[cond]["ti"].append(_top1_id(log))
                acc[cond]["fr"].append(fr); acc[cond]["fl"].append(fl); acc[cond]["fp"].append(fp)
                print(f"     {COND_DISPLAY[cond]:20s}  rank={r:5d}  logit={l:+.4f}  prob={p:.5f}")
                if cond == "ske_preserve_gate":
                    ske_pre_log = log

            if ske_pre_log is not None:
                print(f"     Top-{args.topk_tokens} (skeleton preserve):")
                for rk, (tok, prob) in enumerate(
                        top_tokens(ske_pre_log, tokenizer, args.topk_tokens), 1):
                    marker = " ◄" if tok.strip() == args.target_word.strip() else ""
                    print(f"       {rk:3d}.  {tok!r:25s}  {prob:.5f}{marker}")

    n = len(prompts)
    print(f"\n{sep}\n  Summary ({n} prompts)")
    print(f"  {'':22s}  {'MeanRank':>9}  {'MeanLogit':>11}  {'MeanProb':>10}  {'Top-5':>7}  {'Top-20':>8}")
    summary_rows = []
    for lbl, a in [("Original", ori_acc)] + [
        (COND_DISPLAY[c], acc[c]) for c in ALL_BASELINES if c in active
    ]:
        row = {
            "label":              lbl,
            "mean_rank":          sum(a["r"]) / n,
            "mean_logit":         sum(a["l"]) / n,
            "mean_prob":          sum(a["p"]) / n,
            "top5_count":         sum(r <= 5  for r in a["r"]),
            "top20_count":        sum(r <= 20 for r in a["r"]),
            "mean_family_rank":   sum(a["fr"]) / n,
            "mean_family_logit":  sum(a["fl"]) / n,
            "mean_family_prob":   sum(a["fp"]) / n,
            "top5_family_count":  sum(fr <= 5  for fr in a["fr"]),
            "top20_family_count": sum(fr <= 20 for fr in a["fr"]),
            "n":                  n,
        }
        summary_rows.append(row)
        print(f"  {lbl:22s}  {row['mean_rank']:>9.1f}  "
              f"{row['mean_logit']:>+11.4f}  "
              f"{row['mean_prob']:>10.6f}  "
              f"{row['top5_count']:>4}/{n}  {row['top20_count']:>5}/{n}")
    print(sep)

    # ── Save results to JSON ──────────────────────────────────────────────────
    output_dir = getattr(args, "output_dir", None) or "results/static_ske"
    safe_model  = args.model_name.replace("/", "_")
    safe_target = safe_slug(args.target_word)
    vocab_tag   = "raw_dot"
    out_dir  = os.path.join(output_dir, safe_model)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"{safe_target}_{vocab_tag}_k{args.k}_L{args.start_layer}-{args.end_layer}"
        f"_{PROMPT_FORMAT_TAG}_{RESULT_FORMAT_TAG}.json",
    )
    modified_layers = {
        str(li): int(paths["j_idx"].shape[0])
        for li, paths in pre_selected.items()
    }
    per_prompt = []
    for i in range(n):
        entry = {
            "prompt":   prompts[i],
            "original": {"rank": ori_acc["r"][i], "logit": ori_acc["l"][i],
                         "prob": ori_acc["p"][i],
                         "top1_token": ori_acc["t"][i], "top1_token_id": ori_acc["ti"][i],
                         "family_rank": ori_acc["fr"][i], "family_logit": ori_acc["fl"][i],
                         "family_prob": ori_acc["fp"][i]},
        }
        for cond in ALL_BASELINES:
            if cond in active:
                entry[cond] = {"rank": acc[cond]["r"][i], "logit": acc[cond]["l"][i],
                               "prob": acc[cond]["p"][i],
                               "top1_token": acc[cond]["t"][i], "top1_token_id": acc[cond]["ti"][i],
                               "family_rank": acc[cond]["fr"][i], "family_logit": acc[cond]["fl"][i],
                               "family_prob": acc[cond]["fp"][i]}
        per_prompt.append(entry)

    gemma_style = is_gemma_style(all_blocks)
    result_data = {
        "model_name":        args.model_name,
        "decomposition":     setup.get("decomposition", {"method": "svd"}),
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "result_format_version": RESULT_FORMAT_VERSION,
        "target_word":       args.target_word,
        "target_id":         target_id,
        "target_family_size": len(family_ids),
        "k_percent":         getattr(args, "k_percent", None),
        "modified_total_k":  args.k,
        "modified_layers":   modified_layers,
        "start_layer":       args.start_layer,
        "end_layer":         args.end_layer,
        "n_samples":         n,
        "effector_mode":     "raw_dot",
        "gemma_restore_magnit": gemma_style,
        "gemma_restore_mode": (
            "unclamped_raw_norm_ratio" if gemma_style else "disabled"
        ),
        "gemma_intervention_mode": (
            "norm_match_then_native_post_ffn_norm_then_raw_norm_restore"
            if gemma_style else "raw_down_proj_replacement"
        ),
        "chat_template":     getattr(args, "chat_template", None),
        "baselines":         sorted(active),
        "gemma_approx_intervention_note": (
            GEMMA_APPROX_INTERVENTION_NOTE if gemma_style else None
        ),
        "summary":           summary_rows,
        "per_prompt":        per_prompt,
    }
    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_file}")
    mean_path, mean_complete = write_static_model_mean(
        output_dir=output_dir,
        model_name=args.model_name,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        baselines=active,
        prompt_format_version=PROMPT_FORMAT_VERSION,
        result_format_version=RESULT_FORMAT_VERSION,
    )
    mean_state = "complete" if mean_complete else "incomplete"
    print(f"Model mean summary ({mean_state}) → {mean_path}")
    return summary_rows


def run_static_experiment(args, selection_fn, selection_kwargs, label):
    """Single-k entry point (backward compatible). Loads model once and runs one k value."""
    setup = _setup_static_experiment(args)
    _run_static_k(setup, args, selection_fn, selection_kwargs, label)
