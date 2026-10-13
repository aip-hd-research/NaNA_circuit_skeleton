"""
run_static_ske_per_neuron.py — Per-neuron static skeleton batch experiment

For each neuron j, selects top-k / random / bottom-k subspaces using forward
information flow (sign-gate: h[j]*static_score > 0) during inference.

Selection criterion (auto-detected per model):
  Standard models (no post_feedforward_layernorm):
      rank by |delta[j,i]*eff[i]|,  gate: h[j]*delta[j,i]*eff[i] > 0
  Gemma-2/3 (has post_feedforward_layernorm):
      _eff variants are an explicit control that ranks by |eff[i]| only; all
      conditions share the same gate: h[j]*delta[j,i]*eff[i] > 0. The sparse
      down_proj reconstruction is norm-matched before the native post-FFN norm,
      then its unclamped raw sparse/full norm ratio is restored afterward.

Usage (invoked by run_static_ske_per_neuron.sh, or run directly):

  python scripts_ske_cir/run_static_ske_per_neuron.py \
      --models "openai-community/gpt2-medium" \
      --target "happy" \
      --k_per_neuron_values 1 2 3 5 10 \
      --start_layer 4 \
      --n_samples 50 \
      --gpu 0,1
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import types

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model_config import (
    MODEL_TEMPLATE,
    PROMPT_FORMAT_TAG,
    PROMPT_FORMAT_VERSION,
    safe_slug,
    within_ske_path,
)
from ske_pre_valid.static_skeleton import (
    _setup_static_experiment, _effector_scores, _token_family_ids, family_rank_prob,
    get_token_stats, target_tie_count, validate_precomputed_target_id,
)
from ske_pre_valid.model_adapter import (
    gemma_post_ffn_gamma, is_gemma_style, GEMMA_APPROX_INTERVENTION_NOTE,
)
from ske_pre_valid.forward_hook.base import forward_static_preserve, forward_static_ablation
from ske_pre_valid.forward_hook.forward_per_neuron import (
    select_per_neuron_abs_topk,
    select_per_neuron_random,
    select_per_neuron_middle,
    select_per_neuron_bottom,
    select_per_neuron_eff_topk,
    select_per_neuron_eff_random,
    select_per_neuron_eff_bottom,
)
from result_summary import write_per_neuron_model_mean

CACHE_BASE         = "/nfs/data/shared/pre_compute_skeleton_cache"
DEFAULT_DATASET    = "dataset/model_token_align/gpt2-medium_10000_samples.json"
DEFAULT_OUTPUT_DIR = "results/static_ske_per_neuron"


def _seq_type(dataset_path: str | None, prompt: str | None) -> str:
    if prompt is not None:
        return "prompt_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    return os.path.basename(dataset_path).split("_")[0]


def _model_result_dir(output_dir: str, safe_model: str) -> str:
    """Return the model directory, supporting model-scoped decomposition runs."""
    if os.environ.get("NANA_MODEL_SCOPED_OUTPUT") == "1":
        return output_dir
    return os.path.join(output_dir, safe_model)

# Bump when intervention or metric semantics change without changing input IDs.
# v2 bounds Gemma magnitude restoration and uses tie-aware competition ranks.
# v3 disables both final-vocabulary and attention-logit soft-caps for all Gemma models.
# v4 removes Gemma full-output norm matching/magnitude restoration: sparse down_proj
# activations now flow directly through the model's native post-FFN RMSNorm.
# v5 restores magnitude-preserving Gemma interventions with an unclamped raw-norm ratio.
# v6 preserves tokenizer-provided token_type_ids in Original and intervention forwards.
RESULT_FORMAT_VERSION = 6
RESULT_FORMAT_TAG = f"rv{RESULT_FORMAT_VERSION}"


def _report_mean_summary(path: str, complete: bool) -> None:
    state = "complete" if complete else "incomplete"
    print(f"  model mean summary ({state}) -> {path}")

# Condition order is fixed for consistent JSON output
# top/random/bottom:             rank by |delta*eff|, gate by h*delta*eff > 0
# top_eff/random_eff/bottom_eff: rank by |eff| only,  gate by h*delta*eff > 0
# *_nogate:                      same selection, use_gate=False (all paths kept)
# top_abla_gate/nogate:          complement skeleton of top (forward_static_ablation)
CONDITIONS = (
    "top", "random", "middle", "bottom",
    "top_eff", "random_eff", "bottom_eff",
    "top_nogate", "random_nogate", "bottom_nogate",
    "top_abla_gate", "top_abla_nogate",
)
COND_DISPLAY = {
    "top":             "PerNeuron.top(gate)",
    "random":          "PerNeuron.rnd(gate)",
    "middle":          "PerNeuron.mid(gate)",
    "bottom":          "PerNeuron.bot(gate)",
    "top_eff":         "PerNeuron.top_eff(gate)",
    "random_eff":      "PerNeuron.rnd_eff(gate)",
    "bottom_eff":      "PerNeuron.bot_eff(gate)",
    "top_nogate":      "PerNeuron.top(nogate)",
    "random_nogate":   "PerNeuron.rnd(nogate)",
    "bottom_nogate":   "PerNeuron.bot(nogate)",
    "top_abla_gate":   "PerNeuron.top_abla(gate)",
    "top_abla_nogate": "PerNeuron.top_abla(nogate)",
}

# Conditions dispatched to different forward functions
_NOGATE_CONDS    = frozenset({"top_nogate", "random_nogate", "bottom_nogate"})
_ABLA_GATE_CONDS = frozenset({"top_abla_gate"})
_ABLA_NGTE_CONDS = frozenset({"top_abla_nogate"})


# ── helpers (mirrors run_static_ske.py) ──────────────────────────────────────

def last_layer(within: str) -> int:
    files = sorted(glob.glob(os.path.join(within, "layer_*.pt")))
    if not files:
        raise RuntimeError(f"no layer_*.pt files found in: {within}")
    return max(int(re.search(r"layer_(\d+)", f).group(1)) for f in files)


def needs_bfloat16(within: str) -> bool:
    d = torch.load(os.path.join(within, "layer_000.pt"), weights_only=True, map_location="cpu")
    return int(d["n_neurons"]) > 4096


# ── per-neuron runner ─────────────────────────────────────────────────────────

def _precompute_original_batches(setup, args):
    """Tokenize all prompts and run the unmodified model forward ONCE.

    The Original condition's logits depend only on (model, prompts) — not on
    target_word or k_per_neuron — but _run_per_neuron_k used to redo this full
    forward pass on every k_per_neuron call (and, across multi-target loops,
    on every target too). Precomputing here and reusing across all
    (target, k_per_neuron) combinations removes that redundant work; for an
    8-value K_PER_NEURONS sweep this cuts the Original forward from 8x (or
    8x*n_targets) down to 1x per model run.

    Returns a list of {batch_prompts, batch_ids, last_pos, ori_logs} dicts,
    ori_logs moved to CPU so it can be held across the whole target/kpn loop
    without pinning GPU memory.
    """
    model      = setup["model"]
    tokenizer  = setup["tokenizer"]
    device     = setup["device"]
    prompts    = setup["prompts"]
    add_special_tokens = setup.get("add_special_tokens", False)
    batch_size = getattr(args, "batch_size", 1)

    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    _max_len = (getattr(model.config, "n_positions", None)
                or getattr(model.config, "max_position_embeddings", None)
                or 2048)

    batches = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt",
                         padding=True, padding_side="right",
                         max_length=_max_len, truncation=True,
                         add_special_tokens=add_special_tokens)
        batch_ids = enc["input_ids"]                          # [B, max_len]
        batch_token_type_ids = enc.get("token_type_ids")
        last_pos  = enc["attention_mask"].sum(dim=1) - 1     # [B]

        model_inputs = {
            "input_ids": batch_ids.to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }
        if batch_token_type_ids is not None:
            model_inputs["token_type_ids"] = batch_token_type_ids.to(device)
        with torch.no_grad():
            ori_full = model(**model_inputs).logits          # [B, max_len, vocab]
        lp_dev  = last_pos.to(ori_full.device)
        b_range = torch.arange(len(batch_prompts), device=ori_full.device)
        ori_logs = ori_full[b_range, lp_dev, :].cpu()         # [B, vocab]

        batches.append({
            "batch_start":   batch_start,
            "batch_prompts": batch_prompts,
            "batch_ids":     batch_ids,
            "token_type_ids": batch_token_type_ids,
            "last_pos":      last_pos,
            "ori_logs":      ori_logs,
        })
    return batches


def _precompute_layer_orderings(setup, args):
    """Precompute per-layer effector scores + per-neuron sort order ONCE per target.

    top/bottom/random/eff_top/eff_bottom/eff_random selections for ANY k_per_neuron
    in the sweep are prefix/suffix slices of a sort — so instead of a full
    [n_neurons, n_subspaces] argsort (which for a 13B model is >500MB *per array*,
    in int64 that's 2 arrays x ~36 layers = tens of GB of CPU RAM per target, and
    was the cause of the whole pipeline slowing down / stalling as multiple targets
    accumulated), each is capped at max(k_per_neuron_values) via topk and stored as
    int32 — e.g. capped at 100 instead of a 5120-wide n_subspaces, ~50x smaller.

    "middle" is the one condition that needs a value from the middle of the full
    ranking, not an extreme, so it still needs a full descending argsort — computed
    (and kept, in int32) only when "middle" is actually a requested condition.

    """
    W_emb        = setup["W_emb"]
    target_id    = setup["target_id"]
    cache        = setup["cache"]
    adapter      = setup["adapter"]
    model        = setup["model"]
    all_blocks   = setup.get("all_blocks") or list(adapter.get_blocks(model))

    max_kpn = max(getattr(args, "k_per_neuron_values", None) or [100])
    _requested   = set(getattr(args, "conditions", None) or [])
    needs_middle = not _requested or "middle" in _requested

    orderings = {}
    for li in range(args.start_layer, args.end_layer + 1):
        ld    = cache.layer_skeleton(li)
        block = all_blocks[li]
        post_ffn_w = gemma_post_ffn_gamma(block)

        eff_scores = _effector_scores(
            ld["V"], W_emb, target_id, post_ffn_norm_weight=post_ffn_w)

        delta    = ld["delta"].cpu()
        effector = eff_scores.cpu()
        static_score = delta * effector.unsqueeze(0)                     # [d_ff, n_sp]
        abs_score    = static_score.abs()

        d_ff, n_sp = delta.shape
        k_cap = min(max_kpn, n_sp)

        top_order    = abs_score.topk(k_cap, dim=1).indices.to(torch.int32)               # [d_ff, k_cap]
        bottom_order = abs_score.topk(k_cap, dim=1, largest=False).indices.to(torch.int32)  # [d_ff, k_cap]
        middle_order = (abs_score.argsort(dim=1, descending=True).to(torch.int32)
                         if needs_middle else None)                                        # [d_ff, n_sp] or None

        rng = torch.Generator()
        rng.manual_seed(42 + li)
        rand_scores = torch.rand(d_ff, n_sp, generator=rng)
        rand_order  = rand_scores.topk(k_cap, dim=1).indices.to(torch.int32)              # [d_ff, k_cap]

        eff_order = effector.abs().argsort(descending=True)   # [n_sp] — tiny, keep full

        orderings[li] = {
            "V":             ld["V"],
            "delta":         delta,
            "effector":      effector,
            "static_score":  static_score,
            "top_order":     top_order,
            "bottom_order":  bottom_order,
            "middle_order":  middle_order,
            "rand_order":    rand_order,
            "eff_order":     eff_order,
            "k_cap":         k_cap,
        }
    return orderings


def _slice_per_neuron_order(pre, k_per_neuron, mode):
    """Slice a _precompute_layer_orderings() entry for one (k_per_neuron, mode).

    mode: "top" | "middle" | "bottom" | "random" | "eff_top" | "eff_bottom" | "eff_random"
    Reproduces exactly what select_per_neuron_*(k_per_neuron=k_per_neuron) in
    forward_per_neuron.py would return (same selected (j,i) set), without re-sorting.
    """
    delta, effector, static_score = pre["delta"], pre["effector"], pre["static_score"]
    d_ff, n_sp = delta.shape
    k_each = min(k_per_neuron, n_sp)
    if k_each > pre["k_cap"]:
        raise ValueError(
            f"k_per_neuron={k_per_neuron} exceeds the cached cap {pre['k_cap']} — "
            f"pass the full k_per_neuron_values list via args so _precompute_layer_orderings "
            f"caps at the true max."
        )

    if mode == "eff_top":
        i_idx = pre["eff_order"][:k_each].unsqueeze(0).expand(d_ff, -1).flatten().long()
    elif mode == "eff_bottom":
        i_idx = pre["eff_order"][n_sp - k_each:].unsqueeze(0).expand(d_ff, -1).flatten().long()
    else:
        if mode == "top":
            i_mat = pre["top_order"][:, :k_each]
        elif mode == "bottom":
            i_mat = pre["bottom_order"][:, :k_each]
        elif mode == "middle":
            if pre["middle_order"] is None:
                raise ValueError(
                    "'middle' was not in args.conditions when _precompute_layer_orderings ran, "
                    "so the full sort needed for it wasn't cached."
                )
            start = (n_sp - k_each) // 2
            i_mat = pre["middle_order"][:, start:start + k_each]
        elif mode in ("random", "eff_random"):
            # same fixed-seed permutation as select_per_neuron_random/eff_random
            i_mat = pre["rand_order"][:, :k_each]
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        i_idx = i_mat.flatten().long()

    j_idx = torch.arange(d_ff).unsqueeze(1).expand(-1, k_each).flatten().long()
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    delta[j_idx, i_idx],
        "static_scores": static_score[j_idx, i_idx],
        "effector":      effector,
    }


def _run_per_neuron_k(setup, args, k_per_neuron, label):
    """Run one k_per_neuron value using a pre-loaded setup.

    Three conditions, all with sign-gate (forward information flow):
      top    — top-k subspaces per neuron
      random — k uniformly-random subspaces per neuron
      bottom — bottom-k subspaces per neuron (anti-skeleton)

    All gate-enabled conditions share gate: h[j]*delta[j,i]*eff[i] > 0.
    Selection criterion differs by condition:
      top/random/bottom:             rank by |delta[j,i]*eff[i]|
      top_eff/random_eff/bottom_eff: rank by |eff[i]| only

    Saves results to JSON and returns summary_rows.
    """
    model        = setup["model"]
    tokenizer    = setup["tokenizer"]
    target_id    = setup["target_id"]
    target_info  = setup.get("target_info", {})
    family_ids   = setup.get("family_ids") or [target_id]
    cache        = setup["cache"]
    adapter      = setup["adapter"]
    device       = setup["device"]
    prompts      = setup["prompts"]

    print(f"\nBuilding {label}  [effector=raw_dot]"
          f"  (target={args.target_word!r}) ...")

    # ── select paths for each layer ───────────────────────────────────────────
    # layer_orderings holds the full per-layer sort (effector scores, |delta*eff|
    # descending order, random permutation, |eff| order) — computed once per target
    # in _update_setup_target / run_model, not per k_per_neuron. Slicing it here is
    # O(1) instead of re-sorting the [n_neurons, n_subspaces] matrix on every kpn call.
    layer_orderings = setup.get("layer_orderings")
    if layer_orderings is None:
        layer_orderings = _precompute_layer_orderings(setup, args)
        setup["layer_orderings"] = layer_orderings

    top_selected      = {}
    rnd_selected      = {}
    mid_selected      = {}
    bot_selected      = {}
    top_eff_selected  = {}
    rnd_eff_selected  = {}
    bot_eff_selected  = {}

    for li in range(args.start_layer, args.end_layer + 1):
        pre = layer_orderings[li]

        top_selected[li]     = _slice_per_neuron_order(pre, k_per_neuron, "top")
        rnd_selected[li]     = _slice_per_neuron_order(pre, k_per_neuron, "random")
        # middle_order is only cached by _precompute_layer_orderings when "middle" is
        # actually requested (see needs_middle there) — calling _slice_per_neuron_order
        # unconditionally would raise ValueError whenever --conditions excludes "middle".
        mid_selected[li]     = (_slice_per_neuron_order(pre, k_per_neuron, "middle")
                                 if pre["middle_order"] is not None else None)
        bot_selected[li]     = _slice_per_neuron_order(pre, k_per_neuron, "bottom")
        top_eff_selected[li] = _slice_per_neuron_order(pre, k_per_neuron, "eff_top")
        rnd_eff_selected[li] = _slice_per_neuron_order(pre, k_per_neuron, "eff_random")
        bot_eff_selected[li] = _slice_per_neuron_order(pre, k_per_neuron, "eff_bottom")
        n_paths = top_selected[li]["j_idx"].shape[0]
        print(f"  Layer {li:3d}: {n_paths} paths (k_per_neuron={k_per_neuron})")

    sep = "═" * 70
    print(f"\n{sep}\n  {label}  |  target={args.target_word!r}"
          f"  |  layers {args.start_layer}–{args.end_layer}\n{sep}")

    # ── active conditions (subset of CONDITIONS selected via --conditions) ───────
    _requested = set(getattr(args, "conditions", None) or [])
    active_conds = tuple(c for c in CONDITIONS if (not _requested or c in _requested))
    conds_tag = "-".join(active_conds) if _requested else "all"

    # ── accumulator setup ─────────────────────────────────────────────────────
    all_conds = ("original",) + active_conds
    acc = {
        c: {"r": [], "l": [], "p": [], "tc": [], "t": [], "ti": [],
            "fr": [], "fl": [], "fp": []}
        for c in all_conds
    }

    original_batches = setup.get("original_batches")
    if original_batches is None:
        original_batches = _precompute_original_batches(setup, args)
        setup["original_batches"] = original_batches   # cache for subsequent kpn/target calls

    cond_sel_map = {
        "top":             top_selected,
        "random":          rnd_selected,
        "middle":          mid_selected,
        "bottom":          bot_selected,
        "top_eff":         top_eff_selected,
        "random_eff":      rnd_eff_selected,
        "bottom_eff":      bot_eff_selected,
        "top_nogate":      top_selected,
        "random_nogate":   rnd_selected,
        "bottom_nogate":   bot_selected,
        "top_abla_gate":   top_selected,
        "top_abla_nogate": top_selected,
    }

    def _top1(logit_vec):
        return tokenizer.decode([int(logit_vec.cpu().argmax())])

    def _top1_id(logit_vec):
        return int(logit_vec.cpu().argmax())

    # ── batched forward loop ──────────────────────────────────────────────────
    # batch_ids/last_pos/ori_logs come from the precomputed cache (see
    # _precompute_original_batches) instead of re-tokenizing and re-running
    # the unmodified model forward on every k_per_neuron call.
    for batch in original_batches:
        batch_start   = batch["batch_start"]
        batch_prompts = batch["batch_prompts"]
        batch_ids     = batch["batch_ids"]
        batch_token_type_ids = batch.get("token_type_ids")
        last_pos      = batch["last_pos"]
        ori_logs      = batch["ori_logs"]
        B = len(batch_prompts)

        def _pre_batch(sel):
            """Forward preserve with sign-gate (forward information flow)."""
            full = forward_static_preserve(
                model, batch_ids, cache, sel,
                args.start_layer, args.end_layer, str(device),
                use_bias=True, adapter=adapter, last_pos=last_pos,
                use_gate=True,
                token_type_ids=batch_token_type_ids,
            )
            lp2 = last_pos.to(full.device)
            return full[torch.arange(B, device=full.device), lp2, :]  # [B, vocab]

        def _pre_batch_nogate(sel):
            """Forward preserve without gate (all selected paths kept)."""
            full = forward_static_preserve(
                model, batch_ids, cache, sel,
                args.start_layer, args.end_layer, str(device),
                use_bias=True, adapter=adapter, last_pos=last_pos,
                use_gate=False,
                token_type_ids=batch_token_type_ids,
            )
            lp2 = last_pos.to(full.device)
            return full[torch.arange(B, device=full.device), lp2, :]  # [B, vocab]

        def _pre_abla_batch(sel, use_gate):
            """Forward ablation (complement skeleton), batched via last_pos."""
            full = forward_static_ablation(
                model, batch_ids, cache, sel,
                args.start_layer, args.end_layer, str(device),
                use_bias=True, adapter=adapter, use_gate=use_gate,
                last_pos=last_pos,
                token_type_ids=batch_token_type_ids,
            )
            lp2 = last_pos.to(full.device)
            return full[torch.arange(B, device=full.device), lp2, :]  # [B, vocab]

        cond_logs = {}
        for c in active_conds:
            sel = cond_sel_map[c]
            if c in _NOGATE_CONDS:
                cond_logs[c] = _pre_batch_nogate(sel)
            elif c in _ABLA_GATE_CONDS:
                cond_logs[c] = _pre_abla_batch(sel, use_gate=True)
            elif c in _ABLA_NGTE_CONDS:
                cond_logs[c] = _pre_abla_batch(sel, use_gate=False)
            else:
                cond_logs[c] = _pre_batch(sel)

        for b in range(B):
            idx    = batch_start + b + 1
            prompt = batch_prompts[b]
            o_r, o_l, o_p = get_token_stats(ori_logs[b], target_id)
            o_tc = target_tie_count(ori_logs[b], target_id)
            o_fr, o_fl, o_fp = family_rank_prob(ori_logs[b], family_ids)
            for key, val in zip(("r", "l", "p", "tc", "t", "ti", "fr", "fl", "fp"),
                                (o_r, o_l, o_p, o_tc, _top1(ori_logs[b]),
                                 _top1_id(ori_logs[b]), o_fr, o_fl, o_fp)):
                acc["original"][key].append(val)

            print(f"\n[{idx:2d}] {prompt!r}")
            print(f"     Original               rank={o_r:5d}  logit={o_l:+.4f}  prob={o_p:.5f}")

            for cond in active_conds:
                log = cond_logs[cond][b]
                r, l, p = get_token_stats(log, target_id)
                tc = target_tie_count(log, target_id)
                fr, fl, fp = family_rank_prob(log, family_ids)
                for key, val in zip(("r", "l", "p", "tc", "t", "ti", "fr", "fl", "fp"),
                                    (r, l, p, tc, _top1(log), _top1_id(log), fr, fl, fp)):
                    acc[cond][key].append(val)
                print(f"     {COND_DISPLAY[cond]:25s}  rank={r:5d}  logit={l:+.4f}  prob={p:.5f}")

    # ── summary ───────────────────────────────────────────────────────────────
    n = len(prompts)
    print(f"\n{sep}\n  Summary ({n} prompts)")
    print(f"  {'':27s}  {'MeanRank':>9}  {'MeanLogit':>11}  {'MeanProb':>10}  "
          f"{'Top-5':>7}  {'Top-20':>8}")
    summary_rows = []
    for cond_name, disp in [("original", "Original")] + [
        (c, COND_DISPLAY[c]) for c in active_conds
    ]:
        a = acc[cond_name]
        row = {
            "label":              disp,
            "mean_rank":          sum(a["r"]) / n,
            "mean_logit":         sum(a["l"]) / n,
            "mean_prob":          sum(a["p"]) / n,
            "mean_target_tie_count": sum(a["tc"]) / n,
            "max_target_tie_count":  max(a["tc"]),
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
        print(f"  {disp:27s}  {row['mean_rank']:>9.1f}  "
              f"{row['mean_logit']:>+11.4f}  {row['mean_prob']:>10.6f}  "
              f"{row['top5_count']:>4}/{n}  {row['top20_count']:>5}/{n}")
    print(sep)

    # ── save JSON ─────────────────────────────────────────────────────────────
    output_dir  = getattr(args, "output_dir", None) or DEFAULT_OUTPUT_DIR
    safe_model  = args.model_name.replace("/", "_")
    safe_target  = safe_slug(args.target_word)
    vocab_tag    = "raw_dot"
    seq_type     = _seq_type(args.dataset_path, getattr(args, "prompt", None))
    out_dir      = os.path.join(_model_result_dir(output_dir, safe_model), seq_type)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"{safe_target}_{vocab_tag}_kpn{k_per_neuron}"
        f"_L{args.start_layer}-{args.end_layer}_{conds_tag}"
        f"_{PROMPT_FORMAT_TAG}_{RESULT_FORMAT_TAG}.json",
    )

    per_prompt = []
    for i in range(n):
        entry = {
            "prompt":   prompts[i],
            "original": {
                "rank": acc["original"]["r"][i], "logit": acc["original"]["l"][i],
                "prob": acc["original"]["p"][i], "top1_token": acc["original"]["t"][i],
                "top1_token_id": acc["original"]["ti"][i],
                "target_tie_count": acc["original"]["tc"][i],
                "family_rank": acc["original"]["fr"][i], "family_logit": acc["original"]["fl"][i],
                "family_prob": acc["original"]["fp"][i],
            },
        }
        for cond in active_conds:
            entry[cond] = {
                "rank": acc[cond]["r"][i], "logit": acc[cond]["l"][i],
                "prob": acc[cond]["p"][i], "top1_token": acc[cond]["t"][i],
                "top1_token_id": acc[cond]["ti"][i],
                "target_tie_count": acc[cond]["tc"][i],
                "family_rank": acc[cond]["fr"][i], "family_logit": acc[cond]["fl"][i],
                "family_prob": acc[cond]["fp"][i],
            }
        per_prompt.append(entry)

    gemma_style = is_gemma_style(adapter.get_blocks(model))
    result_data = {
        "model_name":         args.model_name,
        "decomposition":      setup.get("decomposition", {"method": "svd"}),
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "result_format_version": RESULT_FORMAT_VERSION,
        "target_word":        args.target_word,
        "target_id":          target_id,
        "target_family_size": len(family_ids),
        "target_decoded":     target_info.get("target_decoded", tokenizer.decode([target_id])),
        "target_is_single":   target_info.get("target_is_single", True),
        "target_resolve":     target_info.get("target_resolve", "unknown"),
        "k_per_neuron":       k_per_neuron,
        "total_paths_per_layer": {
            str(li): int(top_selected[li]["j_idx"].shape[0])
            for li in top_selected
        },
        "start_layer":        args.start_layer,
        "end_layer":          args.end_layer,
        "n_samples":          n,
        "effector_mode":      "raw_dot",
        "gemma_restore_magnit": gemma_style,
        "gemma_restore_mode": "unclamped_raw_norm_ratio" if gemma_style else "disabled",
        "gemma_intervention_mode": (
            "norm_match_then_native_post_ffn_norm_then_raw_norm_restore"
            if gemma_style else "raw_down_proj_replacement"
        ),
        "gemma_logit_softcaps": {
            "policy": "disabled",
            "original_final": getattr(
                model, "_gemma_disabled_logit_softcaps", {}
            ).get("final"),
            "original_attention": getattr(
                model, "_gemma_disabled_logit_softcaps", {}
            ).get("attention"),
            "cleared_attention_layers": getattr(
                model, "_gemma_disabled_logit_softcaps", {}
            ).get("layers", 0),
        },
        "conditions":         list(active_conds),
        "gemma_approx_intervention_note": (
            GEMMA_APPROX_INTERVENTION_NOTE if gemma_style else None
        ),
        "summary":            summary_rows,
        "per_prompt":         per_prompt,
    }
    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_file}")
    return summary_rows


# ── target update helper ──────────────────────────────────────────────────────

def _update_setup_target(setup: dict, args, target_word: str,
                          precomputed_id: int | None = None) -> None:
    """Update target-dependent fields in a loaded setup without reloading the model.

    Updates in-place: target_id and W_emb_target.
    Model weights, tokenizer, cache, and prompts are unchanged.

    precomputed_id: id already resolved by gen_static_target.py and read straight out
        of the target JSON (see run_model_target_json) — skips re-resolving via
        _resolve_target_info. None falls back to resolving here (e.g. for the
        single-target CLI path, which has no target JSON to read from).
    """
    tokenizer = setup["tokenizer"]
    W_emb     = setup["W_emb"]
    if precomputed_id is not None:
        target_id = validate_precomputed_target_id(
            tokenizer, target_word, precomputed_id
        )
        setup["target_info"] = {
            "target_id":        target_id,
            "target_decoded":   tokenizer.decode([target_id]),
            "target_is_single": True,
            "target_resolve":   "precomputed",
        }
        print(f"  target {target_word!r} → id={target_id} "
              f"({tokenizer.decode([target_id])!r})  resolve=precomputed")
    else:
        from ske_pre_valid.static_skeleton import _resolve_target_info
        tgt_info  = _resolve_target_info(tokenizer, target_word)
        if not tgt_info["target_is_single"]:
            raise ValueError(
                f"target {target_word!r} is not a single token "
                f"(resolve={tgt_info['target_resolve']})"
            )
        target_id = tgt_info["target_id"]
        setup["target_info"] = tgt_info
        print(f"  target {target_word!r} → id={target_id} ({tgt_info['target_decoded']!r})"
              f"  resolve={tgt_info['target_resolve']}  single={tgt_info['target_is_single']}")

    setup["target_id"]    = target_id
    setup["family_ids"]   = _token_family_ids(tokenizer, target_word, target_id=target_id)
    setup["W_emb_target"] = W_emb[target_id].cpu().float()
    args.target_word      = target_word

    # Per-neuron sort order depends only on (target, layer) — recompute on every
    # target switch, reused across all k_per_neuron values for this target.
    setup["layer_orderings"] = _precompute_layer_orderings(setup, args)


# ── per-model orchestration ───────────────────────────────────────────────────

def run_model(model_name: str, target: str, k_per_neuron_values: list,
              n_samples: int, gpu: str,
              dataset: str | None, output_dir: str, cache_base: str,
              start_layer: int,
              batch_size: int = 1,
              gemma_restore_magnit: bool = False,
              conditions: list | None = None,
              prompt: str | None = None) -> None:
    within = within_ske_path(model_name, cache_base)
    if within is None:
        print(f"[SKIP] no within_ske_cache for: {model_name}")
        return

    try:
        end  = last_layer(within)
        bf16 = needs_bfloat16(within)
    except RuntimeError as e:
        print(f"[SKIP] {model_name}: {e}")
        return
    tmpl_key = MODEL_TEMPLATE.get(model_name)

    safe_model  = model_name.replace("/", "_")
    safe_target = safe_slug(target)
    vocab_tag = "raw_dot"
    seq_type    = _seq_type(dataset, prompt)
    out_dir = os.path.join(_model_result_dir(output_dir, safe_model), seq_type)
    os.makedirs(out_dir, exist_ok=True)

    # Filename must encode the requested condition set — otherwise a rerun with a
    # different --conditions baseline set would be silently skipped as "already done"
    # against a file that's missing the newly-requested baselines.
    _requested   = set(conditions) if conditions else set()
    active_conds = tuple(c for c in CONDITIONS if (not _requested or c in _requested))
    conds_tag    = "-".join(active_conds) if _requested else "all"

    def _write_mean_summary() -> None:
        result = write_per_neuron_model_mean(
            output_dir=output_dir,
            model_name=model_name,
            seq_type=seq_type,
            targets=[target],
            k_per_neuron_values=k_per_neuron_values,
            start_layer=start_layer,
            end_layer=end,
            conditions=active_conds,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            result_format_version=RESULT_FORMAT_VERSION,
        )
        _report_mean_summary(*result)

    # Check which k_per_neuron values still need to be run
    pending = []
    for kpn in k_per_neuron_values:
        out_file = os.path.join(
            out_dir,
            f"{safe_target}_{vocab_tag}_kpn{kpn}_L{start_layer}-{end}"
            f"_{conds_tag}_{PROMPT_FORMAT_TAG}_{RESULT_FORMAT_TAG}.json",
        )
        if os.path.exists(out_file):
            print(f"  [SKIP] exists: {out_file}")
        else:
            pending.append(kpn)

    if not pending:
        print(f"[SKIP] all k_per_neuron values done for {model_name}")
        _write_mean_summary()
        return

    print(f"====== {model_name} | target={target!r} | layers {start_layer}-{end} | "
          f"bf16={bf16} | tmpl={tmpl_key or 'none'} | "
          f"{len(pending)}/{len(k_per_neuron_values)} k_per_neuron values to run ======")

    args = types.SimpleNamespace(
        model_name            = model_name,
        cache_dir             = within,
        target_word           = target,
        start_layer           = start_layer,
        end_layer             = end,
        n_samples             = n_samples,
        dataset_path          = dataset,
        prompt                = prompt,
        gpu                   = gpu,
        output_dir            = output_dir,
        gemma_restore_magnit  = gemma_restore_magnit,
        bfloat16              = bf16,
        chat_template         = tmpl_key,
        random_seed           = 42,
        use_bias              = True,
        topk_tokens           = 10,
        batch_size            = batch_size,
        k                     = None,
        k_percent             = None,
        conditions            = conditions,
        k_per_neuron_values   = k_per_neuron_values,
    )

    # Load model ONCE for all pending k_per_neuron values
    model_setup = _setup_static_experiment(args)
    # Original forward and per-neuron sort order don't depend on k_per_neuron —
    # compute once, reuse for every kpn value in the sweep.
    model_setup["original_batches"] = _precompute_original_batches(model_setup, args)
    model_setup["layer_orderings"]  = _precompute_layer_orderings(model_setup, args)

    for kpn in pending:
        _run_per_neuron_k(
            model_setup, args, kpn,
            f"Per-neuron static  k/neuron={kpn}",
        )
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    _write_mean_summary()


def run_model_target_json(
    model_name: str, target_json: str,
    target_categories: list[str] | None, first_n: int,
    k_per_neuron_values: list, n_samples: int, gpu: str,
    dataset: str, output_dir: str, cache_base: str, start_layer: int,
    batch_size: int = 1,
    gemma_restore_magnit: bool = False, conditions: list | None = None,
) -> None:
    """Run static skeleton for targets read from a model_target JSON file.

    The model is loaded ONCE; effector scores are recomputed per target token.

    target_json format:
        {"categories": {"animal": [...], "color": [...]}, "tokens": [...], ...}

    target_categories: list of category names to use (None = all categories).
    first_n: take the first N tokens from each selected category (deterministic).
    """
    with open(target_json) as f:
        target_data = json.load(f)

    categories: dict = target_data.get("categories", {})
    if not categories:
        # fallback: flat token list (old format without categories)
        flat = target_data.get("tokens", [])
        targets = flat[:first_n]
    else:
        selected_cats = target_categories if target_categories else list(categories.keys())
        unknown = [c for c in selected_cats if c not in categories]
        if unknown:
            print(f"  [WARN] unknown categories: {unknown}; available: {list(categories.keys())}")
        seen: set[str] = set()
        targets: list[str] = []
        for cat in selected_cats:
            for tok in categories.get(cat, [])[:first_n]:
                if tok not in seen:
                    seen.add(tok)
                    targets.append(tok)

    print(f"  categories={selected_cats if categories else ['(flat)']}  "
          f"first_n={first_n}  → {len(targets)} targets")

    # Target ids: gen_static_target.py already resolved (and filtered on) exactly
    # this — every token in `targets` was required to be a single valid token for
    # ALL models in the domain (min_models=len(models) by default), and its id per
    # model was saved alongside. Read that id straight out of the JSON instead of
    # re-resolving it here: one resolution site (gen_static_target.py's
    # resolve_single_token_id, which _is_ the encode()-first logic _resolve_target_info
    # uses) means no risk of a duplicated check drifting stricter/looser than what
    # actually runs (see chemical_smiles/MolGPT 'C' vs ' C' incident — a duplicated,
    # stricter pre-filter used to silently skip a model entirely over a conflict the
    # real resolver tolerates).
    precomputed = target_data.get("token_ids", {})
    target_id_map: dict[str, int] = {}
    _filtered = []
    for t in targets:
        tid = precomputed.get(t, {}).get(model_name, {}).get("id")
        if tid is None:
            # Fallback for target_json files predating this field, or a model added
            # to the domain after the json was last generated — resolve on the fly
            # via the same canonical resolver static skeleton itself uses at run time.
            from transformers import AutoTokenizer as _AutoTok
            from ske_pre_valid.static_skeleton import _resolve_target_info
            _tok  = _AutoTok.from_pretrained(model_name)
            info  = _resolve_target_info(_tok, t)
            if not info["target_is_single"]:
                print(f"  [SKIP] {t!r}: not resolvable for {model_name} "
                      f"(resolve={info['target_resolve']}); consider regenerating {target_json}")
                continue
            tid = info["target_id"]
        target_id_map[t] = tid
        _filtered.append(t)
    targets = _filtered
    if not targets:
        print(f"[SKIP] no single-token targets remain for {model_name} after filtering")
        return

    within = within_ske_path(model_name, cache_base)
    if within is None:
        print(f"[SKIP] no within_ske_cache for: {model_name}")
        return

    try:
        end  = last_layer(within)
        bf16 = needs_bfloat16(within)
    except RuntimeError as e:
        print(f"[SKIP] {model_name}: {e}")
        return
    tmpl_key = MODEL_TEMPLATE.get(model_name)

    vocab_tag = "raw_dot"
    safe_model  = model_name.replace("/", "_")
    seq_type    = os.path.basename(dataset).split("_")[0]   # long / short / random
    out_dir     = os.path.join(_model_result_dir(output_dir, safe_model), seq_type)
    os.makedirs(out_dir, exist_ok=True)

    # Filename must encode the requested condition set — otherwise a rerun with a
    # different --conditions baseline set would be silently skipped as "already done"
    # against a file that's missing the newly-requested baselines.
    _requested   = set(conditions) if conditions else set()
    active_conds = tuple(c for c in CONDITIONS if (not _requested or c in _requested))
    conds_tag    = "-".join(active_conds) if _requested else "all"

    def _write_mean_summary() -> None:
        result = write_per_neuron_model_mean(
            output_dir=output_dir,
            model_name=model_name,
            seq_type=seq_type,
            targets=targets,
            k_per_neuron_values=k_per_neuron_values,
            start_layer=start_layer,
            end_layer=end,
            conditions=active_conds,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            result_format_version=RESULT_FORMAT_VERSION,
        )
        _report_mean_summary(*result)

    # Skip entirely if all (target, kpn) combos already exist
    pending_targets = []
    for target in targets:
        safe_target  = safe_slug(target)
        pending_kpns = [
            kpn for kpn in k_per_neuron_values
            if not os.path.exists(os.path.join(
                out_dir,
                f"{safe_target}_{vocab_tag}_kpn{kpn}_L{start_layer}-{end}"
                f"_{conds_tag}_{PROMPT_FORMAT_TAG}_{RESULT_FORMAT_TAG}.json",
            ))
        ]
        if pending_kpns:
            pending_targets.append((target, pending_kpns))

    if not pending_targets:
        print(f"[SKIP] all targets done for {model_name}")
        _write_mean_summary()
        return

    print(f"====== {model_name} | {len(pending_targets)}/{len(targets)} targets to run "
          f"| layers {start_layer}-{end} | bf16={bf16} "
          f"| tmpl={tmpl_key or 'none'} ======")

    args = types.SimpleNamespace(
        model_name           = model_name,
        cache_dir            = within,
        target_word          = pending_targets[0][0],  # updated per target below
        target_id            = target_id_map[pending_targets[0][0]],
        start_layer          = start_layer,
        end_layer            = end,
        n_samples            = n_samples,
        dataset_path         = dataset,
        gpu                  = gpu,
        output_dir           = output_dir,
        gemma_restore_magnit = gemma_restore_magnit,
        bfloat16             = bf16,
        chat_template        = tmpl_key,
        random_seed          = 42,
        use_bias             = True,
        topk_tokens          = 10,
        batch_size           = batch_size,
        k                    = None,
        k_percent            = None,
        conditions           = conditions,
        k_per_neuron_values  = k_per_neuron_values,
    )

    model_setup = _setup_static_experiment(args)
    # Original forward depends only on (model, prompts), not target_word or
    # k_per_neuron — compute once, reuse across every target × kpn combination.
    model_setup["original_batches"] = _precompute_original_batches(model_setup, args)

    for target, pending_kpns in pending_targets:
        print(f"\n-- target={target!r}  ({len(pending_kpns)} k values) --")
        _update_setup_target(model_setup, args, target, precomputed_id=target_id_map.get(target))
        for kpn in pending_kpns:
            _run_per_neuron_k(model_setup, args, kpn, f"Per-neuron static  k/neuron={kpn}")
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    _write_mean_summary()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",              nargs="+", required=True)
    # Target selection: pass exactly one of --target or --target_json
    tgt = p.add_mutually_exclusive_group()
    tgt.add_argument("--target",            default=None,
                     help="Single target word (e.g. 'happy')")
    tgt.add_argument("--target_json",         default=None,
                     help="Path to model_target JSON with semantic categories")
    p.add_argument("--target_categories",    nargs="*", default=None,
                   help="Category names to use from target_json (default: all). "
                        "e.g. --target_categories animal color country")
    p.add_argument("--first_n",              type=int, default=10,
                   help="Take the first N tokens from each selected category (default: 10)")
    p.add_argument("--k_per_neuron_values", nargs="+", type=int, default=[1, 2, 3, 5, 10],
                   help="Number of subspaces to select per neuron (swept in one model load)")
    p.add_argument("--start_layer",         type=int, required=True,
                   help="First MLP layer to include (end_layer is the last cached layer)")
    p.add_argument("--n_samples",           type=int, default=20)
    p.add_argument("--batch_size",          type=int, default=1)
    p.add_argument("--gpu",                 type=str, default="0",
                   help="GPU index or comma-separated indices for multi-GPU (e.g. '0,1')")
    p.add_argument("--dataset_path",        default=DEFAULT_DATASET)
    p.add_argument("--prompt",              default=None,
                   help="Use one prompt directly instead of reading --dataset_path")
    p.add_argument("--output_dir",          default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--cache_base",          default=CACHE_BASE,
                   help="Skeleton cache root: <cache_base>/<model>/within_ske_cache")
    p.add_argument("--gemma_restore_magnit", action="store_true",
                   help="Deprecated compatibility flag; ignored. Gemma-2/3 always use "
                        "norm matching plus unclamped post-norm magnitude restoration.")
    p.add_argument("--conditions", nargs="+", default=None,
                   choices=list(CONDITIONS),
                   help="Subset of conditions to run (default: all). "
                        f"Available: {', '.join(CONDITIONS)}")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.target is None and args.target_json is None:
        args.target = "happy"   # backward-compatible default
    if args.prompt is not None and args.target_json is not None:
        raise SystemExit("error: --prompt requires --target, not --target_json")
    os.environ.setdefault("HF_HOME", "/nfs/data/shared/model_cache")

    if args.target_json is not None:
        for m in args.models:
            run_model_target_json(
                m, args.target_json, args.target_categories, args.first_n,
                args.k_per_neuron_values, args.n_samples, args.gpu,
                args.dataset_path, args.output_dir, args.cache_base, args.start_layer,
                batch_size=args.batch_size,
                gemma_restore_magnit=args.gemma_restore_magnit, conditions=args.conditions,
            )
    else:
        for m in args.models:
            run_model(
                m, args.target, args.k_per_neuron_values,
                args.n_samples, args.gpu,
                args.dataset_path, args.output_dir, args.cache_base,
                args.start_layer,
                batch_size=args.batch_size,
                gemma_restore_magnit=args.gemma_restore_magnit,
                conditions=args.conditions,
                prompt=args.prompt,
            )
