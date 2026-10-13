"""
skeleton_mlp_forward_ori_acv.py

Variant of skeleton_forward where path selection uses activations from the
*original* (unmodified) forward pass rather than the in-progress intervened pass.

Motivation
----------
In skeleton_forward, h at layer L is the activation produced by the already-
intervened residual stream (layers < L have been modified).  With --rely_ori_acv,
we first run the clean original model to collect h at every layer, then use
those stored activations to pre-select paths before running the intervened pass.

Two-pass flow
-------------
Pass 1 (clean):   run original model, hook h at every layer in [start, end]
                  → pre_selected_paths[layer] = topk(h_orig, delta, effector)

Pass 2 (intervened): run modified forward; at each layer use pre_selected_paths
                  instead of re-selecting from the modified h

Architecture support
--------------------
Accepts an optional `adapter` (ModelAdapter) parameter.  If None, auto-detects
from the model.  h is captured by attaching a forward_pre_hook to the output
projection module (c_proj for GPT-2, down_proj for Llama), so inp[0] = h.

Performance
-----------
Path pre-selection: fully vectorised — no Python loops, no list-of-dicts.
Reconstruction:     scatter_add avoids the [k, d_model] V-gather that caused OOM.
Fast path:          k1 >= d_ff*n_sp → skip topk; use h @ delta @ V directly.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from ske_pre_valid.within_ske_cache import WithinSkeCache
from ske_pre_valid.model_adapter import move_to_block_device, gemma_post_ffn_gamma


# ── Pass 1: collect original activations ─────────────────────────────────────

@torch.no_grad()
def _collect_original_h(
    model,
    input_ids: torch.Tensor,
    layers: list,
    device: str,
    adapter,
    token_type_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> dict:
    """Return {layer_idx: h_tensor [batch, seq, d_ff]} for each layer in `layers`.

    Uses a forward_pre_hook on the output projection module (c_proj / down_proj)
    to capture h = input to that projection, which equals the post-activation
    MLP hidden state for both GPT-2 and Llama-style architectures.
    """
    storage = {}
    hooks   = []
    blocks  = adapter.get_blocks(model)

    for li in layers:
        def make_hook(idx):
            def hook_fn(_module, inp):
                storage[idx] = inp[0].detach().cpu()
            return hook_fn
        hooks.append(
            adapter.h_hook_module(blocks[li]).register_forward_pre_hook(make_hook(li))
        )

    model_inputs = {"input_ids": input_ids.to(device)}
    if token_type_ids is not None:
        model_inputs["token_type_ids"] = token_type_ids.to(device)
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask.to(device)
    model(**model_inputs)

    for h in hooks:
        h.remove()

    return storage


# ── Pass 2: intervened forward ────────────────────────────────────────────────

@torch.no_grad()
def skeleton_forward_ori_acv(
    model,
    input_ids: torch.Tensor,
    cache: WithinSkeCache,
    start_layer: int,
    end_layer: int,
    k1: int,
    seq_reduce: str = "last",
    device: str = "cpu",
    intervention: str = "preserve",   # "preserve" | "ablation"
    use_bias: bool = True,
    target_id: int | None = None,
    W_emb: torch.Tensor | None = None,
    token_mode: str = "last",
    pre_norm_effs: dict | None = None,
    verbose: bool = False,
    adapter=None,
    restore_magnit: bool = False,
    layer_gpu_tensors: dict | None = None,
    token_type_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> dict:
    """
    Skeleton forward pass where paths are selected from the *original* activations.

    Parameters
    ----------
    intervention : "preserve" — MLP output = skeleton paths only
                   "ablation" — MLP output = full MLP − skeleton paths
    target_id    : target token id for effector alignment
    W_emb        : embedding matrix [vocab, d_model]
    token_mode   : "last" — intervene on last token only
                   "all"  — intervene on every position
    verbose      : show per-layer tqdm bars
    adapter      : ModelAdapter, optional — auto-detected from model if None
    restore_magnit : deprecated compatibility argument; ignored. Gemma-2/3 always
                     preserve raw replacement magnitude across post_mlp_norm.
    """
    from ske_pre_valid.model_adapter import get_adapter
    if adapter is None:
        adapter = get_adapter(model)

    intervened_layers = list(range(start_layer, end_layer + 1))

    # ── Pass 1: collect h from the clean original model ───────────────────────
    orig_h = _collect_original_h(
        model, input_ids, intervened_layers, device, adapter,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )

    # ── Pre-select paths from original activations (fully vectorised) ─────────
    # pre_selected[layer_idx]:
    #   None                             → fast path: use h @ delta @ V in pass 2
    #   {"j_idx", "i_idx", "delta_vals"} → partial: use scatter_add in pass 2
    pre_selected: dict = {}

    all_blocks = adapter.get_blocks(model)

    layer_iter = (tqdm(intervened_layers, desc="pre-select", unit="layer", leave=False)
                  if verbose else intervened_layers)
    for layer_idx in layer_iter:
        layer_data = cache.layer_skeleton(layer_idx)
        delta = layer_data["delta"]
        V     = layer_data["V"]              # [n_sp, d_model]
        neuron_atoms = layer_data.get("delta_representation") == "identity"
        if neuron_atoms:
            d_ff = n_sp = int(V.shape[0])
            n_total = d_ff
        else:
            d_ff, n_sp = delta.shape
            n_total = d_ff * n_sp

        if k1 >= n_total:
            pre_selected[layer_idx] = None   # flag: all paths
            continue

        # γ scale for Gemma-2/3: (1 + post_feedforward_layernorm.weight)
        block      = all_blocks[layer_idx]
        post_ffn_w = gemma_post_ffn_gamma(block)

        effector_scores = None
        if pre_norm_effs is not None and layer_idx in pre_norm_effs:
            # z-score normalised effectors (can be +/-) — highest priority
            effector_scores = pre_norm_effs[layer_idx].cpu()
        elif target_id is not None and W_emb is not None:
            if post_ffn_w is not None:
                effector_scores = (V.cpu().float() * post_ffn_w) @ W_emb.cpu().float()[target_id]
            else:
                effector_scores = V @ W_emb.cpu().to(dtype=V.dtype)[target_id]

        h_orig = orig_h[layer_idx]           # [batch, seq, d_ff] cpu
        _seq_reduce = "mean" if token_mode == "all" else seq_reduce
        h_reduced = (h_orig[:, -1, :].mean(dim=0) if _seq_reduce == "last"
                     else h_orig.mean(dim=(0, 1)))          # [d_ff]

        score = h_reduced if neuron_atoms else h_reduced.unsqueeze(1) * delta
        if effector_scores is not None:
            score = score * (effector_scores if neuron_atoms
                             else effector_scores.unsqueeze(0))

        _, flat_idx = torch.topk(score.flatten(), k1)
        j_idx = flat_idx if neuron_atoms else flat_idx // n_sp
        i_idx = flat_idx if neuron_atoms else flat_idx % n_sp
        delta_values = (torch.ones(j_idx.numel(), dtype=V.dtype)
                        if neuron_atoms else delta[j_idx, i_idx])
        pre_selected[layer_idx] = {
            "j_idx":      j_idx,
            "i_idx":      i_idx,
            "delta_vals": delta_values,
        }

    # ── Pass 2: intervened forward using pre-selected paths ───────────────────
    input_ids = input_ids.to(device)
    x, position_ids = adapter.embed_with_token_types(
        model, input_ids, device, token_type_ids=token_type_ids
    )
    pos_emb = adapter.prepare_position_embeddings(model, x, position_ids)

    blocks     = adapter.get_blocks(model)
    layer_iter2 = enumerate(blocks)
    if verbose:
        layer_iter2 = tqdm(layer_iter2, total=len(blocks),
                           desc=f"ori_acv/{intervention}", unit="layer", leave=False)

    for layer_idx, block in layer_iter2:
        x, position_ids, pos_emb = move_to_block_device(block, x, position_ids, pos_emb)
        x_pre_attn = x   # snapshot before this layer's attention residual — see mlp_ln_input
        attn_out = adapter.attn_out(block, x, position_ids, pos_emb)
        x = x + attn_out

        if start_layer <= layer_idx <= end_layer:
            # mlp_ln_input is x for sequential architectures (GPT-2, Llama, Gemma, ...) and
            # x_pre_attn for parallel ones (ProGen2) — see ModelAdapter.mlp_ln_input.
            mlp_in       = adapter.mlp_ln_input(x, x_pre_attn)
            normed       = adapter.mlp_ln(block, mlp_in)
            h            = adapter.mlp_h(block, normed)
            full_down    = adapter.mlp_proj(block, h)
            full_mlp_out = adapter.post_mlp_norm(block, full_down)

            bd        = h.device                        # actual block device (multi-GPU safe)
            h_out     = h if token_mode == "all" else h[:, -1:, :]  # [b, s, d_ff]
            act_dtype = h_out.dtype

            if layer_gpu_tensors is not None and layer_idx in layer_gpu_tensors:
                lgt  = layer_gpu_tensors[layer_idx]
                V    = lgt["V"].to(dtype=act_dtype)
                bias = lgt["bias"]
                _delta_gpu = lgt["delta"]
                neuron_atoms = lgt.get("delta_representation") == "identity"
            else:
                layer_data = cache.layer_skeleton(layer_idx)
                bias = layer_data["bias"] if use_bias else None
                V    = layer_data["V"].to(device=bd, dtype=act_dtype)
                _delta_gpu = None
                neuron_atoms = layer_data.get("delta_representation") == "identity"

            paths = pre_selected[layer_idx]
            if paths is None:
                # Fast path: all paths → h @ delta @ V
                if neuron_atoms:
                    ske = h_out @ V
                elif _delta_gpu is not None:
                    delta = _delta_gpu.to(dtype=act_dtype)
                    ske = h_out @ delta @ V
                else:
                    delta = layer_data["delta"].to(device=bd, dtype=act_dtype)
                    ske = h_out @ delta @ V
            else:
                # Partial path: scatter_add — no [k, d_model] V-gather
                j_idx      = paths["j_idx"].to(bd)
                i_idx      = paths["i_idx"].to(bd)
                delta_vals = paths["delta_vals"].to(device=bd, dtype=act_dtype)
                n_sp       = V.shape[0]
                batch, s   = h_out.shape[0], h_out.shape[1]
                coeff = h_out[:, :, j_idx] * delta_vals      # [b, s, k1]
                i_exp = i_idx.view(1, 1, -1).expand(batch, s, -1)
                agg   = torch.zeros(batch, s, n_sp, dtype=act_dtype, device=bd)
                agg.scatter_add_(2, i_exp, coeff)
                ske   = agg @ V                               # [b, s, d_model]

            # bias is a layer-level constant, not a member of any (j,i) path. For
            # "preserve" it's added directly here. For "ablation" it must NOT be added —
            # full_down_ref already carries the true bias, so subtracting a bias-free
            # selected reconstruction (below) leaves bias present in the ablation result
            # too, instead of cancelling it out.
            if bias is not None and intervention != "ablation":
                ske = ske + bias.to(device=bd, dtype=act_dtype)

            full_down_ref = full_down[:, -1:, :] if token_mode == "last" else full_down
            full_down_ref_bd = full_down_ref.to(bd)

            if intervention == "ablation":
                # Subtract BEFORE post_mlp_norm, not after: post_norm(full) − post_norm(selected)
                # ≠ post_norm(full − selected) when post_norm is nonlinear (Gemma-2/3 RMSNorm).
                # The magnitude-preserving post norm below retains the complement's raw size.
                ske = full_down_ref_bd - ske

            ske = adapter.magnitude_preserving_post_mlp_norm(
                block, ske, full_down_ref_bd)

            # ske now holds the final per-condition contribution (post-norm'd) for both
            # preserve and ablation alike — assign it directly, no further subtraction needed.
            if token_mode == "all":
                mlp_out = ske
            else:
                mlp_out = full_mlp_out.clone()
                mlp_out[:, -1:, :] = ske
        else:
            mlp_out = adapter.full_mlp(block, adapter.mlp_ln_input(x, x_pre_attn))

        x = x + mlp_out

    logits = adapter.final_out(model, x)
    return {"logits": logits, "layer_paths": {}}
