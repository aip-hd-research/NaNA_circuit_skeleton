"""
base.py  —  shared static-skeleton forward passes (preserve / ablation).
Used by both forward_per_neuron and forward_global_neuron.

pre_selected[layer_idx] is a tensor dict with keys:
    j_idx         LongTensor[k]   neuron indices
    i_idx         LongTensor[k]   subspace indices
    delta_vals    FloatTensor[k]  delta[j,i] values
    static_scores FloatTensor[k]  delta[j,i]*effector[i] scores (used for sign-gating)

Reconstruction uses scatter_add to avoid materialising the large [k, d_model] gather
(k can be millions, making V[i_idx,:] OOM on GPU).

    coeff[t]  = h[0,-1, j[t]] * delta[t]
    agg[i]    = Σ_{t: i[t]==i}  coeff[t]      (scatter_add, shape [n_sp])
    ske_last  = agg @ V                         ([d_model])

Both functions accept an optional `adapter` parameter (auto-detected if None).
"""

import torch

from ske_pre_valid.model_adapter import move_to_block_device


@torch.no_grad()
def forward_static_preserve(model, input_ids, cache, pre_selected,
                             start_layer, end_layer, device, use_bias=True,
                             anti_align=False, use_gate=True, adapter=None,
                             last_pos=None, restore_magnit=False,
                             token_type_ids=None):
    """Preserve mode: replace last-token MLP output with (optionally sign-gated) skeleton.

    use_gate=True  (default): keep paths where h[j] * static_score > 0  (aligned)
                              or < 0 when anti_align=True
    use_gate=False           : keep all pre-selected paths (no direction filtering)
    adapter  : ModelAdapter, optional — auto-detected from model if None
    last_pos : LongTensor [B], optional — index of last real token per sample.
               Required for batched right-padded input (B > 1).
               When None, uses position -1 (single-sample or left-padded input).
    restore_magnit : deprecated compatibility argument; ignored. Gemma-2/3 always
                     preserve the replacement's raw magnitude across post-MLP RMSNorm.
    """
    from ske_pre_valid.model_adapter import get_adapter
    if adapter is None:
        adapter = get_adapter(model)

    input_ids = input_ids.to(device)
    x, position_ids = adapter.embed_with_token_types(
        model, input_ids, device, token_type_ids=token_type_ids
    )
    pos_emb = adapter.prepare_position_embeddings(model, x, position_ids)

    for layer_idx, block in enumerate(adapter.get_blocks(model)):
        x, position_ids, pos_emb = move_to_block_device(block, x, position_ids, pos_emb)
        x_pre_attn = x   # snapshot before this layer's attention residual — see mlp_ln_input
        attn_out = adapter.attn_out(block, x, position_ids, pos_emb)
        x = x + attn_out

        if start_layer <= layer_idx <= end_layer:
            ld   = cache.layer_skeleton(layer_idx)
            bias = ld["bias"] if use_bias else None

            # mlp_ln_input is x for sequential architectures (GPT-2, Llama, Gemma, ...) and
            # x_pre_attn for parallel ones (ProGen2) — see ModelAdapter.mlp_ln_input.
            mlp_in       = adapter.mlp_ln_input(x, x_pre_attn)
            # post_mlp_norm is identity for most models and the native
            # post_feedforward_layernorm for Gemma-2/3.
            normed       = adapter.mlp_ln(block, mlp_in)
            h            = adapter.mlp_h(block, normed)
            full_down    = adapter.mlp_proj(block, h)
            full_mlp_out = adapter.post_mlp_norm(block, full_down)

            ld_dev = h.device
            V    = ld["V"].to(ld_dev)            # [n_sp, d_model]
            paths         = pre_selected[layer_idx]
            j_idx         = paths["j_idx"].to(ld_dev)
            i_idx         = paths["i_idx"].to(ld_dev)
            delta_vals    = paths["delta_vals"].to(ld_dev)
            static_scores = paths["static_scores"].to(ld_dev)

            act_dtype = h.dtype

            if last_pos is None:
                # ── single-sample path (original) ─────────────────────────────
                # Sign-gate: keep paths where h[j]*static_score has the desired sign
                if use_gate:
                    gate = h[0, -1, j_idx] * static_scores.to(dtype=act_dtype)   # [k]
                    mask = (gate < 0) if anti_align else (gate > 0)
                    i_sel = i_idx[mask]
                    coeff = h[0, -1, j_idx[mask]] * delta_vals[mask].to(dtype=act_dtype)
                else:
                    i_sel = i_idx
                    coeff = h[0, -1, j_idx] * delta_vals.to(dtype=act_dtype)

                # scatter_add with an empty i_sel/coeff (no pre-selected paths, or every
                # candidate gated out) is a valid no-op — agg stays all-zero, so preserve
                # reconstructs a genuine zero skeleton (+ bias) here instead of silently
                # falling back to the untouched full MLP output.
                n_sp = V.shape[0]
                agg  = torch.zeros(n_sp, dtype=act_dtype, device=ld_dev)
                agg.scatter_add_(0, i_sel, coeff)      # [n_sp]
                ske_raw = (agg @ V.to(dtype=act_dtype)).view(1, 1, -1)   # [1, 1, d_model]
                if bias is not None:
                    ske_raw = ske_raw + bias.to(device=ld_dev, dtype=act_dtype)
                full_down_ref = full_down[:, -1:, :].to(ld_dev)
                ske_last = adapter.magnitude_preserving_post_mlp_norm(
                    block, ske_raw, full_down_ref)
                mlp_out            = full_mlp_out.clone()
                mlp_out[:, -1:, :] = ske_last

            else:
                # ── batched path (right-padded, last_pos per sample) ──────────
                B   = h.shape[0]
                lp  = last_pos.to(ld_dev)                       # [B]
                b_i = torch.arange(B, device=ld_dev)

                h_last = h[b_i, lp, :]                          # [B, d_ff]

                if use_gate:
                    gate  = h_last[:, j_idx] * static_scores.to(dtype=act_dtype)  # [B, k]
                    mask  = (gate < 0) if anti_align else (gate > 0)               # [B, k]
                    coeff = h_last[:, j_idx] * delta_vals.to(dtype=act_dtype)      # [B, k]
                    coeff = coeff * mask
                else:
                    coeff = h_last[:, j_idx] * delta_vals.to(dtype=act_dtype)

                n_sp = V.shape[0]
                agg  = torch.zeros(B, n_sp, dtype=act_dtype, device=ld_dev)
                agg.scatter_add_(1, i_idx.unsqueeze(0).expand(B, -1), coeff)       # [B, n_sp]

                ske_raw      = (agg @ V.to(dtype=act_dtype)).unsqueeze(1)          # [B, 1, d_model]
                if bias is not None:
                    ske_raw  = ske_raw + bias.to(device=ld_dev, dtype=act_dtype)
                full_down_ref = full_down[b_i, lp, :].unsqueeze(1)
                ske_last = adapter.magnitude_preserving_post_mlp_norm(
                    block, ske_raw, full_down_ref)                                 # [B, 1, d_model]

                # Write ske_last back for every sample (not just ones with a surviving
                # gated path) — samples where every candidate was gated out already
                # reconstruct an all-zero skeleton via the zeroed-out coeff above, so this
                # stays a genuine preserve of "the selected/gated paths" (zero of them)
                # instead of silently keeping that sample's untouched full MLP output.
                mlp_out = full_mlp_out.clone()
                mlp_out[b_i, lp, :] = ske_last[:, 0, :]
        else:
            mlp_out = adapter.full_mlp(block, adapter.mlp_ln_input(x, x_pre_attn))

        x = x + mlp_out

    return adapter.final_out(model, x)


@torch.no_grad()
def forward_static_ablation(model, input_ids, cache, pre_selected,
                             start_layer, end_layer, device, use_bias=True,
                             use_gate=False, adapter=None, last_pos=None,
                             token_type_ids=None):
    """Ablation mode: replace last-token MLP output with the COMPLEMENT skeleton.

    The complement skeleton = all paths NOT in pre_selected.

    use_gate=False (default): complement skeleton, all paths
                              replacement_raw = full_down − selected_raw
                              Subtraction happens BEFORE post_mlp_norm, not after: for a
                              nonlinear post_norm (Gemma-2/3 RMSNorm), post_norm(full) −
                              post_norm(selected) ≠ post_norm(full − selected). Gemma then
                              norm-matches replacement_raw before RMSNorm and restores its
                              raw replacement/full norm ratio afterward.
    use_gate=True            : complement skeleton, gate-positive paths only
                              mlp_out = post_norm(Σ_all h·δ·eff>0 − Σ_selected h·δ·eff>0)
                              Requires paths["effector"] [n_sp] in pre_selected dicts. This
                              set isn't derivable as "full minus selected" (it excludes
                              gate-negative contributions from both), so it's reconstructed
                              directly rather than by subtracting from full_mlp_out.

    bias handling: bias is a layer-level constant, not a member of any (j,i) path, so it is
    present in both preserve and ablation output (not just preserve):
      use_gate=False — bias is deliberately NOT added to the subtracted "selected" term;
                       full_down already carries it, so full_down − selected(no bias),
                       normalised once, leaves bias in the ablation result.
      use_gate=True  — this branch reconstructs the gate-positive complement directly (it is
                       not a subtraction from full_mlp_out), so bias IS added explicitly here.
    adapter  : ModelAdapter, optional — auto-detected from model if None
    last_pos : LongTensor [B], optional — index of last real token per sample.
               Required for batched right-padded input (B > 1).
               When None, uses position -1 (single-sample or left-padded input).
    """
    from ske_pre_valid.model_adapter import get_adapter
    if adapter is None:
        adapter = get_adapter(model)

    input_ids = input_ids.to(device)
    x, position_ids = adapter.embed_with_token_types(
        model, input_ids, device, token_type_ids=token_type_ids
    )
    pos_emb = adapter.prepare_position_embeddings(model, x, position_ids)

    for layer_idx, block in enumerate(adapter.get_blocks(model)):
        x, position_ids, pos_emb = move_to_block_device(block, x, position_ids, pos_emb)
        x_pre_attn = x   # snapshot before this layer's attention residual — see mlp_ln_input
        attn_out = adapter.attn_out(block, x, position_ids, pos_emb)
        x = x + attn_out

        if start_layer <= layer_idx <= end_layer:
            ld   = cache.layer_skeleton(layer_idx)
            bias = ld["bias"] if use_bias else None

            # mlp_ln_input is x for sequential architectures (GPT-2, Llama, Gemma, ...) and
            # x_pre_attn for parallel ones (ProGen2) — see ModelAdapter.mlp_ln_input.
            mlp_in       = adapter.mlp_ln_input(x, x_pre_attn)
            normed       = adapter.mlp_ln(block, mlp_in)
            h            = adapter.mlp_h(block, normed)
            full_down    = adapter.mlp_proj(block, h)
            full_mlp_out = adapter.post_mlp_norm(block, full_down)

            ld_dev = h.device
            V    = ld["V"].to(ld_dev)            # [n_sp, d_model]
            paths      = pre_selected[layer_idx]
            j_idx      = paths["j_idx"].to(ld_dev)
            i_idx      = paths["i_idx"].to(ld_dev)
            delta_vals = paths["delta_vals"].to(ld_dev)

            if j_idx.numel() == 0:
                mlp_out = full_mlp_out

            elif last_pos is None:
                # ── single-sample path (original) ─────────────────────────────
                act_dtype = h.dtype
                n_sp      = V.shape[0]
                V_dev     = V.to(dtype=act_dtype, device=ld_dev)

                if use_gate:
                    # Complement skeleton, gate-positive paths:
                    #   agg_complement_gate[i] = (Σ_{all j} h[j]·δ[j,i] if h·δ·eff>0)
                    #                          − (Σ_{selected j} h[j]·δ[j,i] if h·δ·eff>0)
                    effector   = paths["effector"].to(device=ld_dev, dtype=act_dtype)  # [n_sp]
                    delta_full = ld["delta"].to(device=ld_dev, dtype=act_dtype)        # [d_ff, n_sp]
                    h_last     = h[0, -1, :].to(dtype=act_dtype)                       # [d_ff]

                    # Full gate-positive aggregate over all (j,i)
                    coeff_mat  = h_last.unsqueeze(1) * delta_full                      # [d_ff, n_sp]
                    pos_mask   = (coeff_mat * effector.unsqueeze(0)) > 0               # [d_ff, n_sp]
                    agg_full   = (coeff_mat * pos_mask.to(act_dtype)).sum(0)           # [n_sp]

                    # Selected gate-positive aggregate
                    static_scores = paths["static_scores"].to(ld_dev)
                    gate_sel  = h[0, -1, j_idx] * static_scores.to(dtype=act_dtype)
                    mask_sel  = gate_sel > 0
                    agg_sel   = torch.zeros(n_sp, dtype=act_dtype, device=ld_dev)
                    if mask_sel.any():
                        coeff_sel = h[0, -1, j_idx[mask_sel]] * delta_vals[mask_sel].to(dtype=act_dtype)
                        agg_sel.scatter_add_(0, i_idx[mask_sel], coeff_sel)

                    ske_raw = ((agg_full - agg_sel) @ V_dev).view(1, 1, -1)
                    if bias is not None:
                        ske_raw = ske_raw + bias.to(device=ld_dev, dtype=act_dtype)
                    full_down_ref = full_down[:, -1:, :].to(ld_dev)
                    ske_last = adapter.magnitude_preserving_post_mlp_norm(
                        block, ske_raw, full_down_ref)
                    mlp_out  = full_mlp_out.clone()
                    mlp_out[:, -1:, :] = ske_last

                else:
                    # Complement skeleton, all paths (no gate). Subtract in raw space before
                    # the magnitude-preserving post norm; post_norm(full) − post_norm(selected)
                    # is not a valid decomposition when post_norm is nonlinear.
                    # bias is a layer-level constant, not a member of any (j,i) path — it's
                    # not part of what's being "selected" here, so it must NOT be subtracted
                    # out. Deliberately NOT added to ske_raw: full_down already carries the
                    # true bias, so full_down − selected(no bias) leaves bias in the
                    # ablation result too, exactly as it's present in preserve's own output.
                    coeff = h[0, -1, j_idx] * delta_vals.to(dtype=act_dtype)
                    agg   = torch.zeros(n_sp, dtype=act_dtype, device=ld_dev)
                    agg.scatter_add_(0, i_idx, coeff)
                    ske_raw      = (agg @ V_dev).view(1, 1, -1)
                    full_down_ref = full_down[:, -1:, :].to(ld_dev)
                    complement_raw = full_down_ref - ske_raw
                    ske_last = adapter.magnitude_preserving_post_mlp_norm(
                        block, complement_raw, full_down_ref)
                    mlp_out  = full_mlp_out.clone()
                    mlp_out[:, -1:, :] = ske_last

            else:
                # ── batched path (right-padded, last_pos per sample) ──────────
                act_dtype = h.dtype
                n_sp      = V.shape[0]
                V_dev     = V.to(dtype=act_dtype, device=ld_dev)
                B    = h.shape[0]
                lp   = last_pos.to(ld_dev)                        # [B]
                b_i  = torch.arange(B, device=ld_dev)
                h_last         = h[b_i, lp, :]                    # [B, d_ff]
                full_down_last = full_down[b_i, lp, :]             # [B, d_model]

                if use_gate:
                    # Same complement-gate formula as the single-sample path, but the
                    # dense [d_ff, n_sp] intermediate (agg_full) is O(GBs) if batched
                    # naively for large models, so it's looped over B here — still a
                    # tiny fraction of cost vs. re-running the whole model per sample,
                    # since embedding/attention/all other layers stay batched above.
                    effector   = paths["effector"].to(device=ld_dev, dtype=act_dtype)  # [n_sp]
                    delta_full = ld["delta"].to(device=ld_dev, dtype=act_dtype)        # [d_ff, n_sp]
                    agg_full = torch.zeros(B, n_sp, dtype=act_dtype, device=ld_dev)
                    for b in range(B):
                        coeff_mat_b = h_last[b].unsqueeze(1) * delta_full               # [d_ff, n_sp]
                        pos_mask_b  = (coeff_mat_b * effector.unsqueeze(0)) > 0
                        agg_full[b] = (coeff_mat_b * pos_mask_b.to(act_dtype)).sum(0)

                    static_scores = paths["static_scores"].to(ld_dev)
                    gate_sel  = h_last[:, j_idx] * static_scores.to(dtype=act_dtype)    # [B, k]
                    mask_sel  = gate_sel > 0
                    coeff_sel = h_last[:, j_idx] * delta_vals.to(dtype=act_dtype) * mask_sel
                    agg_sel   = torch.zeros(B, n_sp, dtype=act_dtype, device=ld_dev)
                    agg_sel.scatter_add_(1, i_idx.unsqueeze(0).expand(B, -1), coeff_sel)

                    ske_raw = (agg_full - agg_sel) @ V_dev                              # [B, d_model]
                    if bias is not None:
                        ske_raw = ske_raw + bias.to(device=ld_dev, dtype=act_dtype)
                    ske_raw  = ske_raw.unsqueeze(1)                                      # [B, 1, d_model]
                    full_down_ref = full_down_last.unsqueeze(1)
                    ske_last = adapter.magnitude_preserving_post_mlp_norm(
                        block, ske_raw, full_down_ref)
                    mlp_out  = full_mlp_out.clone()
                    mlp_out[b_i, lp, :] = ske_last[:, 0, :]

                else:
                    # Subtract in raw space before the magnitude-preserving post norm; see the
                    # single-sample branch above for why bias is not added to ske_raw.
                    coeff = h_last[:, j_idx] * delta_vals.to(dtype=act_dtype)           # [B, k]
                    agg   = torch.zeros(B, n_sp, dtype=act_dtype, device=ld_dev)
                    agg.scatter_add_(1, i_idx.unsqueeze(0).expand(B, -1), coeff)        # [B, n_sp]
                    ske_raw       = (agg @ V_dev).unsqueeze(1)                          # [B, 1, d_model]
                    full_down_ref = full_down_last.unsqueeze(1)
                    complement_raw = full_down_ref - ske_raw
                    ske_last = adapter.magnitude_preserving_post_mlp_norm(
                        block, complement_raw, full_down_ref)
                    mlp_out  = full_mlp_out.clone()
                    mlp_out[b_i, lp, :] = ske_last[:, 0, :]
        else:
            mlp_out = adapter.full_mlp(block, adapter.mlp_ln_input(x, x_pre_attn))

        x = x + mlp_out

    return adapter.final_out(model, x)
