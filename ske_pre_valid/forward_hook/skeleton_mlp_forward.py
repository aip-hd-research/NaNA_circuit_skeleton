import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from ske_pre_valid.within_ske_cache import WithinSkeCache
from ske_pre_valid.within_ske_selector import (
    select_and_reconstruct_skeleton,
    select_and_reconstruct_skeleton_batch,
    select_and_reconstruct_neuron_atoms,
    select_and_reconstruct_neuron_atoms_batch,
)
from ske_pre_valid.model_adapter import move_to_block_device, gemma_post_ffn_gamma
from ske_pre_valid.attention_ske_selector import (
    attention_path_trace,
    capture_attention_projection,
    reconstruct_attention_paths,
    select_attention_paths,
)


def _random_reconstruct(h, delta, V, k1, token_mode, bias, seed, device,
                        last_real_pos=None, return_trace=False,
                        diagonal_only=False):
    """Vectorised random-path reconstruction (no Python loops, no V-gather).

    Fast paths
    ----------
    k1 >= N       : h @ delta @ V (full matmul, no selection needed).
    k1 >  N/2     : complement trick — select random (N-k1) paths on GPU and
                    subtract from full matmul.  Keeps GPU randperm size small.
    k1 <= N/2     : direct GPU randperm[:k1] + scatter_add.

    GPU randperm avoids the multi-second CPU shuffle that dominated for large k1.

    last_real_pos : [b] tensor — when provided, extracts h at the correct last
                    real token per sample (needed for right-padded batches).
    """
    if diagonal_only:
        d_ff = int(V.shape[0])
        n_sp = d_ff
    else:
        d_ff, n_sp = delta.shape
    if last_real_pos is not None:
        b_range = torch.arange(h.shape[0], device=h.device)
        h_out = h[b_range, last_real_pos.to(h.device), :].unsqueeze(1)  # [b, 1, d_ff]
    else:
        h_out = h if token_mode == "all" else h[:, -1:, :]  # [b, s, d_ff]
    act_dtype = h_out.dtype
    bd        = h_out.device
    V     = V.to(device=bd, dtype=act_dtype)

    if diagonal_only:
        n_total = d_ff
        rng = torch.Generator(device=bd)
        rng.manual_seed(seed)
        if k1 >= n_total:
            trace_idx = torch.arange(n_total, device=bd)
            ske = h_out @ V
        else:
            trace_idx = torch.randperm(
                n_total, device=bd, generator=rng
            )[:max(int(k1), 0)]
            ske = h_out[:, :, trace_idx] @ V[trace_idx]
        if bias is not None:
            ske = ske + bias.to(device=bd, dtype=act_dtype)
        if not return_trace:
            return ske
        if h_out.shape[0] != 1:
            raise ValueError("random path tracing currently requires batch size 1")
        h_reduced = h_out.mean(dim=(0, 1))
        trace = {
            "sensor_indices": trace_idx.detach().to("cpu", dtype=torch.int32),
            "effector_indices": trace_idx.detach().to("cpu", dtype=torch.int32),
            "sensor_activations": h_reduced[trace_idx].detach().float().cpu(),
            "delta_values": torch.ones(trace_idx.numel(), dtype=torch.float32),
            "selection_scores": h_reduced[trace_idx].detach().float().cpu(),
        }
        return ske, trace

    delta = delta.to(device=bd, dtype=act_dtype)

    n_total = d_ff * n_sp
    if k1 >= n_total:
        ske = h_out @ delta @ V
        trace_flat_idx = torch.arange(n_total, device=bd)
    else:
        use_complement = k1 * 2 > n_total
        n_select = n_total - k1 if use_complement else k1

        rng = torch.Generator(device=bd)
        rng.manual_seed(seed)
        flat_idx   = torch.randperm(n_total, device=bd, generator=rng)[:n_select]
        j_idx      = flat_idx // n_sp
        i_idx      = flat_idx %  n_sp
        delta_vals = delta[j_idx, i_idx]

        batch, s = h_out.shape[0], h_out.shape[1]
        coeff = h_out[:, :, j_idx] * delta_vals        # [b, s, n_select]
        i_exp = i_idx.view(1, 1, -1).expand(batch, s, -1)
        agg   = torch.zeros(batch, s, n_sp, dtype=act_dtype, device=bd)
        agg.scatter_add_(2, i_exp, coeff)
        partial = agg @ V

        if use_complement:
            ske = (h_out @ delta @ V) - partial
            selected = torch.ones(n_total, dtype=torch.bool, device=bd)
            selected[flat_idx] = False
            trace_flat_idx = torch.nonzero(selected, as_tuple=False).flatten()
        else:
            ske = partial
            trace_flat_idx = flat_idx

    if bias is not None:
        ske = ske + bias.to(device=bd, dtype=act_dtype)
    if not return_trace:
        return ske
    if h_out.shape[0] != 1:
        raise ValueError("random path tracing currently requires batch size 1")
    h_reduced = h_out.mean(dim=(0, 1))
    j_idx, i_idx = trace_flat_idx // n_sp, trace_flat_idx % n_sp
    score = h_reduced[j_idx] * delta[j_idx, i_idx]
    trace = {
        "sensor_indices": j_idx.detach().to("cpu", dtype=torch.int32),
        "effector_indices": i_idx.detach().to("cpu", dtype=torch.int32),
        "sensor_activations": h_reduced[j_idx].detach().float().cpu(),
        "delta_values": delta[j_idx, i_idx].detach().float().cpu(),
        "selection_scores": score.detach().float().cpu(),
    }
    return ske, trace


def _compute_norm_effs(cache, W_emb, target_id, start_layer, end_layer,
                       device, adapter, model):
    """
    Per-layer z-score normalise effectors on `device`.

    Each layer's effectors are normalised by their own mean and std, consistent
    with the static skeleton's norm_eff behaviour in run_static_experiment.

    The underlying effector is the gamma-corrected raw-dot alignment.
    """
    blocks = adapter.get_blocks(model)
    norm_effs = {}
    for li in range(start_layer, end_layer + 1):
        gamma = gemma_post_ffn_gamma(blocks[li], device=device)
        V_li = cache.layer_skeleton(li)["V"].to(device)
        W_f  = W_emb.to(dtype=V_li.dtype)
        if gamma is not None:
            V_li = V_li * gamma.to(dtype=V_li.dtype)
        raw = V_li @ W_f[target_id]
        norm_effs[li] = (raw - raw.mean()) / raw.std().clamp(min=1e-8)
    return norm_effs


@torch.no_grad()
def skeleton_forward(
    model,
    input_ids: torch.Tensor,
    cache: WithinSkeCache,
    start_layer: int,
    end_layer: int,
    k1: int,
    seq_reduce: str = "last",
    device: str = "cpu",
    selection: str = "topk",        # "topk" | "random"
    intervention: str = "preserve", # "preserve" | "ablation"
    seed: int = 0,
    use_bias: bool = True,
    target_id: int | None = None,
    W_emb: torch.Tensor | None = None,
    token_mode: str = "last",       # "last" | "all"
    norm_eff: bool = False,
    precomputed_norm_effs: dict | None = None,
    verbose: bool = False,
    adapter=None,
    restore_magnit: bool = False,
    layer_gpu_tensors: dict | None = None,
    per_sample_effectors: dict | None = None,  # {layer_idx: Tensor[b, n_sp]}
    last_real_pos: torch.Tensor | None = None, # [b] 0-indexed last real token (right-pad batches)
    token_type_ids: torch.Tensor | None = None,
    capture_path_trace: bool = False,
    attention_k1: int | None = None,
    attention_selection: str | None = None,
    attention_intervention: str | None = None,
) -> dict:
    """
    Parameters
    ----------
    selection      : "topk"      — select by h[j]*delta[j,i]*effector[i]
                     "random"    — sample k1 paths uniformly at random
    intervention   : "preserve"  — MLP output = skeleton paths only
                     "ablation"  — MLP output = full MLP − skeleton paths
    restore_magnit : deprecated compatibility argument; ignored. Gemma-2/3 always
                     preserve raw replacement magnitude across post_mlp_norm.
    precomputed_norm_effs : pre-normalised effector dict (shared across calls to avoid recompute)
    adapter      : ModelAdapter, optional — auto-detected from model if None
    attention_k1 : when set, also select/intervene on per-head OV paths before
                   the MLP. The attention output projection is decomposed per head.
    """
    from ske_pre_valid.model_adapter import get_adapter
    if adapter is None:
        adapter = get_adapter(model)

    input_ids = input_ids.to(device)
    x, position_ids = adapter.embed_with_token_types(
        model, input_ids, device, token_type_ids=token_type_ids
    )
    pos_emb = adapter.prepare_position_embeddings(model, x, position_ids)

    # Pre-pass: compute globally normalised effectors (once, shared across preserve/ablation).
    pre_norm_effs: dict = {}
    if precomputed_norm_effs is not None:
        pre_norm_effs = precomputed_norm_effs
    elif norm_eff and selection == "topk" and target_id is not None and W_emb is not None:
        pre_norm_effs = _compute_norm_effs(
            cache, W_emb.to(device), target_id,
            start_layer, end_layer, device, adapter, model,
        )

    layer_paths: dict = {}
    attention_layer_paths: dict = {}
    blocks     = adapter.get_blocks(model)

    layer_iter = enumerate(blocks)
    if verbose:
        layer_iter = tqdm(layer_iter, total=len(blocks),
                          desc=f"{selection}/{intervention}", unit="layer", leave=False)

    for layer_idx, block in layer_iter:
        x, position_ids, pos_emb = move_to_block_device(block, x, position_ids, pos_emb)
        x_pre_attn = x   # snapshot before this layer's attention residual — see mlp_ln_input
        intervene_attention = (
            attention_k1 is not None and start_layer <= layer_idx <= end_layer
        )
        if intervene_attention:
            native_attn, full_attn_raw, attn_values, attn_skeleton, attn_bias = (
                capture_attention_projection(
                    adapter, model, layer_idx, block, x, position_ids, pos_emb
                )
            )
            attn_sel = attention_selection or selection
            attn_itv = attention_intervention or intervention
            target_direction = None
            if target_id is not None and W_emb is not None:
                target_direction = W_emb[target_id]
            attn_mask = select_attention_paths(
                attn_values,
                attn_skeleton,
                attention_k1,
                target_direction,
                selection=attn_sel,
                token_mode=token_mode,
                last_real_pos=last_real_pos,
                seed=seed + 10000 + layer_idx,
            )
            selected_attn_raw = reconstruct_attention_paths(
                attn_values,
                attn_skeleton,
                attn_mask,
                token_mode=token_mode,
                last_real_pos=last_real_pos,
            ).to(device=full_attn_raw.device, dtype=full_attn_raw.dtype)
            if attn_itv == "preserve":
                replacement_raw = selected_attn_raw
                if use_bias and attn_bias is not None:
                    replacement_raw = replacement_raw + attn_bias.to(
                        device=replacement_raw.device, dtype=replacement_raw.dtype
                    )
            elif attn_itv == "ablation":
                if token_mode == "all":
                    full_ref = full_attn_raw
                elif last_real_pos is not None:
                    br = torch.arange(full_attn_raw.shape[0], device=full_attn_raw.device)
                    full_ref = full_attn_raw[
                        br, last_real_pos.to(full_attn_raw.device)
                    ].unsqueeze(1)
                else:
                    full_ref = full_attn_raw[:, -1:]
                replacement_raw = full_ref - selected_attn_raw
            else:
                raise ValueError(f"unknown attention intervention {attn_itv!r}")

            if token_mode == "all":
                full_raw_ref = full_attn_raw
            elif last_real_pos is not None:
                br = torch.arange(full_attn_raw.shape[0], device=full_attn_raw.device)
                full_raw_ref = full_attn_raw[
                    br, last_real_pos.to(full_attn_raw.device)
                ].unsqueeze(1)
            else:
                full_raw_ref = full_attn_raw[:, -1:]
            replacement = adapter.magnitude_preserving_post_attn_norm(
                block, replacement_raw, full_raw_ref
            )
            if token_mode == "all":
                attn_out = replacement
            else:
                attn_out = native_attn.clone()
                if last_real_pos is not None:
                    br = torch.arange(attn_out.shape[0], device=attn_out.device)
                    attn_out[br, last_real_pos.to(attn_out.device)] = replacement.squeeze(1)
                else:
                    attn_out[:, -1:] = replacement
            attention_layer_paths[layer_idx] = (
                attention_path_trace(attn_values, attn_skeleton, attn_mask, target_direction)
                if capture_path_trace else None
            )
        else:
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

            bd = h.device                          # actual block device (multi-GPU safe)

            if layer_gpu_tensors is not None and layer_idx in layer_gpu_tensors:
                # Fast path: tensors already on GPU — no CPU→GPU transfer
                lgt   = layer_gpu_tensors[layer_idx]
                delta = lgt["delta"]
                V     = lgt["V"]
                bias  = lgt["bias"]
                effector_scores = lgt["effector"] if selection == "topk" else None
                neuron_atoms = lgt.get("delta_representation") == "identity"
            else:
                layer_data = cache.layer_skeleton(layer_idx)
                bias  = layer_data["bias"] if use_bias else None
                delta = layer_data["delta"].to(bd)
                V     = layer_data["V"].to(bd)
                neuron_atoms = layer_data.get("delta_representation") == "identity"

                effector_scores = None
                if selection == "topk" and target_id is not None and W_emb is not None:
                    if layer_idx in pre_norm_effs:
                        effector_scores = pre_norm_effs[layer_idx].to(bd)
                    else:
                        # γ correction for post_feedforward_layernorm (Gemma-2/3).
                        post_ffn_gamma = gemma_post_ffn_gamma(block)

                        V_cpu = V.cpu().float()
                        if post_ffn_gamma is not None:
                            V_cpu = V_cpu * post_ffn_gamma
                        W_cpu = W_emb.cpu().float()

                        effector_scores = (V_cpu @ W_cpu[target_id]).to(device=bd)

            _seq_reduce  = "mean" if token_mode == "all" else seq_reduce

            # bias is a layer-level constant, not a member of any (j,i) path. For
            # "preserve" it's added directly into the reconstruction below. For
            # "ablation" it must NOT be added here — full_mlp_out already carries the
            # true bias, so full_mlp_out − selected(no bias) leaves bias present in the
            # ablation result too, instead of cancelling it out.
            recon_bias = None if intervention == "ablation" else bias

            # Batch mode: per-sample topk using pre-computed per-sample effectors
            if (per_sample_effectors is not None
                    and layer_idx in per_sample_effectors
                    and selection == "topk"):
                if neuron_atoms:
                    ske = select_and_reconstruct_neuron_atoms_batch(
                        h, V, k1, per_sample_effectors[layer_idx], recon_bias,
                        last_real_pos=last_real_pos,
                    )
                else:
                    ske = select_and_reconstruct_skeleton_batch(
                        h, delta, V, k1,
                        per_sample_effectors[layer_idx],
                        recon_bias,
                        last_real_pos=last_real_pos,
                    )
            elif selection == "random":
                selected = _random_reconstruct(
                    h, delta, V, k1, token_mode, recon_bias,
                    seed=seed + layer_idx, device=bd,
                    last_real_pos=last_real_pos,
                    return_trace=capture_path_trace,
                    diagonal_only=neuron_atoms,
                )
                if capture_path_trace:
                    ske, layer_paths[layer_idx] = selected
                else:
                    ske = selected
            else:
                if neuron_atoms:
                    selected = select_and_reconstruct_neuron_atoms(
                        h, V, k1, _seq_reduce, effector_scores, recon_bias,
                        token_mode, return_trace=capture_path_trace,
                    )
                else:
                    selected = select_and_reconstruct_skeleton(
                        h, delta, V, k1, _seq_reduce, effector_scores, recon_bias,
                        token_mode, return_trace=capture_path_trace,
                    )
                if capture_path_trace:
                    ske, layer_paths[layer_idx] = selected
                else:
                    ske = selected
            if not capture_path_trace:
                layer_paths[layer_idx] = None

            if last_real_pos is not None:
                _br   = torch.arange(full_down.shape[0], device=full_down.device)
                _last = last_real_pos.to(full_down.device)
                full_down_ref = full_down[_br, _last, :].unsqueeze(1)
            else:
                full_down_ref = full_down[:, -1:, :] if token_mode == "last" else full_down
            full_down_ref_bd = full_down_ref.to(bd)

            if intervention == "ablation":
                # Subtract BEFORE post_mlp_norm, not after: post_norm(full) − post_norm(selected)
                # ≠ post_norm(full − selected) when post_norm is nonlinear (Gemma-2/3 RMSNorm).
                # Bias was not added to ske above (recon_bias), so full_down_ref already carries
                # the true bias and this subtraction leaves it present in the ablation result.
                # The magnitude-preserving post norm below retains the complement's raw size.
                ske = full_down_ref_bd - ske

            ske = adapter.magnitude_preserving_post_mlp_norm(
                block, ske, full_down_ref_bd)

            # ske now holds the final per-condition contribution (post-norm'd) for both
            # preserve and ablation alike — assign it directly, no further subtraction needed.
            if last_real_pos is not None:
                # Batch mode with right-padding: apply intervention at each sample's last real position
                _br   = torch.arange(full_mlp_out.shape[0], device=full_mlp_out.device)
                _last = last_real_pos.to(full_mlp_out.device)
                mlp_out = full_mlp_out.clone()
                mlp_out[_br, _last, :] = ske.squeeze(1)
            elif token_mode == "all":
                mlp_out = ske
            else:
                mlp_out = full_mlp_out.clone()
                mlp_out[:, -1:, :] = ske

        else:
            mlp_out = adapter.full_mlp(block, adapter.mlp_ln_input(x, x_pre_attn))

        x = x + mlp_out

    logits = adapter.final_out(model, x)
    return {
        "logits": logits,
        "layer_paths": layer_paths,
        "attention_layer_paths": attention_layer_paths,
    }
