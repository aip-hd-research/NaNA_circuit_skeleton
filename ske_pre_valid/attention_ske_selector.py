"""Dynamic selection and intervention for per-head attention OV paths."""

from __future__ import annotations

import torch

from circuit_skeleton.attention_skeleton import (
    AttentionSkeleton,
    get_attention_output_projection,
)


def capture_attention_projection(adapter, model, layer_idx, block, x, position_ids, pos_emb):
    """Run native attention and capture W_O input/output without changing routing."""
    projection, layout, heads, head_dim = get_attention_output_projection(model, layer_idx)
    captured = {}

    def pre_hook(_module, inputs):
        captured["values"] = inputs[0]

    def post_hook(_module, _inputs, output):
        captured["raw_output"] = output

    h1 = projection.register_forward_pre_hook(pre_hook)
    h2 = projection.register_forward_hook(post_hook)
    try:
        native_output = adapter.attn_out(block, x, position_ids, pos_emb)
    finally:
        h1.remove()
        h2.remove()
    if "values" not in captured or "raw_output" not in captured:
        raise RuntimeError("attention output projection hooks did not fire")

    values = captured["values"]
    if values.shape[-1] != heads * head_dim:
        raise ValueError(
            f"captured W_O input width {values.shape[-1]} != {heads}*{head_dim}"
        )
    values = values.reshape(*values.shape[:-1], heads, head_dim)
    skeleton = getattr(projection, "_attention_ov_skeleton", None)
    if skeleton is None:
        skeleton = AttentionSkeleton.from_output_weight(
            projection.weight,
            heads,
            head_dim,
            linear_layout=layout,
            compute_dtype=torch.float32,
        )
        # Model weights are frozen during evaluation; cache the decomposition so
        # dataset sweeps do not repeat an SVD for every sample/condition.
        projection._attention_ov_skeleton = skeleton
    return native_output, captured["raw_output"], values, skeleton, projection.bias


def _selection_values(values, token_mode, last_real_pos=None):
    if last_real_pos is not None:
        batch = torch.arange(values.shape[0], device=values.device)
        return values[batch, last_real_pos.to(values.device)]
    if token_mode == "all":
        return values.mean(dim=1)
    return values[:, -1]


def _random_mask(scores, k, seed):
    flat = scores.flatten(start_dim=-3)
    k = max(0, min(int(k), flat.shape[-1]))
    mask = torch.zeros_like(flat)
    if not k:
        return mask.reshape_as(scores)
    generator = torch.Generator(device=flat.device)
    generator.manual_seed(seed)
    # A separate permutation per batch item keeps the path budget per sample.
    for row in mask.reshape(-1, mask.shape[-1]):
        idx = torch.randperm(row.numel(), generator=generator, device=row.device)[:k]
        row[idx] = 1
    return mask.reshape_as(scores)


def select_attention_paths(
    values,
    skeleton,
    k,
    target_direction=None,
    *,
    selection="topk",
    token_mode="last",
    last_real_pos=None,
    seed=0,
):
    """Return one ``[batch, heads, head_dim, rank]`` path mask per sample."""
    selected_values = _selection_values(values, token_mode, last_real_pos)
    scores = skeleton.dynamic_scores(selected_values, target_direction)
    if selection == "topk":
        return skeleton.topk_mask(selected_values, k, target_direction)
    if selection == "random":
        return _random_mask(scores, k, seed)
    raise ValueError(f"unknown attention path selection {selection!r}")


def reconstruct_attention_paths(
    values, skeleton, mask, *, token_mode="last", last_real_pos=None
):
    """Reconstruct selected W_O paths using one fixed mask per batch sample."""
    if token_mode == "all":
        return skeleton.reconstruct(values, mask.unsqueeze(1))
    if last_real_pos is not None:
        batch = torch.arange(values.shape[0], device=values.device)
        selected = values[batch, last_real_pos.to(values.device)].unsqueeze(1)
    else:
        selected = values[:, -1:]
    return skeleton.reconstruct(selected, mask.unsqueeze(1))


def attention_path_trace(values, skeleton, mask, target_direction=None):
    """Return compact CPU tensors describing selected paths (batch size one)."""
    if values.shape[0] != 1 or mask.shape[0] != 1:
        raise ValueError("attention path tracing currently requires batch size 1")
    score = skeleton.dynamic_scores(values[0, -1], target_direction)
    indices = torch.nonzero(mask[0], as_tuple=False)
    h, channel, subspace = indices.unbind(dim=1)
    return {
        "head_indices": h.detach().cpu().to(torch.int16),
        "value_channel_indices": channel.detach().cpu().to(torch.int16),
        "effector_indices": subspace.detach().cpu().to(torch.int16),
        "value_activations": values[0, -1, h, channel].detach().float().cpu(),
        "delta_values": skeleton.delta[h, channel, subspace].detach().float().cpu(),
        "selection_scores": score[h, channel, subspace].detach().float().cpu(),
    }
