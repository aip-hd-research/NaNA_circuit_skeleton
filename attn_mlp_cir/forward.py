"""Norm-correct sequential/parallel Attention+MLP circuit forward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .architecture import describe_model
from .attention_ov import AttentionOVSkeleton
from .audit import assert_audited_for_circuits
from .capture import NativeAttentionCapture
from .mlp import MLPSkeleton
from .norms import BranchNorms
from .types import CircuitMode, PathSelection


@dataclass(frozen=True)
class CircuitForwardConfig:
    start_layer: int
    end_layer: int
    mlp_k: int
    attention_k: int
    mode: CircuitMode = CircuitMode.ATTENTION_MLP
    selection: str = "topk"       # topk | random
    intervention: str = "preserve"  # preserve | ablation
    attention_budget: PathSelection = PathSelection.GLOBAL
    post_norm_mode: str = "magnitude_preserving"  # native | magnitude_preserving
    seed: int = 42


def _random_mask_like(scores, k, seed, mode=PathSelection.GLOBAL):
    # Aggregated Attention scores end in [head, OV-mode].
    working = scores
    start_dim = -2 if mode == PathSelection.GLOBAL else -1
    flat = working.flatten(start_dim=start_dim)
    k = min(max(int(k), 0), flat.shape[-1])
    result = torch.zeros_like(flat)
    generator = torch.Generator(device=flat.device).manual_seed(seed)
    for row in result.reshape(-1, result.shape[-1]):
        if k:
            row[torch.randperm(row.numel(), generator=generator, device=row.device)[:k]] = 1
    return result.reshape_as(working)


def _mlp_mask(scores, k, selection, seed):
    flat = scores.flatten(start_dim=-2)
    k = min(max(int(k), 0), flat.shape[-1])
    mask = torch.zeros_like(flat)
    if k:
        if selection == "topk":
            indices = flat.topk(k, dim=-1).indices
        elif selection == "random":
            generator = torch.Generator(device=flat.device).manual_seed(seed)
            indices = torch.stack([
                torch.randperm(row.numel(), generator=generator, device=row.device)[:k]
                for row in flat.reshape(-1, flat.shape[-1])
            ]).reshape(*flat.shape[:-1], k)
        else:
            raise ValueError(f"unknown selection: {selection}")
        mask.scatter_(-1, indices, 1)
    return mask.reshape_as(scores)


def _mlp_effective_target(block, target_direction):
    """Map the logit direction through a Gemma-style post-FFN affine."""
    from ske_pre_valid.model_adapter import gemma_post_ffn_gamma

    gamma = gemma_post_ffn_gamma(block, device=target_direction.device)
    if gamma is None:
        return target_direction
    return target_direction * gamma.to(dtype=target_direction.dtype)


class DynamicCircuitForward:
    """Execute the manual layer flow while retaining native attention/QK."""

    def __init__(self, model, adapter=None, mlp_cache=None, attention_cache=None):
        if adapter is None:
            from ske_pre_valid.model_adapter import get_adapter
            adapter = get_adapter(model)
        self.model = model
        self.adapter = adapter
        self.layout = describe_model(model)
        self.audit = assert_audited_for_circuits(model)
        self.blocks = adapter.get_blocks(model)
        self._attention = {}
        self._mlp = {}
        self.mlp_cache = mlp_cache
        self.attention_cache = attention_cache

    def _norms(self, block):
        override = None
        if self.layout.residual == "configurable_parallel":
            override = bool(getattr(self.model.config, "use_parallel_residual", False))
        return BranchNorms.from_block(block, self.layout, parallel_override=override)

    def _attention_skeleton(self, layer):
        if layer not in self._attention:
            if self.attention_cache is None:
                self._attention[layer] = AttentionOVSkeleton.from_model(self.model, layer)
            else:
                block = self.blocks[layer]
                device = next(block.parameters()).device
                self._attention[layer] = self.attention_cache.skeleton(
                    layer, device=device, dtype=torch.float32
                )
        return self._attention[layer]

    def _mlp_skeleton(self, layer):
        if layer not in self._mlp:
            if self.mlp_cache is None:
                self._mlp[layer] = MLPSkeleton.from_model(self.model, layer)
            else:
                block = self.blocks[layer]
                device = next(block.parameters()).device
                self._mlp[layer] = MLPSkeleton.from_cache_layer(
                    self.mlp_cache.layer_skeleton(layer), device=device
                )
        return self._mlp[layer]

    @torch.no_grad()
    def __call__(
        self, input_ids, target_direction, config: CircuitForwardConfig,
        *, token_type_ids=None, return_traces: bool = False,
    ):
        device = next(self.model.parameters()).device
        x, position_ids = self.adapter.embed_with_token_types(
            self.model, input_ids.to(device), device, token_type_ids
        )
        pos_emb = self.adapter.prepare_position_embeddings(self.model, x, position_ids)
        attention_traces = [{} for _ in range(input_ids.shape[0])]

        for layer, block in enumerate(self.blocks):
            from ske_pre_valid.model_adapter import move_to_block_device
            x, position_ids, pos_emb = move_to_block_device(block, x, position_ids, pos_emb)
            pre_attention = x
            active = config.start_layer <= layer <= config.end_layer
            use_attention = active and config.mode in {CircuitMode.ATTENTION, CircuitMode.ATTENTION_MLP}
            norms = self._norms(block)

            if use_attention:
                with NativeAttentionCapture(self.model, layer) as capture:
                    native_attention = self.adapter.attn_out(block, x, position_ids, pos_emb)
                    state = capture.state()
                skeleton = self._attention_skeleton(layer)
                probs = state.attention_probs[..., -1:, :]
                scores = skeleton.aggregated_scores(
                    state.normalized_source, probs, target_direction
                )
                if config.selection == "topk":
                    mask = skeleton.select_aggregated(
                        scores, config.attention_k, config.attention_budget
                    ).mask
                else:
                    mask = _random_mask_like(
                        scores, config.attention_k, config.seed + layer,
                        config.attention_budget,
                    )
                selected_raw = skeleton.reconstruct_aggregated(
                    state.normalized_source, probs, mask, include_constants=False
                )
                full_raw = skeleton.reconstruct_aggregated(
                    state.normalized_source, probs, include_constants=True
                )
                if return_traces:
                    for batch_index in range(mask.shape[0]):
                        selected = mask[batch_index, -1]
                        chosen = torch.nonzero(selected, as_tuple=False)
                        per_head = selected.sum(-1).to(torch.int64)
                        attention_traces[batch_index][layer] = {
                            "head_indices": chosen[:, 0].detach().cpu().tolist(),
                            "ov_mode_indices": chosen[:, 1].detach().cpu().tolist(),
                            "scores": scores[batch_index, -1][selected.bool()].detach().float().cpu().tolist(),
                            "selected_count_per_head": per_head.detach().cpu().tolist(),
                        }
                if config.intervention == "preserve":
                    constants = skeleton.constant.sum(0)
                    if skeleton.output_bias is not None:
                        constants = constants + skeleton.output_bias
                    selected_raw = selected_raw + constants
                branch_last = norms.attention_intervention(
                    full_raw, selected_raw, config.intervention, config.post_norm_mode
                )
                attention_out = native_attention.clone()
                attention_out[:, -1:] = branch_last.to(attention_out)
            else:
                attention_out = self.adapter.attn_out(block, x, position_ids, pos_emb)
            post_attention = x + attention_out

            # Sequential models consume the intervened Attention residual here.
            # Parallel models deliberately consume pre_attention instead.
            mlp_source = pre_attention if norms.residual == "parallel" else post_attention
            normed_mlp = norms.mlp_input(mlp_source)
            h = self.adapter.mlp_h(block, normed_mlp)
            full_mlp_raw = self.adapter.mlp_proj(block, h)
            use_mlp = active and config.mode in {CircuitMode.MLP, CircuitMode.ATTENTION_MLP}
            if use_mlp:
                skeleton = self._mlp_skeleton(layer)
                h_last = h[:, -1:]
                effective_target = _mlp_effective_target(block, target_direction)
                scores = skeleton.path_scores(h_last, effective_target)
                mask = _mlp_mask(scores, config.mlp_k, config.selection, config.seed + 10000 + layer)
                selected_raw = skeleton.reconstruct(h_last, mask, include_bias=False)
                if config.intervention == "preserve" and skeleton.bias is not None:
                    selected_raw = selected_raw + skeleton.bias
                full_last = full_mlp_raw[:, -1:]
                branch_last = norms.mlp_intervention(
                    full_last, selected_raw, config.intervention, config.post_norm_mode
                )
                mlp_out = norms.finish_mlp(full_mlp_raw).clone()
                mlp_out[:, -1:] = branch_last.to(mlp_out)
            else:
                mlp_out = norms.finish_mlp(full_mlp_raw)

            # For parallel blocks both branches read pre_attention, but residual
            # addition is still the same sum. For sequential blocks post_attention
            # already contains the modified Attention contribution.
            x = post_attention + mlp_out

        logits = self.adapter.final_out(self.model, x)
        if return_traces:
            traces = attention_traces[0] if len(attention_traces) == 1 else attention_traces
            return {"logits": logits, "attention_paths": traces}
        return logits
