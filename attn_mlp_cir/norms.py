"""Apply branch-output norms at the correct intervention boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .architecture import BlockLayout


def _magnitude_preserving_norm(norm, replacement_raw, full_raw):
    """Use native post norm while retaining the sparse/full raw norm ratio."""
    replacement_f = replacement_raw.float()
    full_f = full_raw.to(replacement_raw.device).float()
    replacement_size = replacement_f.norm(dim=-1, keepdim=True)
    full_size = full_f.norm(dim=-1, keepdim=True)
    valid = (replacement_size > 0) & (full_size > 0)
    tiny = torch.finfo(torch.float32).tiny
    match = torch.where(
        valid, full_size / replacement_size.clamp_min(tiny), torch.ones_like(full_size)
    )
    normalized = norm(replacement_raw * match.to(replacement_raw.dtype))
    restore = torch.where(
        valid, replacement_size / full_size.clamp_min(tiny), torch.ones_like(full_size)
    )
    return normalized * restore.to(normalized.dtype)


def _resolve(root, dotted: str | None):
    if dotted is None:
        return None
    value = root
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


@dataclass(frozen=True)
class BranchNorms:
    attention_input: object
    attention_output: object | None
    mlp_input: object
    mlp_output: object | None
    residual: str

    @classmethod
    def from_block(cls, block, layout: BlockLayout, *, parallel_override: bool | None = None):
        residual = layout.residual
        if residual == "configurable_parallel":
            if parallel_override is None:
                raise ValueError(
                    "this architecture configures parallel residuals per model; "
                    "pass parallel_override=model.config.use_parallel_residual"
                )
            residual = "parallel" if parallel_override else "sequential"
        return cls(
            _resolve(block, layout.attention_input_norm),
            _resolve(block, layout.attention_output_norm),
            _resolve(block, layout.mlp_input_norm),
            _resolve(block, layout.mlp_output_norm),
            residual,
        )

    def attention_branch_input(self, residual):
        return self.attention_input(residual)

    def mlp_branch_input(self, pre_attention_residual, post_attention_residual):
        source = pre_attention_residual if self.residual == "parallel" else post_attention_residual
        return self.mlp_input(source)

    def finish_attention(self, raw_projection):
        return raw_projection if self.attention_output is None else self.attention_output(raw_projection)

    def finish_mlp(self, raw_projection):
        return raw_projection if self.mlp_output is None else self.mlp_output(raw_projection)

    def attention_intervention(
        self, full_raw, selected_raw, intervention: str,
        post_norm_mode: str = "native",
    ):
        """Apply preserve/ablation before any post-attention norm."""
        raw = selected_raw if intervention == "preserve" else full_raw - selected_raw
        if intervention not in {"preserve", "ablation"}:
            raise ValueError(f"unknown intervention: {intervention}")
        if self.attention_output is None:
            return raw
        if post_norm_mode == "native":
            return self.attention_output(raw)
        if post_norm_mode == "magnitude_preserving":
            return _magnitude_preserving_norm(self.attention_output, raw, full_raw)
        raise ValueError(f"unknown post_norm_mode: {post_norm_mode}")

    def mlp_intervention(
        self, full_raw, selected_raw, intervention: str,
        post_norm_mode: str = "native",
    ):
        """Apply preserve/ablation before any post-MLP norm."""
        raw = selected_raw if intervention == "preserve" else full_raw - selected_raw
        if intervention not in {"preserve", "ablation"}:
            raise ValueError(f"unknown intervention: {intervention}")
        if self.mlp_output is None:
            return raw
        if post_norm_mode == "native":
            return self.mlp_output(raw)
        if post_norm_mode == "magnitude_preserving":
            return _magnitude_preserving_norm(self.mlp_output, raw, full_raw)
        raise ValueError(f"unknown post_norm_mode: {post_norm_mode}")
