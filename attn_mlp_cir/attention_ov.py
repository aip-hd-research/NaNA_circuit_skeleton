"""Per-query-head SVD of W_OV with native QK probabilities as routing gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .projections import AttentionProjections, resolve_attention_projections
from .types import PathSelection, PathTrace


def _signed_topk_mask(values: torch.Tensor, k: int) -> torch.Tensor:
    """Select the k largest signed values along the last dimension."""
    k = min(max(int(k), 0), values.shape[-1])
    mask = torch.zeros_like(values)
    if k:
        mask.scatter_(-1, values.topk(k, dim=-1).indices, 1)
    return mask


@dataclass
class AttentionOVSkeleton:
    U: torch.Tensor       # [query_heads, d_model, rank]
    S: torch.Tensor       # [query_heads, rank]
    V: torch.Tensor       # [query_heads, rank, d_model]
    constant: torch.Tensor  # [query_heads, d_model], V-bias followed by O
    output_bias: torch.Tensor | None

    @classmethod
    def from_projections(
        cls, projections: AttentionProjections, *, dtype=torch.float32
    ) -> "AttentionOVSkeleton":
        value = projections.value.to(dtype=dtype)
        output = projections.output.to(dtype=dtype)
        q2kv = projections.query_to_kv.to(value.device)
        per_query_value = value.index_select(1, q2kv).movedim(1, 0)
        # Economy factorisation: never materialise [H,d_model,d_model]. If
        # A=Qa@Ra and B.T=Qb@Rb, then A@B = Qa@(Ra@Rb.T)@Qb.T, so only the
        # [head_dim,head_dim] core needs an SVD.
        qa, ra = torch.linalg.qr(per_query_value, mode="reduced")
        qb, rb = torch.linalg.qr(output.transpose(-2, -1), mode="reduced")
        uc, S, vhc = torch.linalg.svd(ra @ rb.transpose(-2, -1), full_matrices=False)
        U = qa @ uc
        Vh = vhc @ qb.transpose(-2, -1)

        if projections.value_bias is None:
            constant = torch.zeros(
                output.shape[0], output.shape[-1], device=output.device, dtype=dtype
            )
        else:
            vb = projections.value_bias.to(dtype=dtype).index_select(0, q2kv)
            constant = torch.einsum("hd,hdo->ho", vb, output)
        ob = projections.output_bias
        ob = None if ob is None else ob.to(dtype=dtype)
        return cls(U, S, Vh, constant, ob)

    @classmethod
    def from_model(cls, model, layer_idx: int, *, dtype=torch.float32):
        return cls.from_projections(resolve_attention_projections(model, layer_idx), dtype=dtype)

    @property
    def num_heads(self) -> int:
        return self.U.shape[0]

    @property
    def rank(self) -> int:
        return self.S.shape[-1]

    def path_scores(
        self,
        normalized_source: torch.Tensor,
        attention_probs: torch.Tensor,
        target_direction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [..., query, source, head, rank] source-resolved scores.

        ``normalized_source`` is the actual tensor entering V projection,
        shape ``[..., source, d_model]``. ``attention_probs`` has shape
        ``[..., head, query, source]`` and must come from native QK/softmax.
        """
        x = normalized_source.to(device=self.U.device, dtype=self.U.dtype)
        a = attention_probs.to(device=self.U.device, dtype=self.U.dtype)
        if x.shape[-1] != self.U.shape[-2] or a.shape[-3] != self.num_heads:
            raise ValueError("source width or attention head count does not match OV skeleton")
        activation = torch.einsum("...sd,hdr->...shr", x, self.U)
        # a: [..., query, source, head, 1]
        # activation: [..., 1, source, head, rank]
        routed = a.movedim(-3, -1).unsqueeze(-1) * activation.unsqueeze(-4)
        scores = routed * self.S
        if target_direction is not None:
            target = target_direction.to(device=self.V.device, dtype=self.V.dtype)
            if target.ndim == 1:
                effector = torch.einsum("hrd,d->hr", self.V, target)
            else:
                effector = torch.einsum("hrd,...d->...hr", self.V, target)
                while effector.ndim < scores.ndim:
                    effector = effector.unsqueeze(-3)
            scores = scores * effector
        return scores

    def attended_inputs(
        self, normalized_source: torch.Tensor, attention_probs: torch.Tensor
    ) -> torch.Tensor:
        """Return [..., query, head, d_model] after native QK weighted sum."""
        x = normalized_source.to(device=self.U.device, dtype=self.U.dtype)
        a = attention_probs.to(device=self.U.device, dtype=self.U.dtype)
        if x.shape[-1] != self.U.shape[-2] or a.shape[-3] != self.num_heads:
            raise ValueError("source width or attention head count does not match OV skeleton")
        return torch.einsum("...hqs,...sd->...qhd", a, x)

    def aggregated_scores(
        self,
        normalized_source: torch.Tensor,
        attention_probs: torch.Tensor,
        target_direction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [..., query, head, rank] coarse OV-mode path scores.

        QK first forms one weighted input per query head. Unlike
        :meth:`path_scores`, source tokens are not separate selectable paths.
        """
        attended = self.attended_inputs(normalized_source, attention_probs)
        scores = torch.einsum("...qhd,hdr->...qhr", attended, self.U) * self.S
        if target_direction is not None:
            target = target_direction.to(device=self.V.device, dtype=self.V.dtype)
            if target.ndim == 1:
                effector = torch.einsum("hrd,d->hr", self.V, target)
            else:
                effector = torch.einsum("hrd,...d->...hr", self.V, target)
                while effector.ndim < scores.ndim:
                    effector = effector.unsqueeze(-3)
            scores = scores * effector
        return scores

    def reconstruct_aggregated(
        self,
        normalized_source: torch.Tensor,
        attention_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        include_constants: bool = True,
    ) -> torch.Tensor:
        """Reconstruct [..., query, d_model] from selected (head, OV-mode) paths."""
        scores = self.aggregated_scores(normalized_source, attention_probs)
        if mask is not None:
            scores = scores * mask.to(device=scores.device, dtype=scores.dtype)
        out = torch.einsum("...qhr,hrd->...qd", scores, self.V)
        if include_constants:
            out = out + self.constant.sum(dim=0)
            if self.output_bias is not None:
                out = out + self.output_bias
        return out

    def select_aggregated(
        self, scores: torch.Tensor, k: int,
        mode: PathSelection = PathSelection.GLOBAL,
    ) -> PathTrace:
        """Select largest signed contributions globally or inside each head."""
        if mode == PathSelection.GLOBAL:
            mask = _signed_topk_mask(scores.flatten(start_dim=-2), k).reshape_as(scores)
        elif mode == PathSelection.PER_HEAD:
            mask = _signed_topk_mask(scores, k)
        else:
            raise ValueError(f"unknown selection mode: {mode}")
        return PathTrace(scores=scores, mask=mask)

    def reconstruct(
        self,
        normalized_source: torch.Tensor,
        attention_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        include_constants: bool = True,
    ) -> torch.Tensor:
        """Reconstruct [..., query, d_model] from selected (source,head,rank) paths."""
        scores = self.path_scores(normalized_source, attention_probs)
        if mask is not None:
            scores = scores * mask.to(device=scores.device, dtype=scores.dtype)
        coeff = scores.sum(dim=-3)  # source -> [..., query, head, rank]
        out = torch.einsum("...qhr,hrd->...qd", coeff, self.V)
        if include_constants:
            # Softmax sums to one per head, so V bias contributes once per head.
            out = out + self.constant.sum(dim=0)
            if self.output_bias is not None:
                out = out + self.output_bias
        return out

    def select(
        self, scores: torch.Tensor, k: int, mode: PathSelection = PathSelection.GLOBAL,
    ) -> PathTrace:
        if mode == PathSelection.GLOBAL:
            mask = _signed_topk_mask(scores.flatten(start_dim=-3), k).reshape_as(scores)
        elif mode == PathSelection.PER_HEAD:
            # Move head before the jointly ranked source/rank dimensions.
            per_head = scores.movedim(-2, -3)
            shape = per_head.shape
            flat = per_head.flatten(start_dim=-2)
            chosen = _signed_topk_mask(flat, k)
            mask = chosen.reshape(shape).movedim(-3, -2)
        else:
            raise ValueError(f"unknown selection mode: {mode}")
        return PathTrace(scores=scores, mask=mask)
