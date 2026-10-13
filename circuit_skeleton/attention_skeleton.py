"""Low-rank OV circuit skeletons for multi-head causal attention.

The output of one query head is linear in its attended value vector::

    head_out[h] = attended_value[h] @ W_O[h]

where ``W_O[h]`` has shape ``[head_dim, d_model]``.  Decomposing that slice
with SVD gives the same path factorisation used by the MLP skeleton::

    W_O[h] = U[h] diag(S[h]) V[h]
    delta[h, r, i] = U[h, r, i] * S[h, i]

Consequently a path ``(head h, value channel r, subspace i)`` contributes
``value[h,r] * delta[h,r,i] * V[h,i]`` to the residual stream.  Attention
probabilities can be retained as a separate source-token routing factor.

This module deliberately models the OV/content circuit only.  QK routing is
bilinear and passes through a softmax, so it should not be mixed into this
linear reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_ATTENTION_NAMES = ("self_attn", "attn", "attention")
_OUTPUT_NAMES = ("o_proj", "out_proj", "c_proj", "dense", "wo")


def get_attention_output_projection(model, layer_idx: int):
    """Resolve an attention output projection across the repository's adapters.

    Returns ``(module, linear_layout, num_heads, head_dim)``.  The resolver is
    intentionally structural rather than model-name based, covering independent
    QKV, fused-QKV, MHA and GQA models (GQA changes V routing, not W_O slicing).
    """
    from ske_pre_valid.model_adapter import get_adapter

    blocks = get_adapter(model).get_blocks(model)
    block = blocks[layer_idx]
    attention = next((getattr(block, name) for name in _ATTENTION_NAMES if hasattr(block, name)), None)
    if attention is None:
        raise ValueError(f"could not find attention module in layer {layer_idx}")
    # GPTNeoBlock.attn is a dispatch wrapper.  The actual value/output
    # projections live on its GPTNeoSelfAttention child.
    nested = getattr(attention, "attention", None)
    if nested is not None and not any(hasattr(attention, name) for name in _OUTPUT_NAMES):
        attention = nested
    output_name = next((name for name in _OUTPUT_NAMES if hasattr(attention, name)), None)
    if output_name is None:
        raise ValueError(f"could not find attention output projection in {type(attention).__name__}")
    module = getattr(attention, output_name)

    config = model.config
    num_heads = next(
        (int(x) for x in (
            getattr(attention, "num_heads", None),
            getattr(attention, "num_attention_heads", None),
            getattr(config, "num_attention_heads", None),
            getattr(config, "n_head", None),
        ) if x is not None),
        None,
    )
    if num_heads is None:
        raise ValueError("model does not expose its number of attention heads")
    head_dim = next(
        (int(x) for x in (
            getattr(attention, "head_dim", None),
            getattr(config, "head_dim", None),
        ) if x is not None),
        None,
    )
    # Hugging Face Conv1D uses [input, output]; nn.Linear uses [output, input].
    linear_layout = type(module).__name__ != "Conv1D"
    if head_dim is None:
        inner = module.weight.shape[1] if linear_layout else module.weight.shape[0]
        head_dim = inner // num_heads
    return module, linear_layout, num_heads, head_dim


def split_output_projection(
    output_weight: torch.Tensor,
    num_heads: int,
    head_dim: int | None = None,
    *,
    linear_layout: bool = True,
) -> torch.Tensor:
    """Return per-head output matrices with shape ``[heads, head_dim, d_model]``.

    ``nn.Linear`` output projections store ``[d_model, heads * head_dim]``;
    GPT-2 ``Conv1D`` stores the transpose.  Set ``linear_layout=False`` for the
    latter.
    """
    weight = output_weight.detach()
    if weight.ndim != 2:
        raise ValueError(f"output_weight must be rank 2, got {tuple(weight.shape)}")
    matrix = weight.T if linear_layout else weight
    inner, d_model = matrix.shape
    inferred = inner // num_heads
    head_dim = inferred if head_dim is None else head_dim
    if inner != num_heads * head_dim:
        raise ValueError(
            f"projection input width {inner} != num_heads*head_dim "
            f"({num_heads}*{head_dim})"
        )
    return matrix.reshape(num_heads, head_dim, d_model)


@dataclass
class AttentionSkeleton:
    """SVD factors and path operations for the output projection of one layer."""

    U: torch.Tensor       # [heads, head_dim, rank]
    S: torch.Tensor       # [heads, rank]
    V: torch.Tensor       # [heads, rank, d_model]
    delta: torch.Tensor   # [heads, head_dim, rank]

    @classmethod
    def from_model(
        cls,
        model,
        layer_idx: int,
        *,
        compute_dtype: torch.dtype = torch.float32,
    ) -> "AttentionSkeleton":
        """Build the OV skeleton for one layer of a loaded supported model."""
        module, linear_layout, num_heads, head_dim = get_attention_output_projection(
            model, layer_idx
        )
        return cls.from_output_weight(
            module.weight,
            num_heads,
            head_dim,
            linear_layout=linear_layout,
            compute_dtype=compute_dtype,
        )

    @classmethod
    def from_output_weight(
        cls,
        output_weight: torch.Tensor,
        num_heads: int,
        head_dim: int | None = None,
        *,
        linear_layout: bool = True,
        compute_dtype: torch.dtype = torch.float32,
    ) -> "AttentionSkeleton":
        """Build a skeleton from an ``o_proj``/``c_proj`` weight tensor."""
        per_head = split_output_projection(
            output_weight, num_heads, head_dim, linear_layout=linear_layout
        ).to(dtype=compute_dtype)
        U, S, Vh = torch.linalg.svd(per_head, full_matrices=False)
        return cls(U=U, S=S, V=Vh, delta=U * S.unsqueeze(-2))

    @property
    def num_heads(self) -> int:
        return self.U.shape[0]

    @property
    def head_dim(self) -> int:
        return self.U.shape[1]

    @property
    def rank(self) -> int:
        return self.S.shape[1]

    def static_target_scores(self, target_direction: torch.Tensor) -> torch.Tensor:
        """Return ``S * <V, target>`` scores with shape ``[heads, rank]``."""
        return self.S * self.effector_scores(target_direction)

    def effector_scores(self, target_direction: torch.Tensor) -> torch.Tensor:
        """Return residual-to-target alignments ``<V, target>``."""
        direction = target_direction.to(device=self.V.device, dtype=self.V.dtype)
        if direction.ndim != 1 or direction.shape[0] != self.V.shape[-1]:
            raise ValueError("target_direction must have shape [d_model]")
        return torch.einsum("hid,d->hi", self.V, direction)

    def dynamic_scores(
        self,
        value_states: torch.Tensor,
        target_direction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score value-channel paths, returning ``[..., heads, head_dim, rank]``.

        ``value_states`` may be either already-attended values ``[..., H, D]``
        or source-token values after multiplying by attention probabilities.
        The caller owns that routing step so source positions remain explicit.
        """
        values = value_states.to(device=self.delta.device, dtype=self.delta.dtype)
        if values.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"value_states must end in {(self.num_heads, self.head_dim)}, "
                f"got {tuple(values.shape)}"
            )
        scores = values.unsqueeze(-1) * self.delta
        if target_direction is not None:
            scores = scores * self.effector_scores(target_direction).unsqueeze(-2)
        return scores

    def reconstruct(
        self,
        value_states: torch.Tensor,
        path_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct the projected attention output from selected OV paths.

        ``path_mask`` is broadcastable to ``[..., heads, head_dim, rank]``.
        With no mask, the result equals applying the original output projection
        (apart from its optional bias, which belongs at the caller boundary).
        """
        scores = self.dynamic_scores(value_states)
        if path_mask is not None:
            scores = scores * path_mask.to(device=scores.device, dtype=scores.dtype)
        coefficients = scores.sum(dim=-2)  # [..., heads, rank]
        per_head = torch.einsum("...hi,hid->...hd", coefficients, self.V)
        return per_head.sum(dim=-2)

    def topk_mask(
        self,
        value_states: torch.Tensor,
        k: int,
        target_direction: torch.Tensor | None = None,
        *,
        largest_signed: bool = True,
    ) -> torch.Tensor:
        """Select global top-k paths independently for every leading sample."""
        scores = self.dynamic_scores(value_states, target_direction)
        flat = scores.flatten(start_dim=-3)
        k = max(0, min(int(k), flat.shape[-1]))
        mask = torch.zeros_like(flat)
        if k:
            ranked = flat if largest_signed else flat.abs()
            indices = torch.topk(ranked, k, dim=-1).indices
            mask.scatter_(-1, indices, 1)
        return mask.reshape_as(scores)


def route_value_states(
    attention_probs: torch.Tensor,
    value_states: torch.Tensor,
    query_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return source-resolved attended values ``[..., query, source, head, dim]``.

    Inputs use the common shapes ``[..., heads, query, source]`` and
    ``[..., source, kv_heads, head_dim]``.  ``query_to_kv`` supports GQA/MQA;
    it maps each query-head index to its key/value-head index.
    """
    if attention_probs.ndim < 3 or value_states.ndim < 3:
        raise ValueError("attention_probs and value_states must each be rank >= 3")
    q_heads = attention_probs.shape[-3]
    kv_heads = value_states.shape[-2]
    if query_to_kv is None:
        if q_heads % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        query_to_kv = torch.arange(q_heads, device=value_states.device) // (q_heads // kv_heads)
    else:
        query_to_kv = query_to_kv.to(device=value_states.device, dtype=torch.long)
    if query_to_kv.shape != (q_heads,) or query_to_kv.max().item() >= kv_heads:
        raise ValueError("invalid query_to_kv mapping")
    per_query_values = value_states.index_select(-2, query_to_kv)
    # [..., query, source, head, dim]
    probs = attention_probs.movedim(-3, -1).unsqueeze(-1)
    values = per_query_values.unsqueeze(-4)
    return probs * values
