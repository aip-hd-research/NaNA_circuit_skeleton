"""Extract row-vector W_V and W_O matrices from common HF attention layouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionProjections:
    # Row-vector convention: x @ value, concat(heads) @ output.
    value: torch.Tensor                 # [d_model, kv_heads, head_dim]
    output: torch.Tensor                # [query_heads, head_dim, d_model]
    value_bias: torch.Tensor | None     # [kv_heads, head_dim]
    output_bias: torch.Tensor | None    # [d_model]
    query_to_kv: torch.Tensor           # [query_heads]

    @property
    def query_heads(self) -> int:
        return self.output.shape[0]

    @property
    def kv_heads(self) -> int:
        return self.value.shape[1]

    @property
    def head_dim(self) -> int:
        return self.output.shape[1]


def _linear_matrix(module) -> torch.Tensor:
    """Return a module weight in row-vector [input, output] convention."""
    w = module.weight.detach()
    return w if type(module).__name__ == "Conv1D" else w.T


def _head_counts(attention, config, output_width: int) -> tuple[int, int, int]:
    qh = next((int(v) for v in (
        getattr(attention, "num_heads", None),
        getattr(attention, "num_attention_heads", None),
        getattr(config, "num_attention_heads", None),
        getattr(config, "n_head", None),
    ) if v is not None), None)
    if qh is None:
        raise ValueError("cannot determine number of query heads")
    hd = next((int(v) for v in (
        getattr(attention, "head_dim", None), getattr(config, "head_dim", None)
    ) if v is not None), output_width // qh)
    kvh = next((int(v) for v in (
        getattr(attention, "num_key_value_heads", None),
        getattr(config, "num_key_value_heads", None),
    ) if v is not None), qh)
    return qh, kvh, hd


def _find_attention(block):
    for name in ("self_attn", "attn", "attention", "self_attention"):
        if hasattr(block, name):
            attention = getattr(block, name)
            # Hugging Face GPT-Neo inserts a GPTNeoAttention dispatch wrapper
            # between the decoder block and GPTNeoSelfAttention.  The wrapper
            # owns no projections; its ``attention`` child owns q/k/v/out_proj.
            # Resolve that structural wrapper here so projection extraction,
            # native V hooks and norm auditing all see the executable module.
            projection_names = (
                "v_proj", "value", "wv", "c_attn", "query_key_value",
                "qkv_proj", "Wqkv", "wqkv", "o_proj", "out_proj",
                "c_proj", "dense", "wo",
            )
            nested = getattr(attention, "attention", None)
            if nested is not None and not any(
                hasattr(attention, projection) for projection in projection_names
            ):
                attention = nested
            return attention
    raise ValueError(f"cannot find attention module in {type(block).__name__}")


def _find_output(attention):
    for name in ("o_proj", "out_proj", "c_proj", "dense", "wo"):
        if hasattr(attention, name):
            return getattr(attention, name)
    raise ValueError(f"cannot find attention output projection in {type(attention).__name__}")


def _independent_value(attention, kvh: int, hd: int):
    for name in ("v_proj", "value", "wv"):
        if hasattr(attention, name):
            module = getattr(attention, name)
            matrix = _linear_matrix(module)
            if matrix.shape[-1] != kvh * hd:
                raise ValueError(f"{name} width {matrix.shape[-1]} != kv_heads*head_dim")
            bias = getattr(module, "bias", None)
            bias = None if bias is None else bias.detach().reshape(kvh, hd)
            return matrix.reshape(matrix.shape[0], kvh, hd), bias
    return None


def _fused_value(attention, qh: int, kvh: int, hd: int, model_type: str):
    module = next((getattr(attention, n) for n in
                   ("c_attn", "query_key_value", "qkv_proj", "Wqkv", "wqkv")
                   if hasattr(attention, n)), None)
    if module is None:
        raise ValueError(f"cannot find value or fused QKV projection in {type(attention).__name__}")
    matrix = _linear_matrix(module)
    bias = getattr(module, "bias", None)
    bias = None if bias is None else bias.detach()
    total = (qh + 2 * kvh) * hd
    if matrix.shape[-1] != total:
        raise ValueError(
            f"unsupported packed QKV width {matrix.shape[-1]}; expected {total}. "
            "Add an explicit pack-layout resolver for this remote-code model."
        )

    # GPT-NeoX packs [head, Q/K/V, channel]; GPT-2 and standard GQA modules
    # concatenate all Q, then all K, then all V.
    if model_type == "progen" and qh == kvh:
        # ProGen2 remote code reshapes into 8 model-parallel shards and packs
        # [Q,V,K] (note the nonstandard order) inside every shard.
        shards = 8
        if qh % shards:
            raise ValueError("ProGen2 query heads must be divisible by its 8 QKV shards")
        per_shard = qh * hd // shards
        packed = matrix.reshape(matrix.shape[0], shards, 3, per_shard)
        value = packed[:, :, 1, :].reshape(matrix.shape[0], qh, hd)
        value_bias = None if bias is None else (
            bias.reshape(shards, 3, per_shard)[:, 1, :].reshape(qh, hd)
        )
    elif model_type in {"internlm", "internlm2"}:
        # InternLM2 packs each KV group as [its query heads..., K, V].
        queries_per_kv = qh // kvh
        packed = matrix.reshape(matrix.shape[0], kvh, queries_per_kv + 2, hd)
        value = packed[:, :, -1, :]
        value_bias = None if bias is None else (
            bias.reshape(kvh, queries_per_kv + 2, hd)[:, -1, :]
        )
    elif model_type == "gpt_neox" and qh == kvh:
        packed = matrix.reshape(matrix.shape[0], qh, 3, hd)
        value = packed[:, :, 2, :]
        value_bias = None if bias is None else bias.reshape(qh, 3, hd)[:, 2, :]
    else:
        start = (qh + kvh) * hd
        value = matrix[:, start:].reshape(matrix.shape[0], kvh, hd)
        value_bias = None if bias is None else bias[start:].reshape(kvh, hd)
    return value, value_bias


def resolve_attention_projections(model, layer_idx: int) -> AttentionProjections:
    """Resolve per-head V/O projections without executing QK or attention.

    QK, RoPE, masks and softmax remain native model operations.  This function
    extracts only the linear content map needed to form W_OV.
    """
    # Reuse the repository's mature block lookup without changing it.
    from ske_pre_valid.model_adapter import get_adapter

    block = get_adapter(model).get_blocks(model)[layer_idx]
    attention = _find_attention(block)
    output_module = _find_output(attention)
    output_matrix = _linear_matrix(output_module)
    qh, kvh, hd = _head_counts(attention, model.config, output_matrix.shape[0])
    if output_matrix.shape[0] != qh * hd:
        raise ValueError("attention output input width does not match query heads")
    output = output_matrix.reshape(qh, hd, output_matrix.shape[1])

    independent = _independent_value(attention, kvh, hd)
    if independent is None:
        value, value_bias = _fused_value(
            attention, qh, kvh, hd, getattr(model.config, "model_type", "")
        )
    else:
        value, value_bias = independent

    if qh % kvh:
        raise ValueError(f"query heads ({qh}) must be divisible by KV heads ({kvh})")
    query_to_kv = torch.arange(qh, device=value.device) // (qh // kvh)
    output_bias = getattr(output_module, "bias", None)
    output_bias = None if output_bias is None else output_bias.detach()
    return AttentionProjections(value, output, value_bias, output_bias, query_to_kv)
