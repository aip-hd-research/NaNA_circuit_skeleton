"""Non-persistent hooks for collecting native QK routing and V inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .projections import _find_attention


def _value_projection(attention):
    for name in ("v_proj", "value", "wv", "c_attn", "query_key_value", "qkv_proj", "Wqkv", "wqkv"):
        if hasattr(attention, name):
            return getattr(attention, name)
    raise ValueError(f"cannot find a V/QKV projection in {type(attention).__name__}")


@dataclass(frozen=True)
class NativeAttentionState:
    normalized_source: torch.Tensor
    attention_probs: torch.Tensor


class NativeAttentionCapture:
    """Capture the exact input to V and probabilities produced by native QK.

    Use as a context manager around ``model(..., output_attentions=True)``.
    Hooks are always removed on exit and never alter outputs. Some Transformers
    versions require loading with ``attn_implementation='eager'`` before they
    return attention probabilities.
    """

    def __init__(self, model, layer_idx: int):
        from ske_pre_valid.model_adapter import get_adapter

        self.layer_idx = layer_idx
        self.block = get_adapter(model).get_blocks(model)[layer_idx]
        self.attention = _find_attention(self.block)
        self.projection = _value_projection(self.attention)
        self.normalized_source = None
        self.attention_probs = None
        self._handles = []

    def __enter__(self):
        def attention_pre_hook(_module, args, kwargs):
            kwargs["output_attentions"] = True
            kwargs["use_cache"] = False
            return args, kwargs

        def projection_pre_hook(_module, args):
            self.normalized_source = args[0].detach()

        def attention_hook(_module, _args, output):
            candidates = output if isinstance(output, (tuple, list)) else ()
            self.attention_probs = next(
                (x.detach() for x in candidates[1:]
                 if isinstance(x, torch.Tensor) and x.ndim >= 4),
                None,
            )

        self._handles = [
            self.attention.register_forward_pre_hook(attention_pre_hook, with_kwargs=True),
            self.projection.register_forward_pre_hook(projection_pre_hook),
            self.attention.register_forward_hook(attention_hook),
        ]
        return self

    def state(self, fallback_attentions=None) -> NativeAttentionState:
        probs = self.attention_probs
        if probs is None and fallback_attentions is not None:
            probs = fallback_attentions[self.layer_idx].detach()
        if self.normalized_source is None:
            raise RuntimeError("V/QKV projection hook did not fire")
        if probs is None:
            raise RuntimeError(
                "native attention probabilities were not returned; run the model with "
                "output_attentions=True and use an eager attention implementation"
            )
        return NativeAttentionState(self.normalized_source, probs)

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


def collect_native_attention(model, layer_idx: int, **model_inputs):
    """Run one native forward and return outputs plus captured layer state."""
    with NativeAttentionCapture(model, layer_idx) as capture:
        outputs = model(
            **model_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        state = capture.state(getattr(outputs, "attentions", None))
    return outputs, state
