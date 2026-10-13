"""Runtime audit of normalization placement and RMSNorm semantics.

The circuit implementation must be configured from a loaded model, not merely
from a checkpoint-name table.  This module checks every decoder layer and
empirically identifies the affine convention used by RMSNorm implementations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import torch

from .architecture import BlockLayout, describe_model
from .projections import _find_attention


def _resolve(root, dotted: str | None):
    if dotted is None:
        return None
    value = root
    for part in dotted.split("."):
        if not hasattr(value, part):
            raise ValueError(f"{type(root).__name__} is missing expected module {dotted!r}")
        value = getattr(value, part)
    return value


def _eps(module) -> float | None:
    for name in ("variance_epsilon", "eps", "epsilon"):
        value = getattr(module, name, None)
        if value is not None:
            return float(value)
    return None


@dataclass(frozen=True)
class NormInfo:
    path: str
    class_name: str
    kind: str
    eps: float | None
    affine: str
    has_bias: bool


@dataclass(frozen=True)
class LayerNormAudit:
    layer: int
    attention_input: NormInfo
    attention_internal: tuple[NormInfo, ...]
    attention_output: NormInfo | None
    mlp_input: NormInfo
    mlp_output: NormInfo | None


@dataclass(frozen=True)
class ModelNormAudit:
    model_type: str
    residual: str
    layers: tuple[LayerNormAudit, ...]

    def to_json(self, indent=2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


def _rms_affine_semantics(module) -> str:
    """Empirically distinguish weight from (1 + weight) RMSNorm variants."""
    weight = getattr(module, "weight", None)
    eps = _eps(module)
    if weight is None or eps is None:
        return "none"
    width = weight.numel()
    device = weight.device
    # Fixed non-constant input avoids accidental LayerNorm/RMSNorm agreement.
    x = torch.linspace(-1.7, 2.3, width, device=device, dtype=torch.float32).reshape(1, -1)
    param_dtype = weight.dtype
    with torch.no_grad():
        actual = module(x.to(param_dtype)).float()
    rms = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    w = weight.detach().float().reshape(1, -1)
    bias = getattr(module, "bias", None)
    b = 0.0 if bias is None else bias.detach().float().reshape(1, -1)
    direct = rms * w + b
    offset = rms * (1.0 + w) + b
    direct_error = (actual - direct).abs().max().item()
    offset_error = (actual - offset).abs().max().item()
    scale = max(actual.abs().max().item(), 1.0)
    tolerance = 5e-3 * scale if param_dtype in {torch.float16, torch.bfloat16} else 2e-5 * scale
    best, error = ("weight", direct_error) if direct_error <= offset_error else ("one_plus_weight", offset_error)
    return best if error <= tolerance else "custom"


def inspect_norm(module, path: str) -> NormInfo:
    name = type(module).__name__
    lower = name.lower()
    if "rmsnorm" in lower or (
        hasattr(module, "weight") and _eps(module) is not None and "layernorm" not in lower
    ):
        kind = "rmsnorm"
        affine = _rms_affine_semantics(module)
    elif isinstance(module, torch.nn.LayerNorm) or "layernorm" in lower:
        kind = "layernorm"
        affine = "weight"
    else:
        kind = "unknown"
        affine = "unknown"
    return NormInfo(
        path=path,
        class_name=name,
        kind=kind,
        eps=_eps(module),
        affine=affine,
        has_bias=getattr(module, "bias", None) is not None,
    )


def _optional_info(block, path: str | None):
    return None if path is None else inspect_norm(_resolve(block, path), path)


def _internal_attention_norms(block, layout: BlockLayout) -> tuple[NormInfo, ...]:
    attention = _find_attention(block)
    found = []
    for name in ("q_norm", "k_norm", "q_layernorm", "k_layernorm"):
        module = getattr(attention, name, None)
        if module is not None:
            found.append(inspect_norm(module, f"{layout.attention_module}.{name}"))
    return tuple(found)


def audit_model_norms(model, *, require_uniform=True) -> ModelNormAudit:
    """Inspect every layer and validate the expected four branch boundaries."""
    from ske_pre_valid.model_adapter import get_adapter

    layout = describe_model(model)
    blocks = get_adapter(model).get_blocks(model)
    residual = layout.residual
    if residual == "configurable_parallel":
        residual = "parallel" if bool(
            getattr(model.config, "use_parallel_residual", False)
        ) else "sequential"

    audits = []
    for index, block in enumerate(blocks):
        audits.append(LayerNormAudit(
            layer=index,
            attention_input=inspect_norm(
                _resolve(block, layout.attention_input_norm), layout.attention_input_norm
            ),
            attention_internal=_internal_attention_norms(block, layout),
            attention_output=_optional_info(block, layout.attention_output_norm),
            mlp_input=inspect_norm(
                _resolve(block, layout.mlp_input_norm), layout.mlp_input_norm
            ),
            mlp_output=_optional_info(block, layout.mlp_output_norm),
        ))

    if require_uniform and audits:
        signature = lambda a: (
            a.attention_input, a.attention_internal, a.attention_output,
            a.mlp_input, a.mlp_output,
        )
        # Ignore layer index; NormInfo contains only structural/semantic data.
        first = signature(audits[0])
        for current in audits[1:]:
            if signature(current) != first:
                raise ValueError(
                    f"normalization structure changes at layer {current.layer}; "
                    "run with require_uniform=False and handle layers individually"
                )
    return ModelNormAudit(
        model_type=str(getattr(model.config, "model_type", "")),
        residual=residual,
        layers=tuple(audits),
    )


def assert_audited_for_circuits(model) -> ModelNormAudit:
    """Fail before circuit computation if a norm is unknown or custom."""
    audit = audit_model_norms(model)
    for layer in audit.layers:
        norms = (
            layer.attention_input, *layer.attention_internal,
            layer.attention_output, layer.mlp_input, layer.mlp_output,
        )
        for norm in norms:
            if norm is not None and (norm.kind == "unknown" or norm.affine in {"unknown", "custom"}):
                raise ValueError(
                    f"unsupported normalization at layer {layer.layer}, {norm.path}: {norm}"
                )
    return audit
