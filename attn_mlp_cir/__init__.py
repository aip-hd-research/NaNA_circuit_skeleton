"""Source-resolved Attention-OV and MLP circuit primitives.

This package is intentionally independent from the older circuit implementation.
It may read a loaded Hugging Face model, but it does not monkey-patch model classes
or modify files elsewhere in the repository.
"""

from .architecture import BlockLayout, describe_model, supported_model_names
from .attention_ov import AttentionOVSkeleton
from .audit import ModelNormAudit, assert_audited_for_circuits, audit_model_norms
from .capture import NativeAttentionCapture, NativeAttentionState, collect_native_attention
from .cache import AttentionOVCache
from .mlp import MLPSkeleton
from .norms import BranchNorms
from .projections import AttentionProjections, resolve_attention_projections
from .types import CircuitMode, PathSelection

__all__ = [
    "AttentionOVSkeleton",
    "AttentionOVCache",
    "AttentionProjections",
    "BlockLayout",
    "BranchNorms",
    "CircuitMode",
    "MLPSkeleton",
    "ModelNormAudit",
    "NativeAttentionCapture",
    "NativeAttentionState",
    "PathSelection",
    "describe_model",
    "collect_native_attention",
    "assert_audited_for_circuits",
    "audit_model_norms",
    "resolve_attention_projections",
    "supported_model_names",
]
