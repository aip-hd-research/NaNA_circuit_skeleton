from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CircuitMode(str, Enum):
    """Which branch is analysed/intervened on."""

    ATTENTION = "attention"
    MLP = "mlp"
    ATTENTION_MLP = "attention_mlp"


class PathSelection(str, Enum):
    GLOBAL = "global"
    PER_HEAD = "per_head"


@dataclass(frozen=True)
class PathTrace:
    """A selected path tensor and the scores from which it was selected."""

    scores: torch.Tensor
    mask: torch.Tensor
