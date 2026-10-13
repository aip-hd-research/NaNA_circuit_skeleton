"""MLP output-path SVD kept alongside the new Attention-OV skeleton."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _row_matrix(module) -> torch.Tensor:
    weight = module.weight.detach()
    return weight if type(module).__name__ == "Conv1D" else weight.T


@dataclass
class MLPSkeleton:
    delta: torch.Tensor   # [d_ff, rank] = U * S
    V: torch.Tensor       # [rank, d_model]
    bias: torch.Tensor | None

    @classmethod
    def from_output_projection(cls, module, *, dtype=torch.float32):
        matrix = _row_matrix(module).to(dtype=dtype)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        bias = getattr(module, "bias", None)
        bias = None if bias is None else bias.detach().to(dtype=dtype)
        return cls(U * S, Vh, bias)

    @classmethod
    def from_cache_layer(cls, layer: dict, *, device=None, dtype=None):
        delta = layer["delta"]
        V = layer["V"]
        bias = layer.get("bias")
        if device is not None or dtype is not None:
            delta = delta.to(device=device or delta.device, dtype=dtype or delta.dtype)
            V = V.to(device=device or V.device, dtype=dtype or V.dtype)
            if bias is not None:
                bias = bias.to(device=device or bias.device, dtype=dtype or bias.dtype)
        return cls(delta, V, bias)

    @classmethod
    def from_model(cls, model, layer_idx: int, *, dtype=torch.float32):
        from ske_pre_valid.model_adapter import get_adapter

        adapter = get_adapter(model)
        _, output, _, _ = adapter.get_mlp_matrices(model, layer_idx)
        return cls.from_output_projection(output, dtype=dtype)

    def path_scores(self, activated_neurons, target_direction=None):
        h = activated_neurons.to(device=self.delta.device, dtype=self.delta.dtype)
        scores = h.unsqueeze(-1) * self.delta
        if target_direction is not None:
            target = target_direction.to(device=self.V.device, dtype=self.V.dtype)
            if target.ndim == 1:
                effector = torch.einsum("rd,d->r", self.V, target)
            else:
                effector = torch.einsum("rd,...d->...r", self.V, target)
                while effector.ndim < scores.ndim:
                    effector = effector.unsqueeze(-2)
            scores = scores * effector
        return scores

    def reconstruct(self, activated_neurons, mask=None, *, include_bias=True):
        scores = self.path_scores(activated_neurons)
        if mask is not None:
            scores = scores * mask.to(device=scores.device, dtype=scores.dtype)
        coeff = scores.sum(dim=-2)
        output = coeff @ self.V
        if include_bias and self.bias is not None:
            output = output + self.bias
        return output
