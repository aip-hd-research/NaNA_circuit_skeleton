"""Lazy loader for precomputed per-head Attention W_OV SVD factors."""

from __future__ import annotations

import json
import os

import torch

from .attention_ov import AttentionOVSkeleton


class AttentionOVCache:
    def __init__(self, cache_dir: str, expected_model_name: str | None = None):
        self.cache_dir = cache_dir
        self.expected_model_name = expected_model_name
        self._loaded = {}
        metadata_path = os.path.join(cache_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Attention OV cache metadata not found: {metadata_path}; "
                "run pre_compute/precompute_attention_ov.py first"
            )
        with open(metadata_path, encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        cached_name = self.metadata.get("model_name")
        if expected_model_name is not None and cached_name != expected_model_name:
            raise ValueError(
                f"attention_ov_cache mismatch: cached model={cached_name!r}, "
                f"requested model={expected_model_name!r}"
            )
        if self.metadata.get("format_version") != 1:
            raise ValueError(f"unsupported Attention OV cache format: {self.metadata}")

    def _path(self, layer_idx):
        return os.path.join(self.cache_dir, f"layer_{layer_idx:03d}.pt")

    def layer_payload(self, layer_idx: int) -> dict:
        if layer_idx not in self._loaded:
            path = self._path(layer_idx)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Attention OV layer cache not found: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (self.expected_model_name is not None
                    and payload.get("model_name") != self.expected_model_name):
                raise ValueError(
                    f"Attention OV cache model mismatch in {path}: "
                    f"{payload.get('model_name')!r} != {self.expected_model_name!r}"
                )
            if int(payload.get("layer_idx", -1)) != layer_idx:
                raise ValueError(f"Attention OV cache layer mismatch in {path}")
            self._loaded[layer_idx] = payload
        return self._loaded[layer_idx]

    def skeleton(self, layer_idx: int, *, device=None, dtype=torch.float32):
        payload = self.layer_payload(layer_idx)
        move = lambda value: value.to(device=device, dtype=dtype)
        output_bias = payload.get("output_bias")
        return AttentionOVSkeleton(
            U=move(payload["U"]),
            S=move(payload["S"]),
            V=move(payload["V"]),
            constant=move(payload["constant"]),
            output_bias=None if output_bias is None else move(output_bias),
        )

    def clear_layer(self, layer_idx: int):
        self._loaded.pop(layer_idx, None)
