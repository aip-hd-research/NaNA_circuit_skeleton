"""Precompute the coarse per-head W_OV SVD cache for every model layer.

Attention uses a separate cache root from the existing MLP cache:

  <cache_base>/<model_name>/attention_ov_cache/layer_XXX.pt

Only static model information is cached. QK probabilities, weighted inputs,
target alignments and selected paths remain sample-dependent runtime values.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attn_mlp_cir.attention_ov import AttentionOVSkeleton
from attn_mlp_cir.audit import assert_audited_for_circuits
from attn_mlp_cir.projections import resolve_attention_projections
from block_interp.model_load import load_model_and_embeddings
from ske_pre_valid.model_adapter import get_adapter


FORMAT_VERSION = 1


def _config_fingerprint(model) -> str:
    data = model.config.to_dict()
    encoded = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def precompute(model_name, output_dir, device, device_map=None,
               torch_dtype=None, storage_dtype=torch.float32):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = load_model_and_embeddings(
        model_name,
        device=None if device_map else device,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    model.eval()
    adapter = get_adapter(model)
    blocks = adapter.get_blocks(model)

    # Fail before writing cache files when norm placement/RMS semantics is unknown.
    norm_audit = assert_audited_for_circuits(model)
    fingerprint = _config_fingerprint(model)
    metadata = {
        "format_version": FORMAT_VERSION,
        "cache_type": "coarse_per_head_attention_ov_svd",
        "model_name": model_name,
        "model_type": str(getattr(model.config, "model_type", "")),
        "config_sha256": fingerprint,
        "n_layers": len(blocks),
        "storage_dtype": str(storage_dtype).replace("torch.", ""),
        "path_granularity": ["query_head", "ov_mode"],
        "qk_policy": "native_runtime_weighted_sum_last_query",
        "residual": norm_audit.residual,
    }
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity = ("format_version", "model_name", "config_sha256", "n_layers")
        if any(existing.get(key) != metadata.get(key) for key in identity):
            raise ValueError(
                f"existing Attention cache metadata does not match loaded model: {metadata_path}"
            )
    else:
        _atomic_json_save(metadata, metadata_path)

    for layer_idx, block in enumerate(tqdm(blocks, desc="Precomputing Attention W_OV")):
        output_path = output_dir / f"layer_{layer_idx:03d}.pt"
        if output_path.exists():
            continue
        projections = resolve_attention_projections(model, layer_idx)
        if projections.value.device.type == "meta" or projections.output.device.type == "meta":
            raise RuntimeError(
                "attention projection was disk-offloaded as a meta tensor; rerun without "
                "device_map or load the model on CPU"
            )
        skeleton = AttentionOVSkeleton.from_projections(projections, dtype=torch.float32)
        layer_audit = norm_audit.layers[layer_idx]
        payload = {
            "format_version": FORMAT_VERSION,
            "model_name": model_name,
            "model_type": metadata["model_type"],
            "config_sha256": fingerprint,
            "layer_idx": layer_idx,
            "num_query_heads": projections.query_heads,
            "num_kv_heads": projections.kv_heads,
            "head_dim": projections.head_dim,
            "rank": skeleton.rank,
            "d_model": skeleton.U.shape[-2],
            "query_to_kv": projections.query_to_kv.detach().cpu().to(torch.int16),
            "U": skeleton.U.detach().cpu().to(storage_dtype),
            "S": skeleton.S.detach().cpu().to(torch.float32),
            "V": skeleton.V.detach().cpu().to(storage_dtype),
            "constant": skeleton.constant.detach().cpu().to(storage_dtype),
            "output_bias": (
                None if skeleton.output_bias is None
                else skeleton.output_bias.detach().cpu().to(storage_dtype)
            ),
            "attention_input_norm_kind": layer_audit.attention_input.kind,
            "attention_input_norm_eps": layer_audit.attention_input.eps,
            "attention_output_norm_kind": (
                None if layer_audit.attention_output is None
                else layer_audit.attention_output.kind
            ),
            "attention_output_norm_affine": (
                None if layer_audit.attention_output is None
                else layer_audit.attention_output.affine
            ),
        }
        _atomic_torch_save(payload, output_path)
        tqdm.write(
            f"layer {layer_idx}: heads={projections.query_heads}, "
            f"kv_heads={projections.kv_heads}, rank={skeleton.rank} -> {output_path}"
        )
        del projections, skeleton
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Attention OV cache ready: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute per-head Attention W_OV SVD")
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--cache_base",
        default=os.environ.get(
            "ATTENTION_OV_CACHE_BASE",
            "/pfs/work9/workspace/scratch/hd_nf184-nana-cache/pre_compute_skeleton_attn_cache",
        ),
    )
    parser.add_argument("--output_dir")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--torch_dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--storage_dtype", choices=["float32", "bfloat16"], default="float32")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    compute_dtype = torch.float32 if args.torch_dtype == "float32" else torch.bfloat16
    storage_dtype = torch.float32 if args.storage_dtype == "float32" else torch.bfloat16
    output_dir = args.output_dir or os.path.join(
        args.cache_base, args.model_name, "attention_ov_cache"
    )
    precompute(
        args.model_name, output_dir,
        "cuda" if torch.cuda.is_available() else "cpu",
        device_map=args.device_map,
        torch_dtype=compute_dtype,
        storage_dtype=storage_dtype,
    )
