"""
precompute_within_ske.py

Precompute and cache the within-layer skeleton for every neuron in every layer
of the model. For each layer, the skeleton is the per-neuron ranking of
subspaces by |delta[j, i]| = |S[i] * U[j, i]|.

Cache format (one .pt file per layer):
    {
        "layer_idx":              int,
        "model_name":             str,
        "n_neurons":              int,              # d_ff
        "n_subspaces":            int,
        "sorted_subspace_idx":    Tensor[d_ff, n_subspaces],    # argsort of |delta[j,:]| desc  int32
        "sorted_delta":           Tensor[d_ff, n_subspaces],    # signed delta, sorted as above  float32
        "delta":                  Tensor[d_ff, n_subspaces],    # original (unsorted) delta      float32
        "V":                      Tensor[n_subspaces, d_model], # effector directions             float32
        "bias":                   Tensor[d_model],              # output proj bias (float32)
    }

Lookup: load the file for each desired layer, index by neuron j:
    subspace_ranking = sorted_subspace_idx[j]   # rank-1 = most coupled subspace
    delta_values     = sorted_delta[j]

Reconstruction of MLP output via top-k skeleton paths:
    For paths (j, i):  h[j] * delta[j,i] * V[i,:]  (+ bias if use_bias=True)
    where h[j] = post-activation MLP hidden state neuron j
"""

import os
import sys
import gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from tqdm import tqdm

from block_interp.model_load import (
    UntiedOutputEmbeddingOffloadedError,
    load_model_and_embeddings,
)
from ske_pre_valid.model_adapter import get_adapter, gemma_post_ffn_gamma


def precompute_all_layers(model_name: str, output_dir: str, device: str,
                          device_map: str | None = None,
                          torch_dtype: torch.dtype | None = None) -> None:
    """
    Load the model ONCE and iterate over all layers.

    device_map="auto"  — distribute across all available GPUs (required for 70B+).
    device_map=None    — load entire model onto `device` (default, works for ≤13B on 24GB GPU).
    torch_dtype        — override model dtype (e.g. torch.bfloat16 for 7B models to save VRAM).
                         Default: float32 for single-GPU, bfloat16 for device_map loads.
    """
    os.makedirs(output_dir, exist_ok=True)

    cpu_reload_reason = None
    try:
        model, _, W_U = load_model_and_embeddings(
            model_name, device=None if device_map else device,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    except UntiedOutputEmbeddingOffloadedError as exc:
        if not device_map:
            raise
        cpu_reload_reason = str(exc)

    if cpu_reload_reason is not None:
        # The exception traceback (and failed loader frame/model) has been released
        # now that the except block ended, so a second model copy will not overlap it.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [WARN] {cpu_reload_reason} Reloading the model on CPU for exact W_U/SVD weights.")
        model, _, W_U = load_model_and_embeddings(
            model_name, device="cpu", torch_dtype=torch.float32,
        )
        device_map = None
    adapter = get_adapter(model)

    # When device_map="auto" and GPU VRAM is insufficient, accelerate disk-offloads
    # some layers, leaving their weights as meta tensors (no data).  SVD requires
    # real weight values, so detect this and fall back to a plain CPU load.
    # Also check W_U: for tied models, both the output and its valid input-embedding
    # fallback can themselves be offloaded as meta tensors.
    if device_map:
        _, W_out_probe, _, _ = adapter.get_mlp_matrices(model, 0)
        w_u_is_meta = W_U.device.type == "meta"
        mlp_is_meta = W_out_probe.weight.device.type == "meta"
        if w_u_is_meta or mlp_is_meta:
            reason = "W_U (lm_head)" if w_u_is_meta else "MLP layer-0 weights"
            cpu_dtype = torch.float32
            print(f"  [WARN] device_map='auto' triggered disk offloading ({reason} on meta). "
                  f"Reloading onto CPU with dtype={cpu_dtype} for precomputation.")
            del model
            torch.cuda.empty_cache()
            model, _, W_U = load_model_and_embeddings(
                model_name, device="cpu", torch_dtype=cpu_dtype,
            )
            adapter = get_adapter(model)

    n_layers = adapter.num_layers(model)

    # Save W_U for lightweight overlap analysis without reloading the model.
    w_u_path = os.path.join(output_dir, "W_U.pt")
    if not os.path.exists(w_u_path):
        torch.save(W_U.cpu().to(torch.float32), w_u_path)
        print(f"  Saved W_U → {w_u_path}  shape={tuple(W_U.shape)}")

    print(f"Model: {model_name}  |  layers: {n_layers}  |  adapter: {type(adapter).__name__}"
          + (f"  |  device_map={device_map}" if device_map else f"  |  device={device}"))

    for layer_idx in tqdm(range(n_layers), desc="Precomputing within-layer skeletons"):
        out_path = os.path.join(output_dir, f"layer_{layer_idx:03d}.pt")
        if os.path.exists(out_path):
            continue

        W_in_module, W_out_module, pre_mlp_ln, _ = adapter.get_mlp_matrices(model, layer_idx)

        with torch.no_grad():
            ln_w = pre_mlp_ln.weight.detach()

            # SVD of W_out → U [d_ff, r], S [r], V [r, d_model]
            U, S, V = adapter.svd_w_out(W_out_module)

            # LN-fused input projection → [d_ff, d_model]
            W_up_fused = adapter.w_in_fused(W_in_module, ln_w)

        d_ff        = W_up_fused.shape[0]
        n_subspaces = S.shape[0]

        # delta[j, i] = S[i] * U[j, i]
        delta = (U * S.unsqueeze(0)).cpu()   # [d_ff, n_subspaces]
        V_cpu = V.cpu()                       # [n_subspaces, d_model]

        bias_raw = getattr(W_out_module, "bias", None)
        d_model  = V_cpu.shape[1]
        bias = bias_raw.detach().cpu().float() if bias_raw is not None else torch.zeros(d_model)

        # Sort subspaces per neuron by |delta[j,i]| descending
        sorted_idx   = torch.argsort(delta.abs(), dim=1, descending=True)
        sorted_delta = torch.gather(delta, 1, sorted_idx)

        # Save post_feedforward_layernorm scale (1+weight) for Gemma-2/3 so
        # raw-dot effectors can be gamma-corrected without reloading the model.
        block = adapter.get_blocks(model)[layer_idx]
        gamma = gemma_post_ffn_gamma(block)

        payload = {
            "layer_idx":           layer_idx,
            "model_name":          model_name,
            "n_neurons":           d_ff,
            "n_subspaces":         n_subspaces,
            "sorted_subspace_idx": sorted_idx.to(torch.int32),
            "sorted_delta":        sorted_delta.to(torch.float32),
            "delta":               delta.to(torch.float32),
            "V":                   V_cpu.to(torch.float32),
            "bias":                bias,
        }
        if gamma is not None:
            payload["post_ffn_norm_weight"] = gamma  # (1+weight): actual RMSNorm scale for Gemma-2/3
        torch.save(payload, out_path)
        tqdm.write(f"  Layer {layer_idx}: {d_ff} neurons × {n_subspaces} subspaces → {out_path}")

    del model


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute within-layer skeletons for all layers")
    parser.add_argument("--model_name",  default="gpt2-medium")
    parser.add_argument("--output_dir",  default="result/within_ske_cache")
    parser.add_argument("--gpu",         default="0",
                        help="Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'")
    parser.add_argument("--device_map",  default=None,
                        help="Set to 'auto' for models too large for a single GPU (e.g. 70B)")
    parser.add_argument("--torch_dtype", default=None,
                        help="Model dtype: 'bfloat16' for 7B models to save VRAM, default float32")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = None
    if args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == "float32":
        torch_dtype = torch.float32

    print(f"Device: {device}  device_map: {args.device_map}  torch_dtype: {torch_dtype}")
    precompute_all_layers(args.model_name, args.output_dir, device,
                          device_map=args.device_map, torch_dtype=torch_dtype)
    print("Done.")
