"""
subspace_intervention_hook.py

Subspace-specific intervention hook (post-hook on c_proj output) and
patched-activation collector for a single (j → i → k) path.
"""

import numpy as np
import torch

from circuit_skeleton.skeleton_analysis_crossmlp.static_dynamic_crosslayer import collect_activations  # noqa: F401 – re-exported for convenience


def make_subspace_patch_hook(
    neuron_idx: int,
    signal: float,            # delta[j,i] = S[i] * U[j,i]  — scalar coupling amplitude
    eff: torch.Tensor,        # V[i,:]                       — effector direction [d_model]
    intervention: str,
    mean_val: float = 0.0,
    scale: float = 2.0,
):
    """
    Post-hook on c_proj output.
    Contribution of neuron j through ONE subspace i:
        original = a_j * signal * eff
    zero  → subtract  a_j * signal * eff
    mean  → replace a_j with mean_val  (Δ = (mean_val − a_j) * signal * eff)
    scale → replace a_j with a_j*scale (Δ = (scale−1) * a_j * signal * eff)
    """
    def hook_fn(module, inp, output):
        a_j  = inp[0][:, -1, neuron_idx]       # [batch]
        eff_ = eff.to(output.device)            # [d_model]
        out  = output.clone()
        if intervention == "zero":
            out[:, -1, :] -= signal * a_j.unsqueeze(-1) * eff_
        elif intervention == "mean":
            out[:, -1, :] += signal * (mean_val - a_j).unsqueeze(-1) * eff_
        elif intervention == "scale":
            out[:, -1, :] += signal * (scale - 1.0) * a_j.unsqueeze(-1) * eff_
        return out
    return hook_fn


def collect_patched_activations(
    model, tokenizer, prompts: list,
    start_layer: int, target_layers: list,
    neuron_idx: int,
    signal: float,            # delta[j,i]  — scalar coupling amplitude
    eff: torch.Tensor,        # V[i,:]      — effector direction [d_model]
    intervention: str, device: str,
    mean_val: float = 0.0, scale: float = 2.0,
    batch_size: int = 32,
) -> dict:
    """Run forward passes with the single-subspace intervention active.

    Parameters
    ----------
    batch_size : int
        Number of prompts processed per forward pass (default 32).

    Returns
    -------
    dict { layer_idx: np.ndarray [n_samples, d_ff] }
        Post-GELU activations at target layers under the patched model.
    """
    # GPT-2 tokenizer has no pad token by default; use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    storage, hooks = {}, []

    hooks.append(
        model.transformer.h[start_layer].mlp.c_proj
        .register_forward_hook(
            make_subspace_patch_hook(
                neuron_idx, signal, eff, intervention, mean_val, scale)
        )
    )
    for layer_idx in target_layers:
        def make_collect_hook(li):
            def hook_fn(_module, inp, _out):
                # inp[0]: [batch, seq_len, d_ff] → last token → [batch, d_ff]
                storage[li] = inp[0][:, -1, :].detach().cpu()
            return hook_fn
        hooks.append(
            model.transformer.h[layer_idx].mlp.c_proj
            .register_forward_hook(make_collect_hook(layer_idx))
        )

    accumulated = {li: [] for li in target_layers}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch  = prompts[i : i + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            model(**tokens)
            for li in target_layers:
                if li in storage:
                    accumulated[li].append(storage[li].numpy())  # [B, d_ff]

    for h in hooks:
        h.remove()

    # concatenate along sample axis: list of [B, d_ff] → [n_samples, d_ff]
    return {li: np.concatenate(accumulated[li], axis=0) for li in target_layers}
