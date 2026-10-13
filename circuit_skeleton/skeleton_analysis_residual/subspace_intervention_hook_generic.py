"""
subspace_intervention_hook_generic.py

Architecture-agnostic counterpart to subspace_intervention_hook.py.

The original module hooks model.transformer.h[layer].mlp.c_proj directly,
which only exists for GPT-2-style models. This version looks up the
equivalent module (down_proj / c_proj / fc2 / dense_4h_to_h / ...) through
ske_pre_valid.model_adapter.get_adapter(model).h_hook_module(block), which
already knows the right attribute path for every architecture registered
there (GPT-2, Llama-family incl. Qwen/Mistral/Gemma-v1/DeepSeek, Phi-3,
InternLM2, StarCoder2, GPT-Neo, GPT-NeoX, BioGPT, ProGen2).

Math is identical to subspace_intervention_hook.py — only the module lookup
is generalized.
"""

import numpy as np
import torch

from ske_pre_valid.model_adapter import get_adapter
from circuit_skeleton.skeleton_analysis_residual.static_dynamic_crosslayer_generic import (
    ensure_pad_token,
    gather_last_real,
    tokenize_residual_batch,
)


def make_subspace_patch_hook(
    neuron_idx: int,
    signal: float,
    eff: torch.Tensor,
    intervention: str,
    mean_val: float = 0.0,
    scale: float = 2.0,
    position_state: dict | None = None,
):
    """Post-hook on the source layer's down-projection module.

    Identical semantics to subspace_intervention_hook.make_subspace_patch_hook —
    see that module's docstring for the zero/mean/scale formulas.
    """
    def hook_fn(module, inp, output):
        if position_state is None or position_state.get("last_pos") is None:
            raise RuntimeError("last real token positions were not set before patching")
        last_pos = position_state["last_pos"]
        h_last = gather_last_real(inp[0], last_pos)
        a_j = h_last[:, neuron_idx]                                # [batch]
        # cast to output's device AND dtype: eff comes from the (float32, CPU)
        # coupling-strength cache, but the patching model may load in bf16 —
        # in-place ops below require matching dtype with `out`.
        eff_ = eff.to(device=output.device, dtype=output.dtype)    # [d_model]
        a_j  = a_j.to(dtype=output.dtype)
        out  = output.clone()
        pos = last_pos.to(device=output.device)
        rows = torch.arange(output.shape[0], device=output.device)
        if intervention == "zero":
            out[rows, pos, :] -= signal * a_j.unsqueeze(-1) * eff_
        elif intervention == "mean":
            out[rows, pos, :] += signal * (mean_val - a_j).unsqueeze(-1) * eff_
        elif intervention == "scale":
            out[rows, pos, :] += signal * (scale - 1.0) * a_j.unsqueeze(-1) * eff_
        return out
    return hook_fn


def make_batched_subspace_patch_hook(
    neuron_idx: int,
    signals: torch.Tensor,
    effectors: torch.Tensor,
    intervention: str,
    mean_val: float = 0.0,
    scale: float = 2.0,
    position_state: dict | None = None,
):
    """Patch several independent source subspaces in one expanded batch.

    Rows are laid out as ``[prompt0/sp0, prompt0/sp1, ..., prompt1/sp0, ...]``.
    Each row therefore receives exactly one intervention; interventions are
    batched for throughput, never summed together.
    """
    def hook_fn(_module, inp, output):
        if position_state is None or position_state.get("last_pos") is None:
            raise RuntimeError("last real token positions were not set before patching")
        last_pos = position_state["last_pos"]
        n_interventions = int(signals.numel())
        h_last = gather_last_real(inp[0], last_pos)
        a_j = h_last[:, neuron_idx].to(dtype=output.dtype)
        intervention_ids = torch.arange(
            output.shape[0], device=output.device
        ).remainder(n_interventions)
        signal_rows = signals.to(device=output.device, dtype=output.dtype)[intervention_ids]
        eff_rows = effectors.to(device=output.device, dtype=output.dtype)[intervention_ids]
        path_rows = signal_rows.unsqueeze(-1) * a_j.unsqueeze(-1) * eff_rows

        out = output.clone()
        pos = last_pos.to(device=output.device)
        rows = torch.arange(output.shape[0], device=output.device)
        if intervention == "zero":
            out[rows, pos, :] -= path_rows
        elif intervention == "mean":
            replacement = signal_rows * (mean_val - a_j)
            out[rows, pos, :] += replacement.unsqueeze(-1) * eff_rows
        elif intervention == "scale":
            out[rows, pos, :] += (scale - 1.0) * path_rows
        return out
    return hook_fn


def collect_batched_patched_deltas(
    model, tokenizer, prompts: list,
    baseline_acts: dict,
    start_layer: int, target_layers: list,
    neuron_idx: int,
    intervention_specs: list,
    intervention: str, device: str,
    mean_val: float = 0.0, scale: float = 2.0,
    batch_size: int = 32,
    adapter=None,
    add_special_tokens: bool = False,
) -> dict:
    """Return mean absolute target-activation deltas for many interventions.

    ``intervention_specs`` contains ``(cache_key, signal, effector)`` tuples.
    The prompt batch is expanded across these mutually independent patches and
    a single model call evaluates all of them.  Accumulating sums immediately
    avoids retaining ``n_prompts * n_interventions`` target activations.
    """
    if not intervention_specs:
        return {}
    ensure_pad_token(model, tokenizer)
    if adapter is None:
        adapter = get_adapter(model)
    blocks = adapter.get_blocks(model)

    keys = [spec[0] for spec in intervention_specs]
    signals = torch.tensor([spec[1] for spec in intervention_specs], dtype=torch.float32)
    effectors = torch.stack([spec[2].detach().float().cpu() for spec in intervention_specs])
    n_interventions = len(keys)
    storage, hooks = {}, []
    position_state = {"last_pos": None}
    hooks.append(
        adapter.h_hook_module(blocks[start_layer]).register_forward_hook(
            make_batched_subspace_patch_hook(
                neuron_idx, signals, effectors, intervention, mean_val, scale,
                position_state=position_state,
            )
        )
    )
    for layer_idx in target_layers:
        def make_collect_hook(li):
            def hook_fn(_module, inp, _out):
                storage[li] = gather_last_real(
                    inp[0], position_state["last_pos"]
                ).detach().float().cpu()
            return hook_fn
        hooks.append(
            adapter.h_hook_module(blocks[layer_idx]).register_forward_hook(
                make_collect_hook(layer_idx)
            )
        )

    input_device = (
        model.get_input_embeddings().weight.device
        if getattr(model, "hf_device_map", None)
        else device
    )
    sums = {
        li: torch.zeros(
            n_interventions, baseline_acts[li].shape[1], dtype=torch.float64
        )
        for li in target_layers
    }
    count = 0
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                tokens, last_pos = tokenize_residual_batch(
                    model, tokenizer, batch,
                    add_special_tokens=add_special_tokens,
                )
                original_batch = int(last_pos.numel())
                expanded = {
                    name: value.repeat_interleave(n_interventions, dim=0)
                    for name, value in tokens.items()
                }
                position_state["last_pos"] = last_pos.repeat_interleave(n_interventions)
                model(**{name: value.to(input_device) for name, value in expanded.items()})
                for li in target_layers:
                    patched = storage[li].reshape(
                        original_batch, n_interventions, -1
                    )
                    baseline = torch.as_tensor(
                        baseline_acts[li][i:i + original_batch], dtype=torch.float32
                    ).unsqueeze(1)
                    sums[li].add_((patched - baseline).abs().sum(dim=0, dtype=torch.float64))
                count += original_batch
    finally:
        for hook in hooks:
            hook.remove()

    results = {}
    for intervention_idx, key in enumerate(keys):
        results[key] = np.concatenate([
            (sums[li][intervention_idx] / count).float().numpy()
            for li in target_layers
        ])
    return results


def collect_patched_activations(
    model, tokenizer, prompts: list,
    start_layer: int, target_layers: list,
    neuron_idx: int,
    signal: float,
    eff: torch.Tensor,
    intervention: str, device: str,
    mean_val: float = 0.0, scale: float = 2.0,
    batch_size: int = 32,
    adapter=None,
    add_special_tokens: bool = False,
) -> dict:
    """Generic counterpart to subspace_intervention_hook.collect_patched_activations.

    Uses the model adapter to resolve the source/target down-projection modules
    instead of assuming GPT-2's model.transformer.h[...].mlp.c_proj path, so it
    works for any architecture registered in ske_pre_valid.model_adapter.

    `device` is only used to place the tokenized input when the model is not
    sharded via device_map; for device_map-loaded models the input is moved to
    the input-embedding's device instead (see run of this function in
    causal_patching_runner_generic.py).
    """
    ensure_pad_token(model, tokenizer)

    if adapter is None:
        adapter = get_adapter(model)
    blocks = adapter.get_blocks(model)

    storage, hooks = {}, []
    position_state = {"last_pos": None}

    hooks.append(
        adapter.h_hook_module(blocks[start_layer]).register_forward_hook(
            make_subspace_patch_hook(
                neuron_idx, signal, eff, intervention, mean_val, scale,
                position_state=position_state,
            )
        )
    )
    for layer_idx in target_layers:
        def make_collect_hook(li):
            def hook_fn(_module, inp, _out):
                # .float(): numpy has no bfloat16, and the patching model may load in bf16
                storage[li] = gather_last_real(
                    inp[0], position_state["last_pos"]
                ).detach().float().cpu()
            return hook_fn
        hooks.append(
            adapter.h_hook_module(blocks[layer_idx]).register_forward_hook(
                make_collect_hook(layer_idx)
            )
        )

    input_device = (
        model.get_input_embeddings().weight.device
        if getattr(model, "hf_device_map", None)
        else device
    )

    accumulated = {li: [] for li in target_layers}
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                tokens, last_pos = tokenize_residual_batch(
                    model, tokenizer, batch,
                    add_special_tokens=add_special_tokens,
                )
                position_state["last_pos"] = last_pos
                model(**tokens.to(input_device))
                for li in target_layers:
                    if li in storage:
                        accumulated[li].append(storage[li].numpy())
    finally:
        for h in hooks:
            h.remove()

    return {li: np.concatenate(accumulated[li], axis=0) for li in target_layers}
