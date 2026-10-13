"""Generic activation collection for residual-stream cross-MLP experiments.

This module owns batching for the residual experiment.  It deliberately does
not share mutable code with the static/dynamic skeleton extractors, while using
the same prompt contract: right padding, explicit special-token policy, and
per-example indexing of the last non-padding token.
"""

from __future__ import annotations

import numpy as np
import torch

from model_config import NEXT_TOKEN_MAX_LENGTH
from ske_pre_valid.model_adapter import get_adapter


def ensure_pad_token(model, tokenizer) -> None:
    """Install a usable pad token without assuming an EOS token exists."""
    if tokenizer.pad_token_id is not None:
        return

    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    input_embeddings = model.get_input_embeddings()
    if tokenizer.pad_token_id >= input_embeddings.num_embeddings:
        model.resize_token_embeddings(len(tokenizer))


def residual_max_length(model) -> int:
    """Use the shared continuation limit, clipped to the checkpoint context."""
    candidates = [
        getattr(model.config, "n_positions", None),
        getattr(model.config, "max_position_embeddings", None),
    ]
    finite = [int(v) for v in candidates if isinstance(v, int) and 0 < v < 10**7]
    return min([NEXT_TOKEN_MAX_LENGTH, *finite])


def tokenize_residual_batch(
    model,
    tokenizer,
    prompts: list[str],
    add_special_tokens: bool = False,
):
    """Right-pad a batch and return ``(BatchEncoding, last_real_position)``."""
    ensure_pad_token(model, tokenizer)
    tokenizer.padding_side = "right"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=residual_max_length(model),
        add_special_tokens=add_special_tokens,
    )
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        raise ValueError("Tokenizer did not return attention_mask for a padded batch")
    last_pos = attention_mask.long().sum(dim=1) - 1
    if bool((last_pos < 0).any()):
        raise ValueError("Residual experiment received an empty tokenized prompt")
    return enc, last_pos


def gather_last_real(x: torch.Tensor, last_pos: torch.Tensor) -> torch.Tensor:
    """Gather ``x[b, last_pos[b], :]`` without assuming a shared sequence end."""
    pos = last_pos.to(device=x.device)
    rows = torch.arange(x.shape[0], device=x.device)
    return x[rows, pos, :]


def _gated_spec(block):
    """Return modules needed to linearize a gated MLP, or ``None``."""
    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            return {
                "gate": mlp.gate_proj,
                "up": mlp.up_proj,
                "act": mlp.act_fn,
                "fused": False,
            }
        if hasattr(mlp, "gate_up_proj"):
            return {
                "gate_up": mlp.gate_up_proj,
                "act": mlp.activation_fn,
                "fused": True,
            }

    ff = getattr(block, "feed_forward", None)
    if ff is not None and hasattr(ff, "w1") and hasattr(ff, "w3"):
        return {
            "gate": ff.w1,
            "up": ff.w3,
            "act": ff.act_fn,
            "fused": False,
        }
    return None


def _activation_and_derivative(act_fn, gate: torch.Tensor):
    """Evaluate an activation and its elementwise derivative on detached values."""
    with torch.enable_grad():
        gate_var = gate.detach().float().requires_grad_(True)
        activated = act_fn(gate_var)
        derivative = torch.autograd.grad(
            activated,
            gate_var,
            grad_outputs=torch.ones_like(activated),
            create_graph=False,
        )[0]
    return activated.detach(), derivative.detach()


def collect_activations(
    model,
    tokenizer,
    prompts,
    layers,
    device,
    batch_size=32,
    adapter=None,
    add_special_tokens: bool = False,
    collect_target_jacobian_stats: bool = False,
):
    """Collect last-real-token MLP states and optional local derivatives.

    The returned activations are the actual inputs to each down projection.  If
    ``collect_target_jacobian_stats`` is true, the second return value contains
    prompt-averaged coefficients for gated or ordinary activation derivatives.

        d[act(gate) * up] / d(normed residual)

    and is used only by the residual experiment's static candidate ranking.
    """
    ensure_pad_token(model, tokenizer)
    if adapter is None:
        adapter = get_adapter(model)
    blocks = adapter.get_blocks(model)

    storage = {}
    hooks = []
    position_state = {"last_pos": None}
    gated_values = {}
    gated_accum = {}

    for layer_idx in layers:
        spec = _gated_spec(blocks[layer_idx]) if collect_target_jacobian_stats else None
        if collect_target_jacobian_stats and spec is None:
            input_module, _, _, act_fn = adapter.get_mlp_matrices(model, layer_idx)
            spec = {
                "pre_activation": input_module,
                "act": act_fn,
                "fused": False,
                "ungated": True,
            }
        if spec is not None:
            gated_values[layer_idx] = {}

            if spec.get("ungated"):
                def make_pre_activation_hook(li):
                    def hook_fn(_module, _inp, output):
                        gated_values[li]["pre_activation"] = output
                    return hook_fn

                hooks.append(spec["pre_activation"].register_forward_hook(
                    make_pre_activation_hook(layer_idx)
                ))
            elif spec["fused"]:
                def make_fused_hook(li):
                    def hook_fn(_module, _inp, output):
                        gate, up = output.chunk(2, dim=-1)
                        gated_values[li]["gate"] = gate
                        gated_values[li]["up"] = up
                    return hook_fn

                hooks.append(spec["gate_up"].register_forward_hook(
                    make_fused_hook(layer_idx)
                ))
            else:
                def make_component_hook(li, name):
                    def hook_fn(_module, _inp, output):
                        gated_values[li][name] = output
                    return hook_fn

                hooks.append(spec["gate"].register_forward_hook(
                    make_component_hook(layer_idx, "gate")
                ))
                hooks.append(spec["up"].register_forward_hook(
                    make_component_hook(layer_idx, "up")
                ))

        def make_down_hook(li, gated_spec):
            def hook_fn(_module, inp, _out):
                last_pos = position_state["last_pos"]
                storage[li] = gather_last_real(inp[0], last_pos).detach().float().cpu()

                if gated_spec is None:
                    return
                if gated_spec.get("ungated"):
                    pre_activation = gather_last_real(
                        gated_values[li]["pre_activation"], last_pos
                    )
                    _, derivative = _activation_and_derivative(
                        gated_spec["act"], pre_activation
                    )
                    activation_coeff = derivative.sum(dim=0).cpu()
                    acc = gated_accum.setdefault(li, {
                        "kind": "ungated",
                        "activation_coeff_sum": torch.zeros_like(activation_coeff),
                        "n_samples": 0,
                    })
                    acc["activation_coeff_sum"].add_(activation_coeff)
                    acc["n_samples"] += int(pre_activation.shape[0])
                    return
                gate = gather_last_real(gated_values[li]["gate"], last_pos)
                up = gather_last_real(gated_values[li]["up"], last_pos).detach().float()
                activated, derivative = _activation_and_derivative(gated_spec["act"], gate)
                gate_coeff = (derivative * up).sum(dim=0).cpu()
                up_coeff = activated.sum(dim=0).cpu()
                acc = gated_accum.setdefault(li, {
                    "kind": "gated",
                    "gate_coeff_sum": torch.zeros_like(gate_coeff),
                    "up_coeff_sum": torch.zeros_like(up_coeff),
                    "n_samples": 0,
                })
                acc["gate_coeff_sum"].add_(gate_coeff)
                acc["up_coeff_sum"].add_(up_coeff)
                acc["n_samples"] += int(gate.shape[0])
            return hook_fn

        hooks.append(
            adapter.h_hook_module(blocks[layer_idx]).register_forward_hook(
                make_down_hook(layer_idx, spec)
            )
        )

    input_device = (
        model.get_input_embeddings().weight.device
        if getattr(model, "hf_device_map", None)
        else device
    )
    accumulated = {li: [] for li in layers}
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                enc, last_pos = tokenize_residual_batch(
                    model, tokenizer, batch, add_special_tokens=add_special_tokens
                )
                position_state["last_pos"] = last_pos
                enc = enc.to(input_device)
                model(**enc)
                for li in layers:
                    if li in storage:
                        accumulated[li].append(storage[li].numpy())
    finally:
        for hook in hooks:
            hook.remove()

    activations = {
        li: np.concatenate(accumulated[li], axis=0)
        for li in layers
    }
    if not collect_target_jacobian_stats:
        return activations

    gated_stats = {}
    for li, acc in gated_accum.items():
        count = acc["n_samples"]
        if acc["kind"] == "gated":
            gated_stats[li] = {
                "kind": "gated",
                "gate_coeff": acc["gate_coeff_sum"] / count,
                "up_coeff": acc["up_coeff_sum"] / count,
                "n_samples": count,
            }
        else:
            gated_stats[li] = {
                "kind": "ungated",
                "activation_coeff": acc["activation_coeff_sum"] / count,
                "n_samples": count,
            }
    return activations, gated_stats
