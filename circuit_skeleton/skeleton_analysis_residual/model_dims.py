"""
model_dims.py

Look up (n_layers, d_ff at a given layer, total param count) for any model
supported by ske_pre_valid.model_adapter, without materializing real weights:
the model is constructed under accelerate's init_empty_weights() (meta
device), which is enough to read module shapes/counts.

Used by the residual sweep to resolve relative start layers, a bounded target
window, and the start layer's neuron count without loading checkpoint weights.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_TRUST_REMOTE_PATTERNS = ("internlm", "progen", "chemllm")


def _needs_trust_remote_code(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(p in name_lower for p in _TRUST_REMOTE_PATTERNS)


def get_model_dims(model_name: str, start_layer: int, trust_remote_code: bool | None = None) -> dict:
    """Return {"n_layers": int, "d_ff": int, "total_params": int} for model_name.

    d_ff is read from the input width of the start layer's down-projection
    module (adapter.h_hook_module) — the same module every generic hook in
    this experiment attaches to, so this is exactly "the number of neurons in
    the start layer's up-projection matrix" regardless of architecture
    (nn.Linear exposes .in_features directly; GPT-2's Conv1D stores weight as
    [in, out], so .weight.shape[0] gives the same thing).
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from ske_pre_valid.model_adapter import get_adapter

    if trust_remote_code is None:
        trust_remote_code = _needs_trust_remote_code(model_name)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

    adapter  = get_adapter(model)
    n_layers = adapter.num_layers(model)
    if not (0 <= start_layer < n_layers):
        raise ValueError(f"{model_name}: start_layer={start_layer} out of range [0, {n_layers})")

    block = adapter.get_blocks(model)[start_layer]
    w_out = adapter.h_hook_module(block)
    d_ff  = w_out.in_features if hasattr(w_out, "in_features") else w_out.weight.shape[0]

    total_params = sum(p.numel() for p in model.parameters())

    del model
    return {
        "n_layers": n_layers,
        "d_ff": int(d_ff),
        "total_params": int(total_params),
        "model_type": getattr(config, "model_type", None),
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Print n_layers/d_ff/total_params for a model")
    p.add_argument("--model_name", required=True)
    p.add_argument("--start_layer", type=int, required=True)
    args = p.parse_args()
    print(json.dumps(get_model_dims(args.model_name, args.start_layer)))
