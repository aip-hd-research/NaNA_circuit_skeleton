"""Architecture and norm placement metadata for the repository's model suite.

The metadata is an audit contract, not a replacement implementation of every
Transformer block.  Runtime code should execute the model's native attention,
RoPE, masking and normalisation.  These records make the four relevant norm
boundaries explicit and prevent a sequential runner from silently mishandling
parallel architectures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockLayout:
    family: str
    attention_module: str
    attention_input_norm: str
    attention_output_norm: str | None
    mlp_module: str
    mlp_input_norm: str
    mlp_output_norm: str | None
    residual: str = "sequential"
    notes: str = ""


_GPT2 = BlockLayout("gpt2", "attn", "ln_1", None, "mlp", "ln_2", None)
_LLAMA = BlockLayout(
    "llama", "self_attn", "input_layernorm", None,
    "mlp", "post_attention_layernorm", None,
)
_GEMMA23 = BlockLayout(
    "gemma2/3", "self_attn", "input_layernorm", "post_attention_layernorm",
    "mlp", "pre_feedforward_layernorm", "post_feedforward_layernorm",
    notes="Post-branch RMSNorm must be applied after reconstruction/ablation.",
)
_BIOGPT = BlockLayout(
    "biogpt", "self_attn", "self_attn_layer_norm", None,
    "(fc1,fc2)", "final_layer_norm", None,
)
_GPT_NEOX = BlockLayout(
    "gpt_neox", "attention", "input_layernorm", None,
    "mlp", "post_attention_layernorm", None, residual="configurable_parallel",
)
_PROGEN = BlockLayout(
    "progen", "attn", "ln_1", None, "mlp", "ln_1", None,
    residual="parallel", notes="Attention and MLP read the same pre-branch state.",
)
_INTERNLM = BlockLayout(
    "internlm2", "attention", "attention_norm", None,
    "feed_forward", "ffn_norm", None,
)


# The repository currently lists 33 checkpoints (29 base experiment entries plus
# size/long-context variants).  Listing variants explicitly makes audits reproducible.
_MODEL_LAYOUTS: dict[str, BlockLayout] = {
    # Natural language
    "openai-community/gpt2": _GPT2,
    "openai-community/gpt2-medium": _GPT2,
    "Qwen/Qwen3-0.6B": _LLAMA,
    "Qwen/Qwen2.5-1.5B": _LLAMA,
    "microsoft/Phi-3-mini-4k-instruct": _LLAMA,
    "google/gemma-3-1b-it": _GEMMA23,
    "meta-llama/Llama-3.2-1B": _LLAMA,
    "mistralai/Mistral-7B-v0.3": _LLAMA,
    "mistralai/Mistral-7B-v0.1": _LLAMA,
    "deepseek-ai/deepseek-llm-7b-base": _LLAMA,
    "meta-llama/Llama-2-7b-hf": _LLAMA,
    "meta-llama/Llama-2-13b-hf": _LLAMA,
    "google/gemma-2-9b-it": _GEMMA23,
    # Programming
    "bigcode/starcoder2-3b": _LLAMA,
    "codellama/CodeLlama-7b-hf": _LLAMA,
    "deepseek-ai/deepseek-coder-6.7b-base": _LLAMA,
    # Biomedical
    "stanford-crfm/BioMedLM": _GPT2,
    "microsoft/biogpt": _BIOGPT,
    "microsoft/BioGPT-Large": _BIOGPT,
    "liyinghong/BioQwen-0.5B": _LLAMA,
    "liyinghong/BioQwen-1.8B": _LLAMA,
    # Protein
    "nferruz/ProtGPT2": _GPT2,
    "AI4PD/ZymCTRL": _GPT2,
    "hugohrban/progen2-small": _PROGEN,
    "hugohrban/progen2-medium": _PROGEN,
    # SELFIES
    "ncfrey/ChemGPT-4.7M": _GPT2,
    "ncfrey/ChemGPT-19M": _GPT2,
    "ncfrey/ChemGPT-1.2B": _GPT2,
    # SMILES
    "entropy/gpt2_zinc_87m": _GPT2,
    "ChemFM/ChemFM-1B": _GPT2,
    "jonghyunlee/MolGPT_pretrained-by-ZINC15": _GPT2,
    "jonghyunlee/MolGPT_long_context_pretrained-by-ZINC15": _GPT2,
    # Supported but disabled in the current dataset list
    "internlm/internlm2-7b": _INTERNLM,
    "internlm/internlm2_5-7b-chat": _INTERNLM,
}


def supported_model_names() -> tuple[str, ...]:
    return tuple(_MODEL_LAYOUTS)


def describe_model(model_or_name) -> BlockLayout:
    """Return norm/residual metadata by checkpoint name or loaded model."""
    if isinstance(model_or_name, str):
        try:
            return _MODEL_LAYOUTS[model_or_name]
        except KeyError as exc:
            raise ValueError(f"checkpoint is not in the audited model suite: {model_or_name}") from exc

    model_type = getattr(getattr(model_or_name, "config", None), "model_type", "")
    if model_type in {"gemma2", "gemma3", "gemma3_text"}:
        return _GEMMA23
    if model_type in {"llama", "mistral", "qwen2", "qwen3", "phi3", "starcoder2", "gemma"}:
        return _LLAMA
    if model_type == "biogpt":
        return _BIOGPT
    if model_type == "gpt_neox":
        return _GPT_NEOX
    if model_type in {"internlm", "internlm2"}:
        return _INTERNLM
    if model_type == "progen":
        return _PROGEN
    if model_type in {"gpt2", "gpt_neo"}:
        return _GPT2
    raise ValueError(f"unsupported model_type for norm audit: {model_type!r}")
