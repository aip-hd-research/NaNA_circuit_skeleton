import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# DynamicCache.from_legacy_cache was removed in transformers 5.x; internlm2's
# custom modeling_internlm2.py still calls it.  Re-add it as a no-op shim so
# that trust_remote_code models continue to work without modifying the shared
# model-cache files.
if not hasattr(DynamicCache, "from_legacy_cache"):
    @classmethod  # type: ignore[misc]
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    DynamicCache.from_legacy_cache = _from_legacy_cache

if not hasattr(DynamicCache, "to_legacy_cache"):
    def _to_legacy_cache(self):
        return tuple((layer.keys, layer.values) for layer in self.layers)
    DynamicCache.to_legacy_cache = _to_legacy_cache

# transformers >= 4.46 added all_tied_weights_keys (dict) used in
# _move_missing_keys_from_meta_to_device, but trust_remote_code models such as
# ProGen2 were written against older transformers and inherit PreTrainedModel
# without this attribute.  Patching the base class here covers every subclass
# regardless of which dynamic module path from_pretrained loads the class from.
from transformers import PreTrainedModel as _PreTrainedModel
if not hasattr(_PreTrainedModel, "all_tied_weights_keys"):
    # Newer transformers sets all_tied_weights_keys as an instance attribute in post_init.
    # trust_remote_code models (e.g. ProGen2) use an older post_init that never sets it,
    # so _move_missing_keys_from_meta_to_device raises AttributeError.
    # Adding an empty-dict class attribute provides a fallback for those models:
    #   - Standard models: post_init sets the instance attribute, shadowing this class attr.
    #   - ProGen2: no instance attribute → class attr {} is used → missing_keys unchanged.
    # A property would block instance-attribute setting (no setter) and break standard models.
    _PreTrainedModel.all_tied_weights_keys = {}  # type: ignore[attr-defined]


# Models that require trust_remote_code=True to load
_TRUST_REMOTE_PATTERNS = ("internlm", "progen", "chemllm")

# Models whose residual connections are not device-map-aware; device_map="auto"
# splits layers across GPUs and causes "tensors on different devices" errors.
# Models whose residual connections are not device-map-aware; device_map="auto"
# splits layers across GPUs and causes "tensors on different devices" errors.
# Also includes small domain-specific models that are fast enough on one GPU.
_SINGLE_GPU_PATTERNS = ("biogpt", "progen", "chemgpt-4.7", "chemgpt-19")

# Models that should always load in float32, regardless of multi-GPU dtype setting.
# These are small enough that full precision has negligible memory overhead.
_FORCE_FLOAT32_PATTERNS = ("chemgpt-4.7", "chemgpt-19")


class UntiedOutputEmbeddingOffloadedError(RuntimeError):
    """The real untied LM head is meta/offloaded and cannot be read as a weight."""


def _needs_trust_remote_code(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(p in name_lower for p in _TRUST_REMOTE_PATTERNS)


def _needs_single_gpu(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(p in name_lower for p in _SINGLE_GPU_PATTERNS)


def _needs_force_float32(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(p in name_lower for p in _FORCE_FLOAT32_PATTERNS)


def _fix_progen_meta_buffers(model) -> None:
    """Re-initialize ProGenAttention buffers/attrs stranded on meta device.

    transformers >= 4.46 lazy-inits larger models even without device_map, leaving
    non-persistent buffers and plain tensor attributes on meta device.  Each attr is
    initialized with its semantically correct value:

      bias        — lower-triangular bool mask [1,1,max_pos,max_pos] for causal attn.
                    Zero-init would make causal_mask all-False → masked_bias applied
                    everywhere → broken attention.
      masked_bias — scalar -1e9 penalty for masked (future) positions.
                    Zero-init would disable masking entirely.
      scale_attn  — sqrt(head_dim) scalar.  NOT a registered buffer; plain tensor attr,
                    so named_buffers() misses it.  Overwritten by _fix_progen_scale_attn
                    for the dtype-garbage case (non-meta path).
    """
    target_device = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cpu"),
    )
    fixed = 0
    for module in model.modules():
        if type(module).__name__ != "ProGenAttention":
            continue

        bias = getattr(module, "bias", None)
        if bias is not None and bias.device.type == "meta":
            max_pos = bias.shape[-1]
            module.register_buffer(
                "bias",
                torch.tril(torch.ones(max_pos, max_pos, dtype=torch.bool))
                     .view(1, 1, max_pos, max_pos)
                     .to(target_device),
                persistent=False,
            )
            fixed += 1

        masked_bias = getattr(module, "masked_bias", None)
        if masked_bias is not None and masked_bias.device.type == "meta":
            module.register_buffer(
                "masked_bias",
                torch.tensor(-1e9, device=target_device),
                persistent=False,
            )
            fixed += 1

        scale_attn = getattr(module, "scale_attn", None)
        if scale_attn is not None and scale_attn.device.type == "meta":
            correct = torch.sqrt(torch.tensor(module.head_dim, dtype=torch.float32))
            module.scale_attn = correct.to(
                dtype=torch.get_default_dtype(), device=target_device
            )
            fixed += 1

    if fixed:
        print(f"[INFO] progen2: re-initialized {fixed} meta attrs on {target_device}")


def _fix_progen_scale_attn(model) -> None:
    """Correct scale_attn for every progen2 attention layer after from_pretrained.

    progen2's ProGenAttention registers scale_attn = sqrt(head_dim).to(default_dtype)
    as a non-persistent buffer.  When transformers' _fast_init context temporarily
    changes torch.get_default_dtype(), the buffer is created with a garbage value
    (e.g. -1.6e-21) instead of the correct sqrt(head_dim).  Dividing attention
    weights by this tiny value causes float32 overflow → inf → NaN in softmax.
    """
    fixed = 0
    for module in model.modules():
        if hasattr(module, "scale_attn") and hasattr(module, "head_dim"):
            correct = torch.sqrt(torch.tensor(module.head_dim, dtype=torch.float32))
            correct = correct.to(dtype=module.scale_attn.dtype,
                                 device=module.scale_attn.device)
            module.scale_attn.copy_(correct)
            fixed += 1
    if fixed:
        print(f"[INFO] progen2: fixed scale_attn in attention layers")


def _fix_progen_missing_methods(model) -> None:
    """Add PreTrainedModel methods missing from ProGenModel (nn.Module subclass only).

    ProGenModel inherits from nn.Module, not PreTrainedModel, so methods like
    get_head_mask that transformers >= 4.x expects are absent.
    Patched on the class so all instances share the fix.
    """
    for module in model.modules():
        cls = type(module)
        if cls.__name__ == "ProGenModel" and not hasattr(cls, "get_head_mask"):
            def get_head_mask(self, head_mask, num_hidden_layers,
                              is_attention_chunked=False):
                if head_mask is not None:
                    from transformers import PreTrainedModel
                    return PreTrainedModel.get_head_mask(
                        self, head_mask, num_hidden_layers, is_attention_chunked)
                return [None] * num_hidden_layers
            cls.get_head_mask = get_head_mask
            print("[INFO] progen2: patched ProGenModel.get_head_mask")
            break


def load_model_and_embeddings(model_name: str, device: str | None = None,
                              trust_remote_code: bool | None = None,
                              device_map: str | None = None,
                              torch_dtype: torch.dtype | None = None):
    """
    Load a pretrained language model along with its tokenizer and output embedding matrix.

    Returns
    -------
    model      : the loaded LM (eval mode)
    tokenizer  : the corresponding tokenizer
    W_U        : output (un-embedding) weight [vocab, d_model]  — lm_head.weight
                 Use this for effector-direction token projection: direction @ W_U.T
                 For detection-direction analysis (which tokens activate a neuron),
                 use adapter.get_input_embed_weight(model) instead (W_E = embed_tokens).
                 For models with tied embeddings (e.g. GPT-2) W_U == W_E.

    device_map  : pass "auto" to distribute large models (≥70B) across multiple GPUs.
                  When set, the `device` argument is ignored.
    torch_dtype : explicit dtype override. Default: bfloat16 when device_map is set
                  (large multi-GPU models), float32 otherwise. Pass torch.bfloat16
                  explicitly in precompute scripts for 7B models to save VRAM.
    trust_remote_code defaults to auto-detection based on model name.
    """
    if trust_remote_code is None:
        trust_remote_code = _needs_trust_remote_code(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)

    effective_device_map = None if _needs_single_gpu(model_name) else device_map
    if effective_device_map is None and device_map is not None:
        print(f"[INFO] {model_name}: forcing single-GPU (device_map=None)")

    # Models that must run in float32 regardless of the caller's dtype preference
    if _needs_force_float32(model_name) and torch_dtype != torch.float32:
        torch_dtype = torch.float32
        print(f"[INFO] {model_name}: overriding dtype → float32")
    elif torch_dtype is None:
        # bfloat16 only for multi-GPU large-model loads; float32 for single-GPU
        torch_dtype = torch.bfloat16 if effective_device_map else torch.float32
    kwargs = dict(trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
    if effective_device_map:
        kwargs["device_map"] = effective_device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not effective_device_map:
        # When caller passes device=None (multi-GPU path overridden to single-GPU),
        # fall back to cuda rather than silently leaving the model on CPU.
        target_device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = model.to(target_device)

    if "progen" in model_name.lower():
        _fix_progen_meta_buffers(model)   # must run before scale_attn fix
        _fix_progen_scale_attn(model)
        _fix_progen_missing_methods(model)

    model.eval()

    from ske_pre_valid.model_adapter import disable_gemma_logit_softcaps, get_adapter
    cap_state = disable_gemma_logit_softcaps(model)
    if cap_state["disabled"]:
        if cap_state["final"] is not None or cap_state["attention"] is not None:
            print(
                f"[INFO] {model_name}: Gemma logit soft-caps disabled "
                f"(final={cap_state['final']}, attention={cap_state['attention']}, "
                f"attention_layers={cap_state['layers']})"
            )
        else:
            print(f"[INFO] {model_name}: uncapped Gemma policy active (checkpoint caps=None)")

    adapter = get_adapter(model)
    # W_U may span multiple devices under device_map="auto"; move to cpu for uniform access.
    # Falling back to the input embedding is valid only for genuinely tied models.
    # For an untied model, doing so silently changes every effector score and target
    # alignment.  Let callers choose a non-offloaded/CPU load instead.
    W_U = adapter.get_output_embed_weight(model)
    if W_U.device.type == "meta":
        if getattr(model.config, "tie_word_embeddings", False):
            W_U = adapter.get_input_embed_weight(model)
        else:
            raise UntiedOutputEmbeddingOffloadedError(
                f"{model_name}: untied output embedding (lm_head) is offloaded as a "
                "meta tensor; refusing to substitute the input embedding. Reload "
                "without disk offload (for cache building, use the CPU fallback)."
            )
    if device and not device_map:
        W_U = W_U.to(device)
    return model, tokenizer, W_U


def get_mlp_matrices(model, layer_idx: int):
    """
    Extract the MLP projection matrices, pre-MLP layer norm, and activation function
    from a specific transformer layer.

    Supports GPT-2 and Llama-style architectures via the model adapter.

    Returns
    -------
    (W_in_module, W_out_module, pre_mlp_ln, act_fn)

    GPT-2  : (c_fc,      c_proj,    ln_2,                      mlp.act)
    Llama  : (gate_proj, down_proj, post_attention_layernorm,   mlp.act_fn)

    Note for Llama: down_proj.weight has shape [d_model, d_ff] (nn.Linear
    convention).  SVD callers must transpose it to get [d_ff, d_model].
    """
    from ske_pre_valid.model_adapter import get_adapter
    return get_adapter(model).get_mlp_matrices(model, layer_idx)


# Re-export get_adapter so callers can do: from block_interp.model_load import get_adapter
def get_adapter(model):
    from ske_pre_valid.model_adapter import get_adapter as _get_adapter
    return _get_adapter(model)


def parse_topk_subspaces(topk_subspaces, total_subspaces):
    """
    Convert topk_subspaces argument to a list of indices and a label string.

    Args:
        topk_subspaces: int, list[int], or str ("all" or "topN")
        total_subspaces: int, total number of subspaces available

    Returns:
        indices: list of valid subspace indices
        label: string for file naming or display
    """
    if isinstance(topk_subspaces, str):
        if topk_subspaces.lower() == "all":
            indices = list(range(total_subspaces))
            label = "all"
        elif topk_subspaces.lower().startswith("top"):
            try:
                n = int(topk_subspaces[3:])
                indices = list(range(min(n, total_subspaces)))
                label = f"top{len(indices)}"
            except ValueError:
                raise ValueError(f"Invalid topk_subspaces format: {topk_subspaces}")
        else:
            raise ValueError(f"Invalid string for topk_subspaces: {topk_subspaces}")
    elif isinstance(topk_subspaces, int):
        indices = list(range(min(topk_subspaces, total_subspaces)))
        label = f"top{len(indices)}"
    elif isinstance(topk_subspaces, (list, tuple)):
        indices = [i for i in topk_subspaces if 0 <= i < total_subspaces]
        if not indices:
            raise ValueError("topk_subspaces list is empty or out of range.")
        label = "[" + "_".join(str(i) for i in indices) + "]"
    else:
        raise TypeError("topk_subspaces must be int, list[int], or 'all'.")

    return indices, label


def parse_layers_arg(args_layers, model_name: str):
    """
    Parse the --layers argument.
    Supports:
      --layers all
      --layers 4 5 6
    """
    if isinstance(args_layers, list) and len(args_layers) == 1 and args_layers[0].lower() == "all":
        args_layers = "all"

    if isinstance(args_layers, list):
        return [int(x) for x in args_layers]

    elif isinstance(args_layers, str) and args_layers.lower() == "all":
        from ske_pre_valid.model_adapter import get_adapter
        trc = _needs_trust_remote_code(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trc)
        adapter = get_adapter(model)
        n = adapter.num_layers(model)
        del model
        return list(range(n))

    else:
        raise ValueError(f"Invalid --layers argument: {args_layers}")
