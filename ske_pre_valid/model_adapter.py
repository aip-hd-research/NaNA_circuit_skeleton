"""
model_adapter.py — architecture-agnostic interface for transformer forward passes.

Supported adapters
------------------
  GPT2Adapter     — GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
  LlamaAdapter    — Llama-2/3, Mistral, Qwen2, Gemma (v1), DeepSeek dense, StaCoder2
  Phi3Adapter     — Phi-3 (combined gate_up_proj, activation_fn)
  InternLM2Adapter — InternLM2 (feed_forward, w1/w2/w3, tok_embeddings, attention_norm/ffn_norm)

Factory
-------
  get_adapter(model) -> ModelAdapter   (auto-detects from model.config.model_type)

Each adapter provides a uniform interface over architecture differences so that
the forward-pass logic (scatter_add reconstruction, skeleton selection) does not
need to know which model family is being used.

Adding a new architecture
-------------------------
1. Subclass ModelAdapter and implement all abstract methods.
2. Call _register_adapter([list_of_model_type_strings], YourAdapter) at module level.
"""

import torch


GEMMA_APPROX_INTERVENTION_NOTE = (
    "This model has post_feedforward_layernorm (Gemma-2/3 applies RMSNorm after "
    "down_proj, before the residual add). Direct RMSNorm would erase the raw magnitude "
    "of a sparse reconstruction. Intervention outputs therefore first match the full "
    "down_proj norm, pass through the native post-FFN RMSNorm, and then multiply by the "
    "unclamped sparse/full raw-norm ratio. This preserves the raw magnitude information "
    "used by the skeleton experiment; it is a magnitude-preserving intervention, not a "
    "literal native activation replacement."
)


def is_gemma_style(blocks) -> bool:
    """True if any block has post_feedforward_layernorm (Gemma-2/3 RMSNorm-after-MLP).

    Used to attach GEMMA_APPROX_INTERVENTION_NOTE to saved results.
    """
    return any(gemma_post_ffn_gamma(b) is not None for b in blocks)


def gemma_post_ffn_gamma(block, device="cpu", dtype=torch.float32):
    """γ correction for Gemma-2/3 post_feedforward_layernorm, or None elsewhere.

    RMSNorm in Gemma-2/3 computes output = x_norm * (1 + weight); weight is
    zero-initialised so an untrained layer starts as an identity map. The full
    scale factor is therefore (1 + weight), not weight itself. Returns None
    for every other architecture (block has no post_feedforward_layernorm).
    """
    ln = getattr(block, "post_feedforward_layernorm", None)
    if ln is None:
        return None
    return (1.0 + ln.weight.detach()).to(device=device, dtype=dtype)


def _magnitude_preserving_norm(norm_module, replacement_raw, full_raw_ref):
    """Apply an RMS-like post norm while retaining sparse/full raw magnitude."""
    replacement_f = replacement_raw.float()
    full_f = full_raw_ref.to(device=replacement_raw.device).float()
    replacement_norm = replacement_f.norm(dim=-1, keepdim=True)
    full_norm = full_f.norm(dim=-1, keepdim=True)
    valid = (replacement_norm > 0) & (full_norm > 0)
    tiny = torch.finfo(torch.float32).tiny
    match_scale = torch.where(
        valid, full_norm / replacement_norm.clamp_min(tiny), torch.ones_like(full_norm)
    )
    normalized = norm_module(replacement_raw * match_scale.to(replacement_raw.dtype))
    restore_scale = torch.where(
        valid, replacement_norm / full_norm.clamp_min(tiny), torch.ones_like(full_norm)
    )
    return normalized * restore_scale.to(normalized.dtype)


def disable_gemma_logit_softcaps(model) -> dict:
    """Disable final-vocabulary and attention-logit soft-caps for Gemma models.

    Transformers reads the final cap from ``model.config`` on every CausalLM
    forward, but Gemma-2 attention copies its cap into every attention module at
    construction time. Both locations must be cleared; changing only the config
    leaves attention soft-capping active. Gemma-3 checkpoints currently store
    both values as ``None``, so this is a no-op there while enforcing the same
    uncapped policy if a future Gemma-3 checkpoint supplies either value.

    Returns the values that were active before disabling, for logging/metadata.
    Non-Gemma models are left untouched.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in {"gemma", "gemma2", "gemma3", "gemma3_text"}:
        return {"disabled": False, "final": None, "attention": None, "layers": 0}

    config = model.config
    previous_state = getattr(model, "_gemma_disabled_logit_softcaps", None)
    final_cap = (
        previous_state["final"]
        if previous_state is not None
        else getattr(config, "final_logit_softcapping", None)
    )
    attention_cap = (
        previous_state["attention"]
        if previous_state is not None
        else getattr(config, "attn_logit_softcapping", None)
    )
    if hasattr(config, "final_logit_softcapping"):
        config.final_logit_softcapping = None
    if hasattr(config, "attn_logit_softcapping"):
        config.attn_logit_softcapping = None

    cleared_layers = 0
    for module in model.modules():
        if hasattr(module, "attn_logit_softcapping"):
            if module.attn_logit_softcapping is not None:
                cleared_layers += 1
            module.attn_logit_softcapping = None

    state = {
        "disabled": True,
        "final": final_cap,
        "attention": attention_cap,
        "layers": (
            previous_state["layers"]
            if previous_state is not None else cleared_layers
        ),
    }
    model._gemma_disabled_logit_softcaps = state
    return state


def forward_uncapped_logits(model, input_ids, **kwargs):
    """Run a model after disabling every Gemma logit soft-cap.

    This helper also supports models constructed directly in tests or utilities
    instead of through ``load_model_and_embeddings``, which disables the caps at
    load time for all normal experiment paths. Other model families are unchanged.
    """
    disable_gemma_logit_softcaps(model)
    return model(input_ids, **kwargs).logits


def move_to_block_device(block, x, position_ids, pos_emb):
    """Move x, position_ids and pos_emb to the device of `block`.

    With device_map="auto" different blocks live on different GPUs.
    Call this at the top of each layer iteration in manual forward loops so
    that x stays co-located with the block being executed.
    Returns (x, position_ids, pos_emb) on the block's device; is a no-op
    when the block has no parameters (e.g. identity layers) or when all
    tensors are already on the correct device.
    """
    param = next(iter(block.parameters()), None)
    if param is None:
        return x, position_ids, pos_emb
    bd = param.device
    x            = x.to(bd)
    position_ids = position_ids.to(bd)
    def _move(p):
        if isinstance(p, (tuple, list)):
            return type(p)(_move(q) for q in p)
        return p.to(bd)

    if pos_emb is None:
        pass
    elif isinstance(pos_emb, dict):
        # Gemma3Adapter: {"global": (cos, sin), "local": (cos, sin)}
        pos_emb = {k: _move(v) for k, v in pos_emb.items()}
    else:
        pos_emb = _move(pos_emb)
    return x, position_ids, pos_emb


# Architectures that map directly to LlamaAdapter (standard gate/up/down_proj layout):
_LLAMA_TYPES = frozenset({
    "llama", "mistral", "mixtral",
    "qwen2", "qwen2_moe",
    "qwen3",                        # Qwen3 — same LN/MLP layout as Llama; q_norm/k_norm are
                                    # internal to self_attn and do not affect the adapter.
                                    # tie_word_embeddings=True on small models but lm_head.weight
                                    # still exists, so get_output_embed_weight() is fine.
    "gemma",                        # Gemma v1 — same layout as Llama
    "deepseek_v2", "deepseek_v3", "deepseek",
    "cohere",
    # starcoder2 removed — uses c_fc/c_proj (no gate), handled by StarCoder2Adapter
})


class ModelAdapter:
    """Base class — subclass and override all methods for a new architecture."""

    def embed(self, model, input_ids, device):
        """
        Compute the initial token embedding.

        Returns
        -------
        x            : Tensor [batch, seq, d_model]  — initial hidden state
        position_ids : Tensor [1, seq]                — used in attention
        """
        raise NotImplementedError

    def embed_with_token_types(
        self, model, input_ids, device, token_type_ids=None
    ):
        """Embed inputs, including optional tokenizer-provided segment ids.

        Most causal-LM tokenizers do not emit ``token_type_ids`` and most model
        families ignore them. GPT-2/GPT-Neo models, however, implement segment
        embeddings by reusing the token embedding table. ChemGPT's BERT-style
        tokenizer emits an all-zero ``token_type_ids`` tensor, so dropping it
        changes the input and breaks agreement with generated next-token labels.
        """
        return self.embed(model, input_ids, device)

    def get_blocks(self, model):
        """Return the nn.ModuleList / list of transformer blocks."""
        raise NotImplementedError

    def num_layers(self, model) -> int:
        return len(self.get_blocks(model))

    def prepare_position_embeddings(self, model, x, position_ids):
        """
        Pre-compute the rotary (cos, sin) tuple once before the layer loop.

        Returns None for models that handle positions internally (e.g. GPT-2).
        For RoPE models (Llama, Gemma, Phi-3, StarCoder2 …) returns the
        (cos, sin) tensors produced by model.model.rotary_emb.
        The result is forwarded to every attn_out call as `pos_emb`.
        """
        return None

    def attn_out(self, block, x, position_ids, pos_emb=None):
        """
        Compute attention output **without** the residual connection.
        Callers do: x = x + adapter.attn_out(block, x, position_ids, pos_emb)

        pos_emb  — (cos, sin) tuple from prepare_position_embeddings, or None.
        Returns attn_residual [batch, seq, d_model].
        """
        raise NotImplementedError

    def magnitude_preserving_post_attn_norm(self, block, replacement_raw, full_raw_ref):
        """Map a replacement output-projection vector to the residual stream.

        Most architectures add the raw attention projection directly. Gemma-2/3
        override this because they apply a post-attention RMSNorm.
        """
        return replacement_raw

    def mlp_ln_input(self, x, x_pre_attn):
        """Select what mlp_ln/full_mlp should normalise: x (post-attention residual)
        or x_pre_attn (the residual stream *before* this layer's attention output
        was added).

        Sequential architectures (GPT-2, Llama, Gemma, ...) feed the MLP the
        residual stream *after* the attention residual has been added — return x
        (default; identical to always calling mlp_ln(block, x) as before this
        method existed).

        Overridden by ProGenAdapter: ProGen2 is a parallel block where attention
        and MLP both read the same pre-attention input, so it returns x_pre_attn
        instead. See ProGenAdapter.mlp_ln_input for why this matters.
        """
        return x

    def mlp_ln(self, block, x):
        """Pre-MLP layer norm. Returns normed [batch, seq, d_model]."""
        raise NotImplementedError

    def mlp_h(self, block, normed):
        """
        Compute the MLP hidden state h [batch, seq, d_ff] from the normed input.

        GPT-2 : act(c_fc(normed))
        Llama : act_fn(gate_proj(normed)) * up_proj(normed)   (SwiGLU)
        """
        raise NotImplementedError

    def mlp_proj(self, block, h):
        """Output projection W_out(h) → [batch, seq, d_model]."""
        raise NotImplementedError

    def post_mlp_norm(self, block, x):
        """Post-MLP norm applied after down_proj, before residual add.
        Identity for most models; overridden for Gemma 2 (post_feedforward_layernorm)."""
        return x

    def magnitude_preserving_post_mlp_norm(self, block, replacement_raw, full_down_ref):
        """Map a replacement down-projection vector to the residual stream.

        Models without a post-MLP norm retain the replacement's raw magnitude directly.
        Gemma-2/3 override this to compensate for post-FFN RMSNorm's scale invariance.
        """
        return self.post_mlp_norm(block, replacement_raw)

    def full_mlp(self, block, x):
        """
        Full unmodified MLP output (before residual).
        Equivalent to post_mlp_norm(block, mlp_proj(block, mlp_h(block, mlp_ln(block, x)))).
        """
        raise NotImplementedError

    def final_out(self, model, x):
        """Final layer norm + lm_head → logits [batch, seq, vocab]."""
        raise NotImplementedError

    def h_hook_module(self, block):
        """
        Return the module on which to attach a forward_pre_hook to capture h.

        The pre-hook receives inp[0] = h [batch, seq, d_ff].
          GPT-2 : block.mlp.c_proj    (Conv1D; input  = act output)
          Llama : block.mlp.down_proj  (Linear; input  = act_fn(gate)*up)
        """
        raise NotImplementedError

    def get_mlp_matrices(self, model, layer_idx):
        """
        Return (W_in_module, W_out_module, pre_mlp_ln, act_fn) for cache building.

        GPT-2 : (c_fc,      c_proj,    ln_2,                     act)
        Llama : (gate_proj, down_proj, post_attention_layernorm,  act_fn)

        Note for Llama: down_proj.weight is stored as [d_model, d_ff] (nn.Linear
        convention). Use svd_w_out() to get the SVD in the correct orientation.
        Use w_in_fused() to fuse the input projection with the layer-norm weight.
        """
        raise NotImplementedError

    def w_in_fused(self, W_in_module, ln_w):
        """
        LN-fused input projection, shape [d_ff, d_model].

        W_in_fused[j, :] = detection direction of neuron j, scaled by ln_w.

        GPT-2 : c_fc.weight is [d_model, d_ff]  (Conv1D)  →  .T * ln_w
        Llama : gate_proj.weight is [d_ff, d_model] (nn.Linear) →  weight * ln_w
        """
        raise NotImplementedError

    def svd_w_out(self, W_out_module):
        """
        SVD of the output projection.  Returns (U, S, V) where:
            U : [d_ff, r]      left singular vectors
            S : [r]            singular values
            V : [r, d_model]   right singular vectors (effector directions)

        GPT-2 : SVD of c_proj.weight       [d_ff, d_model]  (Conv1D)
        Llama : SVD of down_proj.weight.T  [d_ff, d_model]  (nn.Linear transposed)
        """
        raise NotImplementedError

    # ── Vocabulary projection helpers ──────────────────────────────────────────

    def get_input_embed_weight(self, model) -> "torch.Tensor":
        """
        W_E — input embedding matrix [vocab, d_model].

        Used for detection-direction analysis: which tokens activate a neuron?
          neuron_j fires on token t iff  W_E[t] · w_in_j  is large.

        NOTE: for models with tied embeddings (e.g. GPT-2), W_E == W_U.
        """
        return model.get_input_embeddings().weight.detach()

    def get_output_embed_weight(self, model) -> "torch.Tensor":
        """
        W_U — output (un-embedding) matrix [vocab, d_model].

        Used for effector-direction analysis / logit lens:
          direction d in residual stream writes token t iff  d · W_U[t]  is large.
          → use  V[i] @ W_U.T  for subspace effector scores.
          → use  W_up_fused[j] @ W_U.T  for neuron effector scores.

        NOTE: for models with tied embeddings (e.g. GPT-2), W_U == W_E.
        """
        return model.get_output_embeddings().weight.detach()

    def apply_final_ln(self, model, x: "torch.Tensor") -> "torch.Tensor":
        """
        Apply the final layer norm (the norm that precedes lm_head) to hidden states x.

        Used for logit-lens projection of actual residual-stream activations:
          logits = lm_head(apply_final_ln(model, x))

        For static weight vectors (V, W_up_fused), callers may omit this since
        LN on a weight vector changes its direction in a data-dependent way.
        """
        raise NotImplementedError


# ── GPT-2 ─────────────────────────────────────────────────────────────────────

class GPT2Adapter(ModelAdapter):
    """GPT-2 and variants (gpt2, gpt2-medium, gpt2-large, gpt2-xl)."""

    def embed(self, model, input_ids, device):
        t   = model.transformer
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        x   = t.wte(input_ids) + t.wpe(pos)
        return t.drop(x), pos

    def embed_with_token_types(
        self, model, input_ids, device, token_type_ids=None
    ):
        t   = model.transformer
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        x   = t.wte(input_ids) + t.wpe(pos)
        if token_type_ids is not None:
            x = x + t.wte(token_type_ids.to(device))
        return t.drop(x), pos

    def get_blocks(self, model):
        return model.transformer.h

    def attn_out(self, block, x, position_ids, pos_emb=None):
        return block.attn(block.ln_1(x))[0]

    def mlp_ln(self, block, x):
        return block.ln_2(x)

    def mlp_h(self, block, normed):
        return block.mlp.act(block.mlp.c_fc(normed))

    def mlp_proj(self, block, h):
        return block.mlp.c_proj(h)

    def full_mlp(self, block, x):
        return block.mlp(block.ln_2(x))

    def final_out(self, model, x):
        return model.lm_head(model.transformer.ln_f(x))

    def h_hook_module(self, block):
        return block.mlp.c_proj   # forward_pre_hook: inp[0] = act output = h

    def get_mlp_matrices(self, model, layer_idx):
        block = model.transformer.h[layer_idx]
        return block.mlp.c_fc, block.mlp.c_proj, block.ln_2, block.mlp.act

    def w_in_fused(self, W_in_module, ln_w):
        # Conv1D: weight is [d_model, d_ff] → .T gives [d_ff, d_model] → * ln_w [d_model]
        return W_in_module.weight.T.detach() * ln_w

    def svd_w_out(self, W_out_module):
        from block_interp.mlp_svd_utils import compute_svd
        return compute_svd("c_proj", c_proj=W_out_module)

    def get_input_embed_weight(self, model):
        return model.transformer.wte.weight.detach()   # W_E [vocab, d_model]

    def get_output_embed_weight(self, model):
        return model.lm_head.weight.detach()           # W_U [vocab, d_model] (tied = same tensor)

    def apply_final_ln(self, model, x):
        return model.transformer.ln_f(x)


# ── Llama-style ───────────────────────────────────────────────────────────────

class LlamaAdapter(ModelAdapter):
    """
    Llama-style models: Llama-2/3, Mistral, Qwen2, Gemma (v1), Phi-3,
    DeepSeek dense, StaCoder2, InternLM2.

    Architecture:
      • Embeddings : model.model.embed_tokens
      • Blocks      : model.model.layers
      • Per block   : input_layernorm → self_attn
                      post_attention_layernorm → mlp (gate_proj, up_proj, down_proj)
      • Final norm  : model.model.norm
    """

    def embed(self, model, input_ids, device):
        m   = model.model
        x   = m.embed_tokens(input_ids)
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        return x, pos

    def get_blocks(self, model):
        return model.model.layers

    def prepare_position_embeddings(self, model, x, position_ids):
        return model.model.rotary_emb(x, position_ids=position_ids)

    def attn_out(self, block, x, position_ids, pos_emb=None):
        normed = block.input_layernorm(x)
        # Explicit 4D causal mask: the eager-attention path in this family does not
        # apply any mask when attention_mask=None (same gap as GPT-NeoX/BioGPT — see
        # their attn_out docstrings). Currently loaded models run under SDPA, whose
        # is_causal auto-detection already matches this mask mathematically, so this
        # is a no-op there; it only changes behaviour (fixes bidirectional leakage)
        # if an eager backend is ever selected.
        attn_mask = _make_causal_mask(x.shape[1], x.dtype, x.device)
        out    = block.self_attn(
            hidden_states=normed,
            attention_mask=attn_mask,
            position_embeddings=pos_emb,
            use_cache=False,
        )
        return out[0]

    def mlp_ln(self, block, x):
        return block.post_attention_layernorm(x)

    def mlp_h(self, block, normed):
        return block.mlp.act_fn(block.mlp.gate_proj(normed)) * block.mlp.up_proj(normed)

    def mlp_proj(self, block, h):
        return block.mlp.down_proj(h)

    def full_mlp(self, block, x):
        normed = block.post_attention_layernorm(x)
        return block.mlp.down_proj(
            block.mlp.act_fn(block.mlp.gate_proj(normed)) * block.mlp.up_proj(normed)
        )

    def final_out(self, model, x):
        return model.lm_head(model.model.norm(x))

    def h_hook_module(self, block):
        return block.mlp.down_proj  # forward_pre_hook: inp[0] = act_fn(gate)*up = h

    def get_mlp_matrices(self, model, layer_idx):
        block = model.model.layers[layer_idx]
        return (block.mlp.gate_proj, block.mlp.down_proj,
                block.post_attention_layernorm, block.mlp.act_fn)

    def w_in_fused(self, W_in_module, ln_w):
        # nn.Linear: weight is [d_ff, d_model] → already [d_ff, d_model] → * ln_w [d_model]
        return W_in_module.weight.detach() * ln_w

    def svd_w_out(self, W_out_module):
        from block_interp.mlp_svd_utils import compute_svd
        return compute_svd("down_proj", down_proj=W_out_module)

    def get_input_embed_weight(self, model):
        return model.model.embed_tokens.weight.detach()   # W_E [vocab, d_model]

    def get_output_embed_weight(self, model):
        return model.lm_head.weight.detach()              # W_U [vocab, d_model] (separate from W_E)

    def apply_final_ln(self, model, x):
        return model.model.norm(x)


# ── Phi-3 ─────────────────────────────────────────────────────────────────────

class Phi3Adapter(LlamaAdapter):
    """
    Microsoft Phi-3 family.

    Key difference from LlamaAdapter: the gate and up projections are fused into
    a single matrix  gate_up_proj.weight [2*d_ff, d_model].  The first d_ff rows
    are the gate weights; the second d_ff rows are the up weights.  The activation
    function attribute is named activation_fn (not act_fn).
    """

    def mlp_h(self, block, normed):
        gate_up = block.mlp.gate_up_proj(normed)
        gate, up = gate_up.chunk(2, dim=-1)
        return block.mlp.activation_fn(gate) * up

    def full_mlp(self, block, x):
        normed = block.post_attention_layernorm(x)
        return block.mlp(normed)   # Phi3MLP.forward handles the split internally

    def h_hook_module(self, block):
        return block.mlp.down_proj  # pre-hook: inp[0] = activation_fn(gate)*up = h

    def get_mlp_matrices(self, model, layer_idx):
        block = model.model.layers[layer_idx]
        return (block.mlp.gate_up_proj, block.mlp.down_proj,
                block.post_attention_layernorm, block.mlp.activation_fn)

    def w_in_fused(self, W_in_module, ln_w):
        # gate_up_proj.weight [2*d_ff, d_model]; first d_ff rows = gate weights
        d_ff = W_in_module.weight.shape[0] // 2
        gate_w = W_in_module.weight[:d_ff, :].detach()   # [d_ff, d_model]
        return gate_w * ln_w

    # svd_w_out: down_proj.weight is [d_model, d_ff] — same convention as LlamaAdapter ✓


# ── InternLM2 ─────────────────────────────────────────────────────────────────

class InternLM2Adapter(LlamaAdapter):
    """
    InternLM2 family (requires trust_remote_code=True at load time).

    Attribute differences from LlamaAdapter:
      • Embedding   : model.model.tok_embeddings  (not embed_tokens)
      • Attention   : block.attention             (not self_attn)
      • Pre-attn LN : block.attention_norm        (not input_layernorm)
      • Pre-MLP LN  : block.ffn_norm              (not post_attention_layernorm)
      • MLP module  : block.feed_forward          (not mlp)
      • Gate proj   : feed_forward.w1             (not gate_proj)
      • Up proj     : feed_forward.w3             (not up_proj)
      • Down proj   : feed_forward.w2             (not down_proj)
      • Activation  : feed_forward.act_fn
    Weight shapes match LlamaAdapter (nn.Linear convention), so w_in_fused and
    svd_w_out are inherited unchanged.
    """

    def embed(self, model, input_ids, device):
        x   = model.model.tok_embeddings(input_ids)
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        return x, pos

    def prepare_position_embeddings(self, model, x, position_ids):
        # InternLM2 attention computes RoPE internally from position_ids; nothing to precompute.
        return None

    def attn_out(self, block, x, position_ids, pos_emb=None):
        normed = block.attention_norm(x)
        # See LlamaAdapter.attn_out: explicit mask closes the same eager-backend
        # bidirectional-leakage gap; a no-op under the SDPA backend currently in use.
        attn_mask = _make_causal_mask(x.shape[1], x.dtype, x.device)
        out = block.attention(
            hidden_states=normed,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return out[0]

    def mlp_ln(self, block, x):
        return block.ffn_norm(x)

    def mlp_h(self, block, normed):
        ff = block.feed_forward
        return ff.act_fn(ff.w1(normed)) * ff.w3(normed)

    def mlp_proj(self, block, h):
        return block.feed_forward.w2(h)

    def full_mlp(self, block, x):
        ff = block.feed_forward
        normed = block.ffn_norm(x)
        return ff.w2(ff.act_fn(ff.w1(normed)) * ff.w3(normed))

    def h_hook_module(self, block):
        return block.feed_forward.w2   # pre-hook: inp[0] = act_fn(w1)*w3 = h

    def get_mlp_matrices(self, model, layer_idx):
        block = model.model.layers[layer_idx]
        ff = block.feed_forward
        return (ff.w1, ff.w2, block.ffn_norm, ff.act_fn)

    def get_input_embed_weight(self, model):
        return model.model.tok_embeddings.weight.detach()  # InternLM2 uses tok_embeddings, not embed_tokens

    def get_output_embed_weight(self, model):
        return model.output.weight.detach()   # InternLM2 uses model.output, not model.lm_head

    def final_out(self, model, x):
        return model.output(model.model.norm(x))

    # apply_final_ln: model.model.norm — inherited from LlamaAdapter ✓
    # w_in_fused: w1.weight is [d_ff, d_model] — same as LlamaAdapter.gate_proj ✓
    # svd_w_out : w2.weight is [d_model, d_ff] — same as LlamaAdapter.down_proj ✓


# ── StarCoder2 ────────────────────────────────────────────────────────────────

class StarCoder2Adapter(LlamaAdapter):
    """
    BigCode StarCoder2 family.

    LN positions are identical to LlamaAdapter (input_layernorm, post_attention_layernorm).
    MLP is GPT-2-style (no gate): c_fc → act → c_proj.
    Weight shapes: c_fc [d_ff, d_model], c_proj [d_model, d_ff] (both nn.Linear).
    """

    def mlp_h(self, block, normed):
        return block.mlp.act(block.mlp.c_fc(normed))

    def mlp_proj(self, block, h):
        return block.mlp.c_proj(h)

    def full_mlp(self, block, x):
        normed = block.post_attention_layernorm(x)
        return block.mlp.c_proj(block.mlp.act(block.mlp.c_fc(normed)))

    def h_hook_module(self, block):
        return block.mlp.c_proj   # pre-hook: inp[0] = act(c_fc(normed)) = h

    def get_mlp_matrices(self, model, layer_idx):
        block = model.model.layers[layer_idx]
        return (block.mlp.c_fc, block.mlp.c_proj,
                block.post_attention_layernorm, block.mlp.act)

    def w_in_fused(self, W_in_module, ln_w):
        # c_fc.weight [d_ff, d_model] — no gate split
        return W_in_module.weight.detach() * ln_w

    def svd_w_out(self, W_out_module):
        # c_proj.weight [d_model, d_ff] — same convention as down_proj
        from block_interp.mlp_svd_utils import compute_svd
        return compute_svd("down_proj", down_proj=W_out_module)


# ── Gemma ─────────────────────────────────────────────────────────────────────

class GemmaAdapter(LlamaAdapter):
    """
    Gemma v1 (model_type='gemma').

    Identical to LlamaAdapter except embed_tokens is scaled by sqrt(hidden_size).
    Gemma v1 does NOT have post_attention_layernorm on attn output or
    pre/post_feedforward_layernorm — those appear in Gemma 2 and 3.
    """

    def embed(self, model, input_ids, device):
        m   = model.model
        x   = m.embed_tokens(input_ids) * (model.config.hidden_size ** 0.5)
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        return x, pos


# ── Gemma 2 ───────────────────────────────────────────────────────────────────

class Gemma2Adapter(LlamaAdapter):
    """
    Gemma 2 family (model_type='gemma2').

    Differences from LlamaAdapter:
    - model.embed_tokens is a plain nn.Embedding; Gemma2Model.forward scales its
      output by sqrt(hidden_size) before the first decoder layer
    - post_attention_layernorm is applied to the attn output (post-norm on attn)
    - MLP uses pre_feedforward_layernorm (not post_attention_layernorm)
    - post_feedforward_layernorm is applied after down_proj, before residual add
    - pretrained configs may request attention/final logit soft-caps; this project
      disables both at model load and returns raw logits for circuit analysis
    """

    def embed(self, model, input_ids, device):
        m   = model.model
        x   = m.embed_tokens(input_ids)
        # Match Gemma2Model.forward exactly.  Constructing the normalizer in the
        # hidden-state dtype matters for fp16/bf16 (e.g. sqrt(3072) rounds to
        # 55.5 in fp16); multiplying by a Python float is not equivalent there.
        normalizer = torch.tensor(
            model.config.hidden_size ** 0.5,
            dtype=x.dtype,
            device=x.device,
        )
        x   = x * normalizer
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        return x, pos

    def final_out(self, model, x):
        """Final norm + lm_head, intentionally before final logit soft-capping."""
        return model.lm_head(model.model.norm(x))

    def attn_out(self, block, x, position_ids, pos_emb=None):
        normed = block.input_layernorm(x)
        attn_mask = _make_gemma_attention_mask(block, normed, position_ids)
        out    = block.self_attn(
            hidden_states=normed,
            attention_mask=attn_mask,
            position_embeddings=pos_emb,
            use_cache=False,
        )
        return block.post_attention_layernorm(out[0])

    def mlp_ln(self, block, x):
        return block.pre_feedforward_layernorm(x)

    def magnitude_preserving_post_attn_norm(self, block, replacement_raw, full_raw_ref):
        return _magnitude_preserving_norm(
            block.post_attention_layernorm, replacement_raw, full_raw_ref
        )

    def mlp_proj(self, block, h):
        return block.mlp.down_proj(h)

    def post_mlp_norm(self, block, x):
        return block.post_feedforward_layernorm(x)

    def magnitude_preserving_post_mlp_norm(self, block, replacement_raw, full_down_ref):
        """Apply post-FFN RMSNorm without discarding replacement_raw's relative size.

        The input is first norm-matched to the complete down_proj activation so RMSNorm
        operates at the layer's native scale. After RMSNorm, the original raw-norm ratio
        is multiplied back. The ratio is intentionally not clamped: values above one are
        part of the reconstructed vector's magnitude, not a numerical error.
        """
        return _magnitude_preserving_norm(
            block.post_feedforward_layernorm, replacement_raw, full_down_ref
        )

    def full_mlp(self, block, x):
        normed = block.pre_feedforward_layernorm(x)
        h      = block.mlp.act_fn(block.mlp.gate_proj(normed)) * block.mlp.up_proj(normed)
        return block.post_feedforward_layernorm(block.mlp.down_proj(h))

    def get_mlp_matrices(self, model, layer_idx):
        block = model.model.layers[layer_idx]
        return (block.mlp.gate_proj, block.mlp.down_proj,
                block.pre_feedforward_layernorm, block.mlp.act_fn)


# ── Gemma 3 ───────────────────────────────────────────────────────────────────

class Gemma3Adapter(Gemma2Adapter):
    """
    Gemma 3 family (model_type='gemma3' / 'gemma3_text').

    Same norm structure as Gemma 2 (post_attention_layernorm on attn output,
    pre_feedforward_layernorm as MLP input, post_feedforward_layernorm after
    down_proj).  Unlike Gemma 2 in the installed transformers version,
    Gemma3TextScaledWordEmbedding applies sqrt(hidden_size) internally, so
    embed() must not add another scale.

    Key difference from Gemma 2: global and local_sliding layers use RoPE with
    different frequency bases (rope_theta vs rope_local_base_freq), so a single
    shared rotary embedding is wrong for one or the other. Two transformers API
    shapes have existed for this:
      - current (transformers>=4.5x installed here): two separate modules,
        model.model.rotary_emb (global) and model.model.rotary_emb_local
        (local), each called as rotary(x, position_ids=...); per-layer choice
        is block.self_attn.is_sliding (True -> local).
      - older: a single model.model.rotary_emb whose .forward takes an
        explicit layer_type=("global"|"local_sliding") argument instead.
    prepare_position_embeddings detects which shape is present and computes
    both variants once per forward pass; attn_out picks the right one per
    layer. Silently using one shared rotary for every layer (the previous
    behaviour when neither shape matched) makes attention outputs diverge
    from the real model starting at layer 0 — verified by comparing this
    adapter's hand-rolled forward against real per-layer hook captures.
    """

    def embed(self, model, input_ids, device):
        # Gemma3TextScaledWordEmbedding performs the sqrt(hidden_size) scale.
        m   = model.model
        x   = m.embed_tokens(input_ids)
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        return x, pos

    def prepare_position_embeddings(self, model, x, position_ids):
        m = model.model
        rotary = getattr(m, "rotary_emb", None)
        if rotary is None:
            return None

        rotary_local = getattr(m, "rotary_emb_local", None)
        if rotary_local is not None:
            return {
                "global": rotary(x, position_ids=position_ids),
                "local":  rotary_local(x, position_ids=position_ids),
            }

        import inspect
        sig = inspect.signature(rotary.forward)
        if "layer_type" in sig.parameters:
            self._gemma3_rotary = rotary
            return None
        return rotary(x, position_ids=position_ids)

    def attn_out(self, block, x, position_ids, pos_emb=None):
        if isinstance(pos_emb, dict):
            is_sliding = getattr(block.self_attn, "is_sliding", False)
            pos_emb = pos_emb["local"] if is_sliding else pos_emb["global"]
        elif pos_emb is None:
            rotary     = getattr(self, "_gemma3_rotary", None)
            layer_type = getattr(block.self_attn, "layer_type", None)
            if rotary is not None and layer_type is not None:
                normed  = block.input_layernorm(x)
                pos_emb = rotary(normed, position_ids=position_ids,
                                 layer_type=layer_type)
        normed = block.input_layernorm(x)
        attn_mask = _make_gemma_attention_mask(block, normed, position_ids)
        out = block.self_attn(
            hidden_states=normed,
            attention_mask=attn_mask,
            position_embeddings=pos_emb,
            use_cache=False,
        )
        return block.post_attention_layernorm(out[0])


# ── GPT-Neo ───────────────────────────────────────────────────────────────────

def _make_gemma_attention_mask(block, x, position_ids):
    """Build the eager mask Gemma's model-level forward supplies per layer."""
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    config = block.config
    if bool(getattr(config, "use_bidirectional_attention", False)):
        raise ValueError(
            "bidirectional Gemma-3 attention is not supported by the causal circuit runner"
        )
    cache_position = position_ids[0] if position_ids.ndim == 2 else position_ids
    kwargs = {
        "config": config,
        "input_embeds": x,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    attention_type = getattr(block, "attention_type", "full_attention")
    if attention_type in {"sliding_attention", "local_sliding"}:
        return create_sliding_window_causal_mask(**kwargs)
    return create_causal_mask(**kwargs)


def _make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return [1, 1, seq, seq] additive causal bias: 0 on/below diagonal, -inf above."""
    min_val = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


class GPTNeoAdapter(GPT2Adapter):
    """
    EleutherAI GPT-Neo family (model_type='gpt_neo').

    Same block structure as GPT-2 (model.transformer.h, ln_1/ln_2) but the
    MLP uses nn.Linear (weight [d_ff, d_model]) rather than Conv1D (weight
    [d_model, d_ff]).  Only w_in_fused / svd_w_out need to change.

    Covers: domain models based on GPT-Neo, including all three
    ncfrey/ChemGPT checkpoints used by this repository.
    """

    def w_in_fused(self, W_in_module, ln_w):
        # Linear weight is already [d_ff, d_model] — no transpose needed
        return W_in_module.weight.detach() * ln_w

    def svd_w_out(self, W_out_module):
        # c_proj is Linear: weight [d_model, d_ff] → same as LlamaAdapter.down_proj
        from block_interp.mlp_svd_utils import compute_svd
        return compute_svd("down_proj", down_proj=W_out_module)


# ── GPT-NeoX ──────────────────────────────────────────────────────────────────

class GPTNeoXAdapter(ModelAdapter):
    """
    EleutherAI GPT-NeoX family (model_type='gpt_neox').

    Used by GPT-NeoX checkpoints (distinct from the GPT-Neo-based ChemGPT
    checkpoints), etc.

    Architecture
    ------------
      Blocks     : model.gpt_neox.layers
      Pre-attn LN: block.input_layernorm
      Attention  : block.attention  (shared model-level RoPE tuple)
      Pre-MLP LN : block.post_attention_layernorm
      MLP        : dense_h_to_4h → act → dense_4h_to_h  (GELU)
      Final LN   : model.gpt_neox.final_layer_norm
      LM head    : model.embed_out

    Note: an explicit 4D causal mask is passed to block.attention because the
    GPT-NeoX eager-attention path does not apply a causal mask when
    attention_mask=None (unlike GPT-2 which stores a causal bias buffer).
    Standard GPT-NeoX uses parallel attention/MLP residuals; mlp_ln_input keeps
    that behaviour while still supporting configs that request sequential mode.
    """

    def embed(self, model, input_ids, device):
        x   = model.gpt_neox.embed_in(input_ids)
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        return x, pos

    def get_blocks(self, model):
        self._use_parallel_residual = getattr(
            model.config, "use_parallel_residual", False
        )
        return model.gpt_neox.layers

    def mlp_ln_input(self, x, x_pre_attn):
        # GPT-NeoX normally uses a parallel residual block:
        # x + attn(input_ln(x)) + mlp(post_attn_ln(x)).  Some configs opt into
        # the sequential variant, so retain the model-configured behaviour.
        return x_pre_attn if getattr(self, "_use_parallel_residual", False) else x

    def prepare_position_embeddings(self, model, x, position_ids):
        # Current Transformers computes one shared (cos, sin) tuple in
        # GPTNeoXModel.forward and requires every attention layer to receive it.
        # Older releases computed RoPE internally; passing the tuple explicitly
        # matches the current model-level forward contract.
        return model.gpt_neox.rotary_emb(x, position_ids)

    def attn_out(self, block, x, position_ids, pos_emb=None):
        normed    = block.input_layernorm(x)
        attn_mask = _make_causal_mask(x.shape[1], x.dtype, x.device)
        out = block.attention(
            hidden_states=normed,
            attention_mask=attn_mask,
            position_embeddings=pos_emb,
            use_cache=False,
        )
        return out[0]

    def mlp_ln(self, block, x):
        return block.post_attention_layernorm(x)

    def mlp_h(self, block, normed):
        return block.mlp.act(block.mlp.dense_h_to_4h(normed))

    def mlp_proj(self, block, h):
        return block.mlp.dense_4h_to_h(h)

    def full_mlp(self, block, x):
        normed = block.post_attention_layernorm(x)
        return block.mlp.dense_4h_to_h(block.mlp.act(block.mlp.dense_h_to_4h(normed)))

    def final_out(self, model, x):
        return model.embed_out(model.gpt_neox.final_layer_norm(x))

    def h_hook_module(self, block):
        return block.mlp.dense_4h_to_h  # pre-hook: inp[0] = act(dense_h_to_4h) = h

    def get_mlp_matrices(self, model, layer_idx):
        block = model.gpt_neox.layers[layer_idx]
        return (block.mlp.dense_h_to_4h, block.mlp.dense_4h_to_h,
                block.post_attention_layernorm, block.mlp.act)

    def w_in_fused(self, W_in_module, ln_w):
        # dense_h_to_4h.weight [d_ff, d_model]
        return W_in_module.weight.detach() * ln_w

    def svd_w_out(self, W_out_module):
        from block_interp.mlp_svd_utils import compute_svd
        return compute_svd("down_proj", down_proj=W_out_module)

    def get_input_embed_weight(self, model):
        return model.gpt_neox.embed_in.weight.detach()

    def get_output_embed_weight(self, model):
        return model.embed_out.weight.detach()

    def apply_final_ln(self, model, x):
        return model.gpt_neox.final_layer_norm(x)


# ── BioGPT ────────────────────────────────────────────────────────────────────

class BioGPTAdapter(ModelAdapter):
    """
    Microsoft BioGPT (model_type='biogpt', microsoft/biogpt).

    Architecture
    ------------
      Embedding  : embed_tokens + embed_positions (learned, not RoPE)
      Blocks     : model.biogpt.layers
      Pre-attn LN: block.self_attn_layer_norm
      Attention  : block.self_attn  (BioGptAttention, no RoPE)
      Pre-MLP LN : block.final_layer_norm
      MLP        : fc1 → activation_fn → fc2  (GELU, no gate projection)
      Final LN   : model.biogpt.layer_norm
      LM head    : model.output_projection

    A 4D causal mask is passed to block.self_attn because the BioGPT eager
    attention path requires an explicit mask (unlike SDPA path which uses
    is_causal=True).
    """

    def embed(self, model, input_ids, device):
        m = model.biogpt
        # Learned positional embeddings require an attention mask (all-ones for generation)
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
        inputs_embeds = m.embed_tokens(input_ids)
        positions     = m.embed_positions(attn_mask).to(device)
        x   = inputs_embeds + positions
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        return x, pos

    def get_blocks(self, model):
        return model.biogpt.layers

    def prepare_position_embeddings(self, model, x, position_ids):
        return None  # BioGPT uses learned positional embeddings baked into embed()

    def attn_out(self, block, x, position_ids, pos_emb=None):
        normed    = block.self_attn_layer_norm(x)
        attn_mask = _make_causal_mask(x.shape[1], x.dtype, x.device)
        out = block.self_attn(hidden_states=normed, attention_mask=attn_mask)
        return out[0]

    def mlp_ln(self, block, x):
        return block.final_layer_norm(x)

    def mlp_h(self, block, normed):
        return block.activation_fn(block.fc1(normed))

    def mlp_proj(self, block, h):
        return block.fc2(h)

    def full_mlp(self, block, x):
        normed = block.final_layer_norm(x)
        return block.fc2(block.activation_fn(block.fc1(normed)))

    def final_out(self, model, x):
        return model.output_projection(model.biogpt.layer_norm(x))

    def h_hook_module(self, block):
        return block.fc2  # pre-hook: inp[0] = activation_fn(fc1(normed)) = h

    def get_mlp_matrices(self, model, layer_idx):
        block = model.biogpt.layers[layer_idx]
        return (block.fc1, block.fc2, block.final_layer_norm, block.activation_fn)

    def w_in_fused(self, W_in_module, ln_w):
        # fc1.weight [d_ff, d_model] (Linear)
        return W_in_module.weight.detach() * ln_w

    def svd_w_out(self, W_out_module):
        from block_interp.mlp_svd_utils import compute_svd
        return compute_svd("down_proj", down_proj=W_out_module)

    def get_input_embed_weight(self, model):
        return model.biogpt.embed_tokens.weight.detach()

    def get_output_embed_weight(self, model):
        return model.output_projection.weight.detach()

    def apply_final_ln(self, model, x):
        return model.biogpt.layer_norm(x)


# ── ProGen2 ───────────────────────────────────────────────────────────────────

class ProGenAdapter(ModelAdapter):
    """
    ProGen2 family (model_type='progen', trust_remote_code=True).

    Used by: hugohrban/progen2-small and other ProGen2 variants.

    Architecture (parallel transformer — differs from GPT-2/Llama):
      Blocks     : model.transformer.h
      Embedding  : model.transformer.wte  (no learned positional; RoPE inside attn)
      Per block  : shared ln_1 feeds both attn and MLP in parallel
                   x = x + attn(ln_1(x)) + mlp(ln_1(x))   (NO ln_2)
      MLP        : fc_in → act → fc_out  (no gate projection)
      Causal mask: built into ProGenAttention.bias buffer — no external mask needed
      Final LN   : model.transformer.ln_f
      LM head    : model.lm_head  (not tied to wte)

    Weight shapes:
      fc_in.weight  [d_ff, d_model]  nn.Linear — same as LlamaAdapter.gate_proj
      fc_out.weight [d_model, d_ff]  nn.Linear — same as LlamaAdapter.down_proj

    Note on parallel vs sequential forward:
      precompute_within_ske reads weights only (no forward pass), so parallel/
      sequential ordering does not affect the cache. The shared forward loops
      (base.py, skeleton_mlp_forward*.py) accumulate x sequentially per layer
      (x = x + attn_out, then x = x + mlp_out) — mathematically that sum is the
      same regardless of order, so it doesn't matter for ProGen2 either. What
      does matter is what mlp_ln reads: this class overrides mlp_ln_input to
      return the pre-attention x snapshot (see ModelAdapter.mlp_ln_input), so
      the MLP branch sees the same input the real parallel block gives it,
      instead of the post-attention x a sequential architecture would use.
    """

    def embed(self, model, input_ids, device):
        t   = model.transformer
        x   = t.drop(t.wte(input_ids))
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        return x, pos

    def get_blocks(self, model):
        return model.transformer.h

    def prepare_position_embeddings(self, model, x, position_ids):
        return None  # RoPE computed inside each ProGenAttention block

    def attn_out(self, block, x, position_ids, pos_emb=None):
        # ProGenAttention carries its own causal bias buffer — no external mask needed
        return block.attn(block.ln_1(x))[0]

    def mlp_ln_input(self, x, x_pre_attn):
        """ProGen2 is a parallel block: x_new = x + attn(ln_1(x)) + mlp(ln_1(x)),
        both branches reading the SAME pre-attention x. The shared sequential
        forward loops (base.py, skeleton_mlp_forward*.py) otherwise feed the MLP
        branch x *after* the attention residual has already been added — wrong
        for this architecture. Returning x_pre_attn here (instead of the base
        class's x) makes mlp_ln/full_mlp read the correct, pre-attention input.
        """
        return x_pre_attn

    def mlp_ln(self, block, x):
        return block.ln_1(x)  # shared with attn; no ln_2

    def mlp_h(self, block, normed):
        return block.mlp.act(block.mlp.fc_in(normed))

    def mlp_proj(self, block, h):
        return block.mlp.fc_out(h)

    def full_mlp(self, block, x):
        normed = block.ln_1(x)
        return block.mlp.fc_out(block.mlp.act(block.mlp.fc_in(normed)))

    def final_out(self, model, x):
        return model.lm_head(model.transformer.ln_f(x))

    def h_hook_module(self, block):
        return block.mlp.fc_out  # pre-hook: inp[0] = act(fc_in(normed)) = h

    def get_mlp_matrices(self, model, layer_idx):
        block = model.transformer.h[layer_idx]
        return (block.mlp.fc_in, block.mlp.fc_out, block.ln_1, block.mlp.act)

    def w_in_fused(self, W_in_module, ln_w):
        # fc_in.weight [d_ff, d_model] (nn.Linear) — same as LlamaAdapter.gate_proj
        return W_in_module.weight.detach() * ln_w

    def svd_w_out(self, W_out_module):
        # fc_out.weight [d_model, d_ff] — same convention as LlamaAdapter.down_proj
        from block_interp.mlp_svd_utils import compute_svd
        return compute_svd("down_proj", down_proj=W_out_module)

    def get_input_embed_weight(self, model):
        return model.transformer.wte.weight.detach()

    def get_output_embed_weight(self, model):
        return model.lm_head.weight.detach()

    def apply_final_ln(self, model, x):
        return model.transformer.ln_f(x)


# ── Registry and factory ──────────────────────────────────────────────────────

_ADAPTER_REGISTRY: dict[str, type] = {}


def _register_adapter(model_types, adapter_class):
    for t in model_types:
        _ADAPTER_REGISTRY[t] = adapter_class


_register_adapter(["gpt2"], GPT2Adapter)
_register_adapter(list(_LLAMA_TYPES), LlamaAdapter)
_register_adapter(["starcoder2"], StarCoder2Adapter)
_register_adapter(["gemma"], GemmaAdapter)
_register_adapter(["gemma3", "gemma3_text"], Gemma3Adapter)
_register_adapter(["gemma2"], Gemma2Adapter)
_register_adapter(["phi3"], Phi3Adapter)
_register_adapter(["internlm", "internlm2"], InternLM2Adapter)
_register_adapter(["gpt_neo"], GPTNeoAdapter)
_register_adapter(["gpt_neox"], GPTNeoXAdapter)
_register_adapter(["biogpt"], BioGPTAdapter)
_register_adapter(["progen"], ProGenAdapter)


def get_adapter(model) -> ModelAdapter:
    """
    Auto-detect the correct ModelAdapter from model.config.model_type.

    Falls back to attribute checks if the config type is unregistered:
      • model.transformer.h  → GPT2Adapter
      • model.model.layers   → LlamaAdapter

    Raises ValueError for unsupported architectures.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[model_type]()

    # Fallback via attribute checks
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return GPT2Adapter()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return LlamaAdapter()

    raise ValueError(
        f"Unsupported model type: {model_type!r}. "
        "Implement a ModelAdapter subclass and call _register_adapter() to add support."
    )
