"""Prompt construction rules for causal next-token continuation.

Chat/instruction wrappers are intentionally disabled for this experiment: every
prompt is a prefix whose immediate continuation is scored.  Domain models may
still need a model-native sequence-start prefix, and some base LMs need their
BOS inserted by the tokenizer at token-id level.
"""

# Bump this whenever any rule that changes inference input IDs changes.  Dynamic
# label files use it to invalidate resumable data, while experiment-result files
# include ``pf{version}`` in their names so stale results cannot be skipped or
# mixed with the current prompt format.
PROMPT_FORMAT_VERSION = 3
PROMPT_FORMAT_TAG = f"pf{PROMPT_FORMAT_VERSION}"

# Model-generated target labels and dynamic intervention inference must see the
# same token prefix.  Keep this in the shared prompt policy rather than using a
# literal in only one pipeline.
NEXT_TOKEN_MAX_LENGTH = 512

# key → wrapper string; {prompt} is replaced with the raw input text
CHAT_TEMPLATES = {
    "gemma_it":  "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    "phi3":      "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
    "llama3_it": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                  "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama2_it": "[INST] {prompt} [/INST]",
    "internlm2": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "qwen2_it":  "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "qwen3_it":  "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",  # identical ChatML format
}

# model name → template key.  All entries are None for the continuation task.
# CHAT_TEMPLATES is retained only as reference data; enabling one here changes the
# task from continuation to assistant response generation and is not supported by
# the experiment pipelines.
MODEL_TEMPLATE = {
    "meta-llama/Llama-3.2-1B-Instruct":  None,
    "meta-llama/Llama-2-7b-chat-hf":     None,
    "internlm/internlm2-chat-7b":        None,
    "internlm/internlm2_5-7b-chat":      None,
    "Qwen/Qwen2.5-1.5B-Instruct":        None,
    # google/gemma-3-1b-it, google/gemma-2-9b-it, Qwen/Qwen3-0.6B, microsoft/Phi-3-mini-4k-instruct:
    # officially chat/instruct models, but wrapping the natural_language continuation
    # fragment in a chat template turns "continue this text" into "reply to
    # this message" — confirmed empirically on next_token_short.json: Qwen3-0.6B
    # predicted "<think>" as the kept target 100% of the time (6719/6719),
    # gemma-3-1b-it predicted "<end_of_turn>" 55.1% of the time plus chat-filler
    # openers ("Okay" 16.1%, "Here", "Let"). Phi-3 doesn't degenerate to a meta
    # token, but still answers instead of continuing (test.py, prompt "The cat
    # looks very": raw -> "happy", templated -> "The" — restates the sentence
    # rather than completing it). Feeding the raw prompt (no template) gives
    # ordinary continuation words instead in all four cases. Same fix as
    # liyinghong/BioQwen-* below: leave unregistered (None).
    "google/gemma-3-1b-it":               None,
    "google/gemma-2-9b-it":               None,
    "Qwen/Qwen3-0.6B":                    None,
    "microsoft/Phi-3-mini-4k-instruct":   None,
    "google/gemma-2-9b":                  None,
    # The following are base models that do not require a template
    "openai-community/gpt2":             None,
    "openai-community/gpt2-medium":      None,
    "google/gemma-2b-it":                 None,
    "Qwen/Qwen2.5-1.5B":                 None,
    "meta-llama/Llama-3.2-1B":           None,
    "meta-llama/Llama-2-7b-hf":          None,
    "meta-llama/Llama-2-13b-hf":         None,
    "mistralai/Mistral-7B-v0.1":          None,
    "mistralai/Mistral-7B-v0.3":          None,
    "deepseek-ai/deepseek-llm-7b-base":   None,
    "deepseek-ai/deepseek-coder-6.7b-base": None,
    "codellama/CodeLlama-7b-hf":         None,
    "bigcode/starcoder2-3b":             None,
    "internlm/internlm2-7b":             None,
    # ── Domain-specific biology / chemistry models ────────────────────────────
    # All are base/decoder models; no chat template needed.
    "nferruz/ProtGPT2":                  None,   # GPT-2 arch, protein sequences
    "hugohrban/progen2-small":           None,   # ProGen2 (trust_remote_code=True)
    "hugohrban/progen2-medium":          None,
    "hugohrban/progen2-large":           None,
    "ncfrey/ChemGPT-4.7M":              None,   # GPT-Neo arch, SELFIES
    "ncfrey/ChemGPT-1.2B":              None,
    "ncfrey/ChemGPT-19M":               None,
    "seyonec/ChemBERTa-zinc-base-v1":   None,   # encoder — likely unsupported
    "entropy/gpt2-xl-molecular":         None,   # MolGPT variant (GPT-2 arch)
    "entropy/gpt2_zinc_87m":            None,   # GPT-2 arch, SMILES (ZINC dataset, vocab=2707)
    "Ilyabarigou/DrugGPT":              None,   # GPT-2 arch, drug SMILES
    "ChemFM/ChemFM-1B":                  None,   # Llama arch, SMILES (UniChem, vocab=320, no BOS)
    "jonghyunlee/MolGPT_pretrained-by-ZINC15":              None,  # GPT-2 arch, SMILES (ctx=128)
    "jonghyunlee/MolGPT_long_context_pretrained-by-ZINC15": None,  # GPT-2 arch, SMILES (ctx=256)
    "AI4Chem/ChemLLM-7B-Chat":           None,
    "microsoft/biogpt":                  None,   # BioGPT arch, biomedical text
    "microsoft/BioGPT-Large":            None,
    # liyinghong/BioQwen-0.5B/1.8B: Qwen2 arch, officially a ChatML chat model, but
    # empirically wrapping it in the chat template is actively harmful for this task —
    # it makes the model treat the biomedical_nl short/long fragment as a user question
    # and start composing an unrelated assistant reply (kept-rate 38%→2% on 200 real
    # prompts, top predictions become things like 'The'/'目的'/'我们' instead of sentence
    # continuations). Left unregistered (None = base/no template) so it's fed the raw
    # continuation the same way as the domain's other base models.
    "liyinghong/BioQwen-0.5B":           None,
    "liyinghong/BioQwen-1.8B":           None,
}


# ── Domain-specific sequence prefixes ──────────────────────────────────────────
# NOT chat templates — these are the raw prefix each model's OWN pretraining format
# requires before real sequence content (protein/chemistry base models), so that
# next-token prediction reflects the model's trained distribution instead of an
# out-of-distribution input. Single source of truth: shared by dataset generation
# (dataset_gene/dymn_dst_gene/gen_next_token_labels.py) and the circuit-skeleton
# experiment runners (ske_pre_valid/static_skeleton.py, scripts_ske_cir/run_skeleton_pred.py).

# ProGen2: sequences were always trained with a leading 1-char orientation tag —
# "1" for N→C, "2" for C→N. We always use "1" (standard forward reading direction).
PROGEN2_MODELS = frozenset({
    "hugohrban/progen2-small", "hugohrban/progen2-medium",
    "hugohrban/progen2-large", "hugohrban/progen2-base",
})

# MolGPT (Bagal et al. 2021) SMILES models: custom BPE tokenizer whose trained
# post-processor auto-adds <bos> when add_special_tokens=True (confirmed via
# tokenizer(text) vs tokenizer(text, add_special_tokens=False)) — our pipeline always
# calls with add_special_tokens=False, so the BOS must be prepended manually here.
MOLGPT_ZINC15_MODELS = frozenset({
    "jonghyunlee/MolGPT_pretrained-by-ZINC15",
    "jonghyunlee/MolGPT_long_context_pretrained-by-ZINC15",
})

# ZymCTRL is EC-number-conditional: training format is "{ec}<sep><start>{sequence}...".
# Our sampled sequences carry no real EC annotation, so we use a fixed placeholder EC
# tag — keeps the input in-distribution format-wise even though the tag isn't the
# sequence's true enzyme class.
ZYMCTRL_EC_PLACEHOLDER = "3.5.5.1"

# ProtGPT2 has no dedicated BOS; training data uses EOS (<|endoftext|>) as the
# start-of-sequence marker before each protein. UniRef50 sequences were also stored
# FASTA-style, wrapped at 60 characters per line — confirmed empirically: without the
# wrap, argmax next-token predictions on a >~40-residue prompt are "\n" ~90% of the
# time (the model treats the missing line break as overdue); with 60-char wrapping,
# kept-rate on the "long" (100-residue) subset goes from 10.8% to 63.8%.
PROTGPT2_FASTA_WIDTH = 60

# BioGPT has no dedicated BOS either; its native add_special_tokens=True prepends
# </s> (eos_token_id=2, registered as sep_token too) as the start-of-sequence
# marker — bos_token_id=0 ('<s>') is a vocab entry that is never actually used at
# encode time. Confirmed empirically: tokenizer.encode(x, add_special_tokens=True)
# for both microsoft/biogpt and microsoft/BioGPT-Large starts with id 2, not 0.
BIOGPT_MODELS = frozenset({"microsoft/biogpt", "microsoft/BioGPT-Large"})

# ChemGPT's shipped tokenizer post-processor is BERT-style:
#   [CLS] + sequence + [SEP]
# For causal continuation, [CLS] is the trained sequence-start marker but [SEP]
# would move prediction to "after end of sequence".  We therefore add only the
# literal registered [CLS] token and keep add_special_tokens=False at call sites.
CHEMGPT_MODELS = frozenset({
    "ncfrey/ChemGPT-4.7M", "ncfrey/ChemGPT-19M", "ncfrey/ChemGPT-1.2B",
})


def _fasta_wrap(seq: str, width: int = PROTGPT2_FASTA_WIDTH) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def apply_sequence_prefix(prompt: str, model_name: str, tokenizer=None) -> str:
    """Prepend the model-specific pretraining-format prefix a base sequence model expects.

    Distinct from the chat templates above: these aren't instruction wrappers, they're
    the raw sequence-format convention (orientation tag / conditioning tag / BOS) each
    model was trained with. `tokenizer` is only needed for models whose prefix is a
    dynamic special token (e.g. gpt2_zinc_87m's bos_token) rather than a literal string;
    when it's needed but not given, the prompt is returned unprefixed.
    """
    if model_name == "nferruz/ProtGPT2":
        return "<|endoftext|>" + _fasta_wrap(prompt)
    if model_name in PROGEN2_MODELS:
        return "1" + prompt
    if model_name == "AI4PD/ZymCTRL":
        return f"{ZYMCTRL_EC_PLACEHOLDER}<sep><start>{prompt}"
    if model_name == "entropy/gpt2_zinc_87m" and tokenizer is not None:
        # GPT2TokenizerFast here adds no special tokens even with add_special_tokens=True
        # (per the model card) — the BOS must be prepended manually.
        return tokenizer.bos_token + prompt
    if model_name in MOLGPT_ZINC15_MODELS and tokenizer is not None:
        return tokenizer.bos_token + prompt
    if model_name in BIOGPT_MODELS and tokenizer is not None:
        return tokenizer.eos_token + prompt
    if model_name in CHEMGPT_MODELS:
        return "[CLS]" + prompt
    # ChemFM/ChemFM-1B: tokenizer defines no bos_token at all (only eos) — no prefix
    # convention to replicate, prompt is used as-is (falls through to `return prompt`).
    return prompt


def apply_template(prompt: str, model_name: str, tokenizer=None) -> str:
    """Build a causal-continuation prompt with any model-native sequence prefix.

    Chat wrappers are deliberately absent: the next token must continue ``prompt``
    rather than begin an assistant response.  MODEL_TEMPLATE is kept for reporting
    and compatibility, but all supported continuation models map to None.

    BOS restoration for plain NL/code base models (Llama/Gemma/Mistral/DeepSeek)
    is NOT done here as a string edit — see needs_true_add_special_tokens() below.
    A prior version of this function prepended tokenizer.bos_token as a literal
    string, which is wrong for SentencePiece tokenizers whose normalizer treats
    the first real content token differently depending on whether it sits at
    true string-start or right after a spliced-in special-token string (confirmed
    on Mistral-7B: '<s>' + 'It met the' encodes 'It' as the bare token id 1313,
    while the tokenizer's own add_special_tokens=True gives the space-marked
    '▁It' id 661 — same BOS id, different and wrong content tokens). Restoring
    BOS correctly requires operating at the token-id level, which the actual
    tokenizer(...) call sites do via add_special_tokens (see below), not this
    string-building function.
    """
    prompt = apply_sequence_prefix(prompt, model_name, tokenizer)
    key = MODEL_TEMPLATE.get(model_name)
    if key is not None:
        raise ValueError(
            f"Chat template {key!r} is incompatible with next-token continuation "
            f"for {model_name}; set MODEL_TEMPLATE[{model_name!r}] = None"
        )
    return prompt


def needs_true_add_special_tokens(model_name: str, tokenizer) -> bool:
    """True if the tokenizer(...) call for this model should use
    add_special_tokens=True instead of the project-wide default False.

    This project defaults every tokenizer(...) call to add_special_tokens=False
    (see gen_next_token_labels.py's predict_next_tokens) to dodge tokenizers with
    a stray post-processor that inserts unwanted specials (e.g. ChemGPT's BERT-
    style [CLS]/[SEP]). But several base LMs (Llama-2/3, Mistral, Gemma, DeepSeek)
    were trained to always see a leading BOS, which that default then silently
    drops. Restoring it must happen at the token-id level (add_special_tokens=True
    on the *unprefixed* prompt), not by splicing a BOS string into the prompt text
    — see apply_template's docstring for why the string route is wrong for some
    SentencePiece tokenizers (confirmed on Mistral).

    Only True when apply_sequence_prefix leaves any prompt unchanged for this
    model_name (models with their own explicit prefix — ProtGPT2/ProGen2/ZymCTRL/
    MolGPT/gpt2_zinc_87m/BioGPT/ChemGPT — already handle their own BOS-equivalent via that
    prefix, verified to round-trip correctly as plain string concatenation for all
    of them, so this must not also fire and double it) AND a probe confirms
    add_special_tokens=True changes *nothing* except prepending bos_token_id (no
    trailing EOS or other side effect — e.g. MolGPT's add_special_tokens=True also
    appends a trailing EOS we don't want, so it must keep using its manual string
    prefix with add_special_tokens=False, not this path).

    apply_sequence_prefix's branch taken depends only on model_name (given a
    tokenizer), not on prompt content, so a fixed probe string is representative
    for both checks below.
    """
    if tokenizer is None or tokenizer.bos_token_id is None:
        return False
    probe = "x"
    if apply_sequence_prefix(probe, model_name, tokenizer) != probe:
        return False
    native    = tokenizer.encode(probe, add_special_tokens=True)
    id_splice = [tokenizer.bos_token_id] + tokenizer.encode(probe, add_special_tokens=False)
    return native == id_splice
