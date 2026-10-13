# Attention + MLP dynamic circuits

This directory implements a new circuit definition without changing the older
`circuit_skeleton/` or `ske_pre_valid/` code.

See `NORM_AUDIT.md` for the per-family normalization audit.

## Static cache

Precompute per-head OV factors in the dedicated Attention cache root:

```bash
python pre_compute/precompute_attention_ov.py \
  --model_name openai-community/gpt2 \
  --cache_base /pfs/work9/workspace/scratch/hd_nf184-nana-cache/pre_compute_skeleton_attn_cache
```

This writes `<cache_base>/<model>/attention_ov_cache/{metadata.json,layer_XXX.pt}`.
Each layer stores `U/S/V`, V/O bias constants, query-to-KV mapping, dimensions,
configuration identity and norm metadata. Dynamic QK maps, weighted inputs,
targets and top-k selections are intentionally not cached.

For the last query position `q`, native QK/RoPE/masking/softmax first forms one
weighted input per head. Only the content map is decomposed:

```text
x_bar[q,h] = sum_s A[h,q,s] * x_norm[s]
W_OV[h] = W_V[g(h)] @ W_O[h] = U[h] S[h] V[h]
score[h,r] = <x_bar[q,h], U[h,:,r]> * S[h,r]
```

`g(h)` is the query-to-KV mapping, so MHA, GQA and MQA use the same definition.
With a target/unembedding direction, the score is additionally multiplied by
`<V[h,r], target>`. Source-token contributions remain available diagnostically,
but are not separate top-k circuit paths.

## Norm boundaries

`architecture.py` records the pre/post norm and residual layout for every
checkpoint in the repository's model list.  The important rule is that circuit
selection happens on raw branch projections, while native post-branch norms are
applied after preserve/ablation reconstruction:

- GPT-2/Llama/BioGPT: pre-norm branches, no post-branch norm.
- Gemma 2/3: both attention and MLP have post-branch RMSNorm.
- ProGen2: attention and MLP are parallel and read the same `ln_1(x)`.
- GPT-NeoX: parallel versus sequential is controlled by model config.

`BranchNorms` enforces the intervention order. In particular, Gemma ablation is
`post_norm(full_raw - selected_raw)`, never
`post_norm(full_raw) - post_norm(selected_raw)`. Native and
magnitude-preserving post-norm policies are both available explicitly.

The dataset runner exposes `--post_norm_mode` for Gemma 2/3:

- `native`: apply the checkpoint's post-RMSNorm literally. A very small sparse
  vector is therefore normalised to the branch's RMS scale.
- `magnitude_preserving` (experiment default): norm-match the sparse/complement
  raw vector to the full raw branch, apply the native RMSNorm (including
  Gemma's `1 + weight`), then restore its original sparse/full norm ratio.

The same policy is applied symmetrically to Gemma Attention and MLP post norms.
It has no effect on architectures without a post-branch norm.

Per-head SVD is unconditional. `attention_budget=global` means all
`(head,OV-mode)` paths compete for one top-k budget; `per_head` means
each already-factorised head receives its own equal top-k budget.
For both Attention and MLP, top-k ranks the signed target contribution directly
from largest to smallest. Positive paths therefore rank before negative paths,
but the largest negative paths fill the remaining budget when fewer than k
positive paths exist. Absolute contribution magnitude is never used.

Before an experiment, call `assert_audited_for_circuits(model)`. It walks every
actual decoder block, checks all four branch norm boundaries, records internal
Q/K norms (Qwen3 and Gemma3), epsilon and bias, and empirically distinguishes
ordinary RMSNorm (`weight`) from Gemma-style RMSNorm (`1 + weight`). Unknown or
custom normalization fails before any circuit result is produced.

Runtime collection should execute native model attention with
`output_attentions=True` (and an eager backend when required), hook the actual V
projection input as `x_norm`, and feed the returned probabilities to
`AttentionOVSkeleton`.  This preserves the model's own QK, RoPE, causal/sliding
mask and softmax implementation.

`collect_native_attention(model, layer_idx, **model_inputs)` implements this
collection path. It installs temporary hooks, requests native attention maps,
and removes every hook even if the forward raises.

## Experiment modes

Use all three modes:

1. `attention` and `mlp` separately are necessary attribution controls.
2. `attention_mlp` is the primary end-to-end circuit. For sequential blocks,
   attention is reconstructed/ablated first and MLP is then evaluated from the
   resulting residual state. For ProGen2/parallel GPT-NeoX, both branches must
   read the same pre-branch state before their residual contributions are added.

Reporting only the joint mode cannot distinguish direct Attention effects from
downstream MLP compensation; reporting only separate modes misses their causal
composition.
