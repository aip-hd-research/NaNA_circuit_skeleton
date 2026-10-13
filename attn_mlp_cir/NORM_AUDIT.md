# Normalization audit

The saved `model_structure.txt` files and the installed Transformers model
implementations were checked before defining the runtime audit. The circuit
must consume the **actual output of the branch-input norm**; neither LayerNorm
nor RMSNorm is folded into `W_V`, `W_O`, or the MLP matrices.

| Family / examples | Attention input | Attention internal | Attention output | MLP input | MLP output | Residual |
|---|---|---|---|---|---|---|
| GPT-2, BioMedLM, ProtGPT2, ZymCTRL, small ChemGPT, MolGPT | `ln_1` LayerNorm | none | none | `ln_2` LayerNorm | none | sequential |
| Llama 2/3, CodeLlama, DeepSeek-Llama | `input_layernorm` RMSNorm | none | none | `post_attention_layernorm` RMSNorm | none | sequential |
| Mistral | `input_layernorm` RMSNorm | none | none | `post_attention_layernorm` RMSNorm | none | sequential |
| Qwen2 / BioQwen | `input_layernorm` RMSNorm | none | none | `post_attention_layernorm` RMSNorm | none | sequential |
| Qwen3 | `input_layernorm` RMSNorm | `q_norm`, `k_norm` RMSNorm | none | `post_attention_layernorm` RMSNorm | none | sequential |
| StarCoder2 | `input_layernorm` LayerNorm | none | none | `post_attention_layernorm` LayerNorm | none | sequential |
| Gemma 2 | `input_layernorm` RMSNorm | none | `post_attention_layernorm` RMSNorm | `pre_feedforward_layernorm` RMSNorm | `post_feedforward_layernorm` RMSNorm | sequential |
| Gemma 3 | `input_layernorm` RMSNorm | `q_norm`, `k_norm` RMSNorm | `post_attention_layernorm` RMSNorm | `pre_feedforward_layernorm` RMSNorm | `post_feedforward_layernorm` RMSNorm | sequential |
| InternLM2 | `attention_norm` RMSNorm | none | none | `ffn_norm` RMSNorm | none | sequential |
| GPT-NeoX / large ChemGPT | `input_layernorm` LayerNorm | none | none | `post_attention_layernorm` LayerNorm | none | config-dependent |
| BioGPT | `self_attn_layer_norm` LayerNorm | none | none | `final_layer_norm` LayerNorm | none | sequential |
| ProGen2 | shared `ln_1` LayerNorm | none | none | shared `ln_1` LayerNorm | none | parallel |

## RMSNorm conventions

Installed Llama/Qwen2-style implementations compute

```text
x / sqrt(mean(x^2) + eps) * weight
```

whereas installed Gemma 2/3 implementations compute

```text
x / sqrt(mean(x^2) + eps) * (1 + weight)
```

`audit_model_norms` measures this from each loaded module instead of trusting
the table. It also records epsilon and whether a bias exists. The table is an
expected layout; `assert_audited_for_circuits` is the runtime authority.

## Intervention boundary

For a post-norm branch, selection and ablation operate on the raw projection:

```text
preserve = post_norm(selected_raw)
ablation = post_norm(full_raw - selected_raw)
```

For a branch without output norm, `post_norm` is the identity. In a sequential
block the MLP input is computed from the residual after the Attention
intervention. In a parallel block Attention and MLP are both computed from the
same pre-branch residual.

For sparse-circuit experiments, `magnitude_preserving` additionally matches the
replacement to the full raw branch norm before native post-RMSNorm and restores
the raw sparse/full ratio afterwards. This is applied to both Gemma Attention
and Gemma MLP. `native` remains available for literal model interventions.
