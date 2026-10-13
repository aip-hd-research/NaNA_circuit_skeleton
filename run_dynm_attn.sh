#!/usr/bin/env bash
set -euo pipefail

# GPT-2 Attention and Attention-MLP dynamic circuits on the fixed prompt.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${REPO_ROOT}"
CHECKPOINT_CACHE_ROOT="${REPO_ROOT}/cache"
export HF_HOME="${CHECKPOINT_CACHE_ROOT}/model_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export SKELETON_CACHE_BASE="${CHECKPOINT_CACHE_ROOT}/pre_compute_skeleton_cache"
export ATTENTION_OV_CACHE_BASE="${CHECKPOINT_CACHE_ROOT}/pre_compute_skeleton_attn_cache"

MODEL="openai-community/gpt2"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TYPES=(attn_and_mlp attn)
K_PERCENTS=(1 5 10)
CONDITIONS=(ske_pre ske_abl rnd_pre rnd_abl)
mkdir -p "${HF_HOME}" "${REPO_ROOT}/results/attn_mlp_gpt2"

for circuit_type in "${TYPES[@]}"; do
  echo "[START] GPT-2 dynamic type=${circuit_type}"
  "${PYTHON_BIN}" -u -m attn_mlp_cir.run_dynamic \
    --model "${MODEL}" \
    --prompt "Janell Hobson analysed the imagery presented in the" \
    --target_token " report" \
    --output_dir "${REPO_ROOT}/results/attn_mlp_gpt2/${circuit_type}" \
    --start_layer 4 \
    --k_percents "${K_PERCENTS[@]}" \
    --type "${circuit_type}" \
    --conditions "${CONDITIONS[@]}" \
    --n_samples 1 \
    --batch_size 1 \
    --gpu "${GPU}" \
    --no-rely_ori_acv
done

"${PYTHON_BIN}" Demo/preview_result.py attention \
  --output-dir "${REPO_ROOT}/results/attn_mlp_gpt2" --model "${MODEL}" \
  --prompt "Janell Hobson analysed the imagery presented in the" --target " report"
