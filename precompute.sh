#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${REPO_ROOT}"

CHECKPOINT_CACHE_ROOT="${REPO_ROOT}/cache"
export HF_HOME="${CHECKPOINT_CACHE_ROOT}/model_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export SKELETON_CACHE_BASE="${CHECKPOINT_CACHE_ROOT}/pre_compute_skeleton_cache"
export ATTENTION_OV_CACHE_BASE="${CHECKPOINT_CACHE_ROOT}/pre_compute_skeleton_attn_cache"
RESIDUAL_CACHE_BASE="${CHECKPOINT_CACHE_ROOT}/pre_compute_residual_cache"
mkdir -p "${HF_HOME}" "${SKELETON_CACHE_BASE}" "${ATTENTION_OV_CACHE_BASE}" "${RESIDUAL_CACHE_BASE}"

MODEL="openai-community/gpt2"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[START] GPT-2 MLP SVD cache"
"${PYTHON_BIN}" -u pre_compute/precompute_within_ske.py \
  --model_name "${MODEL}" \
  --output_dir "${SKELETON_CACHE_BASE}/${MODEL}/within_ske_cache" \
  --gpu "${GPU}" \
  --torch_dtype float32

echo "[START] GPT-2 Attention OV cache"
"${PYTHON_BIN}" -u pre_compute/precompute_attention_ov.py \
  --model_name "${MODEL}" \
  --cache_base "${ATTENTION_OV_CACHE_BASE}" \
  --gpu "${GPU}" \
  --torch_dtype float32 \
  --storage_dtype float32

echo "[START] GPT-2 residual circuit cache"
# Keep the model, dataset split, precision, and cache directory aligned with run_residual.sh.
"${PYTHON_BIN}" -u -m pre_compute.precompute_residual.precompute \
  --model_name "${MODEL}" \
  --start_layer 4 \
  --target_layers 5-11 \
  --n_samples 1000 \
  --test_sample_offset 0 \
  --coefficient_n_samples 1000 \
  --coefficient_sample_offset 1000 \
  --batch_size 50 \
  --dataset_path "${REPO_ROOT}/dataset/skeleton_static/natural_language/short_10000samples.json" \
  --gpu "${GPU}" \
  --dtype bfloat16 \
  --coupling_device "${COUPLING_DEVICE:-gpu}" \
  --shared_cache_dir "${RESIDUAL_CACHE_BASE}"

echo "[DONE] GPT-2 MLP, Attention, and residual caches"
printf '[CACHE] MLP: %s\n' "${SKELETON_CACHE_BASE}/${MODEL}/within_ske_cache"
printf '[CACHE] Attention OV: %s\n' "${ATTENTION_OV_CACHE_BASE}/${MODEL}/attention_ov_cache"
printf '[CACHE] Residual: %s\n' "${RESIDUAL_CACHE_BASE}"
