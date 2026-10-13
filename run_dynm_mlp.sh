#!/usr/bin/env bash
set -euo pipefail

# GPT-2 dynamic MLP circuit on the included fixed prompt and target.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${REPO_ROOT}"
CHECKPOINT_CACHE_ROOT="${REPO_ROOT}/cache"
export HF_HOME="${CHECKPOINT_CACHE_ROOT}/model_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export SKELETON_CACHE_BASE="${CHECKPOINT_CACHE_ROOT}/pre_compute_skeleton_cache"

MODEL="openai-community/gpt2"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
K_PERCENTS=(0.01 0.1 1)
CONDITIONS=(ske_pre ske_abl rnd_pre rnd_abl)
OUTPUT_DIR="${REPO_ROOT}/results/dynamic_ske_gpt2"
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

echo '[START] GPT-2 dynamic MLP prompt="Janell Hobson analysed the imagery presented in the" target=" report"'
"${PYTHON_BIN}" -u scripts_ske_cir/run_skeleton_pred.py \
  --models "${MODEL}" \
  --in_seq "Janell Hobson analysed the imagery presented in the" \
  --target " report" \
  --output_dir "${OUTPUT_DIR}" \
  --k_percents "${K_PERCENTS[@]}" --start_layer 4 \
  --n_samples 1 --batch_size 1 --topk_tokens 20 \
  --gpu "${GPU}" --no-rely_ori_acv \
  --conditions "${CONDITIONS[@]}"

"${PYTHON_BIN}" Demo/preview_result.py dynamic_mlp \
  --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
  --prompt "Janell Hobson analysed the imagery presented in the" --target " report"
