#!/usr/bin/env bash
set -euo pipefail

# GPT-2 static per-neuron circuit on 100 natural-language inputs.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${REPO_ROOT}"
CHECKPOINT_CACHE_ROOT="${REPO_ROOT}/cache"
export HF_HOME="${CHECKPOINT_CACHE_ROOT}/model_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export SKELETON_CACHE_BASE="${CHECKPOINT_CACHE_ROOT}/pre_compute_skeleton_cache"

MODEL="openai-community/gpt2"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${REPO_ROOT}/dataset/skeleton_static/natural_language/short_10000samples.json"
OUTPUT_DIR="${REPO_ROOT}/results/static_ske_per_neuron_gpt2"
K_PER_NEURONS=(2)
CONDITIONS=(top random middle bottom)
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

echo '[START] GPT-2 static: first 100 dataset prompts, target=" bird"'
"${PYTHON_BIN}" -u scripts_ske_cir/run_static_ske_per_neuron.py \
  --models "${MODEL}" \
  --target " bird" \
  --k_per_neuron_values "${K_PER_NEURONS[@]}" \
  --start_layer 4 \
  --n_samples 100 \
  --batch_size 1 \
  --gpu "${GPU}" \
  --dataset_path "${DATASET}" \
  --cache_base "${SKELETON_CACHE_BASE}" \
  --output_dir "${OUTPUT_DIR}" \
  --conditions "${CONDITIONS[@]}"

"${PYTHON_BIN}" Demo/preview_result.py static \
  --output-dir "${OUTPUT_DIR}" --model "${MODEL}" \
  --dataset-path "${DATASET}" --target " bird"
