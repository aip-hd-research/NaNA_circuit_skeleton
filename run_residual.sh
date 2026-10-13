#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${REPO_ROOT}"

CHECKPOINT_CACHE_ROOT="${REPO_ROOT}/cache"
export HF_HOME="${CHECKPOINT_CACHE_ROOT}/model_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
RESIDUAL_CACHE_BASE="${CHECKPOINT_CACHE_ROOT}/pre_compute_residual_cache"

MODEL="openai-community/gpt2"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${REPO_ROOT}/dataset/skeleton_static/natural_language/short_10000samples.json"
OUTPUT_ROOT="${REPO_ROOT}/results/residual_gpt2"
mkdir -p "${HF_HOME}" "${RESIDUAL_CACHE_BASE}" "${OUTPUT_ROOT}"

for intervention in zero scale; do
  echo "[START] residual circuit intervention=${intervention}"
  "${PYTHON_BIN}" -u -m circuit_skeleton.skeleton_analysis_residual.run \
    --model_name "${MODEL}" \
    --start_layer 4 \
    --target_layers 5-11 \
    --n_neurons 1 \
    --topk 20 \
    --n_samples 100 \
    --test_sample_offset 0 \
    --coefficient_n_samples 1000 \
    --coefficient_sample_offset 1000 \
    --batch_size 50 \
    --path_chunk_subspaces 16 \
    --intervention "${intervention}" \
    --dataset_path "${DATASET}" \
    --output_base "${OUTPUT_ROOT}/${intervention}" \
    --gpu "${GPU}" \
    --dtype bfloat16 \
    --coupling_device "${COUPLING_DEVICE:-gpu}" \
    --shared_cache_dir "${RESIDUAL_CACHE_BASE}" \
    --no-plots \
    --skeleton_mode complete_skeleton_global
done

"${PYTHON_BIN}" Demo/preview_result.py residual --output-dir "${OUTPUT_ROOT}"
