# Transformer Anatomy: Opening the Black Box of Language Models

This repository provides the source code and example data for the static MLP, dynamic MLP, attention, joint attention–MLP, and cross-layer residual circuit analyses in *Transformer Anatomy*. The supplied scripts demonstrate these analyses using GPT-2; other listed checkpoints can be selected with corresponding model configurations and data.

An interactive demonstration is available in [`Demo/demo.ipynb`](Demo/demo.ipynb).

## 🔹 Repository structure

```text
NaNA_circuit_skeleton/
├── README.md
├── Demo/
│   ├── demo.ipynb                    # Interactive experiment notebook
│   └── preview_result.py             # Concise result summaries
├── requirements.txt
├── precompute.sh                     # MLP, attention, and residual precomputation
├── run_static.sh                     # Static MLP circuits
├── run_dynm_mlp.sh                   # Dynamic MLP circuits
├── run_dynm_attn.sh                  # Attention and joint attention–MLP circuits
├── run_residual.sh                   # Cross-layer residual circuits
├── pre_compute/                      # Precomputation implementations
│   └── precompute_residual/
├── scripts_ske_cir/                  # Static and dynamic MLP experiments
├── attn_mlp_cir/                     # Attention-based experiments
├── block_interp/                     # Shared intervention implementations
├── circuit_skeleton/                 # Circuit computation
│   └── skeleton_analysis_residual/   # Residual circuit analyses
├── model_config/                     # Model configurations
├── ske_pre_valid/                    # Model adapters and validation
├── dataset/                          # Example inputs and targets
│   ├── skeleton_static/
│   ├── skeleton_dymn/
│   └── model_static_target/
└── result_summary.py                 # MLP experiment summaries
```

The scripts generate the following directories, which are not part of the source archive:

```text
cache/
├── model_cache/                       # Downloaded model checkpoints
├── pre_compute_skeleton_cache/       # MLP decomposition
├── pre_compute_skeleton_attn_cache/  # Attention OV decomposition
└── pre_compute_residual_cache/       # Residual coupling and activation data
results/                                # Experiment outputs
```

Additional research and analysis directories may be present; the directories above contain the implementations invoked by the documented entry points.

## 🔹 Environment and installation

The development environment uses Red Hat Enterprise Linux 9.6, Python 3.11, and Bash. Inspected package versions include PyTorch 2.12.1+cu126, Transformers 4.57.6, NumPy 1.26.4, and SciPy 1.14.1. The Python dependencies are specified in `requirements.txt`.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Installing the dependencies from scratch typically takes about one hour, depending on network speed and package caches. Once the model and precomputation caches are available, running the example data typically takes a few minutes; first-time downloads and precomputation take additional time.

The experiments were run on NVIDIA H100 and A100 GPUs. These specific models are not required, but a CUDA-capable GPU is recommended, particularly for precomputation and larger checkpoints; memory requirements depend on the checkpoint and analysis. Scripts use GPU index 0 by default; set `GPU` to select another device or `PYTHON_BIN` to specify the Python interpreter.

On first use, checkpoints are downloaded to `cache/model_cache/`. This requires Hugging Face access unless the models are already cached. Gated models may require authorization.

## 🔹 Interactive demo

Start Jupyter from the repository root on a GPU node and open [`Demo/demo.ipynb`](Demo/demo.ipynb).  

```bash
jupyter lab Demo/demo.ipynb
```

The notebook contains separate cells for dependency installation, precomputation, and the four experiment scripts. Select the intended Python kernel and execute the required cells sequentially. Installation and precomputation may be skipped when the dependencies and corresponding caches are already available.

Each experiment reports progress and a concise result summary, and saves JSON outputs under `results/`. Save the notebook separately to retain its displayed cell outputs.

## 🔹 Experiments

The supplied scripts use `openai-community/gpt2` as an example configuration. Run `precompute.sh` if the required model caches are not already present. Cache generation processes the selected checkpoint.

### 1. Precomputation

```bash
bash precompute.sh
```

Executes `pre_compute/precompute_within_ske.py` for MLP SVD pathways, `pre_compute/precompute_attention_ov.py` for attention OV pathways, and `pre_compute/precompute_residual/precompute.py` for residual coupling and activation caches. MLP and attention caches include `layer_*.pt` files.

### 2. Static MLP circuits

```bash
bash run_static.sh
```

Runs `scripts_ske_cir/run_static_ske_per_neuron.py` on the first 100 prompts in `dataset/skeleton_static/natural_language/short_10000samples.json`, with target token ` bird`.

### 3. Dynamic MLP circuits

```bash
bash run_dynm_mlp.sh
```

Runs `scripts_ske_cir/run_skeleton_pred.py` using the prompt `Janell Hobson analysed the imagery presented in the` and target token ` report`.

### 4. Attention and joint attention–MLP circuits

```bash
bash run_dynm_attn.sh
```

Invokes `python -m attn_mlp_cir.run_dynamic` for `--type attn_and_mlp` and `--type attn`, using the configured prompt and target.

### 5. Cross-layer residual circuits

```bash
bash run_residual.sh
```

Invokes `python -m circuit_skeleton.skeleton_analysis_residual.run` for zero and scaling interventions on short-context prompts, using the precomputed residual cache. This runner excludes Gemma 2/3.

## 🔹 Outputs

Experiment results are written as JSON files under `results/`. The output includes target-token prediction statistics, such as rank, logit, and probability. The MLP experiments also use `result_summary.py` for aggregate summaries. Output filenames and counts depend on the settings in the corresponding scripts.


## 🔹 Supported models

The code supports the following 29 Transformer models:

```text
meta-llama/Llama-2-7b-hf
mistralai/Mistral-7B-v0.3
mistralai/Mistral-7B-v0.1
deepseek-ai/deepseek-llm-7b-base
google/gemma-2-9b-it
codellama/CodeLlama-7b-hf
deepseek-ai/deepseek-coder-6.7b-base
openai-community/gpt2
microsoft/Phi-3-mini-4k-instruct
google/gemma-3-1b-it
meta-llama/Llama-3.2-1B
Qwen/Qwen3-0.6B
Qwen/Qwen2.5-1.5B
bigcode/starcoder2-3b
microsoft/biogpt
microsoft/BioGPT-Large
liyinghong/BioQwen-0.5B
liyinghong/BioQwen-1.8B
nferruz/ProtGPT2
AI4PD/ZymCTRL
hugohrban/progen2-small
hugohrban/progen2-medium
ncfrey/ChemGPT-4.7M
ncfrey/ChemGPT-19M
ncfrey/ChemGPT-1.2B
entropy/gpt2_zinc_87m
ChemFM/ChemFM-1B
jonghyunlee/MolGPT_pretrained-by-ZINC15
jonghyunlee/MolGPT_long_context_pretrained-by-ZINC15
```
