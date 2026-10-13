# FP32 GPU SVD benchmark

This benchmark records per-layer and per-model time, energy (when NVML is
available), average/maximum GPU utilization, peak GPU memory, matrix shape, and
a theoretical SVD-work proxy for the MLP and attention output projections.

Edit the `MODELS = (...)` tuple in `benchmark_gpu_svd.py` by commenting out
models, or override it from the command line:

```bash
python energy_time/benchmark_gpu_svd.py --models openai-community/gpt2 Qwen/Qwen3-0.6B
```

Use a fresh output directory for every independent run because CSV output is
appended:

```bash
python energy_time/benchmark_gpu_svd.py \
  --gpu 0 --repeats 1 \
  --output-dir energy_time/results/run_01
```

The reported SVD time excludes model loading, matrix extraction, and CPU-to-GPU
copy. The model-load duration is retained only as an explicitly excluded audit
field. Models are loaded on CPU in FP32 so only the matrix currently being
decomposed and its SVD workspace occupy GPU memory.

Energy uses the NVIDIA cumulative energy counter when the driver exposes it.
Otherwise it integrates NVML power samples. Very short per-layer operations may
produce `insufficient_power_samples`; use `--repeats` and aggregate repeated
runs for publication-quality measurements. GPU-wide energy can include other
processes, so benchmark on an exclusive idle GPU.

Submit all 12 models on three `gpu_a100_short` nodes (four GPUs per node):

```bash
bash energy_time/submit_gpu_a100_short.sh
```

Slurm sends a signal two minutes before the 30-minute limit. Workers stop and
the array element requeues; on restart, completed per-layer SVD rows are skipped.
