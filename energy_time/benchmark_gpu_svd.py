#!/usr/bin/env python3
"""Benchmark FP32 GPU SVD for every MLP and attention output projection.

Scope
-----
Model loading and CPU->GPU matrix copies are deliberately excluded.  The model
is loaded in FP32 on CPU, one projection is copied to the selected GPU, and only
``torch.linalg.svd(..., full_matrices=False)`` is timed.  This measures the
decomposition cost, not end-to-end preprocessing.

Attention follows this repository's circuit definition: the output projection
is split by head and decomposed as a batched tensor [heads, head_dim, d_model].

Outputs (under --output-dir)
--------------------------------
svd_layers.csv   one row per layer/component/repetition
svd_models.csv   one aggregate row per model
run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from block_interp.model_load import load_model_and_embeddings
from circuit_skeleton.attention_skeleton import (
    get_attention_output_projection,
    split_output_projection,
)
from ske_pre_valid.model_adapter import get_adapter


# Comment/uncomment entries here to select models. Keep the trailing comma for
# a one-model tuple. CLI --models overrides this tuple.
MODELS = (
    "openai-community/gpt2",
    "Qwen/Qwen3-0.6B",
    "AI4PD/ZymCTRL",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B",
    "microsoft/BioGPT-Large",
    "Qwen/Qwen2.5-1.5B",
    "bigcode/starcoder2-3b",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-2-9b-it",
)


CSV_FIELDS = (
    "model_name", "component", "layer_idx", "repeat", "matrix_shape",
    "batch_count", "m", "n", "svd_work_units", "time_s", "energy_j",
    "energy_method", "avg_power_w", "baseline_memory_mib",
    "peak_memory_mib", "incremental_peak_memory_mib", "avg_gpu_util_pct",
    "max_gpu_util_pct", "gpu_name",
)


class NvmlMeter:
    """NVML energy counter when available, otherwise sampled power integration."""

    def __init__(self, physical_index: int, interval_s: float):
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = None
        self.available = False
        self.total_energy_supported = False
        try:
            import pynvml  # type: ignore
            self.nvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
            self.available = True
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.total_energy_supported = True
            except Exception:
                pass
        except Exception:
            self.nvml = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self.energy0_mj = None
        if not self.available:
            return
        if self.total_energy_supported:
            self.energy0_mj = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    mw = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
                    util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                    self.samples.append((time.perf_counter(), mw / 1000.0, float(util)))
                except Exception:
                    pass
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float | None, str, float | None, float | None]:
        if not self.available:
            return None, "unavailable", None, None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_s))
        utils = [sample[2] for sample in self.samples]
        avg_util = sum(utils) / len(utils) if utils else None
        max_util = max(utils) if utils else None
        if self.total_energy_supported:
            e1 = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return (max(0.0, (e1 - self.energy0_mj) / 1000.0),
                    "nvml_total_energy", avg_util, max_util)
        if len(self.samples) < 2:
            return None, "insufficient_power_samples", avg_util, max_util
        energy = sum(
            (t1 - t0) * (p0 + p1) / 2
            for (t0, p0, _), (t1, p1, _) in zip(self.samples, self.samples[1:])
        )
        return energy, "nvml_power_integration", avg_util, max_util


def physical_gpu_index(cuda_index: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        token = visible.split(",")[cuda_index].strip()
        if token.isdigit():
            return int(token)
    return cuda_index


def mlp_matrix(adapter, model, layer_idx: int) -> torch.Tensor:
    _, out, _, _ = adapter.get_mlp_matrices(model, layer_idx)
    w = out.weight.detach()
    # GPT-style Conv1D stores [d_ff, d_model]; nn.Linear stores the transpose.
    return w if type(out).__name__ == "Conv1D" else w.T


def attention_matrix(model, layer_idx: int) -> torch.Tensor:
    out, linear_layout, heads, head_dim = get_attention_output_projection(model, layer_idx)
    return split_output_projection(
        out.weight.detach(), heads, head_dim, linear_layout=linear_layout
    )


def svd_work(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    batch = math.prod(shape[:-2]) if len(shape) > 2 else 1
    m, n = shape[-2:]
    # A hardware-independent proxy, proportional to batched reduced-SVD work.
    return batch, m, n, batch * m * n * min(m, n)


def benchmark_matrix(
    cpu_matrix: torch.Tensor,
    *,
    model_name: str,
    component: str,
    layer_idx: int,
    repeat: int,
    device: torch.device,
    meter: NvmlMeter,
    gpu_name: str,
) -> dict:
    # Transfer/cast is outside the measurement by design.
    matrix = cpu_matrix.detach().to(device=device, dtype=torch.float32).contiguous()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    meter.start()
    start.record()
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    end.record()
    end.synchronize()
    energy_j, energy_method, avg_util, max_util = meter.stop()
    elapsed_s = start.elapsed_time(end) / 1000.0
    peak = torch.cuda.max_memory_allocated(device)
    batch, m, n, work = svd_work(tuple(matrix.shape))

    row = {
        "model_name": model_name,
        "component": component,
        "layer_idx": layer_idx,
        "repeat": repeat,
        "matrix_shape": "x".join(map(str, matrix.shape)),
        "batch_count": batch,
        "m": m,
        "n": n,
        "svd_work_units": work,
        "time_s": elapsed_s,
        "energy_j": energy_j,
        "energy_method": energy_method,
        "avg_power_w": energy_j / elapsed_s if energy_j is not None and elapsed_s else None,
        "baseline_memory_mib": baseline / 2**20,
        "peak_memory_mib": peak / 2**20,
        "incremental_peak_memory_mib": (peak - baseline) / 2**20,
        "avg_gpu_util_pct": avg_util,
        "max_gpu_util_pct": max_util,
        "gpu_name": gpu_name,
    }
    del u, s, vh, matrix
    torch.cuda.empty_cache()
    return row


def append_csv(path: Path, rows: list[dict], fields=CSV_FIELDS) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def existing_layer_rows(path: Path, model_name: str) -> list[dict]:
    """Read valid checkpoint rows and restore numeric types for aggregation."""
    if not path.exists():
        return []
    rows = []
    int_fields = {"layer_idx", "repeat", "batch_count", "m", "n", "svd_work_units"}
    float_fields = {
        "time_s", "energy_j", "avg_power_w", "baseline_memory_mib",
        "peak_memory_mib", "incremental_peak_memory_mib", "avg_gpu_util_pct",
        "max_gpu_util_pct",
    }
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model_name") != model_name:
                continue
            try:
                for key in int_fields:
                    row[key] = int(row[key])
                for key in float_fields:
                    row[key] = float(row[key]) if row.get(key) not in (None, "", "None") else None
            except (KeyError, ValueError):
                continue  # Ignore a truncated final line after forced termination.
            rows.append(row)
    return rows


def write_single_csv(path: Path, row: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    os.replace(tmp, path)


def aggregate(model_name: str, rows: list[dict], load_s: float, wall_s: float) -> dict:
    energies = [r["energy_j"] for r in rows if r["energy_j"] is not None]
    repeats = max(r["repeat"] for r in rows)
    mlp_time = sum(r["time_s"] for r in rows if r["component"] == "mlp")
    attn_time = sum(r["time_s"] for r in rows if r["component"] == "attention")
    return {
        "model_name": model_name,
        "n_layers": len({r["layer_idx"] for r in rows}),
        "n_svd_calls": len(rows),
        "mean_mlp_svd_time_s_per_full_pass": mlp_time / repeats,
        "mean_attention_svd_time_s_per_full_pass": attn_time / repeats,
        "mean_total_svd_time_s_per_full_pass": (mlp_time + attn_time) / repeats,
        "mean_total_svd_energy_j_per_full_pass": (
            sum(energies) / repeats if len(energies) == len(rows) else None
        ),
        "all_repetitions_svd_time_s": mlp_time + attn_time,
        "max_peak_memory_mib": max(r["peak_memory_mib"] for r in rows),
        "svd_work_units_per_full_pass": sum(r["svd_work_units"] for r in rows) // repeats,
        "benchmark_wall_s_excluding_model_load": wall_s,
        "model_load_s_excluded": load_s,
        "repeats": repeats,
        "gpu_name": rows[0]["gpu_name"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", help="Override MODELS in this file")
    p.add_argument("--gpu", type=int, default=0, help="CUDA index after visibility mapping")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--power-interval-ms", type=float, default=20.0)
    p.add_argument("--output-dir", default="energy_time/results_a100short")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip completed layer/component/repeat rows (default: true)")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    models = tuple(args.models) if args.models else MODELS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_csv = out_dir / "svd_layers.csv"
    model_csv = out_dir / "svd_models.csv"
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    meter = NvmlMeter(physical_gpu_index(args.gpu), args.power_interval_ms / 1000.0)

    # Initialize CUDA/cuSOLVER before measurements; no measured model gets a
    # one-off library initialization penalty.
    torch.linalg.svd(torch.randn(32, 16, device=device), full_matrices=False)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "dtype": "float32",
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        "repeats": args.repeats,
        "timing_scope": "GPU SVD only; excludes model load, matrix extraction and H2D copy",
        "attention_definition": "per-head batched SVD of attention output projection",
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for model_name in models:
        print(f"\nLoading {model_name} on CPU in FP32 (excluded from benchmark) ...", flush=True)
        load_start = time.perf_counter()
        model, _, _ = load_model_and_embeddings(
            model_name, device="cpu", torch_dtype=torch.float32
        )
        load_s = time.perf_counter() - load_start
        adapter = get_adapter(model)
        n_layers = adapter.num_layers(model)
        rows: list[dict] = existing_layer_rows(layer_csv, model_name) if args.resume else []
        completed = {
            (r["component"], r["layer_idx"], r["repeat"])
            for r in rows
        }
        if completed:
            print(f"Resuming {model_name}: {len(completed)} SVD calls already complete", flush=True)
        wall_start = time.perf_counter()

        for layer_idx in range(n_layers):
            for component, getter in (("mlp", mlp_matrix), ("attention", attention_matrix)):
                cpu_matrix = getter(adapter, model, layer_idx) if component == "mlp" else getter(model, layer_idx)
                for repeat in range(1, args.repeats + 1):
                    if (component, layer_idx, repeat) in completed:
                        continue
                    row = benchmark_matrix(
                        cpu_matrix, model_name=model_name, component=component,
                        layer_idx=layer_idx, repeat=repeat, device=device,
                        meter=meter, gpu_name=gpu_name,
                    )
                    rows.append(row)
                    append_csv(layer_csv, [row])
                    print(
                        f"  {component:9s} L{layer_idx:02d} r{repeat}: "
                        f"{row['matrix_shape']} {row['time_s']:.4f}s "
                        f"{row['incremental_peak_memory_mib']:.1f}MiB",
                        flush=True,
                    )
                del cpu_matrix

        wall_s = time.perf_counter() - wall_start
        summary = aggregate(model_name, rows, load_s, wall_s)
        write_single_csv(model_csv, summary)
        print(
            f"Finished {model_name}: mean full-pass SVD="
            f"{summary['mean_total_svd_time_s_per_full_pass']:.3f}s",
            flush=True,
        )
        del model, adapter, rows
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
