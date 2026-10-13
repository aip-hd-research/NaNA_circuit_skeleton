"""
causal_patching_runner_generic.py

Architecture-agnostic counterpart to causal_patching_runner.py.

Differences from the GPT-2-only original:
  - Uses the generic (adapter-based) activation collector and per-neuron
    analysis, so it works on any architecture registered in
    ske_pre_valid.model_adapter (GPT-2, Llama-family incl. Qwen/Mistral/
    Gemma-v1/DeepSeek, Phi-3, InternLM2, StarCoder2, GPT-Neo, GPT-NeoX,
    BioGPT, ProGen2).
  - build_coupling_strength() can run its model loads, SVD and gamma products
    on GPU (default) or CPU via --coupling_device.  Completed coupling tensors
    are moved back to CPU before the forward-pass model is loaded so the two
    phases do not retain GPU memory simultaneously.
  - Accepts device_map ("auto") and torch_dtype for the actual patching model
    load, so large models can be sharded across multiple GPUs.
  - Neuron selection: args.n_neurons picks a COUNT, not indices 0..n-1 outright.
    args.neuron_selection controls how that count maps to actual neuron
    indices out of the layer's full d_ff:
      "random" (default) — a fixed-seed (args.neuron_seed) uniform sample,
                            reproducible across resume/replot runs since it's
                            re-derived from (d_ff, n_neurons, seed) rather
                            than persisted separately.
      "first"             — indices 0..n_neurons-1 (old default / historical
                            skeleton_zero1.sh convention).
"""

import hashlib
import fcntl
import json
import os

import numpy as np
import torch

from block_interp.model_load import load_model_and_embeddings
from circuit_skeleton.coupl_strength import build_coupling_strength
from circuit_skeleton.skeleton_analysis_residual.static_dynamic_crosslayer_generic import (
    collect_activations,
    ensure_pad_token,
    residual_max_length,
)
from circuit_skeleton.skeleton_analysis_crossmlp.coupling_path_selector import _parse_layers
from circuit_skeleton.skeleton_analysis_residual.per_neuron_causal_analyse_generic import (
    analyse_one_neuron,
)
from circuit_skeleton.skeleton_analysis_residual.result_aggregate_generic import (
    aggregate_and_print,
    _load_and_aggregate,
)
from model_config import (
    PROMPT_FORMAT_VERSION,
    apply_template,
    needs_true_add_special_tokens,
)
from ske_pre_valid.model_adapter import get_adapter


RESIDUAL_RESULT_FORMAT_VERSION = 5
TARGET_SCORE_VERSION = "heldout_mean_jacobian_v2"
SHARED_CACHE_VERSION = 1


def _move_coupling_to_cpu(cs) -> None:
    """Release coupling tensors from CUDA before loading the forward model."""
    for name in ("U", "S", "V", "W_up_fused", "delta"):
        value = getattr(cs.within_ske, name, None)
        if isinstance(value, torch.Tensor):
            setattr(cs.within_ske, name, value.cpu())
    value = getattr(cs.cross_ske, "V_src", None)
    if isinstance(value, torch.Tensor):
        cs.cross_ske.V_src = value.cpu()
    cs.cross_ske.gamma = {
        layer: value.cpu() for layer, value in cs.cross_ske.gamma.items()
    }
    cs.gamma_stacked = cs.gamma_stacked.cpu()


def _load_activation_cache(
    args, model, tokenizer, prompts, coefficient_prompts, all_model_layers,
    fwd_device, batch_size, add_special_tokens, model_dtype,
    test_offset, coefficient_offset, shared_model_cache_dir,
):
    """Build or load baseline activations and held-out derivative coefficients."""
    def build_activation_statistics():
        baseline = collect_activations(
            model, tokenizer, prompts, all_model_layers, fwd_device,
            batch_size=batch_size,
            add_special_tokens=add_special_tokens,
            collect_target_jacobian_stats=False,
        )
        print(
            f"[Info] Estimating target Jacobian coefficients from "
            f"{len(coefficient_prompts)} held-out coefficient prompts …"
        )
        coefficient_acts, statistics = collect_activations(
            model, tokenizer, coefficient_prompts, all_model_layers, fwd_device,
            batch_size=batch_size,
            add_special_tokens=add_special_tokens,
            collect_target_jacobian_stats=True,
        )
        del coefficient_acts
        return {"baseline_acts": baseline, "gated_stats": statistics}

    if shared_model_cache_dir:
        dataset_group = os.path.basename(os.path.dirname(os.path.abspath(args.dataset_path)))
        dataset_hash = _sha256_file(args.dataset_path)[:16]
        dtype_label = _readable_dtype_name(model_dtype)
        activation_prefix = (
            f"all_layers_L0-L{all_model_layers[-1]}"
            f"__{dataset_group}_long"
            f"__data_{dataset_hash}__test_"
        )
        activation_suffix = (
            f"__coefficient_{coefficient_offset}-{coefficient_offset + len(coefficient_prompts) - 1}"
            f"__{dtype_label}.pt"
        )
        activation_dir = os.path.join(shared_model_cache_dir, "activations")
        activation_name = (
            f"{activation_prefix}{test_offset}-{test_offset + len(prompts) - 1}"
            f"{activation_suffix}"
        )
        exact_path = os.path.join(activation_dir, activation_name)
        if not os.path.isfile(exact_path) and os.path.isdir(activation_dir):
            requested_end = test_offset + len(prompts) - 1
            covering = []
            for name in os.listdir(activation_dir):
                if not name.startswith(activation_prefix) or not name.endswith(activation_suffix):
                    continue
                span = name[len(activation_prefix):-len(activation_suffix)].split("-")
                if len(span) != 2 or not all(part.isdigit() for part in span):
                    continue
                start, end = map(int, span)
                if start <= test_offset and requested_end <= end:
                    covering.append((end - start, start, end, name))
            for _, start, end, name in sorted(covering):
                source_path = os.path.join(activation_dir, name)
                try:
                    source = torch.load(source_path, map_location="cpu", weights_only=False)
                    baseline = source["baseline_acts"]
                    if not isinstance(source["gated_stats"], dict) or not all(
                        layer in baseline and len(baseline[layer]) == end - start + 1
                        for layer in all_model_layers
                    ):
                        continue
                    shift = test_offset - start
                    subset = {
                        "baseline_acts": {
                            layer: baseline[layer][shift:shift + len(prompts)]
                            for layer in all_model_layers
                        },
                        "gated_stats": source["gated_stats"],
                    }
                except (OSError, RuntimeError, EOFError, ValueError, KeyError, TypeError):
                    continue
                print(f"[Cache] Reusing test subset from {source_path}")
                return _load_or_build_shared_cache(
                    exact_path, "baseline/Jacobian subset", lambda: subset,
                )
        return _load_or_build_shared_cache(
            exact_path,
            "baseline/Jacobian", build_activation_statistics,
        )
    return build_activation_statistics()


def resolve_neuron_ids(d_ff: int, n_neurons: int, selection: str = "random", seed: int = 42) -> list:
    """Map a neuron COUNT to actual neuron indices — see module docstring."""
    n = min(n_neurons, d_ff)
    if selection == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(d_ff, size=n, replace=False).tolist())
    return list(range(n))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _slice_prompt_records(raw: dict, offset: int, count: int, label: str):
    """Return deterministic prompt/id slices while rejecting short datasets."""
    if offset < 0:
        raise ValueError(f"{label} sample offset must be non-negative, got {offset}")
    if count <= 0:
        raise ValueError(f"{label} sample count must be positive, got {count}")

    records = list(raw.items())
    end = offset + count
    if end > len(records):
        raise ValueError(
            f"{label} sample range [{offset}, {end}) exceeds dataset size "
            f"{len(records)}"
        )

    selected = records[offset:end]
    sample_ids = [str(sample_id) for sample_id, _ in selected]
    prompts = []
    for sample_id, payload in selected:
        if not isinstance(payload, dict) or "prompt" not in payload:
            raise ValueError(f"dataset record {sample_id!r} has no 'prompt' field")
        prompts.append(payload["prompt"])
    return prompts, sample_ids


def _ids_sha256(sample_ids: list[str]) -> str:
    payload = json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _duplicate_values(values: list[str]) -> list[str]:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _validate_sample_split(
    test_sample_ids: list[str],
    coefficient_sample_ids: list[str],
    raw_test_prompts: list[str],
    raw_coefficient_prompts: list[str],
) -> None:
    """Reject duplicate records within either split and overlap across splits."""
    checks = (
        ("duplicate IDs inside test split", _duplicate_values(test_sample_ids)),
        ("duplicate IDs inside coefficient split", _duplicate_values(coefficient_sample_ids)),
        ("duplicate prompts inside test split", _duplicate_values(raw_test_prompts)),
        ("duplicate prompts inside coefficient split", _duplicate_values(raw_coefficient_prompts)),
        ("overlapping IDs between test and coefficient splits",
         sorted(set(test_sample_ids) & set(coefficient_sample_ids))),
        ("overlapping prompts between test and coefficient splits",
         sorted(set(raw_test_prompts) & set(raw_coefficient_prompts))),
    )
    failures = [(label, values) for label, values in checks if values]
    if failures:
        details = "; ".join(
            f"{label}: count={len(values)}, examples={values[:3]}"
            for label, values in failures
        )
        raise ValueError(f"invalid test/coefficient split: {details}")
    print(
        "[Info] Dataset overlap validation passed: unique IDs and prompts "
        "within each split; no ID or prompt overlap across splits."
    )


def _atomic_json_dump(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _cache_key(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _layer_cache_label(start_layer: int, target_layers: list[int]) -> str:
    targets = sorted(target_layers)
    if targets == list(range(targets[0], targets[-1] + 1)):
        target_label = f"L{targets[0]}-L{targets[-1]}"
    else:
        target_label = "-".join(f"L{layer}" for layer in targets)
    return f"source_L{start_layer}_targets_{target_label}"


def _readable_dtype_name(model_dtype: str) -> str:
    return model_dtype.removeprefix("torch.").replace("/", "_")


def _atomic_torch_save(path: str, payload) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load_or_build_shared_cache(path: str, label: str, builder):
    """Load one shared artifact or build it once under an inter-process lock."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = f"{path}.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if os.path.isfile(path):
            try:
                value = torch.load(path, map_location="cpu", weights_only=False)
                print(f"[Cache] HIT {label}: {path}")
                return value
            except (OSError, RuntimeError, EOFError, ValueError) as exc:
                print(f"[Cache] Invalid {label} cache will be rebuilt: {exc}")
        print(f"[Cache] MISS {label}; building once: {path}")
        value = builder()
        _atomic_torch_save(path, value)
        print(f"[Cache] SAVED {label}: {path}")
        return value


def _result_matches(path: str, experiment_hash: str) -> bool:
    try:
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
        return saved.get("config", {}).get("experiment_hash") == experiment_hash
    except (OSError, ValueError, TypeError):
        return False


def _gated_weight_modules(block):
    """Return (gate weight, up weight) modules for supported gated MLPs."""
    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            return mlp.gate_proj, mlp.up_proj, None
        if hasattr(mlp, "gate_up_proj"):
            return mlp.gate_up_proj, mlp.gate_up_proj, "fused"
    ff = getattr(block, "feed_forward", None)
    if ff is not None and hasattr(ff, "w1") and hasattr(ff, "w3"):
        return ff.w1, ff.w3, None
    return None


def apply_prompt_local_gated_gamma(
    cs,
    model,
    gated_stats: dict,
    target_layers: list[int],
    neuron_chunk: int = 256,
) -> list[int]:
    """Replace input-weight gamma with a prompt-local neuron linearization.

    For h_k = act(g_k) * u_k, the detector with respect to the normalized
    residual is

        E[act'(g_k) * u_k] W_gate[k] + E[act(g_k)] W_up[k].

    Ungated neurons use ``E[act'(preactivation)] W_in``.  The existing
    affine-only pre-MLP normalization approximation is retained in both cases.
    """
    adapter = get_adapter(model)
    blocks = adapter.get_blocks(model)
    v_src = cs.within_ske.V.detach().float().cpu()
    updated = []

    # Drop the old concatenation before replacing per-layer matrices, otherwise
    # a large stale tensor remains live throughout the rebuild.
    cs.gamma_stacked = None
    with torch.no_grad():
        for layer_idx in target_layers:
            stats = gated_stats.get(layer_idx)
            if stats is None:
                continue

            if stats.get("kind") == "ungated":
                activation_coeff = stats["activation_coeff"].float().cpu()
                cs.cross_ske.gamma[layer_idx] = (
                    cs.cross_ske.gamma[layer_idx].detach().float().cpu()
                    * activation_coeff.unsqueeze(0)
                )
                updated.append(layer_idx)
                continue

            modules = _gated_weight_modules(blocks[layer_idx])
            if modules is None:
                continue

            _, up_module, layout = modules
            _, _, pre_mlp_norm, _ = adapter.get_mlp_matrices(model, layer_idx)
            norm_weight = pre_mlp_norm.weight.detach().float().cpu()
            gate_coeff = stats["gate_coeff"].float().cpu()
            up_coeff = stats["up_coeff"].float().cpu()
            d_ff = gate_coeff.numel()
            gate_gamma = cs.cross_ske.gamma[layer_idx].detach().float().cpu()
            chunks = []
            for start in range(0, d_ff, neuron_chunk):
                end = min(start + neuron_chunk, d_ff)
                if layout == "fused":
                    up_w = up_module.weight[d_ff + start:d_ff + end].detach().float().cpu()
                else:
                    up_w = up_module.weight[start:end].detach().float().cpu()
                up_detector = up_coeff[start:end, None] * up_w
                up_detector.mul_(norm_weight)
                gamma_chunk = gate_gamma[:, start:end] * gate_coeff[start:end]
                gamma_chunk.add_(v_src @ up_detector.T)
                chunks.append(gamma_chunk)
            cs.cross_ske.gamma[layer_idx] = torch.cat(chunks, dim=1)
            updated.append(layer_idx)

    cs.gamma_stacked = torch.cat(
        [cs.cross_ske.gamma[layer] for layer in cs.cross_ske.target_layers],
        dim=1,
    )
    return updated


def prepare_gamma_extreme_cache(cs, k: int, chunk_subspaces: int = 16) -> None:
    """Cache per-subspace gamma extremes used by every source neuron.

    For a fixed subspace, multiplying by ``abs(delta[j,i])`` cannot change the
    ordering of its target neurons.  Keeping only that row's k extremes is
    therefore sufficient for the exact global k extremes for every source
    neuron, and avoids rescanning all gamma entries hundreds of times.
    """
    gamma = cs.gamma_stacked.cpu()
    row_k = min(k, gamma.shape[1])
    top_abs_parts = []
    top_cross_parts = []
    bot_abs_parts = []
    bot_cross_parts = []
    for start in range(0, gamma.shape[0], chunk_subspaces):
        values = gamma[start:start + chunk_subspaces].abs()
        top_abs, top_cross = torch.topk(values, row_k, dim=1)
        bot_abs, bot_cross = torch.topk(values, row_k, dim=1, largest=False)
        top_abs_parts.append(top_abs)
        top_cross_parts.append(top_cross)
        bot_abs_parts.append(bot_abs)
        bot_cross_parts.append(bot_cross)
    cs.gamma_extreme_cache = {
        "top_abs": torch.cat(top_abs_parts, dim=0),
        "top_cross": torch.cat(top_cross_parts, dim=0),
        "bot_abs": torch.cat(bot_abs_parts, dim=0),
        "bot_cross": torch.cat(bot_cross_parts, dim=0),
    }


def run_causal_patching(args):
    """Entry point called by circuit_skeleton.skeleton_analysis_residual.run.

    args must additionally provide:
      device_map  : None or "auto"
      torch_dtype : torch.dtype for the patching model load (e.g. torch.bfloat16)
    device_map/torch_dtype are used for the patching-forward-pass model load.
    coupling_device selects CPU or CUDA for coupling construction. The shared
    SVD helper may move its matrix to CPU for decomposition.
    """
    device_map    = getattr(args, "device_map", None)
    torch_dtype   = getattr(args, "torch_dtype", None)
    device        = None if device_map else (
        "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    )
    target_layers = _parse_layers(args.target_layers)

    coupling_choice = getattr(args, "coupling_device", "gpu")
    if coupling_choice == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--coupling_device=gpu requested but CUDA is unavailable")
        coupling_device = "cuda:0"
    else:
        coupling_device = "cpu"
    shared_cache_dir = getattr(args, "shared_cache_dir", None)
    if shared_cache_dir:
        shared_cache_dir = os.path.abspath(shared_cache_dir)
        model_cache_name = args.model_name.replace("/", "_")
        shared_model_cache_dir = os.path.join(shared_cache_dir, model_cache_name)
    else:
        shared_model_cache_dir = None

    def build_cpu_coupling():
        print(f"[Info] Building coupling strength (SVD) on {coupling_device} …")
        value = build_coupling_strength(
            args.model_name, args.start_layer, target_layers, coupling_device
        )
        _move_coupling_to_cpu(value)
        if coupling_device.startswith("cuda"):
            torch.cuda.empty_cache()
        return value

    if shared_cache_dir:
        layer_label = _layer_cache_label(args.start_layer, target_layers)
        cs = _load_or_build_shared_cache(
            os.path.join(shared_model_cache_dir, "coupling", f"{layer_label}.pt"),
            "base coupling/SVD", build_cpu_coupling,
        )
    else:
        cs = build_cpu_coupling()

    cache_only = getattr(args, "cache_only", False)
    if not cache_only:
        d_ff       = cs.within_ske.delta.shape[0]
        selection  = getattr(args, "neuron_selection", "random")
        seed       = getattr(args, "neuron_seed", 42)
        all_neuron_ids = resolve_neuron_ids(d_ff, args.n_neurons, selection, seed)
        shard_count = getattr(args, "neuron_shard_count", 1)
        shard_index = getattr(args, "neuron_shard_index", 0)
        if shard_count <= 0 or not (0 <= shard_index < shard_count):
            raise ValueError(
                f"invalid neuron shard {shard_index}/{shard_count}; require count > 0 "
                "and 0 <= index < count"
            )
        neuron_ids = all_neuron_ids[shard_index::shard_count]
        print(f"[Info] d_ff={d_ff}  neuron_selection={selection}  "
              f"global_n_neurons={len(all_neuron_ids)}  "
              f"shard={shard_index}/{shard_count}  shard_n_neurons={len(neuron_ids)}")

    model, tokenizer, _ = load_model_and_embeddings(
        args.model_name, device, device_map=device_map, torch_dtype=torch_dtype,
    )
    if getattr(model.config, "model_type", None) in {"gemma2", "gemma3", "gemma3_text"}:
        raise ValueError("Gemma 2/3 are intentionally excluded from the residual sweep")
    ensure_pad_token(model, tokenizer)
    add_special_tokens = needs_true_add_special_tokens(args.model_name, tokenizer)
    all_model_layers = list(range(get_adapter(model).num_layers(model)))

    with open(args.dataset_path, encoding="utf-8") as f:
        raw = json.load(f)
    test_offset = getattr(args, "test_sample_offset", 0)
    coefficient_offset = getattr(args, "coefficient_sample_offset", 1000)
    coefficient_n_samples = getattr(args, "coefficient_n_samples", args.n_samples)
    raw_test_prompts, test_sample_ids = _slice_prompt_records(
        raw, test_offset, args.n_samples, "test"
    )
    raw_coefficient_prompts, coefficient_sample_ids = _slice_prompt_records(
        raw, coefficient_offset, coefficient_n_samples, "coefficient"
    )
    _validate_sample_split(
        test_sample_ids,
        coefficient_sample_ids,
        raw_test_prompts,
        raw_coefficient_prompts,
    )

    prompts = [
        apply_template(prompt, args.model_name, tokenizer)
        for prompt in raw_test_prompts
    ]
    coefficient_prompts = [
        apply_template(prompt, args.model_name, tokenizer)
        for prompt in raw_coefficient_prompts
    ]
    print(
        f"[Info] Dataset split: test=[{test_offset}, {test_offset + len(prompts)}) "
        f"coefficient=[{coefficient_offset}, "
        f"{coefficient_offset + len(coefficient_prompts)})"
    )

    batch_size = getattr(args, "batch_size", 32)
    model_dtype = str(next(model.parameters()).dtype)
    if cache_only:
        fwd_device = model.get_input_embeddings().weight.device if device_map else device
        _load_activation_cache(
            args, model, tokenizer, prompts, coefficient_prompts, all_model_layers,
            fwd_device, batch_size, add_special_tokens, model_dtype,
            test_offset, coefficient_offset, shared_model_cache_dir,
        )
        print(
            "[Cache] cache-only mode complete: base coupling/SVD, baseline "
            "activations, and held-out Jacobian coefficients are ready."
        )
        return False

    path_chunk_subspaces = getattr(args, "path_chunk_subspaces", 16)
    skeleton_mode = getattr(args, "skeleton_mode", "complete_skeleton_global")
    candidate_paths_per_source = (
        cs.total_cross
        if skeleton_mode == "complete_skeleton_fullinfo"
        else cs.within_ske.n_subspaces * cs.total_cross
    )
    dataset_sha256 = _sha256_file(args.dataset_path)
    run_config = {
        "result_format_version": RESIDUAL_RESULT_FORMAT_VERSION,
        "model_name": args.model_name,
        "model_type": getattr(model.config, "model_type", None),
        "start_layer": args.start_layer,
        "target_layers": target_layers,
        "n_neurons": len(all_neuron_ids),
        "neuron_ids": all_neuron_ids,
        "neuron_selection": selection,
        "neuron_seed": seed,
        "topk": args.topk,
        "n_samples_requested": args.n_samples,
        "n_samples_actual": len(prompts),
        "test_sample_offset": test_offset,
        "test_sample_ids_sha256": _ids_sha256(test_sample_ids),
        "coefficient_n_samples_requested": coefficient_n_samples,
        "coefficient_n_samples_actual": len(coefficient_prompts),
        "coefficient_sample_offset": coefficient_offset,
        "coefficient_sample_ids_sha256": _ids_sha256(coefficient_sample_ids),
        "dataset_split": "test_first_1000_coeff_next_1000_v1",
        "batch_size": batch_size,
        "intervention": args.intervention,
        "scale": float(args.scale),
        "skeleton_mode": skeleton_mode,
        "candidate_paths_per_source": candidate_paths_per_source,
        "selected_path_fraction": min(args.topk, candidate_paths_per_source) / candidate_paths_per_source,
        "path_groups": ["top", "middle", "random", "bottom"],
        "middle_selection": "sampled_median_coupling_v1",
        "dtype": model_dtype,
        "requested_dtype": str(torch_dtype),
        "coupling_device": coupling_choice,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "add_special_tokens": add_special_tokens,
        "max_length": residual_max_length(model),
        "dataset_path": os.path.abspath(args.dataset_path),
        "dataset_sha256": dataset_sha256,
        "path_chunk_subspaces": path_chunk_subspaces,
        "intervention_chunk_size": getattr(args, "intervention_chunk_size", 1),
        "target_detector_score": TARGET_SCORE_VERSION,
        "shared_cache_version": SHARED_CACHE_VERSION,
    }
    experiment_hash = _config_hash(run_config)
    args.experiment_hash = experiment_hash
    os.makedirs(args.output_base, exist_ok=True)
    _atomic_json_dump(
        os.path.join(args.output_base, "experiment_config.json"),
        {"experiment_hash": experiment_hash, **run_config},
    )

    if args.resume:
        pending = []
        for j in neuron_ids:
            result_path = os.path.join(
                args.output_base, f"neuron_{j}", "patching_results.json"
            )
            if not _result_matches(result_path, experiment_hash):
                pending.append(j)
        print(f"Resume[{experiment_hash[:12]}]: "
              f"{len(neuron_ids) - len(pending)} compatible, {len(pending)} to run.")
    else:
        pending = neuron_ids

    if not pending:
        if getattr(args, "no_finalize", False):
            print("All compatible results for this shard are done.")
            return False
        print("All compatible results are done. Aggregating...")
        _load_and_aggregate(args)
        return True

    fwd_device = model.get_input_embeddings().weight.device if device_map else device
    activation_payload = _load_activation_cache(
        args, model, tokenizer, prompts, coefficient_prompts, all_model_layers,
        fwd_device, batch_size, add_special_tokens, model_dtype,
        test_offset, coefficient_offset, shared_model_cache_dir,
    )
    baseline_acts = activation_payload["baseline_acts"]
    gated_stats = activation_payload["gated_stats"]
    linearized_layers = apply_prompt_local_gated_gamma(
        cs, model, gated_stats, target_layers
    )
    if linearized_layers:
        print(f"[Info] Prompt-local target Jacobian applied at layers {linearized_layers}")
    prepare_gamma_extreme_cache(cs, args.topk, path_chunk_subspaces)
    mean_acts_src = baseline_acts[args.start_layer].mean(axis=0)
    baseline_tgt  = {li: baseline_acts[li] for li in target_layers}

    summaries = []
    failures = []
    for pos, j in enumerate(pending, 1):
        out_dir = os.path.join(args.output_base, f"neuron_{j}")
        print(f"===== Neuron {j:4d}  ({pos}/{len(pending)}) =====")
        try:
            summary = analyse_one_neuron(
                j, cs,
                model, tokenizer, prompts,
                baseline_tgt,
                target_layers, args.start_layer,
                args.topk, args.intervention,
                float(mean_acts_src[j]), args.scale,
                fwd_device, out_dir,
                skeleton_mode=getattr(args, "skeleton_mode", "complete_skeleton_global"),
                batch_size=batch_size,
                print_cpl_delta_abs=getattr(args, "print_cpl_delta_abs", False),
                add_special_tokens=add_special_tokens,
                path_chunk_subspaces=path_chunk_subspaces,
                intervention_chunk_size=getattr(args, "intervention_chunk_size", 1),
                experiment_hash=experiment_hash,
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"  [ERROR] neuron {j}: {exc}")
            failures.append((j, str(exc)))

    for j in set(neuron_ids) - set(pending):
        p = os.path.join(args.output_base, f"neuron_{j}", "patching_results.json")
        if _result_matches(p, experiment_hash):
            with open(p) as f:
                summaries.append(json.load(f))

    summaries.sort(key=lambda s: s["config"]["src_neuron"])
    if failures:
        preview = "; ".join(f"j={j}: {message}" for j, message in failures[:5])
        raise RuntimeError(
            f"{len(failures)} neuron analyses failed; compatible completed results "
            f"were preserved for resume. First failures: {preview}"
        )
    if getattr(args, "no_finalize", False):
        print(f"[Shard complete] {shard_index}/{shard_count}; per-neuron results saved.")
        return False
    aggregate_and_print(summaries, args.output_base, args.intervention)
    return True
