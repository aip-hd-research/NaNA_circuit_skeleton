"""
per_neuron_causal_analyse_generic.py

Architecture-agnostic counterpart to per_neuron_causal_analyse.py. Identical
logic — only the patched-activation collector is swapped for the generic
(adapter-based) version so this works on any architecture registered in
ske_pre_valid.model_adapter, not just GPT-2.

    j  →  subspace i  →  target neuron k

For each path in top-k / middle-k / random-k / bot-k:
  1. Intervene ONLY on the j→i contribution via a post-hook on the source
     layer's down-projection module.
  2. Measure mean |Δa_k| at the corresponding target neuron.
  3. Print a per-path comparison table and save a summary JSON.

"""

import json
import os
import shutil

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm

from circuit_skeleton.skeleton_analysis_residual.subspace_intervention_hook_generic import (
    collect_batched_patched_deltas,
)


def _subspace_checkpoint_path(checkpoint_dir: str, sp) -> str:
    name = "net.npy" if sp is None else f"subspace_{int(sp):05d}.npy"
    return os.path.join(checkpoint_dir, name)


def _load_subspace_checkpoint(path: str, expected_size: int):
    """Load one completed subspace atomically, ignoring stale/corrupt files."""
    try:
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1 or values.size != expected_size:
            return None
        return values
    except (OSError, ValueError, EOFError):
        return None


def _save_subspace_checkpoint(path: str, values: np.ndarray) -> None:
    """Atomically publish a subspace result so preemption cannot corrupt it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _save_json_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _scores_at_flat_paths(
    delta_row: torch.Tensor,
    gamma: torch.Tensor,
    flat_indices,
) -> torch.Tensor:
    """Evaluate factorized ``delta[i] * gamma[i,k]`` paths by flat index."""
    if len(flat_indices) == 0:
        return torch.empty(0, dtype=gamma.dtype)
    idx = torch.as_tensor(flat_indices, dtype=torch.long)
    total_cross = gamma.shape[1]
    sp = torch.div(idx, total_cross, rounding_mode="floor")
    cross = idx.remainder(total_cross)
    return delta_row[sp] * gamma[sp, cross]


def _factorized_extremes(
    delta_row: torch.Tensor,
    gamma: torch.Tensor,
    k: int,
    chunk_subspaces: int,
    extreme_cache: dict | None = None,
) -> tuple[list[int], list[int]]:
    """Return exact global top/bottom paths without building the full matrix."""
    total_cross = gamma.shape[1]
    if extreme_cache is not None:
        delta_abs = delta_row.abs().unsqueeze(1)
        top_candidates = delta_abs * extreme_cache["top_abs"]
        bot_candidates = delta_abs * extreme_cache["bot_abs"]
        count = min(k, top_candidates.numel())
        _, top_local = torch.topk(top_candidates.reshape(-1), count)
        _, bot_local = torch.topk(
            bot_candidates.reshape(-1), count, largest=False
        )
        top_sp = torch.div(top_local, extreme_cache["top_abs"].shape[1], rounding_mode="floor")
        bot_sp = torch.div(bot_local, extreme_cache["bot_abs"].shape[1], rounding_mode="floor")
        top_cross = extreme_cache["top_cross"].reshape(-1)[top_local]
        bot_cross = extreme_cache["bot_cross"].reshape(-1)[bot_local]
        return (
            (top_sp * total_cross + top_cross).tolist(),
            (bot_sp * total_cross + bot_cross).tolist(),
        )

    top_values = torch.empty(0, dtype=gamma.dtype)
    top_indices = torch.empty(0, dtype=torch.long)
    bot_values = torch.empty(0, dtype=gamma.dtype)
    bot_indices = torch.empty(0, dtype=torch.long)

    for start in range(0, gamma.shape[0], chunk_subspaces):
        end = min(start + chunk_subspaces, gamma.shape[0])
        scores = delta_row[start:end, None] * gamma[start:end, :]
        values = scores.abs().reshape(-1)
        local_k = min(k, values.numel())

        chunk_top_values, chunk_top_local = torch.topk(values, local_k)
        chunk_bot_values, chunk_bot_local = torch.topk(
            values, local_k, largest=False
        )
        chunk_top_indices = (
            (chunk_top_local // total_cross + start) * total_cross
            + chunk_top_local.remainder(total_cross)
        )
        chunk_bot_indices = (
            (chunk_bot_local // total_cross + start) * total_cross
            + chunk_bot_local.remainder(total_cross)
        )

        top_values = torch.cat((top_values, chunk_top_values))
        top_indices = torch.cat((top_indices, chunk_top_indices))
        keep = min(k, top_values.numel())
        top_values, keep_pos = torch.topk(top_values, keep)
        top_indices = top_indices[keep_pos]

        bot_values = torch.cat((bot_values, chunk_bot_values))
        bot_indices = torch.cat((bot_indices, chunk_bot_indices))
        keep = min(k, bot_values.numel())
        bot_values, keep_pos = torch.topk(bot_values, keep, largest=False)
        bot_indices = bot_indices[keep_pos]

    return top_indices.tolist(), bot_indices.tolist()


def _sample_random_paths(
    selected_layers: set[int],
    k: int,
    skeleton_mode: str,
    n_subspaces: int,
    total_cross: int,
    layer_offsets: list,
    exclude: set[int],
    seed: int,
) -> list[int]:
    """Uniformly sample the layer-restricted baseline without expanding its pool."""
    parts = [
        np.arange(offset, offset + d_ff, dtype=np.int64)
        for offset, layer, d_ff in layer_offsets
        if layer in selected_layers
    ]
    if not parts or k <= 0:
        return []
    allowed_cross = np.concatenate(parts)
    n_candidates = len(allowed_cross)
    if skeleton_mode != "complete_skeleton_fullinfo":
        n_candidates *= n_subspaces
    target_count = min(k, max(0, n_candidates - len(exclude)))
    rng = np.random.default_rng(seed)
    chosen = set()
    max_attempts = max(100, target_count * 100)
    attempts = 0
    while len(chosen) < target_count and attempts < max_attempts:
        cross = int(allowed_cross[rng.integers(0, len(allowed_cross))])
        if skeleton_mode == "complete_skeleton_fullinfo":
            flat = cross
        else:
            sp = int(rng.integers(0, n_subspaces))
            flat = sp * total_cross + cross
        if flat not in exclude:
            chosen.add(flat)
        attempts += 1
    return sorted(chosen)


def _sample_middle_paths(
    selected_layers: set[int],
    k: int,
    delta_row: torch.Tensor,
    gamma: torch.Tensor,
    n_subspaces: int,
    total_cross: int,
    layer_offsets: list,
    exclude: set[int],
    seed: int,
) -> list[int]:
    """Select reproducible paths near the median absolute coupling.

    The complete global path space can contain billions of entries, so forming
    and sorting it is infeasible.  Estimate its median from a deterministic
    uniform candidate sample, then retain the k sampled paths closest to that
    median.  A large reservoir keeps this distinct from the k-path random
    baseline while bounding selection memory and runtime.
    """
    reservoir_size = min(200_000, max(10_000, 1_000 * k))
    candidates = _sample_random_paths(
        selected_layers, reservoir_size, "complete_skeleton_global",
        n_subspaces, total_cross, layer_offsets, exclude, seed,
    )
    if not candidates:
        return []
    scores = _scores_at_flat_paths(delta_row, gamma, candidates).abs().numpy()
    median = float(np.median(scores))
    count = min(k, len(candidates))
    nearest = np.argsort(np.abs(scores - median), kind="stable")[:count]
    return sorted(candidates[int(pos)] for pos in nearest)


def analyse_one_neuron(
    j: int,
    cs,
    model, tokenizer, prompts: list,
    baseline_acts: dict,
    target_layers: list,
    start_layer: int,
    topk: int,
    intervention: str,
    mean_val: float,
    scale: float,
    device: str,
    output_dir: str,
    skeleton_mode: str = "complete_skeleton_global",
    batch_size: int = 32,
    print_cpl_delta_abs: bool = False,
    add_special_tokens: bool = False,
    path_chunk_subspaces: int = 16,
    intervention_chunk_size: int = 1,
    experiment_hash: str | None = None,
) -> dict:
    """Causal patching analysis for source neuron *j* — see
    per_neuron_causal_analyse.analyse_one_neuron for the full docstring
    (identical semantics; only the hook module lookup is generalized).
    """
    os.makedirs(output_dir, exist_ok=True)
    total_cross = cs.total_cross
    delta_mat   = cs.within_ske.delta.cpu()   # [d_ff, n_sp]
    V_src       = cs.within_ske.V.cpu()       # [n_sp, d_model]

    def _decode(flat_k):
        return cs.flat_to_layer_neuron(flat_k)

    if skeleton_mode == "complete_skeleton_fullinfo":
        flat      = cs.w_contracted([j]).squeeze(0).cpu()    # [total_cross]
        n_flat    = flat.shape[0]
        k_sel     = min(topk, n_flat // 4)
        top_flat  = torch.topk(flat.abs(), k_sel, largest=True ).indices.tolist()
        bot_flat  = torch.topk(flat.abs(), k_sel, largest=False).indices.tolist()
        top_k_pos = top_flat
        bot_k_pos = bot_flat
        top_sp    = [None] * k_sel
        bot_sp    = [None] * k_sel
        top_cpl   = [flat[fi].item() for fi in top_flat]
        order = torch.argsort(flat.abs())
        mid_start = max(0, (len(order) - k_sel) // 2)
        mid_flat = order[mid_start:mid_start + k_sel].tolist()
    else:
        delta_row = delta_mat[j]
        gamma = cs.gamma_stacked.cpu()
        n_flat = int(gamma.shape[0] * gamma.shape[1])
        k_sel     = min(topk, n_flat // 4)
        top_flat, bot_flat = _factorized_extremes(
            delta_row, gamma, k_sel, path_chunk_subspaces,
            extreme_cache=getattr(cs, "gamma_extreme_cache", None),
        )
        top_k_pos = [fi % total_cross for fi in top_flat]
        bot_k_pos = [fi % total_cross for fi in bot_flat]
        top_sp    = [fi // total_cross for fi in top_flat]
        bot_sp    = [fi // total_cross for fi in bot_flat]
        top_cpl = _scores_at_flat_paths(delta_row, gamma, top_flat).tolist()
        mid_flat = _sample_middle_paths(
            {_decode(k)[0] for k in top_k_pos} | {_decode(k)[0] for k in bot_k_pos},
            k_sel, delta_row, gamma, delta_mat.shape[1], total_cross,
            cs.layer_offsets, set(top_flat) | set(bot_flat), seed=1042 + j,
        )

    if skeleton_mode == "complete_skeleton_fullinfo":
        mid_k_pos = mid_flat
        mid_sp = [None] * len(mid_flat)
    else:
        mid_k_pos = [fi % total_cross for fi in mid_flat]
        mid_sp = [fi // total_cross for fi in mid_flat]

    sel_layers = (
        {_decode(k)[0] for k in top_k_pos} |
        {_decode(k)[0] for k in bot_k_pos}
    )
    exclude   = set(top_flat) | set(bot_flat) | set(mid_flat)
    rand_flat = _sample_random_paths(
        sel_layers,
        k_sel,
        skeleton_mode,
        delta_mat.shape[1],
        total_cross,
        cs.layer_offsets,
        exclude,
        seed=42 + j,
    )
    if skeleton_mode == "complete_skeleton_fullinfo":
        rand_k_pos = rand_flat
        rand_sp    = [None] * len(rand_flat)
    else:
        rand_k_pos = [fi % total_cross for fi in rand_flat]
        rand_sp    = [fi // total_cross for fi in rand_flat]

    watched: dict = {}
    for flat_k in top_k_pos:
        li, local_k = _decode(flat_k)
        watched.setdefault(li, []).append(local_k)
    watch_parts = []
    for li, ks in watched.items():
        ks_sorted = sorted(ks)
        preview   = ",".join(str(k) for k in ks_sorted[:6])
        suffix    = f",…(+{len(ks_sorted)-6})" if len(ks_sorted) > 6 else ""
        watch_parts.append(f"L{li}[{preview}{suffix}]")
    print(f"  [{skeleton_mode}] Watching top-{k_sel} targets: {' '.join(watch_parts)}")

    def _signal_and_eff(sp):
        if sp is None:
            net_eff = (delta_mat[[j], :] @ V_src).squeeze(0)   # [d_model]
            return 1.0, net_eff
        signal = delta_mat[j, sp].item()
        eff    = V_src[sp, :]
        return signal, eff

    print(f"  Patching {k_sel} top + {len(mid_k_pos)} middle + "
          f"{len(rand_k_pos)} rand + {k_sel} bot paths …")

    unique_sps = sorted(
        set(top_sp) | set(mid_sp) | set(rand_sp) | set(bot_sp),
        key=lambda x: (x is None, x if x is not None else 0),
    )
    n_passes = len(unique_sps)
    print(f"  {n_passes} unique subspace forward pass(es) "
          f"(vs {k_sel + len(mid_k_pos) + len(rand_k_pos) + k_sel} naive)")

    # A job can be preempted before a whole neuron finishes.  Store each
    # completed subspace separately under the experiment hash and reuse it on
    # the next launch.  The hash prevents mixing incompatible configurations.
    checkpoint_dir = (
        os.path.join(output_dir, ".subspace_checkpoints", experiment_hash)
        if experiment_hash else None
    )
    expected_delta_size = sum(baseline_acts[li].shape[1] for li in target_layers)
    delta_cache: dict = {}
    pending_sps = []
    for sp in unique_sps:
        cached = None
        if checkpoint_dir:
            cached = _load_subspace_checkpoint(
                _subspace_checkpoint_path(checkpoint_dir, sp), expected_delta_size
            )
        if cached is None:
            pending_sps.append(sp)
        else:
            delta_cache[sp] = cached
    if delta_cache:
        print(f"  Resume subspaces: {len(delta_cache)} cached, "
              f"{len(pending_sps)} remaining")

    if intervention_chunk_size <= 0:
        raise ValueError("intervention_chunk_size must be positive")
    subspace_iter = tqdm(
        range(0, len(pending_sps), intervention_chunk_size),
        total=(len(pending_sps) + intervention_chunk_size - 1) // intervention_chunk_size,
        desc=f"    neuron {j} intervention batches",
        unit="batch",
        dynamic_ncols=True,
        disable=n_passes <= 4,
    )

    def collect_specs_with_oom_fallback(specs):
        try:
            return collect_batched_patched_deltas(
                model, tokenizer, prompts, baseline_acts,
                start_layer, target_layers, j, specs, intervention, device,
                mean_val=mean_val, scale=scale,
                batch_size=batch_size,
                add_special_tokens=add_special_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            if len(specs) <= 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(specs) // 2
            print(
                f"  [OOM fallback] intervention batch {len(specs)} -> "
                f"{midpoint} + {len(specs) - midpoint}"
            )
            left = collect_specs_with_oom_fallback(specs[:midpoint])
            right = collect_specs_with_oom_fallback(specs[midpoint:])
            left.update(right)
            return left

    for chunk_start in subspace_iter:
        chunk = pending_sps[chunk_start:chunk_start + intervention_chunk_size]
        specs = []
        for sp in chunk:
            signal, eff = _signal_and_eff(sp)
            specs.append((sp, signal, eff))
        chunk_deltas = collect_specs_with_oom_fallback(specs)
        for sp, delta_values in chunk_deltas.items():
            delta_cache[sp] = delta_values
            if checkpoint_dir:
                _save_subspace_checkpoint(
                    _subspace_checkpoint_path(checkpoint_dir, sp), delta_values
                )

    top_abs  = np.array([delta_cache[sp][k] for sp, k in zip(top_sp,  top_k_pos)])
    mid_abs  = np.array([delta_cache[sp][k] for sp, k in zip(mid_sp,  mid_k_pos)])
    bot_abs  = np.array([delta_cache[sp][k] for sp, k in zip(bot_sp,  bot_k_pos)])
    rand_abs = np.array([delta_cache[sp][k] for sp, k in zip(rand_sp, rand_k_pos)])

    def _path_details(full_indices, subspaces, target_positions, causal_values):
        """Serialize exact route identities alongside their measured effects."""
        if not (
            len(full_indices) == len(subspaces)
            == len(target_positions) == len(causal_values)
        ):
            raise RuntimeError("path metadata and causal-value lengths disagree")

        records = []
        for rank, (full_idx, sp, target_pos, causal_value) in enumerate(
            zip(full_indices, subspaces, target_positions, causal_values), 1
        ):
            target_layer, target_sensor_idx = _decode(int(target_pos))
            if skeleton_mode == "complete_skeleton_fullinfo":
                structural_score = float(flat[int(target_pos)].item())
            else:
                structural_score = float(
                    (delta_row[int(sp)] * gamma[int(sp), int(target_pos)]).item()
                )
            records.append({
                "rank": rank,
                "path_flat_index": int(full_idx),
                "subspace_idx": None if sp is None else int(sp),
                "target_flat_index": int(target_pos),
                "target_layer": int(target_layer),
                "target_sensor_idx": int(target_sensor_idx),
                "structural_score": structural_score,
                "causal_abs_delta": float(causal_value),
            })
        return records

    path_details = {
        "top": _path_details(top_flat, top_sp, top_k_pos, top_abs),
        "middle": _path_details(mid_flat, mid_sp, mid_k_pos, mid_abs),
        "random": _path_details(rand_flat, rand_sp, rand_k_pos, rand_abs),
        "bottom": _path_details(bot_flat, bot_sp, bot_k_pos, bot_abs),
    }

    mw_p_mid = mw_p_rand = mw_p_bot = 1.0
    try:
        if top_abs.std() > 1e-12 and mid_abs.std() > 1e-12:
            _, mw_p_mid = mannwhitneyu(top_abs, mid_abs, alternative="greater")
    except Exception:
        pass
    try:
        if top_abs.std() > 1e-12 and rand_abs.std() > 1e-12:
            _, mw_p_rand = mannwhitneyu(top_abs, rand_abs, alternative="greater")
    except Exception:
        pass
    try:
        if top_abs.std() > 1e-12 and bot_abs.std() > 1e-12:
            _, mw_p_bot = mannwhitneyu(top_abs, bot_abs, alternative="greater")
    except Exception:
        pass
    print(
        f"  j={j:4d} | "
        f"|Δ| Top={top_abs.mean():.5f}  "
        f"Middle={mid_abs.mean():.5f}  "
        f"Rand={rand_abs.mean():.5f}  "
        f"Bot={bot_abs.mean():.5f}  "
        f"| P(T>M)={mw_p_mid:.1e}  P(T>R)={mw_p_rand:.1e}  P(T>B)={mw_p_bot:.1e}"
    )

    if print_cpl_delta_abs:
        n_rows = min(len(top_k_pos), len(rand_k_pos), len(bot_k_pos))
        if skeleton_mode == "complete_skeleton_fullinfo":
            print(f"  {'#':>3}  {'CplStr':>9}  {'Top target':^12}  {'|Δ|Top':>9}  "
                  f"{'Rand target':^12}  {'|Δ|Rand':>9}  "
                  f"{'Bot target':^12}  {'|Δ|Bot':>9}  T>R T>B R>B")
            print(f"  {'─'*106}")
            for rank in range(1, n_rows + 1):
                idx = rank - 1
                tp, rp, bp = top_k_pos[idx], rand_k_pos[idx], bot_k_pos[idx]
                ta, ra, ba = top_abs[idx],   rand_abs[idx],   bot_abs[idx]
                cpl        = top_cpl[idx]
                tl, tn = _decode(tp);  rl, rn = _decode(rp);  bl, bn = _decode(bp)
                w_tr = "✓" if ta > ra else " "
                w_tb = "✓" if ta > ba else " "
                w_rb = "✓" if ra > ba else " "
                print(f"  {rank:>3}  {cpl:>+9.5f}  L{tl}:{tn:<7}  {ta:>9.5f}  "
                      f"L{rl}:{rn:<7}  {ra:>9.5f}  "
                      f"L{bl}:{bn:<7}  {ba:>9.5f}   {w_tr}   {w_tb}  {w_rb}")
        else:
            print(f"  {'#':>3}  {'SP':>4}  {'CplStr':>9}  {'Top target':^12}  {'|Δ|Top':>9}  "
                  f"{'Rand target':^12}  {'|Δ|Rand':>9}  "
                  f"{'Bot target':^12}  {'|Δ|Bot':>9}  T>R T>B R>B")
            print(f"  {'─'*114}")
            for rank in range(1, n_rows + 1):
                idx = rank - 1
                sp          = top_sp[idx]
                tp, rp, bp  = top_k_pos[idx], rand_k_pos[idx], bot_k_pos[idx]
                ta, ra, ba  = top_abs[idx],   rand_abs[idx],   bot_abs[idx]
                cpl         = top_cpl[idx]
                tl, tn = _decode(tp);  rl, rn = _decode(rp);  bl, bn = _decode(bp)
                w_tr = "✓" if ta > ra else " "
                w_tb = "✓" if ta > ba else " "
                w_rb = "✓" if ra > ba else " "
                print(f"  {rank:>3}  i={sp:<3}  {cpl:>+9.5f}  L{tl}:{tn:<7}  {ta:>9.5f}  "
                      f"L{rl}:{rn:<7}  {ra:>9.5f}  "
                      f"L{bl}:{bn:<7}  {ba:>9.5f}   {w_tr}   {w_tb}  {w_rb}")

    # No per-neuron plot here by design: only the pooled summary (2 plots,
    # generated once per job by plot_patching_results_generic.plot_all after
    # every sampled neuron finishes) is written — with N_NEURONS_SAMPLE=300
    # per job, a per-neuron plot would mean 300 extra image writes per job.

    summary = {
        "config": {
            "start_layer":  start_layer,
            "src_neuron":   j,
            "topk":         k_sel,
            "intervention": intervention,
            "experiment_hash": experiment_hash,
            "path_chunk_subspaces": path_chunk_subspaces,
            "intervention_chunk_size": intervention_chunk_size,
        },
        "stats": {
            "top_mean":      float(top_abs.mean()),
            "middle_mean":   float(mid_abs.mean()),
            "rand_mean":     float(rand_abs.mean()),
            "bot_mean":      float(bot_abs.mean()),
            "top_gt_middle": bool(top_abs.mean() > mid_abs.mean()),
            "top_gt_rand":   bool(top_abs.mean() > rand_abs.mean()),
            "top_gt_bot":    bool(top_abs.mean() > bot_abs.mean()),
            "p_top_gt_middle": float(mw_p_mid),
            "p_top_gt_rand": float(mw_p_rand),
            "p_top_gt_bot":  float(mw_p_bot),
        },
        "top_abs":  top_abs.tolist(),
        "middle_abs": mid_abs.tolist(),
        "rand_abs": rand_abs.tolist(),
        "bot_abs":  bot_abs.tolist(),
        "path_details": path_details,
    }

    _save_json_atomic(os.path.join(output_dir, "patching_results.json"), summary)

    # The compact final result now makes the much larger transient per-subspace
    # cache redundant.  Only remove the directory belonging to this exact hash.
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    return summary
