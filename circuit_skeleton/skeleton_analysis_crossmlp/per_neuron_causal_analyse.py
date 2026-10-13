"""
per_neuron_causal_analyse.py

Core causal analysis for a single source neuron j:
    j  →  subspace i  →  target neuron k

For each path in top-k / bot-k / random-k:
  1. Intervene ONLY on the j→i contribution via a post-hook on c_proj.
  2. Measure mean |Δa_k| at the corresponding target neuron.
  3. Print a per-path comparison table and save a summary JSON.

"""

import json
import os

import numpy as np
import torch
from scipy.stats import mannwhitneyu

from circuit_skeleton.skeleton_analysis_crossmlp.subspace_intervention_hook import (
    collect_patched_activations,
)
from circuit_skeleton.skeleton_analysis_crossmlp.coupling_path_selector import (
    _abs_delta_at,
    _layer_restricted_pool,
)
from circuit_skeleton.skeleton_analysis_crossmlp.result_aggregate import _plot_delta_bar


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
) -> dict:
    """Causal patching analysis for source neuron *j*.

    skeleton_mode="complete_skeleton_global"    → rank paths by |delta[j,i]*gamma[i,k]|
    skeleton_mode="complete_skeleton_fullinfo" → rank paths by |Σ_i delta[j,i]*gamma[i,k]|

    Returns a summary dict with per-path abs-delta arrays and aggregate stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_cross = cs.total_cross
    delta_mat   = cs.within_ske.delta.cpu()   # [d_ff, n_sp]
    V_src       = cs.within_ske.V.cpu()       # [n_sp, d_model]

    def _decode(flat_k):
        return cs.flat_to_layer_neuron(flat_k)

    # Select top-k and bot-k paths  
    if skeleton_mode == "complete_skeleton_fullinfo":
        flat      = cs.w_contracted([j]).squeeze(0).cpu()    # [total_cross]
        n_flat    = flat.shape[0]
        k_sel     = min(topk, n_flat // 3)                   # room for top + bot + rand
        top_flat  = torch.topk(flat.abs(), k_sel, largest=True ).indices.tolist()
        bot_flat  = torch.topk(flat.abs(), k_sel, largest=False).indices.tolist()
        top_k_pos = top_flat
        bot_k_pos = bot_flat
        top_sp    = [None] * k_sel
        bot_sp    = [None] * k_sel
        top_cpl   = [flat[fi].item() for fi in top_flat]
        bot_cpl   = [flat[fi].item() for fi in bot_flat]  # noqa: F841
    else:
        full      = cs.w_full_path([j]).reshape(-1).cpu()    # [n_sp * total_cross]
        n_flat    = full.shape[0]
        k_sel     = min(topk, n_flat // 3)
        top_flat  = torch.topk(full.abs(), k_sel, largest=True ).indices.tolist()
        bot_flat  = torch.topk(full.abs(), k_sel, largest=False).indices.tolist()
        top_k_pos = [fi % total_cross for fi in top_flat]
        bot_k_pos = [fi % total_cross for fi in bot_flat]
        top_sp    = [fi // total_cross for fi in top_flat]
        bot_sp    = [fi // total_cross for fi in bot_flat]
        top_cpl   = [full[fi].item() for fi in top_flat]
        bot_cpl   = [full[fi].item() for fi in bot_flat]  # noqa: F841

    # Random baseline (excluding top & bot)  
    sel_layers = (
        {_decode(k)[0] for k in top_k_pos} |
        {_decode(k)[0] for k in bot_k_pos}
    )
    exclude   = set(top_flat) | set(bot_flat)
    available = _layer_restricted_pool(
        sel_layers, n_flat, total_cross,
        cs.layer_offsets, skeleton_mode, exclude,
    )
    rng       = np.random.default_rng(seed=42)   # fixed seed for reproducibility
    rand_flat = (
        rng.choice(available, size=min(k_sel, len(available)), replace=False).tolist()
        if available else []
    )
    if skeleton_mode == "complete_skeleton_fullinfo":
        rand_k_pos = rand_flat
        rand_sp    = [None] * len(rand_flat)
    else:
        rand_k_pos = [fi % total_cross for fi in rand_flat]
        rand_sp    = [fi // total_cross for fi in rand_flat]

    # Print which target neurons are being watched  
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

    #  Per-path helpers
    def _signal_and_eff(sp):
        """
        Return (signal, eff) for one subspace.
          signal = delta[j, sp]  = S[sp] * U[j, sp]   (scalar — coupling amplitude)
          eff    = V_src[sp, :]                         (Tensor [d_model] — effector direction)
        For fullinfo mode (sp=None): net effector with signal=1.
        """
        if sp is None:
            net_eff = (delta_mat[[j], :] @ V_src).squeeze(0)   # [d_model]
            return 1.0, net_eff
        signal = delta_mat[j, sp].item()    # S[sp] * U[j,sp]
        eff    = V_src[sp, :]               # V[sp,:], shape [d_model]
        return signal, eff

    # Run patching: one forward pass per unique subspace, then look up results
    # by indexing the cached full delta vector — O(n_sp) passes instead of O(3*k_sel).
    print(f"  Patching {k_sel} top + {k_sel} bot + {len(rand_k_pos)} rand paths …")

    unique_sps = sorted(
        set(top_sp) | set(bot_sp) | set(rand_sp),
        key=lambda x: (x is None, x if x is not None else 0),
    )
    n_passes = len(unique_sps)
    print(f"  {n_passes} unique subspace forward pass(es) "
          f"(vs {k_sel + k_sel + len(rand_k_pos)} naive)")

    # One forward pass per unique subspace; cache mean|Δ| for every target neuron
    delta_cache: dict = {}   # sp  →  np.ndarray [total_cross]
    for pass_idx, sp in enumerate(unique_sps, 1):
        if n_passes > 4:
            print(f"    subspace {pass_idx}/{n_passes}", end="\r", flush=True)
        signal, eff = _signal_and_eff(sp)
        patched = collect_patched_activations(
            model, tokenizer, prompts,
            start_layer, target_layers,
            j, signal, eff, intervention, device,
            mean_val=mean_val, scale=scale,
            batch_size=batch_size,
        )
        delta_cache[sp] = np.concatenate(
            [np.abs(patched[li] - baseline_acts[li]).mean(axis=0) for li in target_layers]
        )   # [total_cross]

    if n_passes > 4:
        print()  # clear the \r progress line

    top_abs  = np.array([delta_cache[sp][k] for sp, k in zip(top_sp,  top_k_pos)])
    bot_abs  = np.array([delta_cache[sp][k] for sp, k in zip(bot_sp,  bot_k_pos)])
    rand_abs = np.array([delta_cache[sp][k] for sp, k in zip(rand_sp, rand_k_pos)])

    # Statistical tests  
    mw_p_rand = mw_p_bot = 1.0
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
        f"Rand={rand_abs.mean():.5f}  "
        f"Bot={bot_abs.mean():.5f}  "
        f"| P(T>R)={mw_p_rand:.1e}  P(T>B)={mw_p_bot:.1e}"
    )

    # Per-target detail table (only when --print_cpl_delta_abs is set)
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

    # Plot  
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    _plot_delta_bar(
        top_abs, rand_abs, bot_abs,
        j, start_layer, intervention,
        os.path.join(plot_dir, "absolute_delta_comparison.png"),
    )

    # Summary dict + JSON  
    summary = {
        "config": {
            "start_layer":  start_layer,
            "src_neuron":   j,
            "topk":         k_sel,
            "intervention": intervention,
        },
        "stats": {
            "top_mean":      float(top_abs.mean()),
            "rand_mean":     float(rand_abs.mean()),
            "bot_mean":      float(bot_abs.mean()),
            "top_gt_rand":   bool(top_abs.mean() > rand_abs.mean()),
            "top_gt_bot":    bool(top_abs.mean() > bot_abs.mean()),
            "p_top_gt_rand": float(mw_p_rand),
            "p_top_gt_bot":  float(mw_p_bot),
        },
        "top_abs":  top_abs.tolist(),
        "rand_abs": rand_abs.tolist(),
        "bot_abs":  bot_abs.tolist(),
    }

    with open(os.path.join(output_dir, "patching_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
