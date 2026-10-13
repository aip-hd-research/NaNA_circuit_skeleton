"""
eff_select.py  —  effector-alignment-aware static skeleton selection.

All methods use raw logit V[i]@W[target_id] as effector (not signed-abs-softmax).
Softmax over large vocab compresses values to ~1/vocab with max/median < 2.5x;
raw logit has orders-of-magnitude variation with the same sign.

Parameters
----------
delta    : Tensor[d_ff, n_sp]  — signed neuron-subspace coupling strengths
effector : Tensor[n_sp]        — raw logit V[i]@W[target_id], signed
k        : int                 — number of paths to select

Returns
-------
dict: j_idx, i_idx, delta_vals, static_scores, effector
    static_scores carries sign(delta×eff) for use with the gate: h·score > 0
"""

import torch

from ske_pre_valid.forward_hook.forward_global_neuron import allocate_weighted_quotas


def _global_topk(rank_mat: torch.Tensor, delta: torch.Tensor,
                 effector: torch.Tensor, k: int) -> dict:
    """Shared top-k extraction from a [d_ff, n_sp] ranking matrix."""
    d_ff, n_sp = rank_mat.shape
    k = min(k, d_ff * n_sp)
    top_flat = torch.topk(rank_mat.flatten(), k).indices
    j_idx    = (top_flat // n_sp).long()
    i_idx    = (top_flat %  n_sp).long()
    dv       = delta[j_idx, i_idx]
    # Gate sign: sign(delta × effector) for use with forward_static_preserve
    gate_sign    = (dv * effector[i_idx]).sign()
    static_score = gate_sign * rank_mat[j_idx, i_idx]
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    dv,
        "static_scores": static_score,
        "effector":      effector,
    }


def select_eff_delta_rank(delta: torch.Tensor, effector: torch.Tensor,
                          k: int) -> dict:
    """
    Method D — Harmonic within-subspace discounting × effector  (continuous weight_eff).

    score[j,i] = |effector[i]| / (desc_rank_within_sp[j,i] + 1)

    desc_rank_within_sp[j,i] ∈ {0, 1, …, d_ff-1}: 0-based rank of |delta[j,i]| among
    all neurons in subspace i in DESCENDING order (0 = strongest neuron).

    Interpretation: the m-th best neuron in subspace i contributes score |eff[i]|/m.
    This models diminishing returns — adding more paths to the same subspace yields
    progressively less marginal score.

    Relation to ske_preserve_gate_weight_eff
    -----------------------------------------
    weight_eff does two separate integer passes:
      1.  quota[i] = floor(k × |eff[i]| / Σ|eff|)  — hard integer quota per subspace
      2.  within each subspace take top-quota[i] neurons by |delta|

    This method is the exact single-pass continuous equivalent:
      · k < n_sp:  top-k = one path per top-k subspace (by |eff|), same as round-robin
      · k >> n_sp: paths per subspace ∝ |eff[i]|, asymptotically identical to weight_eff
      · no integer quota rounding; no zero-cutoff for low-|eff| subspaces
      · joint global top-k, single pass

    Caller should pass raw logit V[i]@W[target] (not signed-abs-softmax prob).
    """
    delta    = delta.cpu().float()
    effector = effector.cpu().float()
    eff_abs  = effector.abs()                                               # [n_sp]

    # Descending rank within each subspace: 0 = best (largest |delta|), d_ff-1 = worst
    delta_abs  = delta.abs()                                                # [d_ff, n_sp]
    asc_ranks  = torch.argsort(torch.argsort(delta_abs, dim=0), dim=0)    # ascending
    desc_ranks = (delta_abs.shape[0] - 1 - asc_ranks).float()             # descending, [d_ff, n_sp]

    # Harmonic score: best neuron in sp i competes at |eff[i]|/1,
    # 2nd best at |eff[i]|/2, etc.  Creates natural diminishing returns.
    rank_mat = eff_abs.unsqueeze(0) / (desc_ranks + 1.0)                  # [d_ff, n_sp]
    return _global_topk(rank_mat, delta, effector, k)


def select_eff_delta_weight_rank(delta: torch.Tensor, effector: torch.Tensor,
                                  k: int) -> dict:
    """
    Method E — weight_eff quota with rank-normalised effector.

    Equivalent to ske_preserve_gate_weight_eff but replaces the signed-abs-softmax
    effector (nearly uniform, max/median ≈ 1.2x → round-robin) with the
    rank-percentile of the raw logit:

        eff_rank[i] = rank_pct(|eff_raw[i]|)  ∈  {0/n_sp, 1/n_sp, …, 1}

    Quota allocation (same as weight_eff):
        quota[i] = floor(k × eff_rank[i] / Σ eff_rank[i])
        remainder distributed to highest-rank subspaces.

    Within each subspace: take top-quota[i] neurons by |delta[j,i]|.

    Comparison with weight_eff
    --------------------------
    weight_eff    : quota ∝ |eff_softmax| ≈ uniform  → quota[i] ≈ k/n_sp  (round-robin)
    this method   : quota ∝ rank_pct      ∈ [0,1]    → top subspace gets 2× median quota
    max/min ratio : 2× (rank-pct) vs 1.2× (softmax) vs ∞ (raw logit, too extreme)

    Caller should pass raw logit V[i]@W[target] (not signed-abs-softmax prob).
    """
    delta    = delta.cpu().float()
    effector = effector.cpu().float()
    eff_abs  = effector.abs()                                               # [n_sp]
    n_sp     = eff_abs.shape[0]
    d_ff     = delta.shape[0]

    # Rank-percentile of |eff_raw|: ascending 0-based rank / (n_sp-1) ∈ [0, 1]
    asc_eff_ranks = torch.argsort(torch.argsort(eff_abs)).float()          # [n_sp]
    eff_rank_pct  = asc_eff_ranks / max(n_sp - 1, 1)                      # [n_sp] ∈ [0, 1]

    # Capacity-aware quota allocation: if a high-rank subspace fills up, its
    # overflow is reassigned instead of silently reducing the requested k.
    n_paths_per_sp = allocate_weighted_quotas(eff_rank_pct, k, d_ff)

    # Within each subspace: top neurons by |delta|
    j_sorted     = torch.argsort(delta.abs(), dim=0, descending=True)     # [d_ff, n_sp]
    delta_sorted = torch.gather(delta, 0, j_sorted)                        # signed [d_ff, n_sp]

    rank_grid        = torch.arange(d_ff).unsqueeze(1) < n_paths_per_sp.unsqueeze(0)
    neuron_rank_idx, i_idx = rank_grid.nonzero(as_tuple=True)
    j_idx        = j_sorted[neuron_rank_idx, i_idx]
    dv           = delta_sorted[neuron_rank_idx, i_idx]
    gate_sign    = (dv * effector[i_idx]).sign()
    static_score = gate_sign * eff_rank_pct[i_idx]                        # gate-compatible

    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    dv,
        "static_scores": static_score,
        "effector":      effector,
    }

