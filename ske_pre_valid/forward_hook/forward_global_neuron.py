"""
forward_global_neuron.py  —  global (per-layer) static skeleton selection.
Used by scripts_ske_cir/run_per_layer_static_ske.py.

Selection modes (weight_eff / uniform_eff / sqrt_eff / weight_delta / sqrt_delta /
eff_dominate / eff_dominate_threh): see select_global_abs_topk docstring.
"""

import torch


def allocate_weighted_quotas(weights, k, capacity):
    """Allocate as many as ``k`` slots without losing overflow at capacity.

    ``capacity`` may be one scalar shared by every subspace or a length-N
    tensor/list.  The old implementation clamped an already-rounded quota;
    when a high-weight subspace exceeded its capacity, those slots simply
    disappeared instead of being reassigned to another subspace.
    """
    weights = torch.as_tensor(weights).detach().cpu().float().flatten().abs()
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    n_sp = weights.numel()

    if isinstance(capacity, int):
        capacities = torch.full((n_sp,), max(capacity, 0), dtype=torch.long)
    else:
        capacities = torch.as_tensor(capacity).detach().cpu().long().flatten().clamp(min=0)
        if capacities.numel() != n_sp:
            raise ValueError(
                f"capacity has {capacities.numel()} entries, expected {n_sp}"
            )

    target = min(max(int(k), 0), int(capacities.sum().item()))
    quotas = torch.zeros(n_sp, dtype=torch.long)
    remaining = target

    # Each iteration either assigns at least one slot or saturates one or more
    # subspaces.  Recomputing after saturation redistributes their overflow.
    while remaining > 0:
        room = capacities - quotas
        active_idx = (room > 0).nonzero(as_tuple=True)[0]
        if active_idx.numel() == 0:
            break

        active_weights = weights[active_idx]
        if active_weights.sum().item() <= 0:
            active_weights = torch.ones_like(active_weights)

        raw = active_weights / active_weights.sum() * remaining
        additions = torch.minimum(raw.floor().long(), room[active_idx])
        n_added = int(additions.sum().item())
        if n_added > 0:
            quotas[active_idx] += additions
            remaining -= n_added
            continue

        # All proportional shares are below one.  Assign one slot to the
        # largest shares; argsort also gives deterministic tie handling.
        n_take = min(remaining, active_idx.numel())
        chosen = active_idx[torch.argsort(raw, descending=True)[:n_take]]
        quotas[chosen] += 1
        remaining -= n_take

    return quotas


def select_global_abs_topk(delta, V, W_emb_target, k,
                            precomputed_effector=None,
                            weight_eff=False, uniform_eff=False, sqrt_eff=False,
                            soft_gate=False,
                            weight_delta=False, sqrt_delta=False,
                            eff_dominate=False,
                            eff_dominate_threh=False, delta_topk_pct=50.0,
                            filter_pos_eff=False):
    """
    Static path selection for one layer.

    effector computation:
        precomputed_effector is None  → effector[i] = V[i,:] @ W_emb_target
        precomputed_effector given    → use the caller's gamma-corrected raw-dot scores

    default (all models, ske_preserve_gate):
        Global top-k (j,i) by |delta[j,i]*effector[i]|.

    weight_eff=True  (all models, used by ske_preserve_gate_weight_eff baseline):
        Allocate k paths proportionally to |effector[i]|.  Within each subspace
        take neurons in |delta[j,i]| descending order.  Low-|eff| subspaces may
        receive zero paths, preventing denominator inflation under RMSNorm.

    uniform_eff=True  (all models, used by ske_preserve_gate_eff baseline):
        Allocate k // n_sp paths to every subspace (equal quota).  Remainder
        is distributed to the first subspaces.  Within each subspace take
        neurons in |delta[j,i]| descending order.
        static_score = delta * effector[i] for the runtime sign-gate.

    sqrt_eff=True  (all models, used by ske_preserve_gate_sqrteff baseline):
        Allocate k paths proportionally to sqrt(|effector[i]|).  Intermediate
        between uniform_eff and weight_eff — high-|eff| subspaces receive more
        paths but the concentration is compressed vs. linear weighting.
        static_score = delta * effector[i] for the runtime sign-gate.

    weight_delta=True  (all models, used by ske_preserve_gate_weight_delta baseline):
        Allocate k paths proportionally to max_j |delta[j,i]| per subspace.
        Delta magnitudes vary significantly across subspaces (~10× range), giving a smooth but
        non-uniform allocation that reflects each subspace's intrinsic strength.
        static_score = delta * effector[i] for the runtime sign-gate.

    sqrt_delta=True  (all models, used by ske_preserve_gate_sqrtdelta baseline):
        Allocate k paths proportionally to sqrt(max_j |delta[j,i]|).
        Compresses the delta variation further — intermediate between weight_delta
        (moderate concentration) and uniform_eff (flat).
        static_score = delta * effector[i] for the runtime sign-gate.

    eff_dominate=True  (all models, used by ske_preserve_gate_eff_dominate baseline):
        Target-specific allocation: quota ∝ |V[i] @ W_emb[target]| (raw dot product).
        Recomputes the raw-dot effector, so the selected skeleton varies with the target word.
        Within each subspace, neurons are selected in |delta[j,i]| descending order.
        static_score = delta * raw_eff[i] for the runtime sign-gate.

    eff_dominate_threh=True  (all models, used by ske_preserve_gate_eff_dominate_threh):
        Same as eff_dominate but first applies a global percentile threshold on |delta|:
        only neurons where |delta[j,i]| >= global quantile(1 - delta_topk_pct/100) are
        eligible for selection.  Quota per subspace is capped at eligible neuron count.
        Prevents low-delta neurons from diluting the skeleton when quota is large.
        delta_topk_pct=50.0 → top-50% of |delta| values are eligible (default).
        static_score = delta * raw_eff[i] for the runtime sign-gate.

    filter_pos_eff=True  (used by ske_preserve_gate_pos_eff baseline):
        Applies to the default global top-k mode only.  After computing the score matrix,
        zeroes out any subspace i where effector[i] <= 0 (V[i]@W_emb_target, or the
        gamma-corrected/precomputed_effector value when the caller supplies one), so
        paths through negatively-aligning subspaces are never selected.  k stays the
        same as ske_preserve_gate; if positive-eff subspaces cannot fill k slots, the
        remaining slots have static_score=0 and are naturally gated out at runtime.
    """
    delta = delta.cpu()
    if precomputed_effector is not None:
        effector = precomputed_effector.cpu()
    else:
        effector = (V @ W_emb_target).cpu()

    d_ff, n_sp = delta.shape
    k = min(k, d_ff * n_sp)

    def _select_by_quota(delta, effector, n_paths_per_sp, d_ff):
        """Given per-subspace quotas, select top-|delta| neurons within each subspace."""
        j_sorted     = torch.argsort(delta.abs(), dim=0, descending=True)
        delta_sorted = torch.gather(delta, 0, j_sorted)
        rank_grid    = torch.arange(d_ff).unsqueeze(1) < n_paths_per_sp.unsqueeze(0)
        nr_idx, i_idx = rank_grid.nonzero(as_tuple=True)
        j_idx        = j_sorted[nr_idx, i_idx]
        dv           = delta_sorted[nr_idx, i_idx]
        static_score = dv * effector[i_idx]
        return j_idx, i_idx, dv, static_score

    if weight_delta:
        weights      = delta.abs().max(dim=0).values          # [n_sp]
        n_paths      = allocate_weighted_quotas(weights, k, d_ff)
        j_idx, i_idx, dv, static_score = _select_by_quota(delta, effector, n_paths, d_ff)

    elif sqrt_delta:
        weights      = delta.abs().max(dim=0).values.sqrt()   # [n_sp]
        n_paths      = allocate_weighted_quotas(weights, k, d_ff)
        j_idx, i_idx, dv, static_score = _select_by_quota(delta, effector, n_paths, d_ff)

    elif sqrt_eff:
        weights = effector.abs().sqrt()
        n_paths = allocate_weighted_quotas(weights, k, d_ff)
        j_idx, i_idx, dv, static_score = _select_by_quota(delta, effector, n_paths, d_ff)

    elif uniform_eff:
        q      = k // n_sp
        rem    = k % n_sp
        n_paths = torch.full((n_sp,), q, dtype=torch.long)
        if rem > 0:
            n_paths[:rem] += 1
        n_paths = n_paths.clamp(max=d_ff)
        j_idx, i_idx, dv, static_score = _select_by_quota(delta, effector, n_paths, d_ff)

    elif weight_eff:
        weights = effector.abs()
        n_paths = allocate_weighted_quotas(weights, k, d_ff)
        j_idx, i_idx, dv, static_score = _select_by_quota(delta, effector, n_paths, d_ff)

    elif eff_dominate_threh:
        # Uses the already-resolved `effector` (== precomputed_effector when given, e.g.
        # gamma-corrected for Gemma-2/3, else a fresh raw V@W_emb_target dot) instead of
        # recomputing an uncorrected dot here — that used to silently drop gamma even
        # when the caller had already supplied a corrected precomputed_effector.
        raw_eff  = effector.float()                               # [n_sp], target-specific
        # Step 1: global top-delta_topk_pct% threshold on |delta|
        q_val    = 1.0 - delta_topk_pct / 100.0
        thresh   = torch.quantile(delta.abs().flatten(), q_val)
        eligible = delta.abs() >= thresh                          # [d_ff, n_sp] bool mask
        # Step 2: eff-proportional quota, capped at eligible count per subspace
        n_paths  = allocate_weighted_quotas(
            raw_eff.abs(), k, eligible.sum(dim=0).long()
        )
        # Step 3: sort within each subspace — eligible neurons first (by |delta| desc),
        #         ineligible pushed to bottom via sort_key = -1
        sort_key     = delta.abs().masked_fill(~eligible, -1.0)
        j_sorted     = torch.argsort(sort_key, dim=0, descending=True)
        delta_sorted = torch.gather(delta, 0, j_sorted)
        rank_grid    = torch.arange(d_ff).unsqueeze(1) < n_paths.unsqueeze(0)
        nr_idx, i_idx = rank_grid.nonzero(as_tuple=True)
        j_idx        = j_sorted[nr_idx, i_idx]
        dv           = delta_sorted[nr_idx, i_idx]
        static_score = dv * raw_eff[i_idx]

    elif eff_dominate:
        # Target-specific allocation: quota ∝ |effector[i]| (raw dot, or gamma-corrected
        # when the caller supplies precomputed_effector — see eff_dominate_threh above;
        # `effector` is already resolved to whichever applies at the top of this function).
        raw_eff = effector.float()   # [n_sp], target-specific
        n_paths = allocate_weighted_quotas(raw_eff.abs(), k, d_ff)
        j_idx, i_idx, dv, static_score = _select_by_quota(delta, raw_eff, n_paths, d_ff)

    elif soft_gate:
        score_mat = delta.abs().sqrt() * effector.abs().unsqueeze(0)  # [d_ff, n_sp]
        flat      = score_mat.flatten()
        top_flat  = torch.topk(flat, k).indices
        j_idx        = (top_flat // n_sp).long()
        i_idx        = (top_flat %  n_sp).long()
        dv           = delta[j_idx, i_idx]
        static_score = dv * effector[i_idx]

    else:
        static_score_mat = delta * effector.unsqueeze(0)    # [d_ff, n_sp]
        if filter_pos_eff:
            # Zero out subspaces where effector[i] <= 0 so they are never selected. Reuses
            # the already-resolved `effector` (== precomputed_effector when given, e.g.
            # gamma-corrected for Gemma-2/3) rather than recomputing an uncorrected
            # V@W_emb_target dot here, which used to silently drop gamma correction even
            # when the caller had already supplied a corrected precomputed_effector.
            # k is unchanged — the skeleton size stays the same as ske_preserve_gate; if
            # positive-eff subspaces cannot fill k slots, zero-score entries appear last
            # and are gated out at runtime.
            pos_mask = effector > 0                          # [n_sp]
            static_score_mat = static_score_mat * pos_mask.float().unsqueeze(0)
        flat     = static_score_mat.flatten()
        top_flat = torch.topk(flat.abs(), k).indices
        j_idx        = (top_flat // n_sp).long()
        i_idx        = (top_flat %  n_sp).long()
        dv           = delta[j_idx, i_idx]
        static_score = static_score_mat[j_idx, i_idx]

    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    dv,
        "static_scores": static_score,
        "effector":      effector,
    }
