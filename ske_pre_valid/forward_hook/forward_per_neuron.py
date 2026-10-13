"""
forward_per_neuron.py  —  per-neuron static skeleton selection.
Used by scripts_ske_cir/run_per_neuron_static_ske.py and
scripts_ske_cir/run_static_ske_per_neuron.py.

Selection: for each neuron j, top-k / random / bottom-k subspaces by
|delta[j,i] * effector[i]|.
"""

import torch


def select_per_neuron_abs_topk(delta, V, W_emb_target, k_per_neuron, precomputed_effector=None):
    """
    For each neuron j, select top-k subspaces by |delta[j,i] * effector[i]|.

    effector computation:
        precomputed_effector is None  (default / original):
            effector[i] = V[i,:] @ W_emb_target    raw dot product, signed
        precomputed_effector is given:
            use the caller's gamma-corrected raw-dot scores.

    Static: no input h.  Inference gating by h[j] * static_score[j,i] > 0
    is applied in forward_static_preserve (base.py).
    """
    delta = delta.cpu()
    if precomputed_effector is not None:
        effector = precomputed_effector.cpu()
    else:
        effector = (V @ W_emb_target).cpu()
    static_score = delta * effector.unsqueeze(0)   # [d_ff, n_sp]

    d_ff, n_sp = delta.shape
    k_each     = min(k_per_neuron, n_sp)
    top_i_mat  = static_score.abs().topk(k_each, dim=1).indices  # [d_ff, k_each]
    j_idx      = torch.arange(d_ff).unsqueeze(1).expand(-1, k_each).flatten().long()
    i_idx      = top_i_mat.flatten().long()
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    delta[j_idx, i_idx],
        "static_scores": static_score[j_idx, i_idx],
        "effector":      effector,
    }


def select_per_neuron_random(delta, V, W_emb_target, k_per_neuron,
                              precomputed_effector=None, seed=42):
    """For each neuron j, randomly select k subspaces (uniform random baseline).

    Random permutation is generated via a score matrix (vectorised, no Python loop).
    static_scores use the same delta*eff formula as topk so the sign-gate in
    forward_static_preserve applies the same criterion at runtime.
    """
    delta = delta.cpu()
    if precomputed_effector is not None:
        effector = precomputed_effector.cpu()
    else:
        effector = (V @ W_emb_target).cpu()

    d_ff, n_sp = delta.shape
    k_each = min(k_per_neuron, n_sp)

    rng = torch.Generator()
    rng.manual_seed(seed)
    rand_scores = torch.rand(d_ff, n_sp, generator=rng)          # [d_ff, n_sp]
    top_i_mat   = rand_scores.topk(k_each, dim=1).indices        # [d_ff, k_each]

    j_idx = torch.arange(d_ff).unsqueeze(1).expand(-1, k_each).flatten().long()
    i_idx = top_i_mat.flatten().long()
    static_score = delta * effector.unsqueeze(0)                  # [d_ff, n_sp]
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    delta[j_idx, i_idx],
        "static_scores": static_score[j_idx, i_idx],
        "effector":      effector,
    }


def select_per_neuron_middle(delta, V, W_emb_target, k_per_neuron, precomputed_effector=None):
    """For each neuron j, select middle-k subspaces by |delta[j,i]*effector[i]|.

    Sorts all subspaces descending, then takes the k starting at index (n_sp-k)//2,
    i.e. skips the top and bottom tails equally.  Serves as a sanity-check baseline
    between top (highest importance) and bottom (lowest importance).
    """
    delta = delta.cpu()
    if precomputed_effector is not None:
        effector = precomputed_effector.cpu()
    else:
        effector = (V @ W_emb_target).cpu()

    static_score = delta * effector.unsqueeze(0)           # [d_ff, n_sp]
    d_ff, n_sp = delta.shape
    k_each = min(k_per_neuron, n_sp)

    sorted_idx = static_score.abs().argsort(dim=1, descending=True)  # [d_ff, n_sp]
    start = (n_sp - k_each) // 2
    mid_i_mat = sorted_idx[:, start:start + k_each]        # [d_ff, k_each]

    j_idx = torch.arange(d_ff).unsqueeze(1).expand(-1, k_each).flatten().long()
    i_idx = mid_i_mat.flatten().long()
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    delta[j_idx, i_idx],
        "static_scores": static_score[j_idx, i_idx],
        "effector":      effector,
    }


def select_per_neuron_bottom(delta, V, W_emb_target, k_per_neuron, precomputed_effector=None):
    """For each neuron j, select bottom-k subspaces by |delta[j,i]*effector[i]| (worst paths).

    Serves as a lower-bound / anti-skeleton baseline.
    static_scores use the same delta*eff formula so the sign-gate applies at runtime.
    """
    delta = delta.cpu()
    if precomputed_effector is not None:
        effector = precomputed_effector.cpu()
    else:
        effector = (V @ W_emb_target).cpu()

    static_score = delta * effector.unsqueeze(0)                  # [d_ff, n_sp]
    d_ff, n_sp = delta.shape
    k_each = min(k_per_neuron, n_sp)

    bot_i_mat = static_score.abs().topk(k_each, dim=1, largest=False).indices  # [d_ff, k_each]
    j_idx = torch.arange(d_ff).unsqueeze(1).expand(-1, k_each).flatten().long()
    i_idx = bot_i_mat.flatten().long()
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    delta[j_idx, i_idx],
        "static_scores": static_score[j_idx, i_idx],
        "effector":      effector,
    }


# ── eff-only selection baseline ──────────────────────────────────
#
# Selection ranks subspaces by |eff[i]| (target alignment) alone as an explicit
# control independent of the standard |delta*eff| ordering.
# The sign-gate is unified with the non-eff variants: h[j]*delta[j,i]*eff[i] > 0.
# Reconstruction in forward_static_preserve is unchanged: coeff = h[j]*delta[j,i].


def select_per_neuron_eff_topk(delta, V, W_emb_target, k_per_neuron, precomputed_effector=None):
    """All neurons select the same top-k subspaces by target alignment |eff[i]|.

    static_scores = delta[j,i]*eff[i]  →  gate: h[j]*delta[j,i]*eff[i] > 0
    delta_vals kept for use in coeff = h[j]*delta[j,i] reconstruction.
    """
    delta = delta.cpu()
    effector = precomputed_effector.cpu() if precomputed_effector is not None \
               else (V @ W_emb_target).cpu()

    d_ff, n_sp = delta.shape
    k_each = min(k_per_neuron, n_sp)

    top_i = effector.abs().topk(k_each).indices              # [k_each], same for all j
    i_idx = top_i.unsqueeze(0).expand(d_ff, -1).flatten().long()
    j_idx = torch.arange(d_ff).unsqueeze(1).expand(-1, k_each).flatten().long()
    static_score = delta * effector.unsqueeze(0)             # [d_ff, n_sp]
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    delta[j_idx, i_idx],
        "static_scores": static_score[j_idx, i_idx],         # gate: h[j]*delta[j,i]*eff[i] > 0
        "effector":      effector,
    }


def select_per_neuron_eff_random(delta, V, W_emb_target, k_per_neuron,
                                  precomputed_effector=None, seed=42):
    """Gemma-2/3 random: k random subspaces per neuron, gate by h[j]*delta[j,i]*eff[i] > 0."""
    delta = delta.cpu()
    effector = precomputed_effector.cpu() if precomputed_effector is not None \
               else (V @ W_emb_target).cpu()

    d_ff, n_sp = delta.shape
    k_each = min(k_per_neuron, n_sp)

    rng = torch.Generator()
    rng.manual_seed(seed)
    rand_scores = torch.rand(d_ff, n_sp, generator=rng)
    top_i_mat   = rand_scores.topk(k_each, dim=1).indices   # [d_ff, k_each]

    j_idx = torch.arange(d_ff).unsqueeze(1).expand(-1, k_each).flatten().long()
    i_idx = top_i_mat.flatten().long()
    static_score = delta * effector.unsqueeze(0)             # [d_ff, n_sp]
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    delta[j_idx, i_idx],
        "static_scores": static_score[j_idx, i_idx],         # gate: h[j]*delta[j,i]*eff[i] > 0
        "effector":      effector,
    }


def select_per_neuron_eff_bottom(delta, V, W_emb_target, k_per_neuron, precomputed_effector=None):
    """Gemma-2/3 bottom-k: all neurons select the same bottom-k subspaces by |eff[i]|."""
    delta = delta.cpu()
    effector = precomputed_effector.cpu() if precomputed_effector is not None \
               else (V @ W_emb_target).cpu()

    d_ff, n_sp = delta.shape
    k_each = min(k_per_neuron, n_sp)

    bot_i = effector.abs().topk(k_each, largest=False).indices  # [k_each]
    i_idx = bot_i.unsqueeze(0).expand(d_ff, -1).flatten().long()
    j_idx = torch.arange(d_ff).unsqueeze(1).expand(-1, k_each).flatten().long()
    static_score = delta * effector.unsqueeze(0)             # [d_ff, n_sp]
    return {
        "j_idx":         j_idx,
        "i_idx":         i_idx,
        "delta_vals":    delta[j_idx, i_idx],
        "static_scores": static_score[j_idx, i_idx],         # gate: h[j]*delta[j,i]*eff[i] > 0
        "effector":      effector,
    }
