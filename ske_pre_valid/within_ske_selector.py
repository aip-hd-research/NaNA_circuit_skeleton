"""
within_ske_selector.py

For a given MLP layer, select the top-k1 within-layer skeleton paths most
strongly activated by the current input.

Activation of neuron j:
    mlp_in      = ln_2(x)                       [batch, seq, d_model]
    pre_act[j]  = mlp_in @ c_fc.weight.T[:,j] + c_fc.bias[j]
    h[j]        = act(pre_act[j])               scalar per (batch, seq) position

Skeleton path score:
    score[j, i] = h_reduced[j] * delta[j, i]   (activation × skeleton coupling)

Top-k1 selection is global over all (j, i) pairs.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
import torch


def select_top_within_skeleton(
    h: torch.Tensor,                        # [batch, seq, d_ff]
    delta: torch.Tensor,                    # [d_ff, n_subspaces]
    k1: int,
    seq_reduce: str = "last",
    effector_scores: torch.Tensor | None = None,  # [n_subspaces]  V @ W_emb[target_id]
    use_h: bool = True,
) -> list[dict]:
    """
    Select top-k1 within-layer skeleton paths.

    score[j, i] = h_reduced[j] * delta[j, i] (* effector[i])

    Parameters
    ----------
    h               : neuron activations [batch, seq, d_ff]
    delta           : skeleton coupling matrix [d_ff, n_subspaces]
    k1              : number of top paths to return
    seq_reduce      : "last" uses the last token; "mean" averages over all tokens
    effector_scores : optional [n_subspaces] alignment of each subspace with target token
    use_h           : if False, ignore h when scoring

    Returns
    -------
    List of k1 dicts sorted by score descending (largest positive first):
        neuron_idx, subspace_idx, score, abs_score, activation, delta_val, rank
    """
    if use_h:
        if seq_reduce == "last":
            h_reduced = h[:, -1, :].mean(dim=0)   # [d_ff]
        else:
            h_reduced = h.mean(dim=(0, 1))         # [d_ff]
        h_reduced = h_reduced.to(delta.device)
        score = h_reduced.unsqueeze(1) * delta     # [d_ff, n_subspaces]
    else:
        h_reduced = None
        score = delta.clone()                      # [d_ff, n_subspaces], no h

    if effector_scores is not None:
        score = score * effector_scores.to(delta.device).unsqueeze(0)  # broadcast over j

    flat = score.flatten()
    k    = min(k1, flat.shape[0])
    topk_vals, topk_flat_idx = torch.topk(flat, k)

    d_ff, n_sp = score.shape
    results = []
    for rank, (flat_idx, val) in enumerate(
        zip(topk_flat_idx.tolist(), topk_vals.tolist()), 1
    ):
        j = flat_idx // n_sp
        i = flat_idx %  n_sp
        results.append({
            "neuron_idx":   j,
            "subspace_idx": i,
            "score":        val,
            "abs_score":    abs(val),
            "activation":   h_reduced[j].item() if h_reduced is not None else None,
            "delta_val":    delta[j, i].item(),
            "rank":         rank,
        })

    return results


def select_random_skeleton(
    h: torch.Tensor,            # [batch, seq, d_ff]
    delta: torch.Tensor,        # [d_ff, n_subspaces]
    k1: int,
    seq_reduce: str = "last",
    seed: int = 0,
) -> list[dict]:
    """
    Randomly sample k1 (j, i) paths as a control baseline.
    Same k1 and reconstruction interface as select_top_within_skeleton.
    """
    if seq_reduce == "last":
        h_reduced = h[:, -1, :].mean(dim=0)
    else:
        h_reduced = h.mean(dim=(0, 1))

    h_reduced = h_reduced.to(delta.device)
    d_ff, n_sp = delta.shape

    rng = torch.Generator(device=delta.device)
    rng.manual_seed(seed)
    flat_idx = torch.randperm(d_ff * n_sp, generator=rng, device=delta.device)[:min(k1, d_ff * n_sp)]

    results = []
    for rank, idx in enumerate(flat_idx.tolist(), 1):
        j = idx // n_sp
        i = idx %  n_sp
        results.append({
            "neuron_idx":   j,
            "subspace_idx": i,
            "score":        (h_reduced[j] * delta[j, i]).item(),
            "abs_score":    (h_reduced[j] * delta[j, i]).abs().item(),
            "activation":   h_reduced[j].item(),
            "delta_val":    delta[j, i].item(),
            "rank":         rank,
        })
    return results


def reconstruct_mlp_via_skeleton(
    h: torch.Tensor,                        # [batch, seq, d_ff]
    top_paths: list[dict],
    V: torch.Tensor,                        # [n_subspaces, d_model]
    bias: torch.Tensor | None = None,       # [d_model] — add if provided
) -> torch.Tensor:
    """
    Reconstruct the MLP output using only the top-k skeleton paths.

    Output = Σ_{(j,i) ∈ top-k} h[:,:,j] * delta[j,i] * V[i,:]  (+ bias if given)

    Returns Tensor [batch, seq, d_model].
    """
    device = h.device
    dtype  = h.dtype
    V = V.to(device=device, dtype=dtype)

    j_idx      = torch.tensor([p["neuron_idx"]   for p in top_paths], device=device)
    i_idx      = torch.tensor([p["subspace_idx"] for p in top_paths], device=device)
    delta_vals = torch.tensor([p["delta_val"]    for p in top_paths], device=device, dtype=dtype)

    h_selected = h[:, :, j_idx]   # [batch, seq, k]
    V_selected = V[i_idx, :]      # [k, d_model]

    out = (h_selected * delta_vals) @ V_selected
    if bias is not None:
        out = out + bias.to(device=device, dtype=dtype)
    return out


def select_and_reconstruct_skeleton(
    h: torch.Tensor,                        # [batch, seq, d_ff]
    delta: torch.Tensor,                    # [d_ff, n_sp]
    V: torch.Tensor,                        # [n_sp, d_model]
    k1: int,
    seq_reduce: str,
    effector_scores: torch.Tensor | None,
    bias: torch.Tensor | None,
    token_mode: str,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    Fully vectorised topk selection + MLP reconstruction — no Python loops, no list-of-dicts.

    Fast paths
    ----------
    k1 >= d_ff*n_sp : skip topk entirely; reconstruct via h @ delta @ V.
    k1 <  d_ff*n_sp : topk to get (j,i) indices, then scatter_add to aggregate
                      coefficients per subspace before the final @ V.
                      Avoids materialising the huge [k, d_model] V-gather.

    Returns ske_out [batch, seq, d_model] (token_mode="all")
                 or [batch,   1, d_model] (token_mode="last").
    """
    device = h.device
    dtype  = h.dtype
    delta  = delta.to(device=device, dtype=dtype)
    V      = V.to(device=device, dtype=dtype)

    d_ff, n_sp = delta.shape
    h_out = h if token_mode == "all" else h[:, -1:, :]   # [b, s, d_ff]

    # ── fast path: k1 covers all paths → no topk, just matrix multiply ──────
    if k1 >= d_ff * n_sp:
        ske = h_out @ delta @ V                          # [b, s, d_model]
        if bias is not None:
            ske = ske + bias.to(device=device, dtype=dtype)
        if not return_trace:
            return ske
        flat_idx = torch.arange(d_ff * n_sp, device=device)
        h_reduced = (h[:, -1, :].mean(dim=0) if token_mode != "all"
                     else h.mean(dim=(0, 1)))
        score = h_reduced[:, None] * delta
        if effector_scores is not None:
            score = score * effector_scores.to(device=device, dtype=dtype)[None, :]
        j_idx, i_idx = flat_idx // n_sp, flat_idx % n_sp
        trace = _path_trace(j_idx, i_idx, h_reduced, delta, score.flatten()[flat_idx])
        return ske, trace

    # ── partial path: topk selection + scatter_add reconstruction ────────────
    # Complement trick: when k1 > N/2, select the bottom (N-k1) paths and
    # subtract from full matmul.  Halves topk size and scatter_add work.
    n_total = d_ff * n_sp
    use_complement = k1 * 2 > n_total
    n_select = n_total - k1 if use_complement else k1

    _seq_reduce = "mean" if token_mode == "all" else seq_reduce
    h_reduced   = (h[:, -1, :].mean(dim=0) if _seq_reduce == "last"
                   else h.mean(dim=(0, 1)))              # [d_ff]

    score = h_reduced.unsqueeze(1) * delta               # [d_ff, n_sp]
    if effector_scores is not None:
        score = score * effector_scores.to(device=device, dtype=dtype).unsqueeze(0)

    # Complement: select bottom (N-k1); Direct: select top k1
    _, flat_idx = torch.topk(score.flatten(), n_select, largest=not use_complement)
    j_idx      = flat_idx // n_sp                        # [n_select]
    i_idx      = flat_idx %  n_sp                        # [n_select]
    delta_vals = delta[j_idx, i_idx]                     # [n_select]

    # scatter_add: aggregate h[j]*delta[j,i] coefficients per subspace i
    # → avoids [n_select, d_model] V-gather (would be GBs for large k1)
    batch, s  = h_out.shape[0], h_out.shape[1]
    act_dtype = h_out.dtype
    coeff = h_out[:, :, j_idx] * delta_vals.to(dtype=act_dtype)   # [b, s, n_select]
    i_exp = i_idx.view(1, 1, -1).expand(batch, s, -1)
    agg   = torch.zeros(batch, s, n_sp, dtype=act_dtype, device=device)
    agg.scatter_add_(2, i_exp, coeff)                              # [b, s, n_sp]
    partial = agg @ V.to(dtype=act_dtype)                          # [b, s, d_model]

    if use_complement:
        ske = (h_out @ delta @ V.to(dtype=act_dtype)) - partial
    else:
        ske = partial

    if bias is not None:
        ske = ske + bias.to(device=device, dtype=act_dtype)
    if not return_trace:
        return ske

    # ``flat_idx`` denotes excluded bottom paths in complement mode.  The
    # selected skeleton is its complement, so materialise exactly that set.
    if use_complement:
        selected = torch.ones(n_total, dtype=torch.bool, device=device)
        selected[flat_idx] = False
        trace_flat_idx = torch.nonzero(selected, as_tuple=False).flatten()
    else:
        trace_flat_idx = flat_idx
    trace_j, trace_i = trace_flat_idx // n_sp, trace_flat_idx % n_sp
    trace = _path_trace(
        trace_j, trace_i, h_reduced, delta, score.flatten()[trace_flat_idx]
    )
    return ske, trace


def _path_trace(j_idx, i_idx, h_reduced, delta, selection_scores):
    """CPU tensors describing the exact paths used by reconstruction."""
    return {
        "sensor_indices": j_idx.detach().to("cpu", dtype=torch.int32),
        "effector_indices": i_idx.detach().to("cpu", dtype=torch.int32),
        "sensor_activations": h_reduced[j_idx].detach().float().cpu(),
        "delta_values": delta[j_idx, i_idx].detach().float().cpu(),
        "selection_scores": selection_scores.detach().float().cpu(),
    }


def select_and_reconstruct_neuron_atoms(
    h: torch.Tensor,
    V: torch.Tensor,
    k1: int,
    seq_reduce: str,
    effector_scores: torch.Tensor | None,
    bias: torch.Tensor | None,
    token_mode: str,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Top-k for the compact neuron factorization ``W = I @ V``.

    Each valid atom is ``h[j] * V[j, :]``.  Operating directly on those
    ``d_ff`` atoms avoids materialising an otherwise enormous ``d_ff x d_ff``
    identity and prevents structural zeros from entering Top-k.
    """

    device, dtype = h.device, h.dtype
    V = V.to(device=device, dtype=dtype)
    n_neurons = int(V.shape[0])
    count = min(max(int(k1), 0), n_neurons)
    h_out = h if token_mode == "all" else h[:, -1:, :]
    reduced_mode = "mean" if token_mode == "all" else seq_reduce
    h_reduced = (
        h[:, -1, :].mean(dim=0)
        if reduced_mode == "last"
        else h.mean(dim=(0, 1))
    )
    scores = h_reduced
    if effector_scores is not None:
        scores = scores * effector_scores.to(device=device, dtype=dtype)

    if count >= n_neurons:
        selected_idx = torch.arange(n_neurons, device=device)
        output = h_out @ V
    elif count == 0:
        selected_idx = torch.empty(0, device=device, dtype=torch.long)
        output = torch.zeros(
            *h_out.shape[:-1], V.shape[1], device=device, dtype=dtype
        )
    else:
        selected_idx = torch.topk(scores, count).indices
        output = h_out[:, :, selected_idx] @ V[selected_idx]

    if bias is not None:
        output = output + bias.to(device=device, dtype=dtype)
    if not return_trace:
        return output
    trace = {
        "sensor_indices": selected_idx.detach().to("cpu", dtype=torch.int32),
        "effector_indices": selected_idx.detach().to("cpu", dtype=torch.int32),
        "sensor_activations": h_reduced[selected_idx].detach().float().cpu(),
        "delta_values": torch.ones(selected_idx.numel(), dtype=torch.float32),
        "selection_scores": scores[selected_idx].detach().float().cpu(),
    }
    return output, trace


def select_and_reconstruct_neuron_atoms_batch(
    h: torch.Tensor,
    V: torch.Tensor,
    k1: int,
    effector_scores: torch.Tensor | None,
    bias: torch.Tensor | None,
    last_real_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample batched Top-k for the compact ``W = I @ V`` factorization."""

    device, dtype = h.device, h.dtype
    V = V.to(device=device, dtype=dtype)
    batch, n_neurons = h.shape[0], V.shape[0]
    count = min(max(int(k1), 0), int(n_neurons))
    batch_idx = torch.arange(batch, device=device)
    h_last = (
        h[batch_idx, last_real_pos.to(device), :]
        if last_real_pos is not None
        else h[:, -1, :]
    )
    if count >= n_neurons:
        output = (h_last @ V).unsqueeze(1)
    elif count == 0:
        output = torch.zeros(
            batch, 1, V.shape[1], device=device, dtype=dtype
        )
    else:
        scores = h_last
        if effector_scores is not None:
            effector = effector_scores.to(device=device, dtype=dtype)
            if effector.ndim == 1:
                effector = effector.unsqueeze(0)
            scores = scores * effector
        selected = torch.topk(scores, count, dim=1).indices
        coefficients = h_last.gather(1, selected)
        selected_directions = V[selected]
        output = torch.bmm(coefficients.unsqueeze(1), selected_directions)
    if bias is not None:
        output = output + bias.to(device=device, dtype=dtype)
    return output


def select_and_reconstruct_skeleton_batch(
    h: torch.Tensor,                        # [b, seq, d_ff]
    delta: torch.Tensor,                    # [d_ff, n_sp]
    V: torch.Tensor,                        # [n_sp, d_model]
    k1: int,
    effector_scores: torch.Tensor | None,   # [b, n_sp] or [n_sp]
    bias: torch.Tensor | None,
    last_real_pos: torch.Tensor | None = None,  # [b] 0-indexed position of last real token
) -> torch.Tensor:                          # [b, 1, d_model]
    """Per-sample topk selection + reconstruction for a batch.

    Each sample selects its own top-k1 paths scored by its last real-token h.
    Fully vectorised — no Python loop over batch items.

    last_real_pos : when not None, h[b, last_real_pos[b], :] is used per sample
                    (required for right-padded batches so padding tokens are ignored).
                    When None, falls back to h[:, -1, :] (all sequences same length).
    Returns [b, 1, d_model].
    """
    device  = h.device
    dtype   = h.dtype
    b       = h.shape[0]
    delta   = delta.to(device=device, dtype=dtype)
    V       = V.to(device=device, dtype=dtype)
    d_ff, n_sp = delta.shape
    n_total = d_ff * n_sp

    b_range = torch.arange(b, device=device)
    if last_real_pos is not None:
        h_last = h[b_range, last_real_pos.to(device), :]   # [b, d_ff]
    else:
        h_last = h[:, -1, :]                                # [b, d_ff]

    if k1 >= n_total:
        ske = (h_last @ delta @ V).unsqueeze(1)             # [b, 1, d_model]
        if bias is not None:
            ske = ske + bias.to(device=device, dtype=dtype)
        return ske

    # Per-sample score [b, d_ff, n_sp]
    score = h_last.unsqueeze(2) * delta.unsqueeze(0)
    if effector_scores is not None:
        eff = effector_scores.to(device=device, dtype=dtype)
        if eff.dim() == 1:
            eff = eff.unsqueeze(0)               # [1, n_sp] — broadcast over batch
        score = score * eff.unsqueeze(1)         # [b, d_ff, n_sp]

    flat = score.reshape(b, n_total)             # [b, n_total]

    use_complement = k1 * 2 > n_total
    n_select       = n_total - k1 if use_complement else k1

    _, flat_idx = torch.topk(flat, n_select, dim=1, largest=not use_complement)
    j_idx      = flat_idx // n_sp               # [b, n_select]
    i_idx      = flat_idx %  n_sp               # [b, n_select]
    delta_vals = delta[j_idx, i_idx]            # [b, n_select]

    h_gathered = h_last.gather(1, j_idx)        # [b, n_select]
    coeff      = h_gathered * delta_vals        # [b, n_select]

    agg = torch.zeros(b, n_sp, dtype=dtype, device=device)
    agg.scatter_add_(1, i_idx, coeff)           # [b, n_sp]
    partial = (agg @ V).unsqueeze(1)            # [b, 1, d_model]

    if use_complement:
        full = (h_last @ delta @ V).unsqueeze(1)  # [b, 1, d_model]
        ske  = full - partial
    else:
        ske  = partial

    if bias is not None:
        ske = ske + bias.to(device=device, dtype=dtype)
    return ske
