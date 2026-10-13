import torch


def compute_neuron_coupling(y: torch.Tensor, v_j: torch.Tensor) -> torch.Tensor:
    """
    Compute per-neuron coupling strength  p_ij = v_ji * y_i  for subspace j.

    Args:
        y:   activated neuron values f(W_up @ x), shape [n_neurons]
        v_j: detector direction for subspace j (left singular vector of W_down,
             or left singular vector of W_up for c_fc), shape [n_neurons]

    Returns:
        p_j: coupling tensor, shape [n_neurons]
             sum_i p_ij  ==  v_j^T @ y  ==  det_val / sigma_j  (for c_proj)
    """
    return v_j.float() * y.float()


def get_top_neurons_for_subspace(
    p_j: torch.Tensor,
    det_val: float,
    eff_val: float,
    topk: int,
) -> dict:
    """
    Select the top-k W_up neurons that drive subspace j, respecting sign context.

    Sign logic
    ----------
    det_val  >= 0  (positive detector sum):
        High-contribution neurons have the *most positive* p_ij.
        Their individual v_ji > 0 and y_i > 0  (or both negative, giving positive product).

    det_val  <  0  (negative detector sum):
        High-contribution neurons have the *most negative* p_ij.
        Their individual v_ji and y_i have opposite signs such that p_ij < 0,
        which drives the negative det sum; multiplied by negative eff gives
        a positive (beneficial) circuit contribution.

    Both cases can yield a positive overall contrib_val = det_val * eff_val.

    Args:
        p_j:     per-neuron couplings, shape [n_neurons]
        det_val: detector alignment (includes sigma_j factor)
        eff_val: effector alignment t @ u_j
        topk:    number of top neurons to return

    Returns:
        dict with:
            neuron_indices   – 1-indexed list of top neuron positions
            coupling_values  – corresponding p_ij values (negative when det_val < 0)
            det_sign         – "positive" or "negative"
            eff_sign         – "positive" or "negative"
            contrib_sign     – "positive" or "negative"
    """
    n = min(topk, p_j.shape[0])

    if det_val >= 0:
        vals, idx = torch.topk(p_j, n)
    else:
        # Select most-negative p_ij: flip sign, topk, flip back
        vals, idx = torch.topk(-p_j, n)
        vals = -vals

    return {
        "neuron_indices":  [int(i) + 1 for i in idx.tolist()],   # 1-indexed
        "coupling_values": [float(v) for v in vals.tolist()],
        "det_sign":        "positive" if det_val >= 0 else "negative",
        "eff_sign":        "positive" if eff_val >= 0 else "negative",
        "contrib_sign":    "positive" if det_val * eff_val >= 0 else "negative",
    }


