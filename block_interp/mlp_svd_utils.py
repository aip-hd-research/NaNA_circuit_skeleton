import torch


def compute_svd(weight_type: str, c_fc=None, c_proj=None, ln_2=None, down_proj=None):
    """
    Compute the SVD of an MLP weight matrix.

    weight_type values
    ------------------
    "c_proj"    — GPT-2 output projection (Conv1D weight [d_ff, d_model])
    "c_fc"      — GPT-2 input projection fused with LN weight
    "down_proj" — Llama output projection (nn.Linear weight [d_model, d_ff], transposed here)

    Returns (U, S, V) such that  w ≈ U @ diag(S) @ V
      U : [d_ff, r]
      S : [r]
      V : [r, d_model]

    CAVEAT ("c_fc"/"gate_proj" branch): fusing ln_2.weight only applies LayerNorm's
    learnable affine scale (γ), not its bias (β) nor the real per-token mean-
    subtraction/variance-normalization — see reshape_emb_matrix's docstring for
    the same limitation. The resulting detection direction is an approximation.
    """
    if weight_type == "c_proj":
        w = c_proj.weight.detach()
        # Conv1D (GPT-2): weight [d_ff, d_model] — already correct
        # nn.Linear (Llama, ProGen2 fc_out): weight [d_model, d_ff] — needs transpose
        if type(c_proj).__name__ != "Conv1D":
            w = w.T  # → [d_ff, d_model]

    elif weight_type == "down_proj":
        w = down_proj.weight.T.detach()               # [d_model, d_ff].T → [d_ff, d_model]

    elif weight_type in ("c_fc", "gate_proj"):
        w = c_fc.weight.detach()
        # Conv1D (GPT-2): weight is [d_model, d_ff] → .T gives [d_ff, d_model]
        # nn.Linear (Llama gate_proj): weight is [d_ff, d_model] → already correct
        if w.shape[0] == ln_2.weight.shape[0]:   # [d_model, d_ff]: first dim = d_model
            w = w.T
        # w is now [d_ff, d_model]; ln_2.weight is [d_model]
        w = w * ln_2.weight.detach()

    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    device = w.device
    U, S, V = torch.linalg.svd(w.float().cpu(), full_matrices=False)
    U, S, V = U.to(device), S.to(device), V.to(device)
    return U, S, V
 


def reshape_emb_matrix(W_emb: torch.Tensor, c_fc, ln_2, act, use_activation: bool = False, to_cpu: bool = False):
    """
    Project the embedding matrix through the MLP input weights and optionally
    apply the MLP activation function.

    Handles both Conv1D (GPT-2, weight [d_model, d_ff]) and nn.Linear (Llama,
    weight [d_ff, d_model]) conventions.

    CAVEAT: "fuse LayerNorm scale" below only multiplies by ln_2.weight (the
    learnable affine γ) — it does not include ln_2.bias (β), and it cannot
    reproduce LayerNorm's actual per-token mean-subtraction/variance-normalization,
    which depends on the real input at inference time, not just static weights.
    This is a directional approximation of "which tokens activate this neuron",
    not an exact reproduction of ln_2(embedding) @ c_fc, even over the full
    embedding matrix.
    """
    with torch.no_grad():
        W_emb = W_emb * ln_2.weight.detach()        # fuse LayerNorm scale
        w = c_fc.weight.detach()
        # Conv1D: [d_model, d_ff]  →  W_emb @ w  gives [vocab, d_ff]  ✓
        # nn.Linear: [d_ff, d_model]  →  W_emb @ w.T  gives [vocab, d_ff]  ✓
        if w.shape[0] != W_emb.shape[-1]:            # first dim ≠ d_model → nn.Linear
            w = w.T
        reshape_matrix = W_emb @ w
        bias = getattr(c_fc, "bias", None)
        if bias is not None:
            reshape_matrix = reshape_matrix + bias.detach()

        if use_activation:
            reshape_matrix = act(reshape_matrix)

    if to_cpu:
        reshape_matrix = reshape_matrix.to("cpu")

    return reshape_matrix

def parse_topk_subspaces(topk_subspaces, total_subspaces):
    """
    Convert topk_subspaces argument to a list of indices and a label string.

    Args:
        topk_subspaces: int, list[int], or str ("all" or "topN")
        total_subspaces: int, total number of subspaces available

    Returns:
        indices: list of valid subspace indices
        label: string for file naming or display
    """
    if isinstance(topk_subspaces, str):
        if topk_subspaces.lower() == "all":
            indices = list(range(total_subspaces))
            label = "all"
            
        elif topk_subspaces.lower().startswith("top"):
            try:
                n = int(topk_subspaces[3:])
                indices = list(range(min(n, total_subspaces)))
                label = f"top{len(indices)}"
            except ValueError:
                raise ValueError(f"Invalid topk_subspaces format: {topk_subspaces}")
                
        else:
            raise ValueError(f"Invalid string for topk_subspaces: {topk_subspaces}")
            
    elif isinstance(topk_subspaces, int):
        indices = list(range(min(topk_subspaces, total_subspaces)))
        label = f"top{len(indices)}"
        
    elif isinstance(topk_subspaces, (list, tuple)):
        indices = [i for i in topk_subspaces if 0 <= i < total_subspaces]
        if not indices:
            raise ValueError("topk_subspaces list is empty or out of range.")
        label = "[" + "_".join(str(i) for i in indices) + "]"
        
    else:
        raise TypeError("topk_subspaces must be int, list[int], or 'all'.")

    return indices, label
