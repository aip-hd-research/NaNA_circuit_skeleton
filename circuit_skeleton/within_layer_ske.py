import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_interp.model_load import load_model_and_embeddings
from ske_pre_valid.model_adapter import get_adapter


def parse_neurons_arg(neurons_arg, total_neurons: int) -> list:
    """
    Parse neuron selection into a list of indices.

    Supported formats:
        3           (int)    -> [3]
        "3,5,2"     (str)    -> [3, 5, 2]
        "2-19"      (str)    -> [2, 3, ..., 19]
        "ALL"       (str)    -> [0, 1, ..., total_neurons-1]
    """
    if isinstance(neurons_arg, int):
        return [neurons_arg]
    if isinstance(neurons_arg, (list, tuple)):
        return list(neurons_arg)
    if isinstance(neurons_arg, str):
        s = neurons_arg.strip()
        if s.upper() == "ALL":
            return list(range(total_neurons))
        if "," in s:
            return [int(x.strip()) for x in s.split(",")]
        if "-" in s:
            lo, hi = s.split("-", 1)
            return list(range(int(lo), int(hi) + 1))
        return [int(s)]
    raise ValueError(f"Unsupported neurons_arg: {neurons_arg!r}")


class WithinLayerSkeleton:
    """
    Intra-layer MLP circuit skeleton for one transformer layer.

    Precomputes:
        delta[j, i] = S[i] * U[j, i]                  shape: [d_ff, n_subspaces]

    where U, S, V come from SVD of W_out:
      GPT-2 : c_proj.weight       [d_ff, d_model]  (Conv1D convention)
      Llama : down_proj.weight.T  [d_ff, d_model]  (transposed from [d_model, d_ff])

    V : right singular vectors [n_subspaces, d_model] — effector directions in
        the residual stream.
    """

    def __init__(self, model_name: str, layer_idx: int, device: str = "cpu"):
        self.model_name = model_name
        self.layer_idx  = layer_idx
        self.device     = device

        model, _, _ = load_model_and_embeddings(model_name, device)
        adapter     = get_adapter(model)

        W_in_module, W_out_module, pre_mlp_ln, _ = adapter.get_mlp_matrices(model, layer_idx)

        with torch.no_grad():
            ln_w = pre_mlp_ln.weight.detach()
            U, S, V = adapter.svd_w_out(W_out_module)
            self.W_up_fused = adapter.w_in_fused(W_in_module, ln_w).to(device)

        self.U = U.to(device)
        self.S = S.to(device)
        self.V = V.to(device)

        # delta[j, i] = S[i] * U[j, i],  shape: [d_ff, n_subspaces]
        self.delta = self.U * self.S.unsqueeze(0)

        self.d_ff        = self.W_up_fused.shape[0]
        self.n_subspaces = self.S.shape[0]

        del model

    def topk_connections(self, neurons_arg, topk: int) -> list:
        """
        For each specified neuron, return top-k subspaces ranked by |delta_{j,i}|.

        Args:
            neurons_arg : int | str
                          Supported: 3 | "3,5,2" | "2-19" | "ALL"
            topk        : number of top subspaces to return per neuron

        Returns:
            List of dicts, one per neuron:
            {
                "neuron_idx": int,
                "connections": [
                    {
                        "neuron_idx"   : int,
                        "subspace_idx" : int,
                        "coupling_pos" : float,   # max(delta_{j,i}, 0)
                        "coupling_neg" : float,   # min(delta_{j,i}, 0)
                        "abs_strength" : float,
                        "rank"         : int,     # 1-indexed, 1 = strongest
                    },
                    ...
                ]
            }
        """
        neuron_indices = parse_neurons_arg(neurons_arg, self.d_ff)
        k = min(topk, self.n_subspaces)
        results = []

        for j in neuron_indices:
            couplings = self.delta[j]                               # [n_subspaces]
            topk_abs_vals, topk_sub_indices = torch.topk(couplings.abs(), k)

            connections = []
            for rank, (sub_idx, abs_val) in enumerate(
                zip(topk_sub_indices.tolist(), topk_abs_vals.tolist())
            ):
                val = couplings[sub_idx].item()
                connections.append({
                    "neuron_idx":    j,
                    "subspace_idx":  sub_idx,
                    "coupling_pos":  max(val, 0.0),
                    "coupling_neg":  min(val, 0.0),
                    "abs_strength":  abs_val,
                    "rank":          rank + 1,
                })

            results.append({
                "neuron_idx":  j,
                "connections": connections,
            })

        return results
