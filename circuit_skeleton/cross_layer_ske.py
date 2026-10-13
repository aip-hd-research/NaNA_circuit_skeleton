"""
cross_layer_ske.py

Inter-layer MLP circuit skeleton (via residual stream):
    down-projection SVD subspaces at source_layer  ->  up-projection neurons at target_layer(s)

Coupling strength (subspace i at layer l -> neuron k at layer l+n):
    gamma_{i,k}^{(l->l+n)} = effector_i^{(l)} . neuron_k^{(l+n)}
                            = V_src[i, :] @ W_up_fused_tgt[k, :]

When target_layers covers multiple layers, topk is computed globally across all
(layer, neuron) pairs.

CAVEAT: gamma is a static weight-space dot product between two fixed directions.
It does not run any forward pass, so it ignores everything that happens between
source_layer and target_layer (attention, other MLPs, residual accumulation),
the actual per-token LayerNorm/RMSNorm statistics at target_layer (W_up_fused only
fuses the learnable affine scale, not the real input-dependent normalization), and,
for gated-MLP architectures (Llama/Gemma-style), the elementwise gate*up SwiGLU
nonlinearity. Treat gamma as a weight-direction similarity / structural prior for
where to look, not as a measured causal effect of source_layer on target_layer.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_interp.model_load import load_model_and_embeddings, get_mlp_matrices
from ske_pre_valid.model_adapter import get_adapter
from circuit_skeleton.within_layer_ske import parse_neurons_arg


def parse_subspaces_arg(subspaces_arg, total_subspaces: int) -> list:
    """
    Parse subspace selection into a list of indices.

    Supported formats:
        3           (int)    -> [3]
        "3,5,2"     (str)    -> [3, 5, 2]
        "2-19"      (str)    -> [2, 3, ..., 19]
        "ALL"       (str)    -> [0, 1, ..., total_subspaces-1]
    """
    return parse_neurons_arg(subspaces_arg, total_subspaces)


def parse_layers_arg(layers_arg, total_layers: int) -> list:
    """
    Parse layer selection into a list of layer indices.

    Supported formats:
        3           (int)    -> [3]
        "3,5,2"     (str)    -> [3, 5, 2]
        "2-19"      (str)    -> [2, 3, ..., 19]
        "ALL"       (str)    -> [0, 1, ..., total_layers-1]
    """
    return parse_neurons_arg(layers_arg, total_layers)


def _flat_to_layer_neuron(flat_idx: int, layer_offsets: list) -> tuple:
    """Map a flat index into the concatenated coupling vector back to (layer, neuron_idx)."""
    for start, layer, d_ff in layer_offsets:
        if flat_idx < start + d_ff:
            return layer, flat_idx - start
    raise IndexError(f"flat_idx {flat_idx} out of range")


class CrossLayerSkeleton:
    """
    Inter-layer MLP circuit skeleton from source_layer to one or more target layers.

    Precomputes per target layer:
        gamma[tgt_layer][i, k] = V_src[i, :] @ W_up_fused_tgt[k, :]
                                  shape: [n_subspaces, d_ff_tgt]

    where:
        V_src[i, :]        : effector direction of subspace i at source_layer  (shape [d_model])
        W_up_fused_tgt[k,:]: LN-fused detection direction of neuron k at tgt_layer (shape [d_model])

    topk_connections pools all target layers into one global ranking.

    gamma is a static weight-direction similarity, not a measured causal effect —
    see the module docstring's CAVEAT for what it ignores (intermediate computation,
    real per-token normalization statistics, gated-MLP nonlinearity).
    """

    def __init__(
        self,
        model_name: str,
        source_layer: int,
        target_layers_arg,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.source_layer = source_layer
        self.device = device

        model, _, _ = load_model_and_embeddings(model_name, device)
        adapter = get_adapter(model)
        total_layers = adapter.num_layers(model)

        # Parse and validate target layers
        raw_targets = parse_layers_arg(target_layers_arg, total_layers)
        self.target_layers = sorted(
            l for l in raw_targets if source_layer < l < total_layers
        )
        if not self.target_layers:
            raise ValueError(
                f"No valid target layers found (must be > {source_layer} and < {total_layers})."
            )

        #  Source layer: effector directions V_src
        _, W_out_src, _, _ = get_mlp_matrices(model, source_layer)
        _, _, V_src = adapter.svd_w_out(W_out_src)
        self.V_src = V_src.to(device)           # [n_subspaces, d_model]
        self.n_subspaces = self.V_src.shape[0]

        #  Target layers: LN-fused W_up and gamma tensors
        # gamma[tgt_layer]: [n_subspaces, d_ff_tgt]
        self.gamma: dict = {}
        self.d_ff: dict = {}

        for tgt_layer in self.target_layers:
            W_in_tgt, _, ln_tgt, _ = get_mlp_matrices(model, tgt_layer)
            with torch.no_grad():
                W_up_fused = adapter.w_in_fused(
                    W_in_tgt, ln_tgt.weight.detach()
                ).to(device)                            # [d_ff_tgt, d_model]
                self.gamma[tgt_layer] = self.V_src @ W_up_fused.T   # [n_subspaces, d_ff_tgt]
            self.d_ff[tgt_layer] = W_up_fused.shape[0]

        del model

    def topk_connections(self, subspaces_arg, topk: int) -> list:
        """
        For each specified subspace, return top-k (target_layer, neuron) pairs
        ranked by |gamma_{i,k}| globally across all target layers.

        Args:
            subspaces_arg : int | str
                            Supported: 3 | "3,5,2" | "2-19" | "ALL"
            topk          : number of top (layer, neuron) pairs to return per subspace

        Returns:
            List of dicts, one per subspace:
            {
                "source_layer" : int,
                "subspace_idx" : int,
                "connections"  : [
                    {
                        "subspace_idx"  : int,
                        "source_layer"  : int,
                        "neuron_idx"    : int,
                        "target_layer"  : int,
                        "coupling_pos"  : float,   # max(gamma_{i,k}, 0)
                        "coupling_neg"  : float,   # min(gamma_{i,k}, 0)
                        "abs_strength"  : float,
                        "rank"          : int,     # 1-indexed, 1 = strongest
                    },
                    ...
                ]
            }
        """
        subspace_indices = parse_subspaces_arg(subspaces_arg, self.n_subspaces)

        # Build layer-offset table once (same for every subspace)
        layer_offsets = []
        offset = 0
        for tgt_layer in self.target_layers:
            d_ff_tgt = self.d_ff[tgt_layer]
            layer_offsets.append((offset, tgt_layer, d_ff_tgt))
            offset += d_ff_tgt
        total_neurons = offset

        results = []
        for i in subspace_indices:
            # Concatenate couplings across all target layers: [total_neurons]
            all_couplings = torch.cat(
                [self.gamma[tgt][i] for tgt in self.target_layers]
            )

            k = min(topk, total_neurons)
            topk_abs_vals, topk_flat_indices = torch.topk(all_couplings.abs(), k)

            connections = []
            for rank, (flat_idx, abs_val) in enumerate(
                zip(topk_flat_indices.tolist(), topk_abs_vals.tolist())
            ):
                tgt_layer, neuron_idx = _flat_to_layer_neuron(flat_idx, layer_offsets)
                val = all_couplings[flat_idx].item()
                connections.append({
                    "subspace_idx":  i,
                    "source_layer":  self.source_layer,
                    "neuron_idx":    neuron_idx,
                    "target_layer":  tgt_layer,
                    "coupling_pos":  max(val, 0.0),
                    "coupling_neg":  min(val, 0.0),
                    "abs_strength":  abs_val,
                    "rank":          rank + 1,
                })

            results.append({
                "source_layer": self.source_layer,
                "subspace_idx": i,
                "connections":  connections,
            })

        return results
