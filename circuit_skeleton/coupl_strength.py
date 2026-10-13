import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_skeleton.within_layer_ske import WithinLayerSkeleton, parse_neurons_arg
from circuit_skeleton.cross_layer_ske import CrossLayerSkeleton, _flat_to_layer_neuron


class CouplingStrength:
    """
    Total number of target neurons across all target layers.
    """

    def __init__(
        self,
        within_ske: WithinLayerSkeleton,
        cross_ske: CrossLayerSkeleton,
    ) -> None:
        self.within_ske    = within_ske
        self.cross_ske     = cross_ske
        self.source_layer  = within_ske.layer_idx
        self.target_layers = cross_ske.target_layers

        # build layer-offset lookup and stacked gamma once
        self.layer_offsets: list = []
        offset = 0
        for tgt_layer in cross_ske.target_layers:
            d_ff_tgt = cross_ske.d_ff[tgt_layer]
            self.layer_offsets.append((offset, tgt_layer, d_ff_tgt))
            offset += d_ff_tgt
        self.total_cross: int = offset

        # gamma_stacked[i, flat_k] = gamma[L][i, k]   shape [n_sp, total_cross]
        self.gamma_stacked: torch.Tensor = torch.cat(
            [cross_ske.gamma[L] for L in cross_ske.target_layers],
            dim=1,
        )


    def flat_to_layer_neuron(self, flat_idx: int) -> tuple:
        """Map a flat cross-neuron index → (target_layer, neuron_idx)."""
        return _flat_to_layer_neuron(flat_idx, self.layer_offsets)


    def w_contracted(self, neuron_arg) -> torch.Tensor:
        indices = parse_neurons_arg(neuron_arg, self.within_ske.d_ff)
        delta   = self.within_ske.delta[indices, :]   # [n_sel, n_sp]
        return delta @ self.gamma_stacked              # [n_sel, total_cross]


    def w_full_path(
        self,
        neuron_arg,
        subspace_indices=None,
        warn_gb_threshold: float = 4.0,
    ) -> torch.Tensor:

        indices  = parse_neurons_arg(neuron_arg, self.within_ske.d_ff)
        n_sel    = len(indices)
        delta    = self.within_ske.delta[indices, :]   # [n_sel, n_sp]

        if subspace_indices is not None:
            delta = delta[:, subspace_indices]                      # [n_sel, sp_sel]
            gamma = self.gamma_stacked[subspace_indices, :]         # [sp_sel, total_cross]
        else:
            gamma = self.gamma_stacked                              # [n_sp, total_cross]

        n_sp_sel = gamma.shape[0]

        estimated_gb = (n_sel * n_sp_sel * self.total_cross * 4) / (1024 ** 3)
        if estimated_gb > warn_gb_threshold:
            warnings.warn(
                f"w_full_path: estimated tensor size {estimated_gb:.1f} GB "
                f"({n_sel} neurons × {n_sp_sel} subspaces × {self.total_cross} "
                f"cross-neurons).  Consider reducing start_neuron or subspace_indices.",
                ResourceWarning,
                stacklevel=2,
            )

        # outer product: [n_sel, sp, 1] * [1, sp, total_cross]
        return delta.unsqueeze(2) * gamma.unsqueeze(0)  # [n_sel, n_sp_sel, total_cross]


 
def build_coupling_strength(
    model_name: str,
    start_layer: int,
    target_layers,
    device: str = "cpu",
) -> CouplingStrength:

    within_ske = WithinLayerSkeleton(model_name, start_layer, device)
    cross_ske  = CrossLayerSkeleton(model_name, start_layer, target_layers, device)
    return CouplingStrength(within_ske, cross_ske)
