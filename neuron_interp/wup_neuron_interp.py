"""
wup_neuron_interp.py

Direct neuron-level interpretation of the MLP up-projection (W_up / c_fc).
"""

import json
import os

from tqdm.auto import tqdm

from block_interp.model_load import load_model_and_embeddings, get_mlp_matrices
from plot_utils.plot_heatmap import plot_subspace_heatmap

from neuron_interp.neuron_arg_parser import parse_top_neurons
from neuron_interp.neuron_token_scorer import score_neuron, build_results


class WupNeuronInterp:
    """Direct neuron-level interpreter for the MLP up-projection (W_up / c_fc)."""

    def __init__(self, model_name: str, output_dir: str = "result", device=None):
        self.device     = device
        self.model_name = model_name
        self.output_dir = output_dir
        self.model, self.tokenizer, self.W_emb = load_model_and_embeddings(
            model_name, device=device
        )

    #  I/O 
    def _save_json(
        self,
        results: list,
        layer_idx: int,
        neuron_label: str,
        topk_tokens: int,
        sign: str,
        out_dir: str,
    ):
        base_dir = out_dir or self.output_dir
        save_dir = os.path.join(
            base_dir,
            f"{self.model_name}_interp_dir",
            "data",
            f"MLP_wup_neuron_layer{layer_idx}",
        )
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir,
            f"detector_neurons{neuron_label}_top{topk_tokens}tokens_{sign}.json",
        )
        data = {
            "model_name":      self.model_name,
            "layer_idx":       layer_idx,
            "method":          "direct_wup_neuron",
            "sign":            sign,
            "neuron_label":    neuron_label,
            "topk_tokens":     topk_tokens,
            "Interpretability": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved to {path}")


    def run(
        self,
        layer_idx: int,
        top_neurons="top100",
        topk_tokens: int = 50,
        only_english: bool = False,
        only_ascii: bool = True,
        with_negative: bool = False,
        save_file: bool = False,
        return_heatmap: bool = True,
        out_dir: str = None,
    ) -> dict:
        """  Run direct neuron-level detector interpretation on W_up (c_fc). """
        c_fc, _, ln_2, _ = get_mlp_matrices(self.model, layer_idx)

        # LN-fused up-projection: W_up[i] * ln_2.weight
        # Conv1D stores weight as (d_model, 4*d_model); transpose → (4*d_model, d_model)
        W_up = c_fc.weight.T.detach() * ln_2.weight.detach()   # (4*d_model, d_model)

        total_neurons = W_up.shape[0]
        indices, label = parse_top_neurons(top_neurons, total_neurons)

        scores_list = [
            score_neuron(self.W_emb, W_up[idx]).cpu()
            for idx in tqdm(indices, desc=f"Layer {layer_idx} — scoring neurons")
        ]

        # Positive direction
        pos_results = build_results(
            self.tokenizer, indices, scores_list,
            topk_tokens, only_english, only_ascii,
        )
        if save_file:
            self._save_json(pos_results, layer_idx, label, topk_tokens, "positive", out_dir)
        if return_heatmap:
            plot_subspace_heatmap(
                pos_results,
                self.model_name,
                layer_idx=layer_idx,
                weight_type="wup_neuron",
                interp_type="detector",
                sign="positive",
                out_dir=out_dir or self.output_dir,
            )

        # Negative direction
        neg_results = None
        if with_negative:
            neg_scores = [-s for s in scores_list]
            neg_results = build_results(
                self.tokenizer, indices, neg_scores,
                topk_tokens, only_english, only_ascii,
                negate_values=True,
            )
            if save_file:
                self._save_json(neg_results, layer_idx, label, topk_tokens, "negative", out_dir)
            if return_heatmap:
                plot_subspace_heatmap(
                    neg_results,
                    self.model_name,
                    layer_idx=layer_idx,
                    weight_type="wup_neuron",
                    interp_type="detector",
                    sign="negative",
                    out_dir=out_dir or self.output_dir,
                )

        return {"positive": pos_results, "negative": neg_results}
