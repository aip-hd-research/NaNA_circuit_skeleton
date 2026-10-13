import os
import json
import torch
from block_interp.interp_mlp import MLP_DEEF_INTERP
from circuit.neuron_subspace_coupling import compute_neuron_coupling, get_top_neurons_for_subspace
from plot_utils.plot_heatmap import plot_subspace_heatmap
from ske_pre_valid.static_skeleton import _resolve_target_info


def compute_detector_alignments(x, U, S):
    """Compute detector alignment scores for each subspace of W_down (c_proj).

    det_val_j = sigma_j * v_j^T @ y,  where y = x = layer_io["input"] for c_proj.
    """
    num_subspaces = S.shape[0]
    svd_scores = [x @ (S[i] * U[:, i]) for i in range(num_subspaces)]
    return torch.stack(svd_scores, dim=0).squeeze(-1)


def compute_effector_alignments(target_id, W_E, Vh, S):
    """Compute effector alignment scores for each subspace of W_down (c_proj).

    eff_val_j = u_j^T @ t,  u_j = Vh[j, :] (right singular vector of W_down).
    """
    num_subspaces = S.shape[0]
    emb_target = W_E[target_id]
    alignments = torch.tensor([Vh[i, :] @ emb_target for i in range(num_subspaces)])
    return alignments


def analyze_mlp_subspaces(
    model_name,
    tokenizer=None,
    W_E=None,
    U=None,
    S=None,
    Vh=None,
    layer_idx=0,
    layer_io=None,
    target_word=None,
    target_id=None,
    topk_subspaces=10,
    weight_type="c_proj",
    mode="DeEf",
    interp_type=None,
    output_dir="result",
    device=None,
    topk_tokens=20,
    return_heatmap=True,
    do_interp=True,
    use_abs_contribute=False,
    topk_neurons=10,
):
    """
    Analyze MLP subspaces of W_down (c_proj). W_up is accessed only at neuron level.

    Modes:
        "De"   -> Detector alignment only
        "Ef"   -> Effector alignment only
        "DeEf" -> Both; also returns top W_up neurons per subspace via p_ij = v_ji * y_i

    Args:
        topk_neurons: number of top W_up neurons per subspace (DeEf only)
        target_id: exact token id for the target, when the caller already knows it (e.g.
            a dataset item's own target_token_id). Takes priority over re-resolving from
            target_word — decoded text alone is not a reliable identity, since distinct
            tokens can decode to the same string (e.g. leading-space variants), and
            re-resolving by string can silently pick a different id than the sample's own.

    Returns:
        interp_circuit:      MLP_DEEF_INTERP results (None if do_interp=False)
        circuit_point_score: dict with per-subspace contributions and neuron coupling
    """

    circuit_point_score = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "mode": mode,
        "weight_type": weight_type,
        "target_word": target_word,
        "topk_subspaces": topk_subspaces,
        "use_abs_contribute": use_abs_contribute,
        "subspace_results": []
    }

    # Detector Scores — always from W_down (c_proj) SVD
    detector_alignments = None
    if mode in ["De", "DeEf"]:
        detector_alignments = compute_detector_alignments(layer_io["input"], U, S)

    # Effector Scores — always from W_down (c_proj) SVD
    effector_alignments = None
    if mode in ["Ef", "DeEf"]:
        if target_id is not None:
            # Caller already knows the exact id (e.g. dataset item's target_token_id) —
            # use it directly rather than re-resolving from the decoded target_word
            # string, which can't distinguish distinct tokens that decode identically.
            print(f"target_word '{target_word}' -> target_id: {target_id} (caller-supplied)")
        else:
            # encode(...)[0] silently truncated multi-token targets to their first token
            # (scoring the wrong token with no warning). _resolve_target_info rejects
            # anything that isn't a clean single token instead of guessing.
            info = _resolve_target_info(tokenizer, target_word)
            if not info["target_is_single"]:
                raise ValueError(
                    f"target_word {target_word!r} is not a single token for this tokenizer "
                    f"(resolve={info['target_resolve']!r}) — cannot compute effector alignment "
                    f"against a single target_id."
                )
            target_id = info["target_id"]
            print(f"target_word '{target_word}' -> target_id: {target_id}")
        effector_alignments = compute_effector_alignments(target_id, W_E, Vh, S)

    # Select Top-K Subspaces
    subspaces_for_interp = []
    
    def use_abs(x):
        return x.abs() if use_abs_contribute else x

    if mode == "DeEf":
        directional_contrib = detector_alignments.to(effector_alignments.device) * effector_alignments
        top_vals, top_idx = torch.topk(use_abs(directional_contrib), topk_subspaces)

        # layer_io["input"] for c_proj IS the post-activation W_up output: y = f(W_up @ x)
        y_neurons = layer_io["input"].squeeze(0).float()

        print(f"\nTop-{topk_subspaces} subspaces most aligned with '{target_word}':")
        for idx in top_idx:
            det_val = detector_alignments[idx].item()
            eff_val = effector_alignments[idx].item()
            contrib_val = directional_contrib[idx].item()

            # Neuron-to-subspace coupling: p_ij = v_ji * y_i
            v_j = U[:, idx].float().to(y_neurons.device)
            p_j = compute_neuron_coupling(y_neurons, v_j)
            top_neuron_info = get_top_neurons_for_subspace(p_j, det_val, eff_val, topk_neurons)

            print(f"  Subspace {idx+1}: Contribution {contrib_val:.6f} "
                  f"(Detector {det_val:.6f}, Effector {eff_val:.6f})")
                  #f"(neuron {top_neuron_info})")

            circuit_point_score["subspace_results"].append({
                "subspace_index": int(idx.item()) + 1,
                "contribution": float(contrib_val),
                "detector_value": float(det_val),
                "effector_value": float(eff_val),
                **top_neuron_info,
            })

        subspaces_for_interp = top_idx.tolist()

    elif mode == "De":
        top_vals, top_idx = torch.topk(use_abs(detector_alignments), topk_subspaces)
        
        print(f"\nTop-{topk_subspaces} detector subspaces:")
        for idx in top_idx:
            val = detector_alignments[idx].item()
            print(f"  Subspace {idx+1}: Contribution {val:.6f}")
            circuit_point_score["subspace_results"].append({
                "subspace_index": int(idx.item()) + 1,
                "contribution": float(val)
            })
            
        subspaces_for_interp = top_idx.tolist()

    elif mode == "Ef":
        top_vals, top_idx = torch.topk(use_abs(effector_alignments), topk_subspaces)
        
        print(f"\nTop-{topk_subspaces} effector subspaces:")
        for idx in top_idx:
            val = effector_alignments[idx].item()
            print(f"  Subspace {idx+1}: Contribution {val:.6f}")
            circuit_point_score["subspace_results"].append({
                "subspace_index": int(idx.item()) + 1,
                "contribution": float(val)
            })
            
        subspaces_for_interp = top_idx.tolist()

    else:
        raise ValueError(f"Unknown mode: {mode}")


    # Call MLP_DEEF_INTERP
    interp_circuit = None
    if do_interp:
        interp_io = MLP_DEEF_INTERP(model_name, output_dir, device)
        with torch.no_grad():
            interp_circuit = interp_io.mlp_subspace_interp(
                layer_idx=layer_idx,
                out_dir=output_dir,
                topk_tokens=topk_tokens,
                topk_subspaces=subspaces_for_interp,
                weight_type=weight_type,
                interp_type=interp_type,
                with_negative=True,
                save_file=False,
                return_heatmap=return_heatmap 
            )
        
    
    return interp_circuit, circuit_point_score
