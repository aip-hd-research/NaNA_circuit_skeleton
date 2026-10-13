"""
neuron_token_scorer.py

Standalone functions for scoring neurons against the vocabulary and
building interpretability result lists.
"""

import torch
from block_interp.top_tok import top_tokens


def score_neuron(W_emb: torch.Tensor, neuron_vec: torch.Tensor) -> torch.Tensor:
    """Compute token alignment scores for one neuron."""
    return W_emb @ neuron_vec


def extract_top_tokens(
    tokenizer,
    scores: torch.Tensor,
    topk: int,
    only_english: bool,
    only_ascii: bool,
) -> list:
    """Return top-k positive tokens as a list of {"token": …, "value": …} dicts."""
    mask = scores > 0
    if not mask.any():
        return [{"token": "=== No valid tokens ===", "value": None}]
    filtered = scores.clone()
    filtered[~mask] = -float("inf")
    raw = top_tokens(
        tokenizer,
        v_tok=filtered,
        k=topk,
        only_english=only_english,
        only_ascii=only_ascii,
    )
    return [{"token": t, "value": float(v)} for t, v in raw]


def build_results(
    tokenizer,
    indices: list,
    scores_list: list,
    topk_tokens: int,
    only_english: bool,
    only_ascii: bool,
    negate_values: bool = False,
) -> list:
    """Build the standard result list consumed by _save_json and plot_subspace_heatmap.

    Parameters
    ----------
    tokenizer    : HuggingFace tokenizer
    indices      : list of original neuron indices (0-based)
    scores_list  : list of score tensors, one per neuron
    topk_tokens  : how many top tokens to keep
    negate_values: flip sign on token values (used for negative direction)

    Returns
    -------
    list of {"direction": int, "top_tokens": [...]} dicts
    """
    results = []
    for real_idx, score in zip(indices, scores_list):
        top_tok = extract_top_tokens(tokenizer, score, topk_tokens, only_english, only_ascii)
        if negate_values:
            for t in top_tok:
                if t["value"] is not None:
                    t["value"] = -t["value"]
        results.append({"direction": real_idx + 1, "top_tokens": top_tok})
    return results
