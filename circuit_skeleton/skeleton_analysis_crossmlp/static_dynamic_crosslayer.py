import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import torch

from block_interp.model_load import load_model_and_embeddings
from circuit_skeleton.coupl_strength import build_coupling_strength

# activation collection 

def collect_activations(model, tokenizer, prompts, layers, device, batch_size=32):
    """
    Collect post-GELU MLP activations at the last token position.
    Hooks the input of c_proj (= output of act(c_fc(x))) for each layer.

    Parameters
    ----------
    batch_size : int
        Number of prompts processed per forward pass (default 32).
        Since all prompts share the same token length, no padding overhead.

    Returns
    -------
    dict { layer_idx: np.ndarray [n_samples, d_ff] }
    """
    # GPT-2 tokenizer has no pad token by default; use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    storage = {}
    hooks   = []

    for layer_idx in layers:
        def make_hook(li):
            def hook_fn(_module, inp, _out):
                # inp[0]: [batch, seq_len, d_ff] → last token → [batch, d_ff]
                storage[li] = inp[0][:, -1, :].detach().cpu()
            return hook_fn
        hooks.append(
            model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(
                make_hook(layer_idx)
            )
        )

    accumulated = {li: [] for li in layers}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch  = prompts[i : i + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            model(**tokens)
            for li in layers:
                if li in storage:
                    accumulated[li].append(storage[li].numpy())  # [B, d_ff]

    for h in hooks:
        h.remove()

    # concatenate along sample axis: list of [B, d_ff] → [n_samples, d_ff]
    return {li: np.concatenate(accumulated[li], axis=0) for li in layers}

 