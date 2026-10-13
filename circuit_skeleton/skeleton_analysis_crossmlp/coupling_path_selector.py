"""
coupling_path_selector.py

Utilities for selecting top-k / bot-k / random coupling paths for one source
neuron j, given a CouplingStrength object.

Public API
----------
_parse_layers(s)              – parse "17,18,19" or "17-21" → list[int]
_abs_delta_at(...)            – mean |Δa_k| at a flat cross-layer index
_layer_restricted_pool(...)   – candidate flat indices for the random baseline
"""

import numpy as np


def _parse_layers(s: str) -> list:
    """Parse a layer spec string ("17,18,19" or "17-21") into a list of ints."""
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _abs_delta_at(patched_acts, baseline_acts, target_layers, cross_flat):
    """Return mean |Δa_k| at flat position *cross_flat* in the concatenated
    target activation space.

    Parameters
    ----------
    patched_acts, baseline_acts : dict { layer_idx: np.ndarray [n_samples, d_ff] }
    target_layers               : list[int]  (same order used to build the flat space)
    cross_flat                  : int        flat index into the concatenated target space
    """
    offset = 0
    for li in target_layers:
        d_ff = patched_acts[li].shape[1]
        if cross_flat < offset + d_ff:
            local_k = cross_flat - offset
            delta   = patched_acts[li][:, local_k] - baseline_acts[li][:, local_k]
            return float(np.abs(delta).mean())
        offset += d_ff
    return 0.0


def _layer_restricted_pool(
    sel_layers: set,
    n_flat: int,
    total_cross: int,
    layer_offsets: list,
    mode: str,
    exclude: set,
) -> list:
    """Build the candidate pool for random-baseline sampling.

    The pool is restricted to flat indices whose target neuron k belongs to one
    of the layers in *sel_layers* (union of top-k and bot-k target layers).
    Indices already in *exclude* (top-k ∪ bot-k) are removed.

    Flat-index conventions
    ----------------------
    complete_skeleton_fullinfo : flat_idx == cross_flat            (length total_cross)
    complete_skeleton_global   : flat_idx == sp * total_cross + cross_flat
                                                                   (length n_sp * total_cross)
    """
    cross_ranges = [
        (off, off + d_ff)
        for (off, layer, d_ff) in layer_offsets
        if layer in sel_layers
    ]
    if not cross_ranges:
        return []

    if mode == "complete_skeleton_fullinfo":
        parts = [np.arange(start, end, dtype=np.int64)
                 for (start, end) in cross_ranges]
    else:
        n_sp  = n_flat // total_cross
        parts = []
        for sp in range(n_sp):
            base = sp * total_cross
            for (start, end) in cross_ranges:
                parts.append(np.arange(base + start, base + end, dtype=np.int64))

    if not parts:
        return []
    all_flat = np.concatenate(parts)

    if exclude:
        excl_arr = np.array(list(exclude), dtype=np.int64)
        all_flat = all_flat[~np.isin(all_flat, excl_arr)]

    return all_flat.tolist()
