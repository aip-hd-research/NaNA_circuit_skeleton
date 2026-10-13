"""
within_ske_cache.py

Loader for precomputed within-layer skeleton caches produced by
precompute_within_ske.py.  Supports lookup by layer range M-N.

Usage
-----
    cache = WithinSkeCache("result/within_ske_cache", expected_model_name="gpt2-medium")

    # load layers 6-10
    data = cache.load_layers(6, 10)
    # data[layer_idx] = {
    #     "sorted_subspace_idx": Tensor[d_ff, n_subspaces],
    #     "sorted_delta":        Tensor[d_ff, n_subspaces],
    #     "n_neurons":           int,
    #     "n_subspaces":         int,
    # }

    # lookup skeleton for neuron j=3 at layer 8, top-k subspaces
    entry = cache.neuron_skeleton(layer_idx=8, neuron_idx=3, topk=20)
    # entry = list of dicts: {subspace_idx, delta_val, coupling_pos, coupling_neg, abs_strength, rank}
"""

import os
from typing import Optional

import torch
from tqdm import tqdm


class WithinSkeCache:
    def __init__(self, cache_dir: str, expected_model_name: str | None = None):
        """
        expected_model_name : if given, every loaded layer file's stored
            "model_name" (written by precompute_within_ske.py) must match this
            exactly, or loading raises immediately. cache_dir already encodes
            the model via directory naming convention, but that is not enforced
            anywhere — a wrong --cache_dir (stale copy, symlink, typo) would
            otherwise load silently and produce dimension-mismatched or simply
            wrong deltas/V/bias without any error. Layer files saved before this
            field existed have no "model_name" key; those are not validated
            (can't check what was never recorded), not treated as a mismatch.
        """
        self.cache_dir = cache_dir
        self.expected_model_name = expected_model_name
        self._loaded: dict = {}   # layer_idx -> payload

    def _layer_path(self, layer_idx: int) -> str:
        return os.path.join(self.cache_dir, f"layer_{layer_idx:03d}.pt")

    def _load_one(self, layer_idx: int) -> dict:
        if layer_idx not in self._loaded:
            path = self._layer_path(layer_idx)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Cache not found for layer {layer_idx}: {path}\n"
                    f"Run precompute_within_ske.py first."
                )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if self.expected_model_name is not None:
                cached_name = payload.get("model_name")
                if cached_name is not None and cached_name != self.expected_model_name:
                    raise ValueError(
                        f"within_ske_cache mismatch at {path}: cache was built for "
                        f"model_name={cached_name!r}, but this run requested "
                        f"model_name={self.expected_model_name!r}. Refusing to silently "
                        f"mix caches across models/dimensions — point --cache_dir at the "
                        f"correct model's cache or rerun precompute_within_ske.py."
                    )
            self._loaded[layer_idx] = payload
        return self._loaded[layer_idx]

    def preload_layers(self, start_layer: int, end_layer: int,
                       n_workers: int = 8,
                       slim_keys: set | None = None) -> None:
        """Eagerly load all layers in [start, end] using parallel threads.

        Parallel IO is especially effective on NFS where each sequential
        torch.load blocks on a round-trip; threads let multiple files be
        in flight simultaneously.

        slim_keys : if given, keep only these keys after loading each layer.
            Useful for dynamic skeleton which only needs {"delta", "V", "bias"}
            and not the large sorted_subspace_idx / sorted_delta arrays
            (saves ~566 MB × n_layers of CPU RAM for Llama-scale models).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        layers = [li for li in range(start_layer, end_layer + 1)
                  if li not in self._loaded]
        if not layers:
            return
        n_workers = min(n_workers, len(layers))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(self._load_one, li): li for li in layers}
            for f in tqdm(as_completed(futures), total=len(layers),
                          desc="Loading ske cache", unit="layer"):
                f.result()  # re-raise any IO error immediately

        if slim_keys is not None:
            for li in range(start_layer, end_layer + 1):
                d = self._loaded[li]
                for k in [k for k in list(d.keys()) if k not in slim_keys]:
                    del d[k]

    def load_layers(self, layer_start: int, layer_end: int) -> dict:
        """
        Load cached skeletons for layers [layer_start, layer_end] (both inclusive).

        Returns
        -------
        dict[layer_idx -> {
            "sorted_subspace_idx": Tensor[d_ff, n_subspaces],
            "sorted_delta":        Tensor[d_ff, n_subspaces],
            "n_neurons":           int,
            "n_subspaces":         int,
        }]
        """
        result = {}
        for layer_idx in range(layer_start, layer_end + 1):
            payload = self._load_one(layer_idx)
            result[layer_idx] = {
                "sorted_subspace_idx": payload["sorted_subspace_idx"],
                "sorted_delta":        payload["sorted_delta"],
                "n_neurons":           payload["n_neurons"],
                "n_subspaces":         payload["n_subspaces"],
            }
        return result

    def neuron_skeleton(
        self,
        layer_idx: int,
        neuron_idx: int,
        topk: Optional[int] = None,
    ) -> list[dict]:
        """
        Return the within-layer skeleton for a single neuron.

        Parameters
        ----------
        layer_idx  : layer to query
        neuron_idx : neuron j (0-indexed)
        topk       : return only the top-k subspaces; None = all subspaces

        Returns
        -------
        List of dicts sorted by abs_strength descending:
            subspace_idx  : int
            delta_val     : float   (signed)
            coupling_pos  : float   max(delta_val, 0)
            coupling_neg  : float   min(delta_val, 0)
            abs_strength  : float
            rank          : int     1-indexed
        """
        payload = self._load_one(layer_idx)
        sub_idx = payload["sorted_subspace_idx"][neuron_idx]   # [n_subspaces]
        delta   = payload["sorted_delta"][neuron_idx]          # [n_subspaces]

        n = topk if topk is not None else len(sub_idx)
        results = []
        for rank, (sp, dv) in enumerate(
            zip(sub_idx[:n].tolist(), delta[:n].tolist()), 1
        ):
            results.append({
                "subspace_idx":  sp,
                "delta_val":     dv,
                "coupling_pos":  max(dv, 0.0),
                "coupling_neg":  min(dv, 0.0),
                "abs_strength":  abs(dv),
                "rank":          rank,
            })
        return results

    def layer_skeleton(self, layer_idx: int) -> dict:
        """
        Return all cached tensors for an entire layer.

        Returns whatever keys are present in the payload — some may have been
        dropped by preload_layers(slim_keys=...) to save CPU RAM.
        """
        return dict(self._load_one(layer_idx))
