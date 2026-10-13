"""
neuron_arg_parser.py

Utility for parsing the top_neurons argument used across neuron-interpretation
scripts.
"""


def parse_top_neurons(top_neurons, total_neurons: int) -> tuple:
    """Convert a top_neurons argument to (indices, label).

    Accepts
    -------
    int          – first N neurons  → indices = [0, …, N-1]
    list[int]    – explicit neuron indices
    str "all"    – all neurons
    str "topN"   – first N neurons  (e.g. "top100")

    Returns
    -------
    indices : list[int]
    label   : str  (used in filenames / logging)
    """
    if isinstance(top_neurons, str):
        if top_neurons.lower() == "all":
            return list(range(total_neurons)), "all"
        elif top_neurons.lower().startswith("top"):
            try:
                n = int(top_neurons[3:])
            except ValueError:
                raise ValueError(f"Invalid top_neurons format: {top_neurons!r}")
            indices = list(range(min(n, total_neurons)))
            return indices, f"top{len(indices)}"
        else:
            raise ValueError(f"Invalid top_neurons string: {top_neurons!r}")
    elif isinstance(top_neurons, int):
        indices = list(range(min(top_neurons, total_neurons)))
        return indices, f"top{len(indices)}"
    elif isinstance(top_neurons, (list, tuple)):
        indices = [i for i in top_neurons if 0 <= i < total_neurons]
        if not indices:
            raise ValueError("top_neurons list is empty or all indices are out of range.")
        return indices, "[" + "_".join(str(i) for i in indices) + "]"
    else:
        raise TypeError("top_neurons must be int, list[int], or str.")
