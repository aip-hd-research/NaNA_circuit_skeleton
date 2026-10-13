"""
Display and save utilities for the semantic consistency experiment.
"""

import os
import json
import csv

from tabulate import tabulate


def _fmt(tok_list, n=5) -> str:
    return "  ".join(t for t, _ in tok_list[:n])


def print_within_table(exp_result: dict, n_tokens: int = 5) -> None:
    """
    Print within-layer results grouped by neuron.

    For each neuron j:
      header line  — neuron detect(+) / detect(-) tokens
      table rows   — one row per connected subspace i
    """
    cfg = exp_result["config"]
    print(f"\n{'='*70}")
    print(f"  Within-layer  |  {cfg['model_name']}  layer={cfg['start_layer']}")
    print(f"{'='*70}")

    for rec in exp_result["within_layer"]:
        j    = rec["neuron_idx"]
        conn0 = rec["connections"][0]
        print(f"\n  Neuron j={j}")
        print(f"    detect(+) : {_fmt(conn0['neuron_pos_tokens'], n_tokens)}")
        print(f"    detect(-) : {_fmt(conn0['neuron_neg_tokens'], n_tokens)}")

        rows = []
        for conn in rec["connections"]:
            sign = "+" if conn["coupling_pos"] > 0 else "-"
            rows.append([
                f"#{conn['rank']}",
                conn["subspace_idx"],
                f"{sign}{conn['abs_strength']:.4f}",
                _fmt(conn["subspace_pos_tokens"], n_tokens),
                _fmt(conn["subspace_neg_tokens"], n_tokens),
            ])
        print(tabulate(rows,
                       headers=["Rank", "Sub i", "δ", "write(+)", "write(-)"],
                       tablefmt="simple"))


def print_cross_table(exp_result: dict, n_tokens: int = 5) -> None:
    """
    Print cross-layer results grouped by subspace.

    For each subspace i:
      header line  — subspace write(+) / write(-) tokens
      table rows   — one row per connected neuron (layer, k)
    """
    cfg = exp_result["config"]
    print(f"\n{'='*70}")
    print(f"  Cross-layer  |  {cfg['model_name']}  "
          f"L{cfg['start_layer']} -> {cfg['target_layers']}")
    print(f"{'='*70}")

    for rec in exp_result["cross_layer"]:
        i     = rec["subspace_idx"]
        conn0 = rec["connections"][0]
        print(f"\n  Subspace i={i}")
        print(f"    write(+) : {_fmt(conn0['subspace_pos_tokens'], n_tokens)}")
        print(f"    write(-) : {_fmt(conn0['subspace_neg_tokens'], n_tokens)}")

        rows = []
        for conn in rec["connections"]:
            sign = "+" if conn["coupling_pos"] > 0 else "-"
            rows.append([
                f"#{conn['rank']}",
                conn["target_layer"],
                conn["neuron_idx"],
                f"{sign}{conn['abs_strength']:.4f}",
                _fmt(conn["neuron_pos_tokens"], n_tokens),
                _fmt(conn["neuron_neg_tokens"], n_tokens),
            ])
        print(tabulate(rows,
                       headers=["Rank", "Layer", "Neu k", "γ",
                                 "detect(+)", "detect(-)"],
                       tablefmt="simple"))


def save_semantic_labels(exp_result: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(exp_result, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {output_path}")


def save_semantic_labels_csv(exp_result: dict, output_dir: str) -> None:
    """
    Save within-layer and cross-layer results as two CSV files.
    Token columns contain space-separated token strings only (no scores).
    """
    def tok_str(tok_list):
        return " | ".join(t for t, _ in tok_list)

    os.makedirs(output_dir, exist_ok=True)

    # within-layer CSV
    within_path = os.path.join(output_dir, "exp_within_layer.csv")
    with open(within_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "neuron_idx", "subspace_idx", "delta", "sign",
            "neuron_pos", "neuron_neg",
            "subspace_pos", "subspace_neg",
        ])
        for rec in exp_result["within_layer"]:
            for conn in rec["connections"]:
                sign = "+" if conn["coupling_pos"] > 0 else "-"
                writer.writerow([
                    conn["neuron_idx"],
                    conn["subspace_idx"],
                    round(conn["abs_strength"], 6),
                    sign,
                    tok_str(conn["neuron_pos_tokens"]),
                    tok_str(conn["neuron_neg_tokens"]),
                    tok_str(conn["subspace_pos_tokens"]),
                    tok_str(conn["subspace_neg_tokens"]),
                ])
    print(f"Saved -> {within_path}")

    # cross-layer CSV
    cross_path = os.path.join(output_dir, "exp_cross_layer.csv")
    with open(cross_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subspace_idx", "source_layer", "neuron_idx", "target_layer",
            "gamma", "sign",
            "subspace_pos", "subspace_neg",
            "neuron_pos", "neuron_neg",
        ])
        for rec in exp_result["cross_layer"]:
            for conn in rec["connections"]:
                sign = "+" if conn["coupling_pos"] > 0 else "-"
                writer.writerow([
                    rec["subspace_idx"],
                    rec["source_layer"],
                    conn["neuron_idx"],
                    conn["target_layer"],
                    round(conn["abs_strength"], 6),
                    sign,
                    tok_str(conn["subspace_pos_tokens"]),
                    tok_str(conn["subspace_neg_tokens"]),
                    tok_str(conn["neuron_pos_tokens"]),
                    tok_str(conn["neuron_neg_tokens"]),
                ])
    print(f"Saved -> {cross_path}")
