import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_skeleton.skeleton import load_skeleton


# node ID helpers 

def neuron_id(layer: int, idx: int) -> str:
    return f"N_l{layer}_j{idx}"


def subspace_id(layer: int, idx: int) -> str:
    return f"S_l{layer}_i{idx}"


# graph builder  
def build_skeleton_graph(skeleton_result: dict) -> nx.DiGraph:
    """
    Build a directed weighted graph from a skeleton result dict.

    Parameters
    ----------
    skeleton_result : dict returned by compute_skeleton() or loaded via load_skeleton()

    Returns
    -------
    G : nx.DiGraph
        Nodes carry: type, layer, idx
        Edges carry: signed_weight, abs_strength, coupling_pos, coupling_neg, sign,
                     edge_type, rank
    """
    G = nx.DiGraph()
    start_layer = skeleton_result["config"]["start_layer"]
    mode        = skeleton_result["config"].get("mode", "")

    # ── complete_fullinfo: direct neuron → neuron edges (no subspace) ─────────
    if mode == "complete_fullinfo":
        for neuron_rec in skeleton_result["within_layer"]:
            j   = neuron_rec["neuron_idx"]
            src = neuron_id(start_layer, j)
            G.add_node(src, type="neuron", layer=start_layer, idx=j)

            for conn in neuron_rec["connections"]:
                tgt_layer = conn["target_layer"]
                k         = conn["tgt_neuron"]
                dst       = neuron_id(tgt_layer, k)
                G.add_node(dst, type="neuron", layer=tgt_layer, idx=k)

                signed = conn["coupling_pos"] + conn["coupling_neg"]
                G.add_edge(
                    src, dst,
                    signed_weight = signed,
                    abs_strength  = conn["abs_strength"],
                    coupling_pos  = conn["coupling_pos"],
                    coupling_neg  = conn["coupling_neg"],
                    sign          = "+" if signed >= 0 else "-",
                    edge_type     = "direct",
                    rank          = conn["rank"],
                    source_layer  = start_layer,
                    target_layer  = tgt_layer,
                )
        return G

    #  within-layer edges: neuron -> subspace
    for neuron_rec in skeleton_result["within_layer"]:
        j = neuron_rec["neuron_idx"]
        nid = neuron_id(start_layer, j)
        G.add_node(nid, type="neuron", layer=start_layer, idx=j)

        for conn in neuron_rec["connections"]:
            i   = conn["subspace_idx"]
            sid = subspace_id(start_layer, i)
            G.add_node(sid, type="subspace", layer=start_layer, idx=i)

            signed = conn["coupling_pos"] + conn["coupling_neg"]  # pos + neg = raw value
            G.add_edge(
                nid, sid,
                signed_weight = signed,
                abs_strength  = conn["abs_strength"],
                coupling_pos  = conn["coupling_pos"],
                coupling_neg  = conn["coupling_neg"],
                sign          = "+" if signed >= 0 else "-",
                edge_type     = "within_layer",
                rank          = conn["rank"],
            )

    # cross-layer edges: subspace -> neuron
    for space_rec in skeleton_result["cross_layer"]:
        src_layer = space_rec["source_layer"]
        i         = space_rec["subspace_idx"]
        sid       = subspace_id(src_layer, i)
        G.add_node(sid, type="subspace", layer=src_layer, idx=i)

        for conn in space_rec["connections"]:
            tgt_layer = conn["target_layer"]
            k         = conn["neuron_idx"]
            nid       = neuron_id(tgt_layer, k)
            G.add_node(nid, type="neuron", layer=tgt_layer, idx=k)

            signed = conn["coupling_pos"] + conn["coupling_neg"]
            G.add_edge(
                sid, nid,
                signed_weight = signed,
                abs_strength  = conn["abs_strength"],
                coupling_pos  = conn["coupling_pos"],
                coupling_neg  = conn["coupling_neg"],
                sign          = "+" if signed >= 0 else "-",
                edge_type     = "cross_layer",
                rank          = conn["rank"],
                source_layer  = src_layer,
                target_layer  = tgt_layer,
            )

    return G


#   convenience accessors  

def get_nodes_by_type(G: nx.DiGraph, node_type: str) -> list:
    """Return all node IDs of a given type ("neuron" or "subspace")."""
    return [n for n, d in G.nodes(data=True) if d.get("type") == node_type]


def get_nodes_by_layer(G: nx.DiGraph, layer: int) -> list:
    """Return all node IDs belonging to a given layer."""
    return [n for n, d in G.nodes(data=True) if d.get("layer") == layer]


def get_edges_sorted(G: nx.DiGraph, edge_type: str = None) -> list:
    """
    Return edges sorted by abs_strength (descending).

    Parameters
    ----------
    edge_type : "within_layer", "cross_layer", or None (all edges)

    Returns
    -------
    List of (src, dst, attr_dict) tuples.
    """
    edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if edge_type is None or d.get("edge_type") == edge_type
    ]
    return sorted(edges, key=lambda x: x[2]["abs_strength"], reverse=True)


def graph_summary(G: nx.DiGraph) -> dict:
    """Return a brief summary of the graph."""
    neuron_nodes   = get_nodes_by_type(G, "neuron")
    subspace_nodes = get_nodes_by_type(G, "subspace")
    within_edges   = [e for e in G.edges(data=True) if e[2]["edge_type"] in ("within_layer", "direct")]
    cross_edges    = [e for e in G.edges(data=True) if e[2]["edge_type"] == "cross_layer"]
    pos_edges      = [e for e in G.edges(data=True) if e[2]["sign"] == "+"]
    neg_edges      = [e for e in G.edges(data=True) if e[2]["sign"] == "-"]

    layers = sorted({d["layer"] for _, d in G.nodes(data=True)})

    return {
        "n_nodes":         G.number_of_nodes(),
        "n_neuron_nodes":  len(neuron_nodes),
        "n_subspace_nodes":len(subspace_nodes),
        "n_edges":         G.number_of_edges(),
        "n_within_edges":  len(within_edges),
        "n_cross_edges":   len(cross_edges),
        "n_pos_edges":     len(pos_edges),
        "n_neg_edges":     len(neg_edges),
        "layers_present":  layers,
    }


# from file 

def build_graph_from_file(path: str) -> nx.DiGraph:
    """Load a saved skeleton JSON and build the graph directly."""
    return build_skeleton_graph(load_skeleton(path))
