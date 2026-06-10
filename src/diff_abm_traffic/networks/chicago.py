from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ._network_utils import (
    build_link_centroids_and_normalize,
    filter_and_remap,
    parse_tntp_net_file,
    parse_tntp_node_file,
    random_drop_virtual_links,
    report_link_statistics,
    stretch_virtual_link_lengths,
    validate_filtered_network,
)


_VIRT_LEN_MULTIPLIER = 1e3
_VIRTUAL_DROP_SEED = 2021

def build_network_chicago(
    script_dir: Optional[str] = None,
    net_path: Optional[str] = None,
    node_path: Optional[str] = None,
    forbid_uturn: bool = True,
) -> dict:

    if script_dir is None:
        script_dir = os.getcwd()
    if net_path is None:
        net_path = os.path.normpath(
            os.path.join(script_dir, "chicagosketch", "ChicagoSketch_net.tntp")
        )
    if node_path is None:
        node_path = os.path.normpath(
            os.path.join(script_dir, "chicagosketch", "ChicagoSketch_node.tntp")
        )

    if not os.path.exists(net_path):
        raise FileNotFoundError(f"Chicago net file not found: {net_path}")

    # 1) Parse TNTP files.
    init_nodes, term_nodes, lengths, free_flow_times, speeds, link_types = (
        parse_tntp_net_file(net_path)
    )
    orig_n_links = init_nodes.size

    base_adj = (
        term_nodes.reshape(-1, 1) == init_nodes.reshape(1, -1)
    ).astype(float)

    # 2) Boundary detection (degree-1 junctions).
    print("Detecting boundary junctions...")
    unique_junctions = np.unique(np.concatenate([init_nodes, term_nodes]))
    junction_neighbors = {int(j): set() for j in unique_junctions}
    for s, e in zip(init_nodes, term_nodes):
        junction_neighbors[int(s)].add(int(e))
        junction_neighbors[int(e)].add(int(s))
    degree_one_junctions = [
        jid for jid, neigh in junction_neighbors.items() if len(neigh) == 1
    ]
    in_node_ids = degree_one_junctions.copy()
    out_node_ids = degree_one_junctions.copy()

    virtual_in_indices = sorted(
        set(np.where(np.isin(init_nodes, in_node_ids))[0].tolist())
    )
    virtual_out_indices = sorted(
        set(np.where(np.isin(term_nodes, out_node_ids))[0].tolist())
    )
    print(
        f"Before random drop: virtual_in={len(virtual_in_indices)}, "
        f"virtual_out={len(virtual_out_indices)}"
    )

    # 3) Random drop of overlapping virtual links per boundary node.
    virtual_in_indices, virtual_out_indices, dropped_links = (
        random_drop_virtual_links(
            init_nodes,
            term_nodes,
            orig_n_links,
            virtual_in_indices,
            virtual_out_indices,
            seed=_VIRTUAL_DROP_SEED,
            forbid_uturn=forbid_uturn,
            vin_key="init",
            vout_key="term",
        )
    )

    # 4) Filter + remap.
    filt = filter_and_remap(
        init_nodes,
        term_nodes,
        base_adj,
        lengths,
        free_flow_times,
        speeds,
        link_types,
        orig_n_links,
        dropped_links,
        virtual_in_indices,
        virtual_out_indices,
        forbid_uturn=forbid_uturn,
    )

    # 5) Stretch virtual link lengths so they act as buffers.
    l_np, l, virt_len_val = stretch_virtual_link_lengths(
        filt["l_np"],
        filt["virtual_in_indices"],
        filt["virtual_out_indices"],
        multiplier=_VIRT_LEN_MULTIPLIER,
    )

    # 6) Print statistics (excluding virtual links).
    report_link_statistics(
        l_np,
        np.asarray(filt["u"]),
        filt["virtual_in_indices"],
        filt["virtual_out_indices"],
        filt["n_links"],
    )

    # 7) Load + normalise node coordinates for visualisation.
    junction_xy_raw = parse_tntp_node_file(node_path)
    node_xy, junction_xy_norm = build_link_centroids_and_normalize(
        junction_xy_raw,
        filt["init_nodes"],
        filt["term_nodes"],
        filt["old_to_new"],
        convert_latlon=False,
    )

    # 8) Validate.
    validate_filtered_network(
        filt["init_nodes"],
        filt["term_nodes"],
        filt["kept_indices"],
        orig_n_links,
        filt["n_links"],
        filt["adj_matrix"],
        l,
        filt["u"],
        filt["virtual_in_indices"],
        filt["virtual_out_indices"],
    )

    print(
        f"Built filtered Chicago network with {orig_n_links} original links, "
        f"kept {filt['n_links']} links, "
        f"{len(filt['virtual_in_indices'])} virtual in links, "
        f"{len(filt['virtual_out_indices'])} virtual out links."
    )

    vin = filt["virtual_in_indices"]
    vout = filt["virtual_out_indices"]
    return {
        "init_nodes": filt["init_nodes"],
        "term_nodes": filt["term_nodes"],
        "orig_n_links": orig_n_links,
        "n_links": filt["n_links"],
        "adj_matrix": filt["adj_matrix"],
        "virtual_in_idx": int(vin[0]) if vin else None,
        "virtual_out_idx": int(vout[0]) if vout else None,
        "virtual_in_indices": vin,
        "virtual_out_indices": vout,
        "node_xy": node_xy,
        "link_type": filt["link_types"],
        "junction_xy": junction_xy_norm,
        "degree_one_junctions": degree_one_junctions,
        "junction_neighbors": {
            int(k): list(v) for k, v in junction_neighbors.items()
        },
        "l": l,
        "u": filt["u"],
        "l_np": l_np,
        "virt_len_val": None,
    }
