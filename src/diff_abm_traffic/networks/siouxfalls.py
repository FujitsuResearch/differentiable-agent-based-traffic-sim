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


_VIRT_LEN_MULTIPLIER = 1e2
_VIRTUAL_DROP_SEED = 2021


def build_network_siouxfalls(
    script_dir: Optional[str] = None,
    net_path: Optional[str] = None,
    node_path: Optional[str] = None,
    forbid_uturn: bool = True,
) -> dict:

    if script_dir is None:
        script_dir = os.getcwd()
    if net_path is None:
        net_path = os.path.normpath(
            os.path.join(script_dir, "SiouxFalls", "SiouxFalls_net.tntp")
        )
    if node_path is None:
        node_path = os.path.normpath(
            os.path.join(script_dir, "SiouxFalls", "SiouxFalls_node.tntp")
        )

    if not os.path.exists(net_path):
        raise FileNotFoundError(f"SiouxFalls net file not found: {net_path}")

    # 1) Parse TNTP files (with fall-backs for missing columns).
    init_nodes, term_nodes, lengths, free_flow_times, speeds, link_types = (
        parse_tntp_net_file(net_path)
    )
    real_n_links = init_nodes.size
    if lengths is None:
        lengths = np.ones((real_n_links,), dtype=float)
    if free_flow_times is None:
        free_flow_times = np.ones((real_n_links,), dtype=float)
    if speeds is None:
        speeds = np.zeros((real_n_links,), dtype=float)

    # 2) Detect junctions.
    print("Detecting junction structure...")
    unique_junctions = np.unique(np.concatenate([init_nodes, term_nodes]))
    junction_neighbors = {int(j): set() for j in unique_junctions}
    for s, e in zip(init_nodes, term_nodes):
        junction_neighbors[int(s)].add(int(e))
        junction_neighbors[int(e)].add(int(s))
    degree_one_junctions = [
        jid for jid, neigh in junction_neighbors.items() if len(neigh) == 1
    ]
    all_junctions = sorted(int(j) for j in unique_junctions.tolist())

    # 3) Append synthetic virtual_in / virtual_out links for every junction.
    max_real_node_id = max(all_junctions) if all_junctions else int(
        max(np.max(init_nodes), np.max(term_nodes))
    )
    virtual_in_node_by_junction = {}
    virtual_out_node_by_junction = {}

    virtual_init_nodes = []
    virtual_term_nodes = []
    virtual_type_names = []
    for i, junction_id in enumerate(all_junctions):
        vin_virtual_node = max_real_node_id + 2 * i + 1
        vout_virtual_node = max_real_node_id + 2 * i + 2
        virtual_in_node_by_junction[junction_id] = vin_virtual_node
        virtual_out_node_by_junction[junction_id] = vout_virtual_node
        # virtual_in : external -> junction (entry to real network)
        # virtual_out: junction -> external (exit from real network)
        virtual_init_nodes.append(vin_virtual_node)
        virtual_term_nodes.append(junction_id)
        virtual_type_names.append("virtual_in")
        virtual_init_nodes.append(junction_id)
        virtual_term_nodes.append(vout_virtual_node)
        virtual_type_names.append("virtual_out")

    n_virtual_links_added = len(virtual_init_nodes)
    if n_virtual_links_added > 0:
        default_len = (
            float(np.median(lengths[lengths > 0])) if np.any(lengths > 0) else 1.0
        )
        default_fftt = (
            float(np.median(free_flow_times[free_flow_times > 0]))
            if np.any(free_flow_times > 0)
            else 1.0
        )
        init_nodes = np.concatenate(
            [init_nodes, np.asarray(virtual_init_nodes, dtype=int)]
        )
        term_nodes = np.concatenate(
            [term_nodes, np.asarray(virtual_term_nodes, dtype=int)]
        )
        lengths = np.concatenate(
            [lengths, np.full((n_virtual_links_added,), default_len, dtype=float)]
        )
        free_flow_times = np.concatenate(
            [
                free_flow_times,
                np.full((n_virtual_links_added,), default_fftt, dtype=float),
            ]
        )
        speeds = np.concatenate(
            [speeds, np.zeros((n_virtual_links_added,), dtype=float)]
        )

        if link_types is None:
            link_types = np.asarray(
                ["1"] * real_n_links + virtual_type_names, dtype=str
            )
        else:
            if isinstance(link_types, list):
                link_types = np.asarray(link_types, dtype=str)
            link_types = np.concatenate(
                [link_types.astype(str), np.asarray(virtual_type_names, dtype=str)]
            )

    orig_n_links = init_nodes.size
    base_adj = (
        term_nodes.reshape(-1, 1) == init_nodes.reshape(1, -1)
    ).astype(float)

    virtual_in_indices = list(range(real_n_links, orig_n_links, 2))
    virtual_out_indices = list(range(real_n_links + 1, orig_n_links, 2))
    print(
        f"Before random drop: real_links={real_n_links}, "
        f"added_virtual={n_virtual_links_added}, "
        f"virtual_in={len(virtual_in_indices)}, "
        f"virtual_out={len(virtual_out_indices)}"
    )

    # 4) Random drop.  
    virtual_in_indices, virtual_out_indices, dropped_links = (
        random_drop_virtual_links(
            init_nodes,
            term_nodes,
            orig_n_links,
            virtual_in_indices,
            virtual_out_indices,
            seed=_VIRTUAL_DROP_SEED,
            forbid_uturn=forbid_uturn,
            vin_key="term",
            vout_key="init",
        )
    )

    # 5) Filter + remap.
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

    # 6) Stretch virtual link lengths.
    l_np, l, virt_len_val = stretch_virtual_link_lengths(
        filt["l_np"],
        filt["virtual_in_indices"],
        filt["virtual_out_indices"],
        multiplier=_VIRT_LEN_MULTIPLIER,
    )

    # 7) Stats.
    report_link_statistics(
        l_np,
        np.asarray(filt["u"]),
        filt["virtual_in_indices"],
        filt["virtual_out_indices"],
        filt["n_links"],
    )

    # 8) Coordinates
    junction_xy_raw = parse_tntp_node_file(node_path)
    extra_junctions = {
        jid: (
            virtual_in_node_by_junction.get(jid),
            virtual_out_node_by_junction.get(jid),
        )
        for jid in all_junctions
    }
    node_xy, junction_xy_norm = build_link_centroids_and_normalize(
        junction_xy_raw,
        filt["init_nodes"],
        filt["term_nodes"],
        filt["old_to_new"],
        convert_latlon=True,
        all_junctions=all_junctions,
        extra_junctions=extra_junctions,
    )

    # 9) Validate.
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
        f"Built filtered SiouxFalls network with {orig_n_links} links "
        f"(including appended virtual links), kept {filt['n_links']} links, "
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
        "virt_len_val": virt_len_val,
    }
