from __future__ import annotations

import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import jax.numpy as jnp

from ..sim_core import MILE_TO_M


# ---------------------------------------------------------------------------
# TNTP parsing
# ---------------------------------------------------------------------------
def parse_tntp_net_file(net_path: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]
]:
    try:
        df = pd.read_csv(net_path, skiprows=8, sep="\t", engine="python")
        df.columns = [s.strip().lower() for s in df.columns]
        if not all(c in df.columns for c in ("init_node", "term_node")):
            raise ValueError(
                "pandas parse did not yield required init_node/term_node columns"
            )

        init_nodes = df["init_node"].astype(int).to_numpy()
        term_nodes = df["term_node"].astype(int).to_numpy()

        lengths = (
            df["length"].astype(float).to_numpy() * MILE_TO_M
            if "length" in df.columns
            else None
        )
        free_flow_times = (
            df["free_flow_time"].astype(float).to_numpy()
            if "free_flow_time" in df.columns
            else None
        )
        speeds = (
            df["speed"].astype(float).to_numpy() if "speed" in df.columns else None
        )

        link_types: Optional[np.ndarray] = None
        if "link_type" in df.columns:
            try:
                link_types = df["link_type"].astype(str).to_numpy()
            except Exception:
                link_types = np.asarray(df["link_type"].astype(str).to_list())
    except Exception as exc:
        print("Failed to parse network file:", str(exc))
        sys.exit(1)

    return (
        np.asarray(init_nodes, dtype=int),
        np.asarray(term_nodes, dtype=int),
        lengths,
        free_flow_times,
        speeds,
        link_types,
    )


def parse_tntp_node_file(node_path: str) -> Dict[int, Tuple[float, float]]:
    """Parse a TNTP ``*_node.tntp`` coordinate file.

    Returns a ``{node_id: (x, y)}`` mapping in the file's native units.
    Lines with HTML-style tags or comments are stripped first.
    """
    junction_xy: Dict[int, Tuple[float, float]] = {}
    if not os.path.exists(node_path):
        return junction_xy

    tag_re = re.compile(r"<[^>]+>")
    with open(node_path, "r") as fh:
        for line in fh:
            s = tag_re.sub("", line).strip()
            if s == "" or s.lower().startswith("node"):
                continue
            tok = s.replace("\t", " ").split()
            if len(tok) < 3:
                continue
            try:
                nid = int(tok[0])
                x = float(tok[1])
                y = float(tok[2].rstrip(";"))
                junction_xy[nid] = (x, y)
            except Exception:
                continue
    return junction_xy


# ---------------------------------------------------------------------------
# Random drop of overlapping virtual boundary links
# ---------------------------------------------------------------------------
def random_drop_virtual_links(
    init_nodes: np.ndarray,
    term_nodes: np.ndarray,
    orig_n_links: int,
    virtual_in_indices: List[int],
    virtual_out_indices: List[int],
    *,
    seed: int = 2021,
    forbid_uturn: bool = True,
    vin_key: str = "init",  # which side identifies the junction
    vout_key: str = "term",
) -> Tuple[List[int], List[int], set]:

    dropped_links: set = set()
    if not virtual_in_indices or not virtual_out_indices:
        return list(virtual_in_indices), list(virtual_out_indices), dropped_links

    def _side(side_key: str, idx: int) -> int:
        return int(init_nodes[idx]) if side_key == "init" else int(term_nodes[idx])

    vin_by_node: Dict[int, List[int]] = {}
    for idx in virtual_in_indices:
        vin_by_node.setdefault(_side(vin_key, idx), []).append(idx)

    vout_by_node: Dict[int, List[int]] = {}
    for idx in virtual_out_indices:
        vout_by_node.setdefault(_side(vout_key, idx), []).append(idx)

    overlap_nodes = sorted(set(vin_by_node) & set(vout_by_node))
    if not overlap_nodes:
        return list(virtual_in_indices), list(virtual_out_indices), dropped_links

    virtual_link_set = set(virtual_in_indices) | set(virtual_out_indices)
    start_links_by_node: Dict[int, List[int]] = {}
    for i in range(orig_n_links):
        start_links_by_node.setdefault(int(init_nodes[i]), []).append(i)

    check_link_indices = [
        i for i in range(orig_n_links) if i not in virtual_link_set
    ] or list(range(orig_n_links))

    def has_non_uturn_successor(link_idx: int, removed_links: set) -> bool:
        s = int(init_nodes[link_idx])
        t = int(term_nodes[link_idx])
        for nxt in start_links_by_node.get(t, []):
            if nxt in removed_links:
                continue
            if forbid_uturn and (
                s == int(term_nodes[nxt]) and t == int(init_nodes[nxt])
            ):
                continue
            return True
        return False

    baseline_has_successor = {
        i: has_non_uturn_successor(i, set()) for i in check_link_indices
    }

    unsafe_vout_nodes: set = set()
    for n in overlap_nodes:
        removed_links = set(vout_by_node[n])
        for i in check_link_indices:
            if i in removed_links:
                continue
            if baseline_has_successor[i] and not has_non_uturn_successor(
                i, removed_links
            ):
                unsafe_vout_nodes.add(n)
                break

    rng = np.random.default_rng(seed)
    drop_vin: set = set()
    drop_vout: set = set()
    skipped_nodes: List[int] = []

    print(f"\nJunctions with BOTH virtual_in and virtual_out: {len(overlap_nodes)}")
    print(f"Random seed for virtual link dropping: {seed} (fixed for reproducibility)")

    for n in overlap_nodes:
        if n in unsafe_vout_nodes:
            skipped_nodes.append(n)
            continue
        if rng.random() < 0.5:
            drop_vin.update(vin_by_node[n])
        else:
            drop_vout.update(vout_by_node[n])

    if skipped_nodes:
        print(
            f"\nSkipped {len(skipped_nodes)} nodes from random drop "
            f"(dropping virtual_out would create dead-end/U-turn-only states)"
        )

    dropped_links = drop_vin | drop_vout
    vin_after = sorted(set(virtual_in_indices) - drop_vin)
    vout_after = sorted(set(virtual_out_indices) - drop_vout)

    print(
        f"After random drop: virtual_in={len(vin_after)}, "
        f"virtual_out={len(vout_after)}"
    )
    print(f"Dropped {len(dropped_links)} virtual links")

    return vin_after, vout_after, dropped_links


# ---------------------------------------------------------------------------
# Filtering and link parameter computation
# ---------------------------------------------------------------------------
def filter_and_remap(
    init_nodes: np.ndarray,
    term_nodes: np.ndarray,
    base_adj: np.ndarray,
    lengths: np.ndarray,
    free_flow_times: np.ndarray,
    speeds: np.ndarray,
    link_types: Optional[np.ndarray],
    orig_n_links: int,
    dropped_links: set,
    virtual_in_indices: Sequence[int],
    virtual_out_indices: Sequence[int],
    *,
    forbid_uturn: bool = True,
) -> dict:

    all_indices = set(range(orig_n_links))
    kept_indices = sorted(all_indices - dropped_links)
    print(
        f"Filtering network: keeping {len(kept_indices)} links out of "
        f"{orig_n_links} (removed {len(dropped_links)} dropped virtual links)"
    )

    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    n_links = len(kept_indices)
    kept_indices_arr = np.asarray(kept_indices, dtype=np.int64)

    # Topology
    init_nodes_f = np.asarray(init_nodes)[kept_indices_arr]
    term_nodes_f = np.asarray(term_nodes)[kept_indices_arr]

    # Adjacency (filtered + U-turn suppression + no self-loops)
    adj_np = np.asarray(base_adj)[np.ix_(kept_indices, kept_indices)]
    if forbid_uturn:
        rev_mask = (
            init_nodes_f.reshape(-1, 1) == term_nodes_f.reshape(1, -1)
        ) & (term_nodes_f.reshape(-1, 1) == init_nodes_f.reshape(1, -1))
        adj_np[:n_links, :n_links][rev_mask] = 0.0
    np.fill_diagonal(adj_np, 0.0)
    adj_matrix = jnp.asarray(adj_np)

    # Parameters
    l_np = np.asarray(lengths, dtype=float)[kept_indices_arr]
    speeds_np = np.asarray(speeds, dtype=float)[kept_indices_arr]
    fftt_np = np.asarray(free_flow_times, dtype=float)[kept_indices_arr]

    u_np = _compute_link_speeds(l_np, speeds_np, fftt_np)
    l = jnp.asarray(l_np)
    u = jnp.asarray(u_np)

    link_types_f: Optional[np.ndarray] = None
    if link_types is not None:
        if isinstance(link_types, list):
            link_types_arr = np.asarray(link_types, dtype=str)
        else:
            link_types_arr = link_types
        link_types_f = link_types_arr[kept_indices_arr]

    # Remap virtual indices
    vin_remap = sorted(old_to_new[vi] for vi in virtual_in_indices if vi in old_to_new)
    vout_remap = sorted(
        old_to_new[vo] for vo in virtual_out_indices if vo in old_to_new
    )

    return {
        "init_nodes": init_nodes_f,
        "term_nodes": term_nodes_f,
        "adj_matrix": adj_matrix,
        "l": l,
        "l_np": l_np,
        "u": u,
        "link_types": link_types_f,
        "n_links": n_links,
        "kept_indices": kept_indices,
        "old_to_new": old_to_new,
        "virtual_in_indices": vin_remap,
        "virtual_out_indices": vout_remap,
    }


def _compute_link_speeds(
    l_np: np.ndarray, speeds_np: np.ndarray, fftt_np: np.ndarray
) -> np.ndarray:

    u_from_speed = speeds_np * MILE_TO_M / 3600.0
    u_np = np.copy(u_from_speed)

    zero_mask = u_np <= 1e-6
    fftt_seconds = fftt_np * 60.0
    alt_speed = np.zeros_like(u_np)
    valid_fftt = fftt_seconds > 1e-6
    alt_speed[valid_fftt] = l_np[valid_fftt] / fftt_seconds[valid_fftt]
    u_np[zero_mask] = alt_speed[zero_mask]

    positive_mask = u_np > 1e-6
    median_speed = float(np.median(u_np[positive_mask])) if np.any(positive_mask) else 1.0
    u_np[~positive_mask] = median_speed
    return u_np


# ---------------------------------------------------------------------------
# Virtual-link length adjustment + statistics
# ---------------------------------------------------------------------------
def stretch_virtual_link_lengths(
    l_np: np.ndarray,
    virtual_in_indices: Sequence[int],
    virtual_out_indices: Sequence[int],
    multiplier: float,
) -> Tuple[np.ndarray, jnp.ndarray, float]:

    max_orig_len = float(np.max(l_np)) if l_np.size > 0 else 100.0
    virt_len_val = max_orig_len * multiplier
    if virt_len_val > 1e8:
        print(
            f"yy Warning: original max length {max_orig_len:.2f} is large; "
            f"using virt_len_val={virt_len_val:.2e}"
        )
    for vi in list(virtual_in_indices) + list(virtual_out_indices):
        if vi is not None and 0 <= int(vi) < l_np.size:
            l_np[int(vi)] = virt_len_val
    return l_np, jnp.asarray(l_np), virt_len_val


def report_link_statistics(
    l_np: np.ndarray,
    u_np: np.ndarray,
    virtual_in_indices: Sequence[int],
    virtual_out_indices: Sequence[int],
    n_links: int,
) -> None:

    virtual_link_set = set(virtual_in_indices) | set(virtual_out_indices)
    real_link_indices = np.asarray(
        [i for i in range(n_links) if i not in virtual_link_set]
    )
    if real_link_indices.size == 0:
        return
    l_real = l_np[real_link_indices]
    u_real = u_np[real_link_indices]

    print("\n=== Link Parameter Statistics (excluding virtual links) ===")
    print(
        f"Total links: {n_links}, Real links: {len(real_link_indices)}, "
        f"Virtual links: {len(virtual_link_set)}"
    )
    print("\nLink Length (meters):")
    print(f"  min={np.min(l_real):.2f}, max={np.max(l_real):.2f}")
    print(f"  mean={np.mean(l_real):.2f}, std={np.std(l_real):.2f}")
    print("\nLink Speed (m/s):")
    print(f"  min={np.min(u_real):.4f}, max={np.max(u_real):.4f}")
    print(f"  mean={np.mean(u_real):.4f}, std={np.std(u_real):.4f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------
def build_link_centroids_and_normalize(
    junction_xy_raw: Dict[int, Tuple[float, float]],
    init_nodes: np.ndarray,
    term_nodes: np.ndarray,
    old_to_new: Dict[int, int],
    *,
    convert_latlon: bool = False,
    all_junctions: Optional[Iterable[int]] = None,
    extra_junctions: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple[Optional[Dict[int, Tuple[float, float]]], Optional[Dict[int, Tuple[float, float]]]]:

    if not junction_xy_raw:
        return None, None

    junction_xy = dict(junction_xy_raw)

    if convert_latlon:
        xs_j = [c[0] for c in junction_xy.values()]
        ys_j = [c[1] for c in junction_xy.values()]
        looks_like_latlon = (
            np.max(np.abs(xs_j)) <= 180.0 and np.max(np.abs(ys_j)) <= 90.0
        )
        if looks_like_latlon:
            R = 6371000.0  # Earth radius in metres
            lat0 = float(np.median(ys_j))
            lon0 = float(np.median(xs_j))
            cos_lat0 = float(np.cos(np.deg2rad(lat0)))
            for jid, (lon, lat) in list(junction_xy.items()):
                x_m = (float(lon) - lon0) * cos_lat0 * R * (np.pi / 180.0)
                y_m = (float(lat) - lat0) * R * (np.pi / 180.0)
                junction_xy[jid] = (x_m, y_m)

    # Inject synthetic virtual nodes if requested (Sioux Falls).
    if extra_junctions and all_junctions:
        xs_j = [c[0] for c in junction_xy.values()]
        ys_j = [c[1] for c in junction_xy.values()]
        span_x = float(np.max(xs_j) - np.min(xs_j)) if len(xs_j) > 1 else 1.0
        span_y = float(np.max(ys_j) - np.min(ys_j)) if len(ys_j) > 1 else 1.0
        offset_x = max(span_x * 0.03, 1e-4)
        offset_y = max(span_y * 0.03, 1e-4)
        for jid, vin_vout in extra_junctions.items():
            if jid not in junction_xy:
                continue
            x, y = junction_xy[jid]
            vin, vout = vin_vout
            if vin is not None:
                junction_xy[vin] = (x - offset_x, y - offset_y)
            if vout is not None:
                junction_xy[vout] = (x - 1.8 * offset_x, y - 1.8 * offset_y)

    # Compute per-link centroid in filtered numbering.
    link_xy: Dict[int, Tuple[float, float]] = {}
    for old_i, new_i in old_to_new.items():
        s = int(init_nodes[new_i])
        e = int(term_nodes[new_i])
        if s in junction_xy and e in junction_xy:
            xs_s, ys_s = junction_xy[s]
            xs_e, ys_e = junction_xy[e]
            link_xy[new_i] = ((xs_s + xs_e) / 2.0, (ys_s + ys_e) / 2.0)

    if not link_xy:
        return None, None

    xs = [c[0] for c in link_xy.values()]
    ys = [c[1] for c in link_xy.values()]
    median_abs = (np.median(np.abs(xs)) + np.median(np.abs(ys))) / 2.0
    # If coordinates are in feet (large absolute values), convert to metres.
    conv = 0.3048 if median_abs > 1e5 else 1.0

    for k, (x, y) in list(link_xy.items()):
        link_xy[k] = (x * conv, y * conv)

    xs = [c[0] for c in link_xy.values()]
    ys = [c[1] for c in link_xy.values()]
    min_x = float(np.min(xs)) if xs else 0.0
    min_y = float(np.min(ys)) if ys else 0.0
    if not (min_x == 0.0 and min_y == 0.0):
        for k, (x, y) in list(link_xy.items()):
            link_xy[k] = (float(x - min_x), float(y - min_y))

    junction_xy_norm: Dict[int, Tuple[float, float]] = {}
    for jid, (jx, jy) in junction_xy.items():
        jx_c = float(jx * conv)
        jy_c = float(jy * conv)
        junction_xy_norm[jid] = (float(jx_c - min_x), float(jy_c - min_y))

    return link_xy, junction_xy_norm


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_filtered_network(
    init_nodes: np.ndarray,
    term_nodes: np.ndarray,
    kept_indices: Sequence[int],
    old_n_links: int,
    n_links: int,
    adj_matrix,
    l,
    u,
    virtual_in_indices: Sequence[int],
    virtual_out_indices: Sequence[int],
) -> None:

    issues: List[str] = []
    warnings: List[str] = []

    if init_nodes is not None and len(init_nodes) != n_links:
        issues.append(f"init_nodes size ({len(init_nodes)}) != n_links ({n_links})")
    if term_nodes is not None and len(term_nodes) != n_links:
        issues.append(f"term_nodes size ({len(term_nodes)}) != n_links ({n_links})")

    l_arr = np.asarray(l)
    if l_arr.shape[0] != n_links:
        issues.append(f"l array size ({l_arr.shape[0]}) != n_links ({n_links})")

    u_arr = np.asarray(u)
    if u_arr.shape[0] != n_links:
        issues.append(f"u array size ({u_arr.shape[0]}) != n_links ({n_links})")

    adj_arr = np.asarray(adj_matrix)
    if adj_arr.shape != (n_links, n_links):
        issues.append(
            f"adj_matrix shape {adj_arr.shape} != ({n_links}, {n_links})"
        )

    if virtual_in_indices:
        if max(virtual_in_indices) >= n_links:
            issues.append(
                f"max virtual_in_indices ({max(virtual_in_indices)}) >= n_links ({n_links})"
            )
        if min(virtual_in_indices) < 0:
            issues.append(
                f"min virtual_in_indices ({min(virtual_in_indices)}) < 0"
            )
    else:
        warnings.append(
            "No virtual inflow links retained - network may be disconnected from inflow"
        )

    if virtual_out_indices:
        if max(virtual_out_indices) >= n_links:
            issues.append(
                f"max virtual_out_indices ({max(virtual_out_indices)}) >= n_links ({n_links})"
            )
        if min(virtual_out_indices) < 0:
            issues.append(
                f"min virtual_out_indices ({min(virtual_out_indices)}) < 0"
            )

    if n_links >= old_n_links:
        warnings.append(
            f"Network size did not reduce: {n_links} >= {old_n_links} "
            "(expected reduction)"
        )

    print("\n===== Network Validation Report =====")
    if issues:
        print(f"ERRORS ({len(issues)}):")
        for issue in issues:
            print(f"  X {issue}")
        raise ValueError(f"Network validation failed with {len(issues)} errors")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")
    if not issues and not warnings:
        print("All validation checks passed")
    print("=" * 40)
