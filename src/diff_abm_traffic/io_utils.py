"""Utilities for saving calibration outputs to disk."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .simulation import SimContext


def save_param_results(
    output_dir: str,
    suffix_method: str,
    n_agent: int,
    n_links: int,
    *,
    beta_true,
    beta_b,
    beta_est,
    u_true,
    u_b,
    u_est,
    kappa_true,
    kappa_b,
    kappa_est,
    mp_true,
    mp_b,
    mp_est,
) -> Dict[str, str]:
    """Save per-link CSVs of true / background / estimated parameters."""
    paths: Dict[str, str] = {}

    paths["beta"] = os.path.join(
        output_dir, f"assim_joint_beta_results{suffix_method}_n{n_agent}.csv"
    )
    pd.DataFrame(
        {
            "link_idx": np.arange(n_links),
            "beta_true": np.asarray(beta_true),
            "beta_b": np.asarray(beta_b),
            "beta_est": np.asarray(beta_est),
        }
    ).to_csv(paths["beta"], index=False)
    print(f"Beta results saved to: {paths['beta']}")

    paths["u"] = os.path.join(
        output_dir, f"assim_joint_u_results{suffix_method}_n{n_agent}.csv"
    )
    pd.DataFrame(
        {
            "link_idx": np.arange(n_links),
            "u_true": np.asarray(u_true),
            "u_b": np.asarray(u_b),
            "u_est": np.asarray(u_est),
        }
    ).to_csv(paths["u"], index=False)
    print(f"U results saved to: {paths['u']}")

    paths["kappa"] = os.path.join(
        output_dir, f"assim_joint_kappa_results{suffix_method}_n{n_agent}.csv"
    )
    pd.DataFrame(
        {
            "link_idx": np.arange(n_links),
            "kappa_true": np.asarray(kappa_true),
            "kappa_b": np.asarray(kappa_b),
            "kappa_est": np.asarray(kappa_est),
        }
    ).to_csv(paths["kappa"], index=False)
    print(f"Kappa results saved to: {paths['kappa']}")

    paths["mp"] = os.path.join(
        output_dir,
        f"assim_joint_merge_priority_results{suffix_method}_n{n_agent}.csv",
    )
    pd.DataFrame(
        {
            "link_idx": np.arange(n_links),
            "merge_priority_true": np.asarray(mp_true),
            "merge_priority_b": np.asarray(mp_b),
            "merge_priority_est": np.asarray(mp_est),
        }
    ).to_csv(paths["mp"], index=False)
    print(f"Merge priority results saved to: {paths['mp']}")

    return paths


def load_param_results(
    output_dir: str, suffix_method: str, n_agent: int, n_links: int
):
    """Reload previously saved parameter CSVs.  Returns ``None`` if any
    file is missing or has the wrong number of rows."""
    files = {
        name: os.path.join(
            output_dir, f"assim_joint_{name}_results{suffix_method}_n{n_agent}.csv"
        )
        for name in ("beta", "u", "kappa", "merge_priority")
    }
    if not all(os.path.exists(p) for p in files.values()):
        return None
    try:
        dfs = {name: pd.read_csv(p) for name, p in files.items()}
        if not all(len(df) == n_links for df in dfs.values()):
            return None
        import jax.numpy as jnp

        return (
            jnp.asarray(dfs["beta"]["beta_est"].values),
            jnp.asarray(dfs["u"]["u_est"].values),
            jnp.asarray(dfs["kappa"]["kappa_est"].values),
            jnp.asarray(dfs["merge_priority"]["merge_priority_est"].values),
        )
    except Exception as exc:
        print("Existing parameter files invalid:", exc)
        return None


def save_forward_opt_snapshot(
    output_dir: str,
    suffix_method: str,
    n_agent: int,
    n_links: int,
    beta_fwd_opt,
    u_fwd_opt,
    kappa_fwd_opt,
    mp_fwd_opt,
) -> str:
    """Save the post-forward-optimisation parameter snapshot."""
    path = os.path.join(
        output_dir, f"assim_joint_forward_opt_params{suffix_method}_n{n_agent}.csv"
    )
    pd.DataFrame(
        {
            "link_idx": np.arange(n_links),
            "beta_forward_opt": np.asarray(beta_fwd_opt),
            "u_forward_opt": np.asarray(u_fwd_opt),
            "kappa_forward_opt": np.asarray(kappa_fwd_opt),
            "merge_priority_forward_opt": np.asarray(mp_fwd_opt),
        }
    ).to_csv(path, index=False)
    print(f"Forward optimization parameter snapshot saved to: {path}")
    return path


def save_cumulative_series(
    output_dir: str,
    suffix_method: str,
    n_agent: int,
    n_links: int,
    obs_step: int,
    cums_true,
    cums_bg,
    cums_est,
) -> Dict[str, str]:
    """Save the calibration-time cumulative-count series (per scenario)."""
    obs_times = np.arange(np.asarray(cums_true).shape[0]) * obs_step
    cols = [f"link_{i}" for i in range(n_links)]

    paths: Dict[str, str] = {}
    for name, arr in (("true", cums_true), ("background", cums_bg), ("est", cums_est)):
        df = pd.DataFrame(np.asarray(arr), columns=cols)
        df.insert(0, "time_step", obs_times)
        path = os.path.join(
            output_dir,
            f"assim_joint_cum_series_{name}{suffix_method}_n{n_agent}.csv",
        )
        df.to_csv(path, index=False)
        paths[name] = path
    print("Saved cumulative series CSVs:", *paths.values())
    return paths


def save_forward_cumulative_series(
    output_dir: str,
    suffix_method: str,
    n_agent: int,
    n_links: int,
    obs_step: int,
    n_steps: int,
    cums_fwd_true,
    cums_fwd_est,
    cums_fwd_opt=None,
) -> Dict[str, str]:
    """Save the forward-prediction cumulative-count series."""
    forward_times = (
        np.arange(np.asarray(cums_fwd_true).shape[0]) * obs_step + n_steps
    )
    cols = [f"link_{i}" for i in range(n_links)]
    paths: Dict[str, str] = {}

    df_true = pd.DataFrame(np.asarray(cums_fwd_true), columns=cols)
    df_true.insert(0, "time_step", forward_times)
    paths["forward_true"] = os.path.join(
        output_dir,
        f"assim_joint_cum_series_forward_true{suffix_method}_n{n_agent}.csv",
    )
    df_true.to_csv(paths["forward_true"], index=False)

    df_est = pd.DataFrame(np.asarray(cums_fwd_est), columns=cols)
    df_est.insert(0, "time_step", forward_times)
    paths["forward_est"] = os.path.join(
        output_dir,
        f"assim_joint_cum_series_forward_est{suffix_method}_n{n_agent}.csv",
    )
    df_est.to_csv(paths["forward_est"], index=False)
    print(f"Saved forward cumulative series (true/est): {paths['forward_true']}, "
          f"{paths['forward_est']}")

    if cums_fwd_opt is not None:
        df_opt = pd.DataFrame(np.asarray(cums_fwd_opt), columns=cols)
        df_opt.insert(0, "time_step", forward_times)
        paths["forward_opt"] = os.path.join(
            output_dir,
            f"assim_joint_cum_series_forward_opt{suffix_method}_n{n_agent}.csv",
        )
        df_opt.to_csv(paths["forward_opt"], index=False)
        print(f"Saved forward cumulative series (optimized): {paths['forward_opt']}")
    return paths


def save_trajectories(
    traj_dir: str,
    suffix_method: str,
    n_agent: int,
    trajectories: Dict[str, Optional[dict]],
) -> None:
    """Save agent trajectories as compressed ``.npz`` files."""
    os.makedirs(traj_dir, exist_ok=True)
    for name, traj_data in trajectories.items():
        if traj_data is None:
            continue
        filepath = os.path.join(
            traj_dir, f"agent_trajectories_{name}{suffix_method}_n{n_agent}.npz"
        )
        np.savez_compressed(
            filepath,
            link_indices=traj_data["link_indices"],
            positions=traj_data["positions"],
            n_timesteps=traj_data["n_timesteps"],
            n_agents=traj_data["n_agents"],
            n_links=traj_data["n_links"],
        )
        print(f"  Saved {name} agent trajectories: {filepath}")


def save_link_counts(
    traj_dir: str,
    suffix_method: str,
    n_agent: int,
    counts: Dict[str, Optional[np.ndarray]],
) -> None:
    """Save raw per-link cumulative counts as ``.npy`` files."""
    os.makedirs(traj_dir, exist_ok=True)
    for name, arr in counts.items():
        if arr is None:
            continue
        filepath = os.path.join(
            traj_dir, f"agent_counts_{name}{suffix_method}_n{n_agent}.npy"
        )
        np.save(filepath, np.asarray(arr))
        print(f"  Saved {name} link counts: {filepath}")


def save_virtual_inflow_occupancy(
    traj_dir: str,
    suffix_method: str,
    n_agent: int,
    cum_save_step: int,
    occupancies: Dict[str, tuple],
) -> None:
    """Save the per-timestep virtual-inflow occupancy series."""
    os.makedirs(traj_dir, exist_ok=True)
    for name, (occ_data, time_offset) in occupancies.items():
        if occ_data is None:
            continue
        occ_np = np.asarray(occ_data)
        npy_path = os.path.join(
            traj_dir,
            f"virtual_inflow_occupancy_{name}{suffix_method}_n{n_agent}.npy",
        )
        np.save(npy_path, occ_np)
        times = np.arange(occ_np.shape[0]) * cum_save_step + time_offset
        csv_path = os.path.join(
            traj_dir,
            f"virtual_inflow_occupancy_{name}{suffix_method}_n{n_agent}.csv",
        )
        pd.DataFrame(
            {"time_step": times, "virtual_inflow_occupancy": occ_np}
        ).to_csv(csv_path, index=False)
        print(f"  Saved {name} virtual inflow occupancy: {npy_path}, {csv_path}")


def save_replay_metadata(
    output_dir: str,
    suffix_method: str,
    n_agent: int,
    seed: int,
    backend: str,
    n_steps: int,
    n_steps_forward: int,
    obs_step: int,
    cum_save_step: int,
    keys: Dict[str, Optional[Sequence[int]]],
) -> str:
    """Persist the JAX PRNG keys used so the run can be replayed."""
    def _key_to_list(k):
        if k is None:
            return None
        return [int(v) for v in np.asarray(k).tolist()]

    meta = {
        "seed": int(seed),
        "backend": str(backend),
        "n_steps": int(n_steps),
        "n_steps_forward": int(n_steps_forward),
        "obs_step": int(obs_step),
        "cum_save_step": int(cum_save_step),
        "keys": {name: _key_to_list(k) for name, k in keys.items()},
    }
    path = os.path.join(
        output_dir, f"replay_metadata{suffix_method}_n{n_agent}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved replay metadata: {path}")
    return path


# ---------------------------------------------------------------------------
# Network / agent-state exports for external visualisation tools
# ---------------------------------------------------------------------------
def export_network_and_agent_states(
    ctx: SimContext,
    junction_xy: Optional[Dict[int, tuple]],
    virtual_link_mask: np.ndarray,
    export_dir: str,
    suffix_method: str,
    n_agent: int,
    trajectories: Dict[str, Optional[dict]],
    cums_true=None,
    cums_bg=None,
    cums_est=None,
    cums_fwd_true=None,
    cums_fwd_est=None,
    cums_fwd_opt=None,
    beta_true=None,
    beta_est=None,
    beta_fwd_opt=None,
) -> None:
    """Write a compact set of files describing the network and agent states."""

    os.makedirs(export_dir, exist_ok=True)
    n_links = ctx.n_links
    init_nodes = ctx.init_nodes
    term_nodes = ctx.term_nodes
    l = np.asarray(ctx.l_np)

    def _node_coord(nid):
        nid_i = int(nid)
        if isinstance(junction_xy, dict) and nid_i in junction_xy:
            x, y = junction_xy[nid_i]
            return float(x), float(y)
        return np.nan, np.nan

    vin_nodes = set(
        int(init_nodes[idx])
        for idx in ctx.virtual_in_indices
        if idx is not None and int(idx) < len(init_nodes)
    )
    vout_nodes = set(
        int(term_nodes[idx])
        for idx in ctx.virtual_out_indices
        if idx is not None and int(idx) < len(term_nodes)
    )

    # 1) Network links table.
    link_rows = []
    for li in range(n_links):
        s = int(init_nodes[li]) if li < len(init_nodes) else -1
        t = int(term_nodes[li]) if li < len(term_nodes) else -1
        sx, sy = _node_coord(s)
        tx, ty = _node_coord(t)
        is_virtual = bool(virtual_link_mask[li]) if li < len(virtual_link_mask) else False
        link_rows.append(
            {
                "link_id": li,
                "start_node_id": s,
                "end_node_id": t,
                "start_x": sx,
                "start_y": sy,
                "end_x": tx,
                "end_y": ty,
                "link_attr": "virtual" if is_virtual else "real",
            }
        )
    links_csv = os.path.join(
        export_dir, f"network_links{suffix_method}_n{n_agent}.csv"
    )
    pd.DataFrame(link_rows).to_csv(links_csv, index=False)
    links_npz = os.path.join(
        export_dir, f"network_links{suffix_method}_n{n_agent}.npz"
    )
    np.savez_compressed(
        links_npz,
        link_id=np.asarray([r["link_id"] for r in link_rows], dtype=np.int32),
        start_node_id=np.asarray(
            [r["start_node_id"] for r in link_rows], dtype=np.int32
        ),
        end_node_id=np.asarray(
            [r["end_node_id"] for r in link_rows], dtype=np.int32
        ),
        start_x=np.asarray([r["start_x"] for r in link_rows], dtype=np.float32),
        start_y=np.asarray([r["start_y"] for r in link_rows], dtype=np.float32),
        end_x=np.asarray([r["end_x"] for r in link_rows], dtype=np.float32),
        end_y=np.asarray([r["end_y"] for r in link_rows], dtype=np.float32),
        link_attr=np.asarray([r["link_attr"] for r in link_rows], dtype="U8"),
    )

    # 2) Network nodes table.
    node_ids = sorted(
        set(
            int(v)
            for v in np.concatenate(
                [np.asarray(init_nodes).ravel(), np.asarray(term_nodes).ravel()]
            )
        )
    )
    node_rows = []
    for nid in node_ids:
        x, y = _node_coord(nid)
        in_flag = nid in vin_nodes
        out_flag = nid in vout_nodes
        if in_flag and out_flag:
            attr = "inflow_outflow"
        elif in_flag:
            attr = "inflow"
        elif out_flag:
            attr = "outflow"
        else:
            attr = "normal"
        node_rows.append({"node_id": nid, "x": x, "y": y, "node_attr": attr})
    nodes_csv = os.path.join(
        export_dir, f"network_nodes{suffix_method}_n{n_agent}.csv"
    )
    pd.DataFrame(node_rows).to_csv(nodes_csv, index=False)
    nodes_npz = os.path.join(
        export_dir, f"network_nodes{suffix_method}_n{n_agent}.npz"
    )
    np.savez_compressed(
        nodes_npz,
        node_id=np.asarray([r["node_id"] for r in node_rows], dtype=np.int32),
        x=np.asarray([r["x"] for r in node_rows], dtype=np.float32),
        y=np.asarray([r["y"] for r in node_rows], dtype=np.float32),
        node_attr=np.asarray([r["node_attr"] for r in node_rows], dtype="U16"),
    )
    print(
        f"  Saved network data: {links_csv}, {nodes_csv}, {links_npz}, {nodes_npz}"
    )

    def _link_pos_to_xy(link_idx_arr, pos_arr):
        n_t, n_a = link_idx_arr.shape
        xs = np.full((n_t, n_a), np.nan, dtype=np.float32)
        ys = np.full((n_t, n_a), np.nan, dtype=np.float32)
        for li in range(n_links):
            s = int(init_nodes[li]) if li < len(init_nodes) else -1
            t = int(term_nodes[li]) if li < len(term_nodes) else -1
            sx, sy = _node_coord(s)
            tx, ty = _node_coord(t)
            if not (
                np.isfinite(sx) and np.isfinite(sy) and np.isfinite(tx) and np.isfinite(ty)
            ):
                continue
            mask = link_idx_arr == li
            if not np.any(mask):
                continue
            length = float(l[li]) if li < len(l) else 1.0
            length = max(length, 1e-6)
            ratio = np.clip(pos_arr[mask] / length, 0.0, 1.0)
            xs[mask] = sx + (tx - sx) * ratio
            ys[mask] = sy + (ty - sy) * ratio
        return xs, ys

    def _export_agent_states(name, traj_data):
        if traj_data is None:
            return
        link_idx = np.asarray(traj_data["link_indices"])
        pos = np.asarray(traj_data["positions"])
        xs, ys = _link_pos_to_xy(link_idx, pos)
        valid = link_idx >= 0
        timestep_idx, agent_ids = np.where(valid)
        state_npz = os.path.join(
            export_dir, f"agent_states_{name}{suffix_method}_n{n_agent}.npz"
        )
        np.savez_compressed(
            state_npz,
            timestep=timestep_idx.astype(np.int32),
            agent_id=agent_ids.astype(np.int32),
            x=xs[valid].astype(np.float32),
            y=ys[valid].astype(np.float32),
            n_timesteps=np.int32(link_idx.shape[0]),
            n_agents=np.int32(link_idx.shape[1]),
        )
        print(
            f"  Saved agent states ({name}) npz: {state_npz} "
            f"(records={timestep_idx.size})"
        )

    for name in ("true", "background", "est", "forward_true", "forward_est", "forward_opt"):
        _export_agent_states(name, trajectories.get(name))

    # 3) Per-link cumulative series across scenarios.
    def _save_link_series(arr_true, arr_bg, arr_est, arr_opt, base_name):
        if arr_true is None or arr_est is None:
            return
        n_t = min(
            arr_true.shape[0],
            arr_est.shape[0],
            arr_opt.shape[0] if arr_opt is not None else arr_true.shape[0],
        )
        arr_true = np.asarray(arr_true)[:n_t, :].astype(np.float32, copy=False)
        arr_bg = (
            np.asarray(arr_bg)[:n_t, :].astype(np.float32, copy=False)
            if arr_bg is not None
            else np.full((n_t, n_links), np.nan, dtype=np.float32)
        )
        arr_est = np.asarray(arr_est)[:n_t, :].astype(np.float32, copy=False)
        arr_opt_save = (
            np.asarray(arr_opt)[:n_t, :].astype(np.float32, copy=False)
            if arr_opt is not None
            else np.full((n_t, n_links), np.nan, dtype=np.float32)
        )
        path = os.path.join(
            export_dir,
            f"link_cumulative_series_{base_name}{suffix_method}_n{n_agent}.npz",
        )
        np.savez_compressed(
            path,
            link_id=np.arange(n_links, dtype=np.int32),
            true=arr_true,
            background=arr_bg,
            est=arr_est,
            opt=arr_opt_save,
            n_timesteps=np.int32(n_t),
        )
        print(
            f"  Saved link cumulative series npz: {path} "
            f"(timesteps={n_t}, base={base_name})"
        )

    if cums_true is not None and cums_est is not None:
        _save_link_series(cums_true, cums_bg, cums_est, None, "calibration")
    if cums_fwd_true is not None and cums_fwd_est is not None:
        _save_link_series(
            cums_fwd_true, cums_bg, cums_fwd_est, cums_fwd_opt, "forward"
        )

    # 4) Per-link beta parameters across variants.
    if beta_true is not None and beta_est is not None:
        beta_export = pd.DataFrame(
            {
                "link_id": np.arange(n_links),
                "beta_true": np.asarray(beta_true),
                "beta_est": np.asarray(beta_est),
                "beta_opt": np.asarray(
                    beta_fwd_opt if beta_fwd_opt is not None else beta_est
                ),
            }
        )
        beta_csv = os.path.join(
            export_dir, f"link_beta_params{suffix_method}_n{n_agent}.csv"
        )
        beta_export.to_csv(beta_csv, index=False)
        beta_npz = os.path.join(
            export_dir, f"link_beta_params{suffix_method}_n{n_agent}.npz"
        )
        np.savez_compressed(
            beta_npz,
            link_id=np.arange(n_links, dtype=np.int32),
            beta_true=np.asarray(beta_true, dtype=np.float32),
            beta_est=np.asarray(beta_est, dtype=np.float32),
            beta_opt=np.asarray(
                beta_fwd_opt if beta_fwd_opt is not None else beta_est,
                dtype=np.float32,
            ),
        )
        print(f"  Saved beta parameters: {beta_csv}, {beta_npz}")


def save_stage_timing_summary(
    output_dir: str,
    suffix_method: str,
    n_agent: int,
    t_calib: float,
    t_forward: Optional[float],
    t_opt: Optional[float],
    n_steps_forward: int,
    run_forward_opt: bool,
) -> str:
    """Save the per-stage wall-clock summary used in the paper's figures."""
    rows = [
        {"stage": "calibration", "elapsed_sec": float(t_calib), "executed": True},
        {
            "stage": "forward_prediction",
            "elapsed_sec": float(t_forward) if t_forward is not None else np.nan,
            "executed": bool(n_steps_forward > 0),
        },
        {
            "stage": "forward_optimization",
            "elapsed_sec": float(t_opt) if t_opt is not None else np.nan,
            "executed": bool(n_steps_forward > 0 and run_forward_opt),
        },
    ]
    path = os.path.join(
        output_dir, f"stage_timing_summary{suffix_method}_n{n_agent}.csv"
    )
    pd.DataFrame(rows).to_csv(path, index=False)

    print("\n" + "=" * 60)
    print("Stage timing summary")
    print("=" * 60)
    print(f"  [STEP 1] Calibration: {t_calib:.3f} sec")
    if t_forward is not None:
        print(f"  [STEP 2] Now casting:        {t_forward:.3f} sec")
    else:
        print("  [STEP 2] Now casting:        skipped")
    if t_opt is not None:
        print(f"  [STEP 3] Forward optimization: {t_opt:.3f} sec")
    else:
        print("  [STEP 3] Forward optimization: skipped")
    print(f"Saved stage timing summary: {path}")
    return path
