"""Forward simulation routines shared by both calibration scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import checkpoint, jit, random

from .sim_core import BatchLinkForward_fast_vmap_sort, transfer_fast


try:  
    import nvtx as _nvtx
except Exception:
    _nvtx = None


# ---------------------------------------------------------------------------
# Simulation context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimContext:

    n_links: int
    l: jnp.ndarray
    l_np: np.ndarray
    adj_matrix: jnp.ndarray
    init_nodes: np.ndarray
    term_nodes: np.ndarray
    delta_t: float
    delta_n: float
    eps_sig: float
    use_checkpoint: bool
    chunk_size: int
    virtual_in_indices: List[int] = field(default_factory=list)
    virtual_out_indices: List[int] = field(default_factory=list)
    agent_traj_stride: int = 1


# ---------------------------------------------------------------------------
# Forward simulation with cumulative counting
# ---------------------------------------------------------------------------
def forward_and_count_simple(
    ctx: SimContext,
    key: jnp.ndarray,
    x0: jnp.ndarray,
    n_steps: int,
    beta: jnp.ndarray,
    u: jnp.ndarray,
    kappa_param: jnp.ndarray,
    merge_priority_param: jnp.ndarray,
    obs_step: int,
    eps_override: Optional[float] = None,
    delta_n_override: Optional[float] = None,
    delta_t_override: Optional[float] = None,
    save_cums: bool = True,
    return_x_final: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    if not save_cums and not return_x_final:
        raise ValueError(
            "forward_and_count_simple: at least one of save_cums / return_x_final must be True"
        )

    n_links = ctx.n_links
    l = ctx.l
    adj_matrix = ctx.adj_matrix
    delta_t = ctx.delta_t
    delta_n = ctx.delta_n
    eps_sig = ctx.eps_sig
    use_checkpoint = ctx.use_checkpoint
    chunk_size_cfg = ctx.chunk_size

    link_length_matrix = jnp.ones((x0.shape[0], x0.shape[1]), dtype=float) * l

    default_eps = float(eps_sig)
    use_eps = default_eps if eps_override is None else float(eps_override)
    use_eps = jnp.clip(use_eps, 1e-12, 0.5)

    scale_per_link = 1.0 / (0.2 * l)
    center = l / 2.0

    def step_fn_with_cum(carry, t):
        """Single simulation step + cumulative count update."""
        rng, x_prev, cum = carry
        rng, subkey = random.split(rng)

        delta_n_use = delta_n if delta_n_override is None else float(delta_n_override)
        delta_t_use = delta_t if delta_t_override is None else float(delta_t_override)
        mp_reshaped = jnp.reshape(merge_priority_param, (n_links, 1))

        if _nvtx is not None:
            _nvtx.push_range("BatchLinkForward_forward_step")
        x_curr = BatchLinkForward_fast_vmap_sort(
            x_prev, u, kappa_param, l, delta_t=delta_t_use, delta_n=delta_n_use
        )
        if _nvtx is not None:
            _nvtx.pop_range()

        rng, x_curr = transfer_fast(
            subkey,
            x_curr,
            adj_matrix,
            beta,
            mp_reshaped,
            kappa_param,
            link_length_matrix,
            delta_t=delta_t_use,
            delta_n=delta_n_use,
        )

        ind_prev_hard = jnp.where(x_prev >= center, 1.0, 0.0)
        ind_curr_hard = jnp.where(x_curr >= center, 1.0, 0.0)
        ind_prev_soft = jax.nn.sigmoid(scale_per_link * (x_prev - center))
        ind_curr_soft = jax.nn.sigmoid(scale_per_link * (x_curr - center))
        ind_prev = ind_prev_soft + ind_prev_hard - jax.lax.stop_gradient(ind_prev_soft)
        ind_curr = ind_curr_soft + ind_curr_hard - jax.lax.stop_gradient(ind_curr_soft)

        inc = jnp.sum(jnp.maximum(0.0, ind_curr - ind_prev), axis=0)
        cum = cum + inc * delta_n_use 
        return (rng, x_curr, cum), None

    def step_fn_no_cum(carry, t):
        """Single simulation step without the counter (used when only the
        final state is needed)."""
        rng, x_prev = carry
        rng, subkey = random.split(rng)
        delta_n_use = delta_n if delta_n_override is None else float(delta_n_override)
        delta_t_use = delta_t if delta_t_override is None else float(delta_t_override)
        mp_reshaped = jnp.reshape(merge_priority_param, (n_links, 1))

        if _nvtx is not None:
            _nvtx.push_range("BatchLinkForward_forward_step")
        x_curr = BatchLinkForward_fast_vmap_sort(
            x_prev, u, kappa_param, l, delta_t=delta_t_use, delta_n=delta_n_use
        )
        if _nvtx is not None:
            _nvtx.pop_range()
        rng, x_curr = transfer_fast(
            subkey,
            x_curr,
            adj_matrix,
            beta,
            mp_reshaped,
            kappa_param,
            link_length_matrix,
            delta_t=delta_t_use,
            delta_n=delta_n_use,
        )
        return (rng, x_curr), None

    # ------------------------------------------------------------------
    # save_cums = True branch
    # ------------------------------------------------------------------
    if save_cums:
        step_fn = jit(checkpoint(step_fn_with_cum)) if use_checkpoint else jit(step_fn_with_cum)

        rng = key
        cum0 = jnp.zeros((n_links,), dtype=float)
        carry = (rng, x0, cum0)

        n_obs = n_steps // obs_step
        saved_cums = []
        chunk_size = min(obs_step, int(chunk_size_cfg))

        for _ in range(n_obs):
            remaining = obs_step
            while remaining > 0:
                current_chunk = min(remaining, chunk_size)
                carry, _ = jax.lax.scan(step_fn, carry, jnp.arange(current_chunk))
                remaining -= current_chunk
            saved_cums.append(carry[2])

        cum_f = carry[2]
        cums = (
            jnp.stack(saved_cums)
            if saved_cums
            else jnp.zeros((0, n_links), dtype=float)
        )
        if return_x_final:
            return cum_f, cums, carry[1]
        return cum_f, cums

    # ------------------------------------------------------------------
    # save_cums = False branch
    # ------------------------------------------------------------------
    step_fn = jit(checkpoint(step_fn_no_cum)) if use_checkpoint else jit(step_fn_no_cum)

    carry = (key, x0)
    remaining = int(n_steps)
    chunk_size = min(max(1, int(chunk_size_cfg)), max(1, remaining))
    while remaining > 0:
        current_chunk = min(remaining, chunk_size)
        carry, _ = jax.lax.scan(step_fn, carry, jnp.arange(current_chunk))
        remaining -= current_chunk
    return carry[1]


# ---------------------------------------------------------------------------
# Observation perturbation
# ---------------------------------------------------------------------------
def apply_obs_perturbation(
    obs_cum: np.ndarray, obs_perturb_pct: float, rng: np.random.Generator
) -> np.ndarray:

    obs_cum_np = np.asarray(obs_cum)
    n_obs_steps, n_links = obs_cum_np.shape

    if obs_perturb_pct <= 0.0 or n_obs_steps <= 1:
        return obs_cum_np

    increments = np.diff(obs_cum_np, axis=0, prepend=np.zeros((1, n_links)))
    noise = rng.normal(loc=0.0, scale=obs_perturb_pct, size=(n_obs_steps, n_links))
    perturbed_increments = np.maximum(increments * (1.0 + noise), 0.0)
    obs_cum_perturbed = np.round(np.cumsum(perturbed_increments, axis=0))

    print(f"Observation perturbation applied (obs_perturb={obs_perturb_pct*100:.1f}%)")
    return obs_cum_perturbed


# ---------------------------------------------------------------------------
# Initial agent placement
# ---------------------------------------------------------------------------
def make_initial_x(
    ctx: SimContext,
    n_agent: int,
    start_idx: Union[int, Sequence[int]],
    delta_n_override: Optional[float] = None,
    delta_t_override: Optional[float] = None,
    u_speed_array: Optional[Sequence[float]] = None,
) -> jnp.ndarray:

    l = ctx.l
    n_links = ctx.n_links
    delta_t = ctx.delta_t
    delta_n = ctx.delta_n

    delta_n_use = delta_n if delta_n_override is None else float(delta_n_override)
    delta_t_use = delta_t if delta_t_override is None else float(delta_t_override)
    n_agent_effective = int(n_agent / delta_n_use)

    if u_speed_array is None:
        raise ValueError(
            "u_speed_array must be provided to make_initial_x "
            "for proper spacing calculation."
        )

    if isinstance(start_idx, (list, tuple, np.ndarray)):
        start_list = [int(s) for s in start_idx]
    else:
        start_list = [int(start_idx)]
    if not start_list:
        start_list = [0]
    n_starts = len(start_list)

    base = n_agent_effective // n_starts
    rem = n_agent_effective % n_starts
    counts = [base + (1 if i < rem else 0) for i in range(n_starts)]

    def _get_scalar(obj, idx):
        if obj is None:
            return None
        a = np.asarray(obj)
        if getattr(a, "ndim", 0) == 0:
            return float(a.item())
        if idx < 0 or idx >= a.shape[0]:
            return None
        v = a[idx]
        v_arr = np.asarray(v)
        if v_arr.size == 1:
            return float(v_arr.item())
        return float(v_arr.ravel()[0])

    x0 = jnp.full((n_agent_effective, n_links), -9999.0)
    idx_offset = 0
    for count, s_idx in zip(counts, start_list):
        if count <= 0:
            continue
        Ls = float(l[int(s_idx)]) if 0 <= int(s_idx) < len(l) else float(jnp.max(l))
        u_speed = _get_scalar(u_speed_array, int(s_idx))
        base_spacing = (
            (u_speed * delta_t_use) if (u_speed is not None) else float(delta_n_use)
        )
        if not np.isfinite(base_spacing) or base_spacing <= 0:
            base_spacing = float(delta_n_use)
        spacing = min(base_spacing, max(Ls, 1.0))

        eps = 1e-3
        start_pos = max(0.0, Ls - eps)

        required_len = (count - 1) * spacing
        if required_len > Ls:
            raise ValueError(
                f"make_initial_x: cannot fit {count} agents on link {int(s_idx)} "
                f"(L={Ls:.3f}) with spacing={spacing:.3f}; "
                f"required_len={required_len:.3f} exceeds link length"
            )

        end_positions = jnp.maximum(
            0.0,
            jnp.arange(start_pos, start_pos - count * spacing, -spacing),
        )
        if end_positions.shape[0] != count:
            end_positions = jnp.array(
                [max(0.0, Ls - float(i) * spacing - spacing) for i in range(count)],
                dtype=float,
            )

        rows = slice(idx_offset, idx_offset + count)
        x0 = x0.at[rows, int(s_idx)].set(end_positions)
        idx_offset += count

    return x0


# ---------------------------------------------------------------------------
# Pre-built `beta` favouring a given route
# ---------------------------------------------------------------------------
def setup_route_beta(
    ctx: SimContext,
    route_nodes: Sequence[int],
    value: float = 3.0,
    virtual_out_idx: Optional[int] = None,
) -> np.ndarray:

    init_nodes = ctx.init_nodes
    term_nodes = ctx.term_nodes
    n_links = ctx.n_links

    allowed = set()
    for a, b in zip(route_nodes[:-1], route_nodes[1:]):
        matches = np.where((init_nodes == a) & (term_nodes == b))[0]
        if matches.size == 0:
            print(f"Warning: no link found for arc {a}->{b}")
        else:
            for idx in matches:
                allowed.add(int(idx))
    if virtual_out_idx is not None:
        allowed.add(int(virtual_out_idx))

    beta_np = np.zeros((n_links,), dtype=float)
    for idx in allowed:
        if 0 <= idx < n_links:
            beta_np[idx] = value
    return beta_np


# ---------------------------------------------------------------------------
# Visualisation-friendly forward simulation
# ---------------------------------------------------------------------------
def forward_cumulative_at_step(
    ctx: SimContext,
    key_in: jnp.ndarray,
    x0_in: jnp.ndarray,
    beta_in: jnp.ndarray,
    u_in: jnp.ndarray,
    kappa_in: jnp.ndarray,
    mp_in: jnp.ndarray,
    sim_steps: int,
    save_step: int,
    delta_n_override: Optional[float] = None,
    save_traj: bool = False,
    output_dir: Optional[str] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[dict], jnp.ndarray, jnp.ndarray]:

    n_links = ctx.n_links
    l = ctx.l
    adj_matrix = ctx.adj_matrix
    delta_t = ctx.delta_t
    delta_n = ctx.delta_n
    use_checkpoint = ctx.use_checkpoint
    agent_stride = ctx.agent_traj_stride
    virtual_in_indices = ctx.virtual_in_indices

    link_length_matrix = jnp.ones((x0_in.shape[0], x0_in.shape[1]), dtype=float) * l
    center = l / 2.0

    virtual_in_idx_arr = np.asarray(
        [int(v) for v in virtual_in_indices if v is not None], dtype=np.int32
    )
    virtual_in_idx_jax = (
        jnp.asarray(virtual_in_idx_arr, dtype=jnp.int32)
        if virtual_in_idx_arr.size > 0
        else None
    )

    def step_fn(carry, t):
        rng, x_prev, cum = carry
        rng, subkey = random.split(rng)
        delta_n_use = (
            delta_n if delta_n_override is None else float(delta_n_override)
        )
        mp_reshaped = jnp.reshape(mp_in, (n_links, 1))

        if _nvtx is not None:
            _nvtx.push_range("BatchLinkForward_forward_step")
        x_curr = BatchLinkForward_fast_vmap_sort(
            x_prev, u_in, kappa_in, l, delta_t=delta_t, delta_n=delta_n_use
        )
        if _nvtx is not None:
            _nvtx.pop_range()
        rng, x_curr = transfer_fast(
            subkey,
            x_curr,
            adj_matrix,
            beta_in,
            mp_reshaped,
            kappa_in,
            link_length_matrix,
            delta_t=delta_t,
            delta_n=delta_n_use,
        )
    
        ind_prev = jnp.where(x_prev >= center, 1.0, 0.0)
        ind_curr = jnp.where(x_curr >= center, 1.0, 0.0)
        inc = jnp.sum(jnp.maximum(0.0, ind_curr - ind_prev), axis=0)
        cum = cum + inc * delta_n_use
        return (rng, x_curr, cum), None

    step_fn_use = checkpoint(step_fn) if use_checkpoint else step_fn

    rng_local = key_in
    cum0 = jnp.zeros((n_links,), dtype=float)
    carry_init = (rng_local, x0_in, cum0)

    n_save = sim_steps // save_step
    remainder = sim_steps - n_save * save_step

    def run_block(carry, _):
        carry, _ = jax.lax.scan(step_fn_use, carry, jnp.arange(save_step))
        x_curr = carry[1]
        delta_n_use = (
            delta_n if delta_n_override is None else float(delta_n_override)
        )
        if virtual_in_idx_jax is not None:
            on_virtual_in = jnp.any(
                x_curr[:, virtual_in_idx_jax] >= 0.0, axis=1
            )
            virtual_in_occupancy = jnp.sum(on_virtual_in.astype(float)) * delta_n_use
        else:
            virtual_in_occupancy = jnp.array(0.0, dtype=float)
        if save_traj:
            x_saved = carry[1]
            if agent_stride > 1:
                x_saved = x_saved[::agent_stride, :]
            return carry, (carry[2], x_saved, virtual_in_occupancy)
        return carry, (carry[2], virtual_in_occupancy)

    if n_save > 0:
        carry_final, saved = jax.lax.scan(run_block, carry_init, jnp.arange(n_save))
        if save_traj:
            saved_cums, saved_x, saved_vi_occ = saved
        else:
            saved_cums, saved_vi_occ = saved
            saved_x = None
    else:
        carry_final = carry_init
        saved_cums = jnp.zeros((0, n_links), dtype=float)
        saved_x = (
            jnp.zeros((0,) + x0_in.shape, dtype=float) if save_traj else None
        )
        saved_vi_occ = jnp.zeros((0,), dtype=float)

    if remainder > 0:
        carry_final, _ = jax.lax.scan(
            step_fn_use, carry_final, jnp.arange(remainder)
        )

    final_x = carry_final[1]
    if not save_traj:
        return carry_final[2], saved_cums, None, final_x, saved_vi_occ

    saved_x_np = np.array(saved_x)
    original_n_agents = x0_in.shape[0]
    agent_id_map = {
        subsampled_idx: original_idx
        for subsampled_idx, original_idx in enumerate(
            range(0, original_n_agents, agent_stride if agent_stride > 1 else 1)
        )
    }
    if agent_stride <= 1:
        agent_id_map = {idx: idx for idx in range(original_n_agents)}

    n_timesteps, n_agents_out, n_links_out = saved_x_np.shape
    valid_mask_3d = saved_x_np >= 0.0
    n_valid = np.sum(valid_mask_3d, axis=2)
    has_valid = n_valid > 0

    best_idx_single = np.argmax(valid_mask_3d, axis=2).astype(np.int32)
    best_idx_multi = np.argmax(saved_x_np, axis=2).astype(np.int32)
    is_multi = n_valid > 1
    link_indices = np.where(
        has_valid, np.where(is_multi, best_idx_multi, best_idx_single), -1
    ).astype(np.int32)

    pos_pick = np.take_along_axis(
        saved_x_np,
        np.clip(link_indices, 0, n_links_out - 1)[..., np.newaxis],
        axis=2,
    ).squeeze(axis=2)
    positions = np.where(has_valid, pos_pick, -9999.0).astype(np.float32)

    error_count = int(np.sum(is_multi))
    if error_count > 0:
        print(
            f"[ERROR] Trajectory validation failed: {error_count} "
            f"cases where an agent exists on multiple links"
        )
    else:
        print(
            "[OK] Trajectory validation passed: all agents exist on at "
            "most one link per timestep"
        )

    if output_dir is not None:
        import csv
        import os

        os.makedirs(output_dir, exist_ok=True)
        csv_output_path = os.path.join(output_dir, "trajectories_all_agents_detailed.csv")
        with open(csv_output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["agent_subsampled", "agent_original", "timestep", "link_idx", "position"]
            )
            for a in range(n_agents_out):
                original_a = agent_id_map.get(a, a)
                for t in range(n_timesteps):
                    link_idx = link_indices[t, a]
                    position = positions[t, a]
                    if link_idx >= 0 and position > -0.01:
                        writer.writerow(
                            [a, original_a, t, link_idx, f"{position:.2f}"]
                        )
        logging.info("Exported detailed trajectories to %s", csv_output_path)

    traj_compact = {
        "link_indices": link_indices,
        "positions": positions,
        "n_timesteps": n_timesteps,
        "n_agents": n_agents_out,
        "n_links": n_links_out,
    }
    return carry_final[2], saved_cums, traj_compact, final_x, saved_vi_occ
