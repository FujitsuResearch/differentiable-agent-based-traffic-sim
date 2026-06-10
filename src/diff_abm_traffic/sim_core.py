"""Core differentiable agent-based traffic simulator primitives."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------
INF: float = 9999.0
NEG_INF: float = -9999.0
MILE_TO_M: float = 1609.344
EPS_MASK: float = -1e-2
inf = INF
negative_inf = NEG_INF
eps_mask = EPS_MASK


# ---------------------------------------------------------------------------
# Per-link forward update (intra-link motion)
# ---------------------------------------------------------------------------
def compute_headway_by_sort(x_on_link: jnp.ndarray, inf_val: float = INF) -> jnp.ndarray:
    """Compute headway (distance to the next agent ahead) on a single link."""

    active_mask = x_on_link >= EPS_MASK

    # Push inactive agents to a very large negative position so they sort
    # behind every active agent.
    fill_val = -1e12
    pos_filled = jnp.where(active_mask, x_on_link, fill_val)

    # Sort descending so the front-most agent comes first.
    order = jnp.argsort(-pos_filled)
    sorted_pos = pos_filled[order]

    # Differences between consecutive agents in sorted order = headways.
    front_inf = jnp.array([inf_val], dtype=sorted_pos.dtype)
    diffs = sorted_pos[:-1] - sorted_pos[1:]
    diffs_full = jnp.concatenate([front_inf, diffs])

    # Scatter back to the original agent order.
    headway_full = jnp.full_like(x_on_link, inf_val)
    headway_full = headway_full.at[order].set(diffs_full)

    # Inactive agents always report `inf_val` (no constraint).
    headway_full = jnp.where(active_mask, headway_full, inf_val)
    return headway_full


def BatchLinkForward_fast_vmap_sort(
    x: jnp.ndarray,
    u: jnp.ndarray,
    kappa: jnp.ndarray,
    l: jnp.ndarray,
    delta_t: float = 1.0,
    delta_n: float = 1.0,
) -> jnp.ndarray:
    """Advance every agent's position by one timestep using a car-following
    update applied independently per link."""

    x_transposed = x.T  # (n_links, n_agent)

    def process_one_link(x_on_link, u_val, kappa_val, l_val, link_idx):
        # Headway to the agent ahead (or `INF` if none).
        min_headway_1d = compute_headway_by_sort(x_on_link, inf_val=INF)
        min_headway = jnp.reshape(min_headway_1d, (x_on_link.shape[0], 1))

        # Minimum spacing dictated by jam density.
        delta_spacing = 1.0 / kappa_val * delta_n
        u_times_dt = u_val * delta_t

        forward_mask = x_on_link >= EPS_MASK
        forward_mask_3d = forward_mask[:, None]

        free_flow_dx = u_times_dt * forward_mask_3d
        cong_flow_dx = jnp.maximum(min_headway - delta_spacing, 0.0) * forward_mask_3d
        dx = jnp.minimum(free_flow_dx, cong_flow_dx)

        dx_1d = dx.squeeze(-1)
        x_new = x_on_link + dx_1d

        # Clip to link length using a straight-through trick so the
        # gradient of `x_new` flows even when the clip becomes active.
        l_limiter = jnp.full_like(x_new, l_val)
        l_limiter = l_limiter + x_new - jax.lax.stop_gradient(x_new)
        x_new = jnp.minimum(x_new, l_limiter)
        return x_new

    link_indices = jnp.arange(x_transposed.shape[0])
    x_new_transposed = jax.vmap(process_one_link)(
        x_transposed, u, kappa, l, link_indices
    )
    return x_new_transposed.T


# ---------------------------------------------------------------------------
# Inter-link transfer (junctions)
# ---------------------------------------------------------------------------
def _batch_gumbel_sample(rng: jnp.ndarray, shape) -> jnp.ndarray:
    """Sample standard Gumbel(0, 1) noise of the given shape."""
    EPS = 1e-10
    uniform = random.uniform(rng, shape=shape)
    return -jnp.log(EPS - jnp.log(uniform + EPS))


def _batch_gumbel_softmax_sample(
    rng: jnp.ndarray, z: jnp.ndarray, temp: float = 0.01, hard: bool = True
) -> jnp.ndarray:
    """Differentiable categorical sample via Gumbel-Softmax."""

    gumbels = _batch_gumbel_sample(rng, z.shape)
    gumbels = (jnp.log(z + 1e-8) + gumbels) / temp
    choice_soft = jax.nn.softmax(gumbels, axis=1)
    if hard:
        index = jnp.argmax(choice_soft, axis=1)
        choice_hard = jax.nn.one_hot(index, num_classes=choice_soft.shape[1])
        return choice_hard - jax.lax.stop_gradient(choice_soft) + choice_soft
    return choice_soft


def transfer_fast(
    key: jnp.ndarray,
    x: jnp.ndarray,
    adj_matrix: jnp.ndarray,
    beta: jnp.ndarray,
    merge_priority: jnp.ndarray,
    kappa: jnp.ndarray,
    link_length_matrix: jnp.ndarray,
    save_debug: bool = False,
    debug_dir=None,
    delta_t: float = 1.0,
    delta_n: float = 1.0,
):
    """Probabilistic inter-link transfer at junctions."""
    existence_matrix = x >= EPS_MASK

    # Agents that have reached (or passed) the end of their current link.
    is_arrived_raw = jnp.where(
        x >= EPS_MASK, x - EPS_MASK >= link_length_matrix, False
    )
    is_arrived = jnp.asarray(
        [(jnp.sum(is_arrived_raw, axis=1) > 0).astype(jnp.float32)]
    ).T

    # Links with enough downstream headroom to accept a new agent.
    is_vacant = jnp.min(jnp.where(x < EPS_MASK, INF, x), axis=0)
    is_vacant = is_vacant > (1.0 / kappa * delta_n)

    potential_next_link = jnp.matmul(existence_matrix, adj_matrix)
    is_connected = jnp.asarray([jnp.max(potential_next_link, axis=1)]).T

    # Softmax over candidate downstream links weighted by beta.
    # link choice utility uses beta directly. (cost is set to one for every link).
    link_utility = NEG_INF * (potential_next_link < 1) + beta * potential_next_link
    link_choice_prob = jax.nn.softmax(link_utility, axis=1)
    link_choice_prob = jnp.where(potential_next_link >= 1, link_choice_prob, 0.0)
    link_choice_prob = link_choice_prob / (
        jnp.sum(link_choice_prob, axis=1, keepdims=True) + 1e-10
    )

    newkey, subkey = random.split(key)
    link_choice_hard = _batch_gumbel_softmax_sample(
        subkey, link_choice_prob, hard=True
    )
    rng = newkey

    # Mask: only agents that have arrived AND whose chosen link is vacant.
    link_choice_hard_mod = is_connected * link_choice_hard
    link_choice_hard_mod = is_arrived * link_choice_hard_mod
    link_choice_hard_mod = link_choice_hard_mod * is_vacant

    agent_choice_mask = jnp.asarray([jnp.max(link_choice_hard_mod, axis=0)]).T

    # Per-link competition: one agent wins each contested downstream link.
    agent_priority = jnp.matmul(existence_matrix, merge_priority)
    agent_priority_matrix = (agent_priority * link_choice_hard_mod).T

    zeromask_agent_priority_matrix = agent_priority_matrix == 0
    agent_utility = NEG_INF * zeromask_agent_priority_matrix + agent_priority_matrix
    agent_choice_prob = jax.nn.softmax(agent_utility, axis=1)

    newkey, subkey = random.split(rng)
    agent_choice_hard = _batch_gumbel_softmax_sample(
        subkey, agent_choice_prob, hard=True
    )
    rng = newkey

    masked_agent_choice_hard = agent_choice_mask * agent_choice_hard
    transferring_agent_mask = jnp.asarray(
        [jnp.sum(masked_agent_choice_hard.T, axis=1)]
    ).T

    # Apply transfer: vacate old link entry, write new link entry at position 0.
    x1 = existence_matrix * transferring_agent_mask
    x2 = masked_agent_choice_hard.T

    x_now = jnp.asarray([jnp.max(x, axis=1)]).T
    scale1 = NEG_INF - x_now
    scale2 = INF + x_now - jax.lax.stop_gradient(x_now)

    new_x = x + scale1 * x1 + scale2 * x2
    return rng, new_x
