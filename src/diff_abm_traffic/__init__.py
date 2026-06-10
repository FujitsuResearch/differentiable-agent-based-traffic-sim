"""Differentiable agent-based traffic simulator."""

from .config import Config, load_config_from_env, setup_jax_memory_env
from .sim_core import (
    BatchLinkForward_fast_vmap_sort,
    INF,
    NEG_INF,
    MILE_TO_M,
    transfer_fast,
)
from .simulation import (
    SimContext,
    apply_obs_perturbation,
    forward_and_count_simple,
    forward_cumulative_at_step,
    make_initial_x,
    setup_route_beta,
)
from .assimilation import assimilate
from .lr_schedule import create_learning_rate_schedule, find_optimal_lr
from .networks import build_network_chicago, build_network_siouxfalls

__all__ = [
    # config
    "Config",
    "load_config_from_env",
    "setup_jax_memory_env",
    # sim core
    "BatchLinkForward_fast_vmap_sort",
    "transfer_fast",
    "INF",
    "NEG_INF",
    "MILE_TO_M",
    # simulation
    "SimContext",
    "apply_obs_perturbation",
    "forward_and_count_simple",
    "forward_cumulative_at_step",
    "make_initial_x",
    "setup_route_beta",
    # assimilation
    "assimilate",
    # learning-rate utilities
    "create_learning_rate_schedule",
    "find_optimal_lr",
    # network builders
    "build_network_chicago",
    "build_network_siouxfalls",
]
