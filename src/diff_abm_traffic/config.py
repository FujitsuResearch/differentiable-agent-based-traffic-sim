"""Run-time configuration backed by environment variables."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Config:
    SEED: int = 1234
    F_HEADER: str = "chicago"
    EPS_SIG: float = 0.1
    N_AGENT: int = 200
    N_STEPS: int = 1800
    OBS_STEP: int = 300
    ASSIM_METHOD: str = "adam"
    N_VMAP_SAMPLES: int = 1
    MAX_GRAD_NORM: float = 100.0
    AD_MODE: str = "reverse"
    DEBUG_VJP: bool = False
    GRAD_TOL: float = 1e-8
    LOSS_TOL: float = 1e-8
    LOSS_REL_TOL: float = 1e-6
    PATIENCE: int = 20
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.999
    ADAM_EPS: float = 1e-8
    ADAM_LR: float = 0.03
    ADAM_OPTIMIZER: str = "adam"
    ADAM_WEIGHT_DECAY: float = 1e-5
    ADAM_LR_SCHEDULE: str = "constant"
    ADAM_LR_DECAY_RATE: float = 0.96
    ADAM_LR_DECAY_STEPS: int = 50
    ADAM_LR_WARMUP_STEPS: int = 0
    ADAM_LR_END_VALUE: float = 1e-5
    USE_PARAM_REPARAM: bool = False
    PARAM_BETA_SCALE: float = 1.0
    PARAM_U_SCALE: float = 20.0
    PARAM_KAPPA_SCALE: float = 0.2
    PARAM_MP_SCALE: float = 1.0
    PARAM_BETA_MIN: float = 0.0
    PARAM_U_MIN: float = 0.0
    PARAM_KAPPA_MIN: float = 0.15
    PARAM_KAPPA_MAX: float = 0.25
    PARAM_MP_MIN: float = 1e-3
    OPTIMIZE_BETA: bool = True
    OPTIMIZE_U: bool = True
    OPTIMIZE_KAPPA: bool = True
    OPTIMIZE_MP: bool = True
    CHUNK_SIZE: int = 9999
    DELTA_T: float = 1.0
    DELTA_N: float = 1.0
    DELTA_T_OBS: float = 1.0
    DELTA_N_OBS: float = 1.0
    OBS_PERTURB: float = 0.05
    USE_OBS_FILES: bool = False
    CUM_SAVE_STEP: int = 1
    FORCE_ASSIM: bool = False
    ASSIM_LR: float = 0.03
    ASSIM_N_ITER: int = 200
    USE_CHECKPOINT: bool = True
    AGENT_TRAJ_STRIDE: int = 1
    N_STEPS_FORWARD: int = 0
    RUN_FORWARD_OPT: bool = True
    FORWARD_TARGET_RATIO: float = 0.5
    FORWARD_OPT_N_ITER: int = 200
    FORWARD_OPT_LR: float = 0.03
    SAVE_FORWARD_TRAJ: bool = True

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config_from_env() -> Config:
    return Config(
        SEED=_env_int("SEED", "1234"),
        F_HEADER=_env_str("F_HEADER", "chicago"),
        EPS_SIG=_env_float("EPS_SIG", "0.1"),
        N_AGENT=_env_int("N_AGENT", "200"),
        N_STEPS=_env_int("N_STEPS", "1800"),
        OBS_STEP=_env_int("OBS_STEP", "300"),
        ASSIM_METHOD=_env_str("ASSIM_METHOD", "adam").lower(),
        N_VMAP_SAMPLES=_env_int("N_VMAP_SAMPLES", "1"),
        MAX_GRAD_NORM=_env_float("MAX_GRAD_NORM", "100.0"),
        AD_MODE=_env_str("AD_MODE", "reverse").lower(),
        DEBUG_VJP=_env_bool("DEBUG_VJP", "0"),
        GRAD_TOL=_env_float("GRAD_TOL", "1e-8"),
        LOSS_TOL=_env_float("LOSS_TOL", "1e-8"),
        LOSS_REL_TOL=_env_float("LOSS_REL_TOL", "1e-6"),
        PATIENCE=_env_int("PATIENCE", "20"),
        ADAM_BETA1=_env_float("ADAM_BETA1", "0.9"),
        ADAM_BETA2=_env_float("ADAM_BETA2", "0.999"),
        ADAM_EPS=_env_float("ADAM_EPS", "1e-8"),
        ADAM_LR=_env_float("ADAM_LR", "0.03"),
        ADAM_OPTIMIZER=_env_str("ADAM_OPTIMIZER", "adam").lower(),
        ADAM_WEIGHT_DECAY=_env_float("ADAM_WEIGHT_DECAY", "1e-5"),
        ADAM_LR_SCHEDULE=_env_str("ADAM_LR_SCHEDULE", "constant").lower(),
        ADAM_LR_DECAY_RATE=_env_float("ADAM_LR_DECAY_RATE", "0.96"),
        ADAM_LR_DECAY_STEPS=_env_int("ADAM_LR_DECAY_STEPS", "50"),
        ADAM_LR_WARMUP_STEPS=_env_int("ADAM_LR_WARMUP_STEPS", "0"),
        ADAM_LR_END_VALUE=_env_float("ADAM_LR_END_VALUE", "1e-5"),
        USE_PARAM_REPARAM=_env_bool("USE_PARAM_REPARAM", "0"),
        PARAM_BETA_SCALE=_env_float("PARAM_BETA_SCALE", "1.0"),
        PARAM_U_SCALE=_env_float("PARAM_U_SCALE", "20.0"),
        PARAM_KAPPA_SCALE=_env_float("PARAM_KAPPA_SCALE", "0.2"),
        PARAM_MP_SCALE=_env_float("PARAM_MP_SCALE", "1.0"),
        PARAM_BETA_MIN=_env_float("PARAM_BETA_MIN", "0.0"),
        PARAM_U_MIN=_env_float("PARAM_U_MIN", "0.0"),
        PARAM_KAPPA_MIN=_env_float("PARAM_KAPPA_MIN", "0.15"),
        PARAM_KAPPA_MAX=_env_float("PARAM_KAPPA_MAX", "0.25"),
        PARAM_MP_MIN=_env_float("PARAM_MP_MIN", "1e-3"),
        OPTIMIZE_BETA=_env_bool("OPTIMIZE_BETA", "1"),
        OPTIMIZE_U=_env_bool("OPTIMIZE_U", "1"),
        OPTIMIZE_KAPPA=_env_bool("OPTIMIZE_KAPPA", "1"),
        OPTIMIZE_MP=_env_bool("OPTIMIZE_MP", "1"),
        CHUNK_SIZE=_env_int("CHUNK_SIZE", "9999"),
        DELTA_T=_env_float("DELTA_T", "1.0"),
        DELTA_N=_env_float("DELTA_N", "1.0"),
        DELTA_T_OBS=_env_float("DELTA_T_OBS", "1.0"),
        DELTA_N_OBS=_env_float("DELTA_N_OBS", "1.0"),
        OBS_PERTURB=_env_float("OBS_PERTURB", "0.05"),
        USE_OBS_FILES=_env_bool("USE_OBS_FILES", "0"),
        CUM_SAVE_STEP=_env_int("CUM_SAVE_STEP", "1"),
        FORCE_ASSIM=_env_bool("FORCE_ASSIM", "0"),
        ASSIM_LR=_env_float("ASSIM_LR", "0.03"),
        ASSIM_N_ITER=_env_int("ASSIM_N_ITER", "200"),
        USE_CHECKPOINT=_env_bool("USE_CHECKPOINT", "1"),
        AGENT_TRAJ_STRIDE=_env_int("AGENT_TRAJ_STRIDE", "1"),
        N_STEPS_FORWARD=_env_int("N_STEPS_FORWARD", "0"),
        RUN_FORWARD_OPT=_env_bool("RUN_FORWARD_OPT", "1"),
        FORWARD_TARGET_RATIO=_env_float("FORWARD_TARGET_RATIO", "0.5"),
        FORWARD_OPT_N_ITER=int(
            os.environ.get(
                "FORWARD_OPT_N_ITER", os.environ.get("ASSIM_N_ITER", "200")
            )
        ),
        FORWARD_OPT_LR=float(
            os.environ.get("FORWARD_OPT_LR", os.environ.get("ASSIM_LR", "0.03"))
        ),
        SAVE_FORWARD_TRAJ=_env_bool("SAVE_FORWARD_TRAJ", "1"),
    )


def setup_jax_memory_env() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if os.environ.get("ENABLE_GPU_DETERMINISTIC_OPS", "0") == "1":
        if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
            os.environ["XLA_FLAGS"] = (
                os.environ.get("XLA_FLAGS", "")
                + " --xla_gpu_deterministic_ops=true"
            ).strip()


def print_jax_memory_settings() -> None:
    print("JAX Memory Settings:")
    for var in (
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "ENABLE_GPU_DETERMINISTIC_OPS",
        "XLA_FLAGS",
    ):
        print(f"  {var}: {os.environ.get(var, 'default')}")