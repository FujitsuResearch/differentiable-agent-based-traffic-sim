#!/usr/bin/env python3
from __future__ import annotations

from diff_abm_traffic.config import setup_jax_memory_env, print_jax_memory_settings

setup_jax_memory_env()
print_jax_memory_settings()

from diff_abm_traffic._calibration_runner import ( 
    CalibrationOptions,
    run_calibration,
)
from diff_abm_traffic.config import load_config_from_env
from diff_abm_traffic.networks import build_network_siouxfalls


def main() -> None:
    config = load_config_from_env()
    opts = CalibrationOptions(
        network_builder=lambda script_dir: build_network_siouxfalls(
            script_dir=script_dir
        ),
        city_label="SiouxFalls network",
        adjust_obs_schedule=True,
    )
    run_calibration(opts, config)


if __name__ == "__main__":
    main()
