"""Shared calibration driver used by both city-specific entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .assimilation import assimilate
from .config import Config
from .io_utils import (
    export_network_and_agent_states,
    load_param_results,
    save_cumulative_series,
    save_forward_cumulative_series,
    save_forward_opt_snapshot,
    save_link_counts,
    save_param_results,
    save_replay_metadata,
    save_stage_timing_summary,
    save_trajectories,
    save_virtual_inflow_occupancy,
)
from .simulation import (
    SimContext,
    apply_obs_perturbation,
    forward_and_count_simple,
    forward_cumulative_at_step,
    make_initial_x,
)


@dataclass
class CalibrationOptions:
    network_builder: Callable[[Optional[str]], dict]
    city_label: str
    adjust_obs_schedule: bool = False


def run_calibration(opts: CalibrationOptions, config: Config) -> None:
    # 1) Network & simulation context
    script_dir = os.getcwd()
    net = opts.network_builder(script_dir)
    n_links = int(net["n_links"])
    orig_n_links = int(net["orig_n_links"])
    init_nodes = np.asarray(net["init_nodes"])
    term_nodes = np.asarray(net["term_nodes"])
    adj_matrix = net["adj_matrix"]
    l = net["l"]
    l_np = np.asarray(net["l_np"])
    virtual_in_indices = list(net["virtual_in_indices"])
    virtual_out_indices = list(net["virtual_out_indices"])
    virtual_in_idx = net.get("virtual_in_idx")
    virtual_out_idx = net.get("virtual_out_idx")
    junction_xy = net.get("junction_xy")
    u_org = net["u"] 

    delta_t = float(config["DELTA_T"])
    delta_n = float(config["DELTA_N"])

    # Boolean virtual-link mask (used by the loss to drop virtual links).
    virtual_link_mask_np = np.zeros((n_links,), dtype=bool)
    for vi in virtual_in_indices + virtual_out_indices:
        if vi is not None and 0 <= int(vi) < n_links:
            virtual_link_mask_np[int(vi)] = True
    virtual_link_mask_jax = jnp.asarray(virtual_link_mask_np)

    ctx = SimContext(
        n_links=n_links,
        l=l,
        l_np=l_np,
        adj_matrix=adj_matrix,
        init_nodes=init_nodes,
        term_nodes=term_nodes,
        delta_t=delta_t,
        delta_n=delta_n,
        eps_sig=float(config["EPS_SIG"]),
        use_checkpoint=bool(config["USE_CHECKPOINT"]),
        chunk_size=int(config["CHUNK_SIZE"]),
        virtual_in_indices=virtual_in_indices,
        virtual_out_indices=virtual_out_indices,
        agent_traj_stride=int(config["AGENT_TRAJ_STRIDE"]),
    )

    # 2) Run-level configuration
    key = random.PRNGKey(int(config["SEED"]))
    key_pred = random.split(key, 2)[1]

    f_header = config["F_HEADER"]
    n_agent = int(config["N_AGENT"])
    n_steps = int(config["N_STEPS"])
    obs_step = int(config["OBS_STEP"])
    method = str(config["ASSIM_METHOD"])
    n_vmap_samples = int(config["N_VMAP_SAMPLES"])
    max_grad_norm = float(config["MAX_GRAD_NORM"])
    assim_lr = float(config["ASSIM_LR"])
    assim_n_iter = int(config["ASSIM_N_ITER"])
    n_steps_forward = int(config["N_STEPS_FORWARD"])
    run_forward_opt = bool(config["RUN_FORWARD_OPT"])
    forward_target_ratio = float(config["FORWARD_TARGET_RATIO"])
    forward_opt_n_iter = int(config["FORWARD_OPT_N_ITER"])
    forward_opt_lr = float(config["FORWARD_OPT_LR"])
    loss_tol = float(config["LOSS_TOL"])
    loss_rel_tol = float(config["LOSS_REL_TOL"])
    patience = int(config["PATIENCE"])
    obs_perturb = float(config["OBS_PERTURB"])
    delta_n_obs = float(config["DELTA_N_OBS"])

    if opts.adjust_obs_schedule:
        if delta_n_obs != delta_n:
            delta_t_obs = delta_t * delta_n_obs / delta_n
            n_steps_obs = int(np.round(n_steps * delta_n / delta_n_obs))
            obs_step_obs = int(np.round(obs_step * delta_n / delta_n_obs))
            cum_save_step_obs = int(
                np.round(int(config["CUM_SAVE_STEP"]) * delta_n / delta_n_obs)
            )
            total_time_sim = n_steps * delta_t
            total_time_obs = n_steps_obs * delta_t_obs
            if abs(total_time_sim - total_time_obs) > 1e-6:
                raise ValueError(
                    f"Time calculation error: sim_time={total_time_sim} "
                    f"!= obs_time={total_time_obs}"
                )
        else:
            delta_t_obs = delta_t
            n_steps_obs = n_steps
            obs_step_obs = obs_step
            cum_save_step_obs = int(config["CUM_SAVE_STEP"])
        print(
            f"Observation params: delta_n_obs={delta_n_obs}, "
            f"delta_t_obs={delta_t_obs:.4f}"
        )
        print(
            f"  => n_steps_obs={n_steps_obs}, obs_step_obs={obs_step_obs}, "
            f"cum_save_step_obs={cum_save_step_obs}"
        )
        print(
            f"  Total simulation time: {n_steps * delta_t:.2f} (sim) = "
            f"{n_steps_obs * delta_t_obs:.2f} (obs)"
        )
    else:
        delta_t_obs = float(config["DELTA_T_OBS"])
        n_steps_obs = n_steps
        obs_step_obs = obs_step
        cum_save_step_obs = int(config["CUM_SAVE_STEP"]) 

    print(
        f"Configuration: n_agent={n_agent}, n_steps={n_steps}, "
        f"n_steps_forward={n_steps_forward}, obs_step={obs_step}, "
        f"method={method}, n_vmap_samples={n_vmap_samples}, "
        f"max_grad_norm={max_grad_norm}"
    )
    print(
        f"Delta parameters: delta_t={delta_t}, delta_n={delta_n} "
        f"(effective agents={int(n_agent / delta_n)})"
    )
    print(
        f"Network: {opts.city_label} loaded, n_links={n_links}, "
        f"orig_n_links={orig_n_links}"
    )

    # Agent placement at every virtual inflow link.
    start_idxs = [int(v) for v in virtual_in_indices if v is not None] or [
        int(virtual_in_idx) if virtual_in_idx is not None else 0
    ]
    print(f"Placing agents evenly across virtual inflow indices: {start_idxs}")

    # 3) True / perturbed / background parameters
    rng = np.random.default_rng(2026)

    print("\n" + "=" * 60)
    print("TRUE PARAMETERS")
    print("=" * 60)
    u_true_np = rng.uniform(13.9, 22.2, size=(n_links,))
    u_true = jnp.asarray(u_true_np)
    print(f"u_true: mean={np.mean(u_true_np):.2f}, std={np.std(u_true_np):.2f}")

    beta_true_np = rng.uniform(0.0, 5.0, size=(n_links,))
    beta_true = jnp.asarray(beta_true_np)
    print(
        f"beta_true: mean={np.mean(beta_true_np):.2f}, "
        f"std={np.std(beta_true_np):.2f}, {np.sum(beta_true_np > 0)} active links"
    )

    kappa_true_np = rng.uniform(0.18, 0.22, size=(n_links,))
    kappa_true = jnp.asarray(kappa_true_np)
    print(
        f"kappa_true: mean={np.mean(kappa_true_np):.4f}, "
        f"std={np.std(kappa_true_np):.4f}"
    )

    merge_priority_true_np = rng.uniform(0.0, 5.0, size=(n_links,))
    merge_priority_true = jnp.asarray(merge_priority_true_np)
    print(
        f"merge_priority_true: mean={np.mean(merge_priority_true_np):.2f}, "
        f"std={np.std(merge_priority_true_np):.2f}"
    )

    print("\n" + "=" * 60)
    print("PERTURBED PARAMETERS (true + gaussian noise)")
    print("=" * 60)
    perturb = 0.0 
    u_perturb_np = u_true_np * (1.0 + rng.normal(loc=0.0, scale=perturb, size=(n_links,)))
    u_perturb = jnp.asarray(u_perturb_np)
    beta_perturb_np = beta_true_np * (
        1.0 + rng.normal(loc=0.0, scale=perturb, size=(n_links,))
    )
    beta_perturb = jnp.asarray(beta_perturb_np)
    kappa_perturb_np = kappa_true_np * (
        1.0 + rng.normal(loc=0.0, scale=perturb, size=(n_links,))
    )
    kappa_perturb = jnp.asarray(kappa_perturb_np)
    merge_priority_perturb_np = merge_priority_true_np * (
        1.0 + rng.normal(loc=0.0, scale=perturb, size=(n_links,))
    )
    merge_priority_perturb = jnp.asarray(merge_priority_perturb_np)
    print(
        f"u_perturb: mean={np.mean(u_perturb_np):.2f}, std={np.std(u_perturb_np):.2f}"
    )
    print(
        f"beta_perturb: mean={np.mean(beta_perturb_np):.2f}, "
        f"std={np.std(beta_perturb_np):.2f}"
    )
    print(
        f"kappa_perturb: mean={np.mean(kappa_perturb_np):.4f}, "
        f"std={np.std(kappa_perturb_np):.4f}"
    )
    print(
        f"merge_priority_perturb: mean={np.mean(merge_priority_perturb_np):.2f}, "
        f"std={np.std(merge_priority_perturb_np):.2f}"
    )

    print("\n" + "=" * 60)
    print("BACKGROUND PARAMETERS (initial estimates)")
    print("=" * 60)
    u_b_np = np.full((n_links,), 18.05)
    u_b = jnp.asarray(u_b_np)
    print(f"u_b (background): {u_b_np[0]:.2f} (uniform)")
    beta_b_np = np.full((n_links,), 2.5)
    beta_b = jnp.asarray(beta_b_np)
    print("beta_b (background): uniform 2.5")
    kappa_b_np = np.full((n_links,), 0.2)
    kappa_b = jnp.asarray(kappa_b_np)
    print("kappa_b (background): uniform 0.2")
    merge_priority_b_np = np.full((n_links,), 2.5)
    merge_priority_b = jnp.asarray(merge_priority_b_np)
    print("merge_priority_b (background): uniform 2.5")

    # 4) Setup output directory and observation generation
    file_tag = (
        f"{f_header}_n{n_agent}_s{n_steps}_o{obs_step}_m{method}"
        f"_v{n_vmap_samples}_dn{int(delta_n)}"
    )
    output_dir = os.path.join(script_dir, "results", file_tag)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    obs_file = os.path.join(output_dir, f"obs_cum_{file_tag}.npy")
    cum_file = os.path.join(output_dir, f"cum_f_{file_tag}.npy")
    use_obs_files = bool(config["USE_OBS_FILES"])

    u_tmp = np.full((n_links,), 18.05)
    if opts.adjust_obs_schedule:
        x0_true = make_initial_x(
            ctx, n_agent, start_idxs,
            delta_n_override=delta_n_obs,
            delta_t_override=delta_t_obs,
            u_speed_array=u_tmp,
        )
    else:
        x0_true = make_initial_x(
            ctx, n_agent, start_idxs,
            delta_n_override=delta_n_obs,
            u_speed_array=u_tmp,
        )

    if use_obs_files and os.path.exists(obs_file) and os.path.exists(cum_file):
        print(f"Loading observations from files: {obs_file}, {cum_file}")
        obs_cum_np = np.load(obs_file)
        cum_f_np = np.load(cum_file)
        obs_cum = jnp.asarray(obs_cum_np)
    else:
        print(f"Generating observations with {n_agent} agents")
        if opts.adjust_obs_schedule:
            cum_f, obs_cum = forward_and_count_simple(
                ctx, key, x0_true, n_steps_obs,
                beta_perturb, u_perturb, kappa_perturb, merge_priority_perturb,
                obs_step_obs,
                delta_n_override=delta_n_obs,
                delta_t_override=delta_t_obs,
            )
        else:
            cum_f, obs_cum = forward_and_count_simple(
                ctx, key, x0_true, n_steps,
                beta_perturb, u_perturb, kappa_perturb, merge_priority_perturb,
                obs_step,
                delta_n_override=delta_n_obs,
            )
        obs_cum_np = np.array(obs_cum)
        cum_f_np = np.array(cum_f)
        np.save(obs_file, obs_cum_np)
        np.save(cum_file, cum_f_np)
        print(f"Saved observations to: {obs_file}, {cum_file}")
        print(f"Observations shape: {obs_cum.shape}")

    if obs_perturb > 0.0:
        print(f"\nApplying observation perturbation (obs_perturb={obs_perturb*100:.2f}%)")
        obs_cum_np = apply_obs_perturbation(obs_cum_np, obs_perturb, rng)
        obs_cum = jnp.asarray(obs_cum_np)
    else:
        print(f"\nNo observation perturbation applied (obs_perturb={obs_perturb*100:.2f}%)")

    x0 = x0_true

    optimize_beta = bool(config["OPTIMIZE_BETA"])
    optimize_u = bool(config["OPTIMIZE_U"])
    optimize_kappa = bool(config["OPTIMIZE_KAPPA"])
    optimize_mp = bool(config["OPTIMIZE_MP"])
    print(
        f"Optimization targets: beta={optimize_beta}, u={optimize_u}, "
        f"kappa={optimize_kappa}, merge_priority={optimize_mp}"
    )

    if not optimize_beta:
        beta_b = beta_true
        print("Calibration setup: beta is fixed -> beta_b is set to beta_true")
    if not optimize_u:
        u_b = u_true
        print("Calibration setup: u is fixed -> u_b is set to u_true")
    if not optimize_kappa:
        kappa_b = kappa_true
        print("Calibration setup: kappa is fixed -> kappa_b is set to kappa_true")
    if not optimize_mp:
        merge_priority_b = merge_priority_true
        print(
            "Calibration setup: merge_priority is fixed -> merge_priority_b is set to merge_priority_true"
        )

    params_init = jnp.concatenate([beta_b, u_b, kappa_b, merge_priority_b])

    print("\n" + "=" * 60)
    print("Starting joint beta+u+kappa+merge_priority data assimilation")
    print("=" * 60)
    t_calib_start = perf_counter()
    suffix_method = f"_{method}" if method != "gd" else ""
    force_assim = bool(config["FORCE_ASSIM"])

    loaded = None if force_assim else load_param_results(
        output_dir, suffix_method, n_agent, n_links
    )
    if loaded is not None:
        print("Found existing estimated parameter files. Loading and skipping assimilation.")
        beta_est, u_est, kappa_est, mp_est = loaded
    else:
        print("Running assimilation to estimate parameters.")
        params_est = assimilate(
            ctx, config, virtual_link_mask_jax,
            params_init, x0, n_steps, obs_cum,
            beta_b, u_b, kappa_b, merge_priority_b,
            obs_step,
            optimize_beta=optimize_beta,
            optimize_u=optimize_u,
            optimize_kappa=optimize_kappa,
            optimize_mp=optimize_mp,
            method=method, lr=assim_lr, n_iter=assim_n_iter,
            key=key, n_agent=n_agent, n_vmap_samples=n_vmap_samples,
            max_grad_norm=max_grad_norm,
            grad_tol=float(config["GRAD_TOL"]),
            loss_tol=loss_tol, loss_rel_tol=loss_rel_tol, patience=patience,
            file_tag=file_tag, output_dir=output_dir,
        )
        beta_est = params_est[:n_links]
        u_est = params_est[n_links : 2 * n_links]
        kappa_est = params_est[2 * n_links : 3 * n_links]
        mp_est = params_est[3 * n_links : 4 * n_links]
    t_calib_elapsed = perf_counter() - t_calib_start

    save_param_results(
        output_dir, suffix_method, n_agent, n_links,
        beta_true=beta_true, beta_b=beta_b, beta_est=beta_est,
        u_true=u_true, u_b=u_b, u_est=u_est,
        kappa_true=kappa_true, kappa_b=kappa_b, kappa_est=kappa_est,
        mp_true=merge_priority_true, mp_b=merge_priority_b, mp_est=mp_est,
    )
    save_forward_opt_snapshot(
        output_dir, suffix_method, n_agent, n_links,
        beta_est, u_est, kappa_est, mp_est,
    )

    beta_rmse = float(jnp.sqrt(jnp.mean((beta_est - beta_true) ** 2)))
    u_rmse = float(jnp.sqrt(jnp.mean((u_est - u_true) ** 2)))
    kappa_rmse = float(jnp.sqrt(jnp.mean((kappa_est - kappa_true) ** 2)))
    mp_rmse = float(jnp.sqrt(jnp.mean((mp_est - merge_priority_true) ** 2)))
    print(
        f"\nRMSE: beta={beta_rmse:.6f}, u={u_rmse:.6f}, "
        f"kappa={kappa_rmse:.6f}, merge_priority={mp_rmse:.6f}"
    )

    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    # 7) Forward prediction & optional forward optimisation
    beta_fwd_opt = beta_est
    u_fwd_opt = u_est
    kappa_fwd_opt = kappa_est
    mp_fwd_opt = mp_est
    cums_fwd_true = cums_fwd_est = cums_fwd_opt = None
    x_final_true = x_final_est = None
    t_forward_elapsed = t_opt_elapsed = None
    key_fwd_true = key_fwd = key_fwd_eval = None
    key_state_true = key_state_est = key_fwd_opt_key = None

    x0_true_sim = x0_est_sim = x0_bg_sim = x0_true

    if n_steps_forward > 0:
        print("\n" + "=" * 60)
        print(f"[STEP 2] Now casting / nowcast from calibrated state "
              f"(n_steps_forward={n_steps_forward})")
        print("=" * 60)
        key_pred_local = key_pred

        key_pred_local, key_state_true = random.split(key_pred_local)
        x_final_true = forward_and_count_simple(
            ctx, key_state_true, x0_true_sim, n_steps,
            beta_true, u_true, kappa_true, merge_priority_true, obs_step,
            delta_n_override=delta_n_obs,
            save_cums=False, return_x_final=True,
        )

        key_pred_local, key_state_est = random.split(key_pred_local)
        x_final_est = forward_and_count_simple(
            ctx, key_state_est, x0_est_sim, n_steps,
            beta_est, u_est, kappa_est, mp_est, obs_step,
            save_cums=False, return_x_final=True,
        )

        key_pred_local, key_fwd_true = random.split(key_pred_local)
        cum_f_fwd_true, cums_fwd_true = forward_and_count_simple(
            ctx, key_fwd_true, x_final_true, n_steps_forward,
            beta_true, u_true, kappa_true, merge_priority_true, obs_step,
        )

        t_forward_start = perf_counter()
        key_pred_local, key_fwd = random.split(key_pred_local)
        cum_f_fwd_est, cums_fwd_est = forward_and_count_simple(
            ctx, key_fwd, x_final_est, n_steps_forward,
            beta_est, u_est, kappa_est, mp_est, obs_step,
        )
        t_forward_elapsed = perf_counter() - t_forward_start

        non_virtual_indices = np.where(~np.asarray(virtual_link_mask_np))[0]
        cum_f_fwd_est_np = np.array(cum_f_fwd_est)
        candidate_indices = (
            non_virtual_indices
            if non_virtual_indices.size > 0
            else np.arange(cum_f_fwd_est_np.shape[0])
        )
        candidate_totals = cum_f_fwd_est_np[candidate_indices]
        order = np.argsort(-candidate_totals)
        rank_pos = 0 if order.size >= 2 else 0
        max_link_idx = int(candidate_indices[order[rank_pos]])
        max_link_total = float(cum_f_fwd_est_np[max_link_idx])
        target_total = forward_target_ratio * max_link_total
        print(
            f"Forward optimization target link (rank={rank_pos+1}): "
            f"link_{max_link_idx}, baseline_total={max_link_total:.6f}, "
            f"target_ratio={forward_target_ratio:.3f}, target_total={target_total:.6f}"
        )

        cums_fwd_target = np.array(cums_fwd_est)
        cums_fwd_target[:, max_link_idx] = (
            cums_fwd_target[:, max_link_idx] * forward_target_ratio
        )
        cums_fwd_target_jax = jnp.asarray(cums_fwd_target)

        import pandas as pd
        pd.DataFrame(
            [{
                "link_idx": max_link_idx,
                "baseline_total": max_link_total,
                "target_ratio": forward_target_ratio,
                "target_total": target_total,
            }]
        ).to_csv(
            os.path.join(
                output_dir,
                f"forward_dominant_link_summary{suffix_method}_n{n_agent}.csv",
            ),
            index=False,
        )

        if run_forward_opt:
            print("\n" + "=" * 60)
            print("[STEP 3] Forward optimization on the dominant link")
            print("=" * 60)
            params_init_fwd_opt = jnp.concatenate(
                [beta_est, u_est, kappa_est, mp_est]
            )

            key_pred_local, key_fwd_opt_key = random.split(key_pred_local)
            t_opt_start = perf_counter()
            params_fwd_opt = assimilate(
                ctx, config, virtual_link_mask_jax,
                params_init_fwd_opt, x_final_est, n_steps_forward,
                cums_fwd_target_jax,
                beta_est, u_est, kappa_est, mp_est,
                obs_step,
                optimize_beta=True,
                optimize_u=False,
                optimize_kappa=False,
                optimize_mp=False,
                method=method, lr=forward_opt_lr, n_iter=forward_opt_n_iter,
                key=key_fwd_opt_key, n_agent=n_agent,
                n_vmap_samples=n_vmap_samples, max_grad_norm=max_grad_norm,
                grad_tol=float(config["GRAD_TOL"]),
                loss_tol=loss_tol, loss_rel_tol=loss_rel_tol, patience=patience,
                file_tag=f"{file_tag}_fwdopt_link{max_link_idx}",
                output_dir=output_dir,
                obs_keep_ratio=1.0, obs_link_indices=[max_link_idx],
                objective_label="forward_target",
            )
            t_opt_elapsed = perf_counter() - t_opt_start
            beta_fwd_opt = params_fwd_opt[:n_links]
            u_fwd_opt = params_fwd_opt[n_links : 2 * n_links]
            kappa_fwd_opt = params_fwd_opt[2 * n_links : 3 * n_links]
            mp_fwd_opt = params_fwd_opt[3 * n_links : 4 * n_links]
            save_forward_opt_snapshot(
                output_dir, suffix_method, n_agent, n_links,
                beta_fwd_opt, u_fwd_opt, kappa_fwd_opt, mp_fwd_opt,
            )

            key_pred_local, key_fwd_eval = random.split(key_pred_local)
            cum_f_fwd_opt, cums_fwd_opt = forward_and_count_simple(
                ctx, key_fwd_eval, x_final_est, n_steps_forward,
                beta_fwd_opt, u_fwd_opt, kappa_fwd_opt, mp_fwd_opt, obs_step,
            )
            dominant_after = float(np.array(cum_f_fwd_opt)[max_link_idx])
            reduction_ratio = (
                dominant_after / max_link_total if max_link_total > 1e-12 else np.nan
            )
            print(
                f"Dominant link after forward optimization: link_{max_link_idx}, "
                f"optimized_total={dominant_after:.6f}, "
                f"achieved_ratio={reduction_ratio:.6f}"
            )
            pd.DataFrame(
                [{
                    "link_idx": max_link_idx,
                    "baseline_total": max_link_total,
                    "target_ratio": forward_target_ratio,
                    "target_total": target_total,
                    "optimized_total": dominant_after,
                    "achieved_ratio": reduction_ratio,
                }]
            ).to_csv(
                os.path.join(
                    output_dir,
                    f"forward_optimization_summary{suffix_method}_n{n_agent}.csv",
                ),
                index=False,
            )
        else:
            print("RUN_FORWARD_OPT=0: skipped forward-window optimization.")
    else:
        print("N_STEPS_FORWARD<=0: skipped forward prediction and forward-window optimization.")

    # 8) Visualisation outputs (cumulative series, trajectories, etc.)
    print("\nComputing visualization outputs (cumulative series, trajectories, occupancy)...")
    cum_save_step = int(config["CUM_SAVE_STEP"])

    _, cums_true_v = forward_and_count_simple(
        ctx, key, x0_true_sim, n_steps,
        beta_true, u_true, kappa_true, merge_priority_true, obs_step,
        delta_n_override=delta_n_obs,
    )
    _, cums_bg_v = forward_and_count_simple(
        ctx, key, x0_true, n_steps,
        beta_b, u_b, kappa_b, merge_priority_b, obs_step,
    )
    _, cums_est_v = forward_and_count_simple(
        ctx, key, x0_true, n_steps,
        beta_est, u_est, kappa_est, mp_est, obs_step,
    )

    cum_f_true, _, traj_true, x_final_true_vis, vi_occ_true = forward_cumulative_at_step(
        ctx, key, x0_true_sim, beta_true, u_true, kappa_true, merge_priority_true,
        n_steps, cum_save_step, delta_n_override=delta_n_obs, save_traj=True,
        output_dir=output_dir,
    )
    cum_f_bg, _, traj_bg, x_final_bg_vis, vi_occ_bg = forward_cumulative_at_step(
        ctx, key, x0_bg_sim, beta_b, u_b, kappa_b, merge_priority_b,
        n_steps, cum_save_step, save_traj=True,
    )
    cum_f_est, _, traj_est, x_final_est_vis, vi_occ_est = forward_cumulative_at_step(
        ctx, key, x0_est_sim, beta_est, u_est, kappa_est, mp_est,
        n_steps, cum_save_step, save_traj=True,
    )

    save_cumulative_series(
        output_dir, suffix_method, n_agent, n_links, obs_step,
        cums_true_v, cums_bg_v, cums_est_v,
    )

    traj_fwd_true = traj_fwd_est = traj_fwd_opt = None
    vi_occ_fwd_true = vi_occ_fwd_est = vi_occ_fwd_opt = None
    if n_steps_forward > 0 and x_final_true is not None and x_final_est is not None:
        save_traj_flag = bool(config["SAVE_FORWARD_TRAJ"])
        _, _, traj_fwd_true, _, vi_occ_fwd_true = forward_cumulative_at_step(
            ctx, key_fwd_true, x_final_true,
            beta_true, u_true, kappa_true, merge_priority_true,
            n_steps_forward, cum_save_step, save_traj=save_traj_flag,
        )
        _, _, traj_fwd_est, _, vi_occ_fwd_est = forward_cumulative_at_step(
            ctx, key_fwd, x_final_est,
            beta_est, u_est, kappa_est, mp_est,
            n_steps_forward, cum_save_step, save_traj=save_traj_flag,
        )
        save_forward_cumulative_series(
            output_dir, suffix_method, n_agent, n_links, obs_step, n_steps,
            cums_fwd_true, cums_fwd_est, cums_fwd_opt,
        )
        if run_forward_opt and cums_fwd_opt is not None and key_fwd_eval is not None:
            _, _, traj_fwd_opt, _, vi_occ_fwd_opt = forward_cumulative_at_step(
                ctx, key_fwd_eval, x_final_est,
                beta_fwd_opt, u_fwd_opt, kappa_fwd_opt, mp_fwd_opt,
                n_steps_forward, cum_save_step, save_traj=save_traj_flag,
            )

    print("\nSaving agent trajectories for visualization...")
    save_trajectories(
        traj_dir, suffix_method, n_agent,
        {"true": traj_true, "background": traj_bg, "est": traj_est},
    )
    save_trajectories(
        traj_dir, suffix_method, n_agent,
        {
            "forward_true": traj_fwd_true,
            "forward_est": traj_fwd_est,
            "forward_opt": traj_fwd_opt,
        },
    )

    save_link_counts(
        traj_dir, suffix_method, n_agent,
        {"true": cums_true_v, "background": cums_bg_v, "est": cums_est_v,
         "forward_est": cums_fwd_est, "forward_true": cums_fwd_true,
         "forward_opt": cums_fwd_opt},
    )

    save_virtual_inflow_occupancy(
        traj_dir, suffix_method, n_agent, cum_save_step,
        {
            "true": (vi_occ_true, 0),
            "background": (vi_occ_bg, 0),
            "est": (vi_occ_est, 0),
            "forward_true": (vi_occ_fwd_true, n_steps),
            "forward_est": (vi_occ_fwd_est, n_steps),
            "forward_opt": (vi_occ_fwd_opt, n_steps),
        },
    )

    save_replay_metadata(
        output_dir, suffix_method, n_agent,
        seed=int(config["SEED"]),
        backend=str(jax.default_backend()),
        n_steps=n_steps, n_steps_forward=n_steps_forward,
        obs_step=obs_step, cum_save_step=cum_save_step,
        keys={
            "key_main": key, "key_pred": key_pred,
            "key_state_true": key_state_true, "key_state_est": key_state_est,
            "key_fwd_true": key_fwd_true, "key_fwd_est": key_fwd,
            "key_fwd_opt_assim": key_fwd_opt_key,
            "key_fwd_opt_eval": key_fwd_eval,
        },
    )

    print("\nAdditional exports for external post-processing / visualization ...")
    export_dir = os.path.join(output_dir, "exports")
    export_network_and_agent_states(
        ctx, junction_xy, virtual_link_mask_np,
        export_dir, suffix_method, n_agent,
        trajectories={
            "true": traj_true, "background": traj_bg, "est": traj_est,
            "forward_true": traj_fwd_true, "forward_est": traj_fwd_est,
            "forward_opt": traj_fwd_opt,
        },
        cums_true=cums_true_v, cums_bg=cums_bg_v, cums_est=cums_est_v,
        cums_fwd_true=cums_fwd_true, cums_fwd_est=cums_fwd_est,
        cums_fwd_opt=cums_fwd_opt,
        beta_true=beta_true, beta_est=beta_est, beta_fwd_opt=beta_fwd_opt,
    )

    save_stage_timing_summary(
        output_dir, suffix_method, n_agent,
        t_calib=t_calib_elapsed,
        t_forward=t_forward_elapsed,
        t_opt=t_opt_elapsed,
        n_steps_forward=n_steps_forward,
        run_forward_opt=run_forward_opt,
    )
    print("=" * 60)
    print("All processing completed successfully")
    print("=" * 60)