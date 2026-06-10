"""Gradient-based joint calibration of (beta, u, kappa, merge_priority)."""

from __future__ import annotations

import os
from time import perf_counter
from typing import List, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import numpy as np
import optax
import pandas as pd
from jax import random

from .config import Config
from .lr_schedule import create_learning_rate_schedule, find_optimal_lr
from .simulation import SimContext, forward_and_count_simple


def assimilate(
    ctx: SimContext,
    config: Config,
    virtual_link_mask_jax: jnp.ndarray,
    params_init: jnp.ndarray,
    x0: jnp.ndarray,
    n_steps: int,
    obs_cum: jnp.ndarray,
    beta_b: jnp.ndarray,
    u_b: jnp.ndarray,
    kappa_b: jnp.ndarray,
    merge_priority_b: jnp.ndarray,
    obs_step: int,
    *,
    optimize_beta: bool = True,
    optimize_u: bool = True,
    optimize_kappa: bool = True,
    optimize_mp: bool = True,
    method: str = "adam",
    lr: float = 0.01,
    n_iter: int = 50,
    key: jnp.ndarray = random.PRNGKey(0),
    n_agent: Optional[int] = None,
    n_vmap_samples: int = 1,
    max_grad_norm: float = 10.0,
    grad_tol: float = 1e-8,
    loss_tol: float = 1e-8,
    loss_rel_tol: float = 1e-6,
    patience: int = 5,
    file_tag: str = "",
    output_dir: Optional[str] = None,
    obs_keep_ratio: float = 0.8,
    obs_link_indices: Optional[Sequence[int]] = None,
    objective_label: str = "observation",
) -> jnp.ndarray:
    n_links = ctx.n_links
    save_dir = output_dir if output_dir is not None else os.getcwd()

    loss_hist: List[float] = []
    grad_norm_hist: List[float] = []
    params_hist: List[np.ndarray] = []
    cum_hist: List[np.ndarray] = []
    time_hist: List[float] = []
    start_time = perf_counter()

    method_name = method.upper()
    suffix = f"_{method}" if method != "gd" else ""
    if file_tag:
        suffix += f"_{file_tag}"

    print(
        f"Joint beta+u+kappa+merge_priority assimilation method: {method_name}, "
        f"n_iter={n_iter}, lr={lr}"
    )
    print(f"  n_links={n_links}")

    best_loss = float("inf")
    no_improvement_count = 0
    out_scale = 1.0
    print(f"Output scaling factor: {float(out_scale):.3f}")

    non_virtual_indices = np.where(~np.asarray(virtual_link_mask_jax))[0]
    n_non_virtual = len(non_virtual_indices)
    if obs_link_indices is None:
        n_to_keep = max(1, min(n_non_virtual, int(np.ceil(n_non_virtual * float(obs_keep_ratio)))))
        key, obs_select_key = random.split(key)
        perm = random.permutation(obs_select_key, jnp.arange(n_non_virtual))
        keep_indices = non_virtual_indices[np.asarray(perm[:n_to_keep])]
        print(
            f"Loss mask ({objective_label}): selected_links={n_to_keep}"
            f"/{n_non_virtual} non-virtual links (ratio={float(obs_keep_ratio):.2f})"
        )
    else:
        keep_indices = np.asarray(obs_link_indices, dtype=int)
        keep_indices = keep_indices[(keep_indices >= 0) & (keep_indices < n_links)]
        if keep_indices.size == 0:
            raise ValueError("obs_link_indices is empty after bounds filtering.")
        print(
            f"Loss mask ({objective_label}): explicit links count={keep_indices.size}, "
            f"sample={keep_indices[:10].tolist()}"
        )

    obs_link_mask_np = np.zeros(n_links, dtype=np.float32)
    obs_link_mask_np[keep_indices] = 1.0
    obs_link_mask = jnp.array(obs_link_mask_np)
    obs_link_mask_2d = obs_link_mask[jnp.newaxis, :]
    n_obs_steps = obs_cum.shape[0]
    n_obs_links = jnp.sum(obs_link_mask)
    obs_norm_denom = jnp.maximum(1.0, n_obs_steps * n_obs_links)
    print(
        f"Loss normalization ({objective_label}): "
        f"n_obs_steps={int(n_obs_steps)}, n_obs_links={int(np.array(n_obs_links))}"
    )

    obs_links_out = os.path.join(save_dir, f"assim_selected_obs_links{suffix}.csv")
    try:
        pd.DataFrame(
            {
                "link_idx": np.asarray(keep_indices, dtype=int),
                "objective_label": [objective_label] * int(len(keep_indices)),
            }
        ).to_csv(obs_links_out, index=False)
        print(f"Saved selected observation links to: {obs_links_out}")
    except Exception as exc:
        print(
            f"Warning: failed to save selected observation links ({obs_links_out}): {exc}"
        )

    print(
        f"  optimize: beta={optimize_beta}, u={optimize_u}, "
        f"kappa={optimize_kappa}, merge_priority={optimize_mp}"
    )

    use_param_reparam = bool(config["USE_PARAM_REPARAM"])
    beta_scale = float(config["PARAM_BETA_SCALE"])
    u_scale = float(config["PARAM_U_SCALE"])
    kappa_scale = float(config["PARAM_KAPPA_SCALE"])
    mp_scale = float(config["PARAM_MP_SCALE"])
    beta_min = float(config["PARAM_BETA_MIN"])
    u_min = float(config["PARAM_U_MIN"])
    kappa_min = float(config["PARAM_KAPPA_MIN"])
    kappa_max = float(config["PARAM_KAPPA_MAX"])
    mp_min = float(config["PARAM_MP_MIN"])
    if kappa_max <= kappa_min:
        kappa_max = kappa_min + 1e-6
        print(
            f"WARNING: PARAM_KAPPA_MAX <= PARAM_KAPPA_MIN. "
            f"Adjusted PARAM_KAPPA_MAX to {kappa_max:.6f}"
        )
    print(
        f"Param reparam: enabled={use_param_reparam}, "
        f"scales=(beta={beta_scale}, u={u_scale}, kappa={kappa_scale}, mp={mp_scale}), "
        f"mins=(beta={beta_min}, u={u_min}, kappa={kappa_min}, mp={mp_min}), "
        f"kappa_max={kappa_max}"
    )

    def _to_physical(raw, scale, minv):
        if not use_param_reparam:
            return raw
        return jnp.maximum(raw, minv)

    def _to_raw(phys, scale, minv):
        return phys

    def _to_physical_kappa(raw):
        if not use_param_reparam:
            return raw
        return jnp.clip(raw, kappa_min, kappa_max)

    def _to_raw_kappa(phys):
        return phys

    param_mask = []
    reduced_init = []
    if optimize_beta:
        param_mask.append(("beta", jnp.arange(n_links)))
        reduced_init.append(_to_raw(params_init[:n_links], beta_scale, beta_min))
    if optimize_u:
        param_mask.append(("u", jnp.arange(n_links)))
        reduced_init.append(
            _to_raw(params_init[n_links : 2 * n_links], u_scale, u_min)
        )
    if optimize_kappa:
        param_mask.append(("kappa", jnp.arange(n_links)))
        reduced_init.append(_to_raw_kappa(params_init[2 * n_links : 3 * n_links]))
    if optimize_mp:
        param_mask.append(("merge_priority", jnp.arange(n_links)))
        reduced_init.append(
            _to_raw(params_init[3 * n_links : 4 * n_links], mp_scale, mp_min)
        )

    reduced_params_init = (
        jnp.concatenate(reduced_init) if reduced_init else jnp.array([])
    )
    n_reduced = reduced_params_init.shape[0]
    print(
        f"  Optimization parameters: {n_reduced} / {4 * n_links} "
        f"(excluded {4 * n_links - n_reduced} fixed params)"
    )

    def reconstruct_params(reduced_params):
        beta_out = beta_b
        u_out = u_b
        kappa_out = kappa_b
        mp_out = merge_priority_b
        offset = 0
        for ptype, indices in param_mask:
            n_p = indices.shape[0]
            slc = reduced_params[offset : offset + n_p]
            if ptype == "beta":
                beta_out = _to_physical(slc, beta_scale, beta_min)
            elif ptype == "u":
                u_out = _to_physical(slc, u_scale, u_min)
            elif ptype == "kappa":
                kappa_out = _to_physical_kappa(slc)
            elif ptype == "merge_priority":
                mp_out = _to_physical(slc, mp_scale, mp_min)
            offset += n_p
        return beta_out, u_out, kappa_out, mp_out

    def loss_and_cums_fn(rng_key, reduced_params):
        beta, u, kappa_param, mp_param = reconstruct_params(reduced_params)
        cum_f, cums = forward_and_count_simple(
            ctx,
            rng_key,
            x0,
            n_steps,
            beta,
            u,
            kappa_param,
            mp_param,
            obs_step,
        )
        resid = (cums - obs_cum) / out_scale
        resid2_masked = (resid ** 2) * obs_link_mask_2d
        J_obs = 0.5 * (jnp.sum(resid2_masked) / obs_norm_denom)
        loss = J_obs
        return loss, (cums, J_obs, None, None, None, None)

    AD_MODE = str(config["AD_MODE"])
    print(f"Automatic differentiation mode: {AD_MODE}")
    if AD_MODE != "reverse":
        raise ValueError(f"Unknown AD mode: {AD_MODE}")

    if bool(config["DEBUG_VJP"]):
        print("DEBUG_VJP mode: enabled")

        def grad_fn(rng_key, params):
            t0 = perf_counter()
            loss, aux = loss_and_cums_fn(rng_key, params)
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None,
                (loss, aux),
            )
            t1 = perf_counter()

            def scalar_loss(p):
                return loss_and_cums_fn(rng_key, p)[0]

            g = jax.grad(scalar_loss)(params)
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None,
                g,
            )
            t2 = perf_counter()
            try:
                print(
                    f"DEBUG_VJP reverse: forward_time={t1 - t0:.4f}s, "
                    f"pullback_time={t2 - t1:.4f}s"
                )
            except Exception:
                pass
            return (loss, aux), g

    else:
        grad_fn = jax.value_and_grad(loss_and_cums_fn, argnums=1, has_aux=True)

    if method != "adam":
        print("Error")
        raise NotImplementedError(
            f"Only the 'adam' method is implemented; got '{method}'."
        )

    params_curr = reduced_params_init

    base_lr = float(config["ADAM_LR"])
    schedule_type = str(config["ADAM_LR_SCHEDULE"])
    decay_rate = float(config["ADAM_LR_DECAY_RATE"])
    decay_steps = int(config["ADAM_LR_DECAY_STEPS"])
    warmup_steps = int(config["ADAM_LR_WARMUP_STEPS"])
    end_value = float(config["ADAM_LR_END_VALUE"])
    beta1 = float(config["ADAM_BETA1"])
    beta2 = float(config["ADAM_BETA2"])
    eps_adam = float(config["ADAM_EPS"])
    adam_optimizer = str(config["ADAM_OPTIMIZER"])
    adam_weight_decay = float(config["ADAM_WEIGHT_DECAY"])
    print(f"Adam optimizer type: {adam_optimizer}")

    learning_rate_schedule = create_learning_rate_schedule(
        base_lr=base_lr,
        n_iter=n_iter,
        schedule_type=schedule_type,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        end_value=end_value,
    )

    auto_lr = False
    print(f"AUTO_LR is set to {auto_lr}")
    if auto_lr:
        print(f"Auto-tuning learning rate (initial: {base_lr:.2e})")
        optimal_lr = find_optimal_lr(
            jax.jit(lambda params, key: loss_and_cums_fn(key, params)[0]),
            params_curr,
            key,
        )
        if optimal_lr > 0:
            base_lr = optimal_lr
            print(f"Updated base LR to {base_lr:.2e}")
            learning_rate_schedule = create_learning_rate_schedule(
                base_lr=base_lr,
                n_iter=n_iter,
                schedule_type=schedule_type,
                decay_rate=decay_rate,
                decay_steps=decay_steps,
                warmup_steps=warmup_steps,
                end_value=end_value,
            )

    if adam_optimizer == "adamw":
        optimizer = optax.adamw(
            learning_rate=learning_rate_schedule,
            b1=beta1,
            b2=beta2,
            eps=eps_adam,
            weight_decay=adam_weight_decay,
        )
    else:
        optimizer = optax.adam(
            learning_rate=learning_rate_schedule, b1=beta1, b2=beta2, eps=eps_adam
        )
    opt_state = optimizer.init(params_curr)

    if n_vmap_samples > 1:
        batched_grad_fn = jax.vmap(grad_fn, in_axes=[0, None], out_axes=0)
        rng = key
        print(
            f"Starting {method_name} joint assimilation: n_iter={n_iter}, "
            f"base_lr={base_lr}, schedule={schedule_type}, optimizer={adam_optimizer}, "
            f"weight_decay={adam_weight_decay}, n_vmap_samples={n_vmap_samples}, "
            f"max_grad_norm={max_grad_norm}"
        )
    else:
        loss_grad_fn = grad_fn if bool(config["DEBUG_VJP"]) else jax.jit(grad_fn)
        print(
            f"Starting {method_name} joint assimilation: n_iter={n_iter}, "
            f"base_lr={base_lr}, schedule={schedule_type}, optimizer={adam_optimizer}, "
            f"weight_decay={adam_weight_decay}"
        )

    no_improvement_count = 0
    for it in range(n_iter):
        if n_vmap_samples > 1:
            rng, subkey = random.split(rng)
            batched_keys = random.split(subkey, n_vmap_samples)
            (loss_vals, aux_vals), grad_vals = batched_grad_fn(batched_keys, params_curr)
            lval = float(jnp.mean(loss_vals))
            g = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grad_vals)
            aux_avg = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), aux_vals)
            cums_pred, J_obs, J_beta, J_u, J_kappa, J_mp = aux_avg

            grad_squared_sum = sum(jnp.sum(x ** 2) for x in tree_util.tree_leaves(g))
            global_norm = jnp.sqrt(grad_squared_sum)
            clip_coeff = jnp.minimum(1.0, max_grad_norm / (global_norm + 1e-6))
            g = tree_util.tree_map(lambda x: x * clip_coeff, g)
        else:
            (lval, (cums_pred, J_obs, J_beta, J_u, J_kappa, J_mp)), g = loss_grad_fn(
                key, params_curr
            )
            lval = float(lval)
            grad_squared_sum = sum(jnp.sum(x ** 2) for x in tree_util.tree_leaves(g))
            global_norm = jnp.sqrt(grad_squared_sum + 1e-12)
            clip_coeff = jnp.minimum(1.0, max_grad_norm / (global_norm + 1e-6))
            g = tree_util.tree_map(lambda x: x * clip_coeff, g)

        grad_norm = float(global_norm)
        cum_np = (
            np.array(cums_pred[-1])
            if cums_pred.shape[0] > 0
            else np.full((n_links,), np.nan)
        )
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None,
            (cums_pred, g),
        )
        elapsed = perf_counter() - start_time
        iter_time = elapsed - (time_hist[-1] if time_hist else 0.0)
        current_lr = learning_rate_schedule(it)

        print(
            f"  iter {it+1}/{n_iter}: loss={lval:.6f} "
            f"(obs={float(J_obs):.6f}, grad_norm={grad_norm:.6f}, "
            f"lr={current_lr:.6e}, time={elapsed:.2f}s, iter_time={iter_time:.3f}s"
        )

        beta_full, u_full, kappa_full, mp_full = reconstruct_params(params_curr)
        params_full = jnp.concatenate([beta_full, u_full, kappa_full, mp_full])
        loss_hist.append(lval)
        grad_norm_hist.append(grad_norm)
        params_hist.append(np.array(params_full))
        cum_hist.append(cum_np)
        time_hist.append(elapsed)

        if best_loss == float("inf"):
            best_loss = lval
            no_improvement_count = 0
        else:
            min_improve = max(loss_tol, loss_rel_tol * abs(best_loss))
            if lval < best_loss - min_improve:
                best_loss = lval
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(
                        f"  Converged (no improvement for {patience} steps) "
                        f"at iteration {it+1}"
                    )
                    break

        updates, opt_state = optimizer.update(g, opt_state, params_curr)
        params_curr = optax.apply_updates(params_curr, updates)

        if grad_norm < grad_tol:
            print(
                f"  Converged (grad_norm) at iteration {it+1}: "
                f"{grad_norm:.3e} < {grad_tol:.3e}"
            )
            break

    if loss_hist:
        best_idx = int(np.argmin(loss_hist))
        params_est = jnp.asarray(params_hist[best_idx], dtype=float)
        print(
            f"  Using parameters from best iteration {best_idx+1} "
            f"with loss={loss_hist[best_idx]:.6f}"
        )
    else:
        beta_full, u_full, kappa_full, mp_full = reconstruct_params(params_curr)
        params_est = jnp.concatenate([beta_full, u_full, kappa_full, mp_full])

    final_lr = (
        learning_rate_schedule(len(loss_hist) - 1) if loss_hist else base_lr
    )
    print(f"{method_name} optimization completed:")
    print(f"  Final loss: {lval:.6f}")
    print(f"  Final grad_norm: {grad_norm:.6f}")
    print(f"  Iterations: {len(loss_hist)}")
    print(
        f"  Initial LR: {base_lr:.6e}, Final LR: {final_lr:.6e}, "
        f"Schedule: {schedule_type}"
    )

    loss_df = pd.DataFrame(
        {
            "iter": range(len(loss_hist)),
            "loss": loss_hist,
            "grad_norm": grad_norm_hist,
            "time": time_hist,
        }
    )
    loss_df.to_csv(
        os.path.join(save_dir, f"assim_joint_loss_history{suffix}.csv"), index=False
    )

    cum_df = pd.DataFrame(
        np.stack(cum_hist) if cum_hist else np.zeros((0, n_links))
    )
    cum_df.columns = [f"link_{i}" for i in range(n_links)]
    cum_df.to_csv(
        os.path.join(save_dir, f"assim_joint_cum_history{suffix}.csv"),
        index_label="iter",
    )

    obs_final = obs_cum[-1] if obs_cum.shape[0] > 0 else np.zeros((n_links,))
    pd.DataFrame(
        [obs_final], columns=[f"link_{i}" for i in range(n_links)]
    ).to_csv(
        os.path.join(save_dir, f"assim_joint_obs_cum{suffix}.csv"), index=False
    )

    if params_hist:
        params_array = np.stack(params_hist)
        params_dict = {}
        for i in range(n_links):
            params_dict[f"beta_{i}"] = params_array[:, i]
        for i in range(n_links):
            params_dict[f"u_{i}"] = params_array[:, n_links + i]
        for i in range(n_links):
            params_dict[f"kappa_{i}"] = params_array[:, 2 * n_links + i]
        for i in range(n_links):
            params_dict[f"merge_priority_{i}"] = params_array[:, 3 * n_links + i]
        pd.DataFrame(params_dict).to_csv(
            os.path.join(save_dir, f"assim_joint_params_history{suffix}.csv"),
            index=False,
        )
        print(
            f"Saved parameter history to: assim_joint_params_history{suffix}.csv"
        )

    print(f"Saved histories to: {save_dir}/assim_joint_*{suffix}.csv")
    return params_est