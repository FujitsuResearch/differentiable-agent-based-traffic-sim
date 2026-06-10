"""Learning rate schedules and a simple LR finder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def create_learning_rate_schedule(
    base_lr: float,
    n_iter: int,
    schedule_type: str = "constant",
    decay_rate: float = 0.96,
    decay_steps: int = 50,
    warmup_steps: int = 0,
    end_value: float = 1e-5,
):

    if schedule_type == "exponential":
        main_schedule = optax.exponential_decay(
            init_value=base_lr,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False,
        )
    elif schedule_type == "cosine":
        main_schedule = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=n_iter - warmup_steps,
            alpha=end_value / base_lr,
        )
    elif schedule_type == "polynomial":
        main_schedule = optax.polynomial_schedule(
            init_value=base_lr,
            end_value=end_value,
            power=1.0,
            transition_steps=n_iter - warmup_steps,
        )
    else:
        main_schedule = optax.constant_schedule(base_lr)

    if warmup_steps > 0:
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=base_lr,
            transition_steps=warmup_steps,
        )
        return optax.join_schedules([warmup, main_schedule], boundaries=[warmup_steps])
    return main_schedule


def find_optimal_lr(
    loss_fn,
    params_init,
    key,
    max_lr: float = 1.0,
    num_steps: int = 20,
    lr_mult: float = 1.2,
) -> float:

    print("Finding optimal initial learning rate...")

    lr = 1e-6 
    best_lr = lr
    min_loss = float("inf")
    losses = []
    lrs = []

    params = params_init

    for step in range(num_steps):
        try:
            loss_val = float(loss_fn(params, key))
        except Exception as exc:
            print(f"Error evaluating loss at LR {lr}: {exc}")
            break

        losses.append(loss_val)
        lrs.append(lr)

        if loss_val < min_loss:
            min_loss = loss_val
            best_lr = lr

        if step > 2 and loss_val > 1.5 * min_loss:
            print(f"Loss increasing at LR {lr:.2e}, stopping LR finder")
            break

        try:
            grad = jax.grad(loss_fn)(params, key)
            params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grad)
            params = jnp.maximum(params, 0.0)
        except Exception as exc:
            print(f"Error in gradient step at LR {lr}: {exc}")
            break

        lr = min(lr * lr_mult, max_lr)

    if len(losses) > 1:
        for i in range(1, len(losses)):
            if losses[i] > losses[i - 1]:
                optimal_lr = lrs[i - 1]
                print(
                    f"Optimal LR found: {optimal_lr:.2e} "
                    f"(loss: {losses[i-1]:.6f})"
                )
                return optimal_lr

    fallback_lr = best_lr
    print(f"Using fallback LR: {fallback_lr:.2e}")
    return fallback_lr
