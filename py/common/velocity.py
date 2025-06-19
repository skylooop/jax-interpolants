"""
Nicholas M. Boffi
6/29/25

Basic routines for sampling from a flow model.
"""

import functools
from typing import Callable, Dict

import jax
import jax.numpy as jnp


Parameters = Dict[str, Dict]


def sample_euler(
    apply_velocity: Callable, variables: Dict, x0: jnp.ndarray, N: int, label: int
) -> jnp.ndarray:
    """Euler integration."""
    ts = jnp.linspace(0.0, 1.0, N + 1)
    dts = jnp.diff(ts)

    def step(x, idx):
        return (
            x
            + dts[idx]
            * apply_velocity(variables, ts[idx], x, label=label, train=False),
            None,
        )

    final_state, _ = jax.lax.scan(step, x0, jnp.arange(N))
    return final_state


def sample(
    apply_velocity: Callable,
    variables: Dict,
    x0: jnp.ndarray,
    N: int,
    label: int,
) -> jnp.ndarray:
    """
    Second-order Heun (explicit trapezoidal) integrator for dx/dt = v(t, x).
    """
    ts = jnp.linspace(0.0, 1.0, N + 1, dtype=x0.dtype)  # shape (N+1,)
    dts = jnp.diff(ts)  # shape (N,)

    def step(x, idx):
        t = ts[idx]
        dt = dts[idx]

        v0 = apply_velocity(variables, t, x, label=label, train=False)
        x_pred = x + dt * v0  # Euler predictor
        v1 = apply_velocity(variables, t + dt, x_pred, label=label, train=False)

        x_next = x + 0.5 * dt * (v0 + v1)  # trapezoidal corrector
        return x_next, None

    x_T, _ = jax.lax.scan(step, x0, jnp.arange(N))
    return x_T


@functools.partial(jax.jit, static_argnums=(0, 3))
@functools.partial(jax.vmap, in_axes=(None, None, 0, None, 0))
def batch_sample(
    apply_velocity, variables: Dict, x0s: jnp.ndarray, N: int, label: int
) -> jnp.ndarray:
    """Batch unconditional sampling."""
    return sample(apply_velocity, variables, x0s, N, label)
