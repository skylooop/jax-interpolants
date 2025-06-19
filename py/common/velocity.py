"""
Nicholas M. Boffi
4/25/25

Basic routines for velocity class.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from ml_collections import config_dict

from . import edm2_net, network_utils

Parameters = Dict[str, Dict]


class Velocity(nn.Module):
    """Basic class for a velocity model.
    Uses a two-time network for direct comparison to flow map models.
    """

    config: config_dict.ConfigDict

    def setup(self):
        """Set up the flow map."""
        self.velocity = network_utils.setup_network(self.config)

    def __call__(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        """Apply the flow map."""
        return self.velocity(t, t, x, label, train, calc_weight)

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        """Compute the b term for the velocity."""
        return self.velocity.calc_b(t, x, label, train, calc_weight)

    def calc_weight(self, s: float, t: float) -> jnp.ndarray:
        """Compute the weights for the flow map."""
        del s
        return self.velocity.calc_weight(t, t)


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


def initialize_velocity(
    network_config: config_dict.ConfigDict, ex_input: jnp.ndarray, prng_key: jnp.ndarray
) -> Tuple[nn.Module, Parameters, jnp.ndarray]:
    # define the network
    net = Velocity(network_config)

    # initialize the parameters
    ex_t = 0.0
    ex_label = 0
    params = net.init(
        {"params": prng_key},
        ex_t,
        ex_input,
        ex_label,
        train=False,
        calc_weight=True,
    )
    prng_key = jax.random.split(prng_key)[0]

    print(f"Number of parameters: {jax.flatten_util.ravel_pytree(params)[0].size}")

    if network_config.network_type == "edm2":
        params = edm2_net.project_to_sphere(params)

    return net, params, prng_key
