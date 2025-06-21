"""
Nicholas M. Boffi
6/19/25

Standardized stochastic interpolant implementation.
"""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from ml_collections import config_dict


@dataclasses.dataclass
class Interpolant:
    """Basic class for a stochastic interpolant, following the mathematical
    description in https://arxiv.org/abs/2303.08797.
    Assumes that the base distribution is a Gaussian, avoiding the need for
    the \\gamma term.
    """

    alpha: Callable[[jnp.ndarray | float], jnp.ndarray | float]
    beta: Callable[[jnp.ndarray | float], jnp.ndarray | float]
    alpha_dot: Callable[[jnp.ndarray | float], jnp.ndarray | float]
    beta_dot: Callable[[jnp.ndarray | float], jnp.ndarray | float]

    def calc_It(
        self, t: jnp.ndarray | float, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_It_dot(
        self, t: jnp.ndarray | float, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    def calc_target(
        self, t: jnp.ndarray | float, x0: jnp.ndarray, x1: jnp.ndarray, target_type: str
    ) -> jnp.ndarray:
        """Compute the target for learning."""
        if target_type == "velocity":
            return self.calc_It_dot(t, x0, x1)
        elif target_type == "score":
            return -x0 / self.alpha(t)
        elif target_type == "noise":
            return x0
        elif target_type == "denoiser":
            return x1
        else:
            raise ValueError(f"Target type {target_type} not recognized.")

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It(
        self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return self.calc_It(t, x0, x1)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It_dot(
        self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return self.calc_It_dot(t, x0, x1)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
    def batch_calc_target(
        self,
        t: jnp.ndarray,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        target_type: str,
    ) -> jnp.ndarray:
        return self.calc_target(t, x0, x1, target_type)

    def __hash__(self):
        return hash((self.alpha, self.beta))

    def __eq__(self, other):
        return self.alpha == other.alpha and self.beta == other.beta


def setup_interpolant(cfg: config_dict.ConfigDict) -> Interpolant:
    if cfg.problem.interp_type == "linear":
        interp = Interpolant(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )
    elif cfg.problem.interp_type == "trig":
        interp = Interpolant(
            alpha=lambda t: jnp.cos(jnp.pi * t / 2),
            beta=lambda t: jnp.sin(jnp.pi * t / 2),
            alpha_dot=lambda t: -0.5 * jnp.pi * jnp.sin(jnp.pi * t / 2),
            beta_dot=lambda t: 0.5 * jnp.pi * jnp.cos(jnp.pi * t / 2),
        )
    elif cfg.problem.interpolant_type == "vp_diffusion":
        return Interpolant(
            alpha=lambda t: jnp.sqrt(1 - jnp.exp(2 * (t - cfg.problem.tmax))),
            beta=lambda t: jnp.exp(t - cfg.problem.tmax),
            alpha_dot=lambda t: -jnp.exp(2 * (t - cfg.problem.tmax))
            / jnp.sqrt(1 - jnp.exp(2 * (t - cfg.problem.tmax))),
            beta_dot=lambda t: jnp.exp(t - cfg.problem.tmax),
        )

    elif cfg.problem.interpolant_type == "vp_diffusion_logscale":
        return Interpolant(
            alpha=lambda t: jnp.sqrt(1 - t**2),
            beta=lambda t: t,
            alpha_dot=lambda t: -t / jnp.sqrt(1 - t**2),
            beta_dot=lambda t: 1,
        )

    elif cfg.problem.interpolant_type == "ve_diffusion":
        return Interpolant(
            alpha=lambda t: cfg.problem.tf - t,
            beta=lambda t: 1,
            alpha_dot=lambda t: -1,
            beta_dot=lambda t: 0,
        )

    else:
        raise ValueError(f"Interpolant type {cfg.interpolant_type} not recognized.")

    return interp
