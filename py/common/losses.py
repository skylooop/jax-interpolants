"""
Nicholas M. Boffi
6/19/25

Loss functions for learning.
"""

import functools
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import interpolant as interpolant

Parameters = dict[str, dict]


def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return jnp.mean(batched_outputs)

    return wrapper


def loss(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    model: nn.Module,
    loss_type: str,
) -> float:
    """Loss for learning the drift field b."""

    # compute interpolant and the target
    It = interp.calc_It(t, x0, x1)
    target = interp.calc_target(t, x0, x1, loss_type)

    # compute the weighted loss
    bt = model.apply(params, t, It, label, train=True, rngs=rng)
    loss = jnp.sum((bt - target) ** 2)
    weight = model.apply(params, t, method="calc_weight")
    return jnp.exp(-weight) * loss + weight


def setup_loss(
    cfg: config_dict.ConfigDict,
    model: nn.Module,
    interp: interpolant.Interpolant,
) -> Callable:
    """Setup the loss functions."""

    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def empirical_loss(
        params: Parameters,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray,
        t: float,
        rng: jnp.ndarray,
    ) -> float:
        """Compute the empirical loss."""
        return loss(
            params,
            x0,
            x1,
            label,
            t,
            rng,
            interp=interp,
            model=model,
            loss_type=cfg.training.loss_type,
        )

    return empirical_loss
