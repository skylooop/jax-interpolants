"""
Nicholas M. Boffi
3/20/25

Loss functions for learning.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import flow_map as flow_map
from . import interpolant as interpolant
from . import velocity as velocity

Parameters = Dict[str, Dict]


def sum_reduce(func):
    """
    A decorator that computes the sum of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return jnp.sum(batched_outputs)

    return wrapper


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


def diagonal_term(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Compute the diagonal (interpolant) term of the loss."""

    # compute interpolant and the target
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)

    # compute the weighted loss
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)

    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def pfmm_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    u: float,
    h: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    pfmm_type: str,
    stopgrad_type: str,
) -> float:
    """Compute the PFMM term of the loss. Note usage of fixed dropout pattern."""
    Is = interp.calc_It(s, x0, x1)

    # compute the full jump
    X_st, phi_st = X.apply(
        params, s, t, Is, label, train=True, rngs=rng, return_X_and_phi=True
    )

    # break it down into two jumps
    if stopgrad_type == "convex":
        X_su, phi_su = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                u,
                Is,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )

        X_ut, phi_ut = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                u,
                t,
                X_su,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )
    elif stopgrad_type == "none":
        X_su, phi_su = X.apply(
            params,
            s,
            u,
            Is,
            label,
            train=True,
            rngs=rng,
            return_X_and_phi=True,
        )

        X_ut, phi_ut = X.apply(
            params,
            u,
            t,
            X_su,
            label,
            train=True,
            rngs=rng,
            return_X_and_phi=True,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    if pfmm_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut
    elif pfmm_type == "lmd":
        student = X_st
        teacher = X_ut
    elif pfmm_type == "shortcut":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)

    pfmm_loss = jnp.sum((student - teacher) ** 2)

    if pfmm_type == "lmd":
        pfmm_loss /= h**2

    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * pfmm_loss + weight_st


def lsd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
    rescale_lsd: bool,
) -> float:
    """Compute the LSD term of the loss. Note iusage of same dropout pattern."""
    Is = interp.calc_It(s, x0, x1)

    # now compute the "distillation" loss
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=True, method="partial_t", rngs=rng
    )

    # compute the self teacher at the evaluation point
    if stopgrad_type == "convex":
        Xst_Is = jax.lax.stop_gradient(Xst_Is)
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                t,
                Xst_Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
    else:
        if stopgrad_type == "lmd":
            b_eval = X.apply(
                teacher_params,
                t,
                Xst_Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        elif stopgrad_type == "none":
            b_eval = X.apply(
                params,
                t,
                Xst_Is,
                label,
                train=True,
                method="calc_b",
                rngs=rng,
            )
        else:
            raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    weight_st = X.apply(params, s, t, method="calc_weight")

    if rescale_lsd:
        error = (b_eval - dt_Xst) / (t - s)
    else:
        error = b_eval - dt_Xst

    lsd_loss = jnp.sum(error**2)
    return jnp.exp(-weight_st) * lsd_loss + weight_st


def csd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Compute the LSD term of the loss. Note iusage of same dropout pattern."""
    Is = interp.calc_It(s, x0, x1)

    # compute the derivative with respect to the first time
    _, ds_Xst = X.apply(
        params, s, t, Is, label, train=True, method="partial_s", rngs=rng
    )

    # compute the self teacher at the evaluation point
    if stopgrad_type == "convex":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            s,
            Is,
            label,
            train=True,
            method="calc_b",
            rngs=rng,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    # compute the advective term
    _, grad_Xst_b = jax.jvp(
        lambda x: X.apply(params, s, t, x, label, train=True, rngs=rng),
        primals=(Is,),
        tangents=(b_eval,),
    )

    csd_loss = jnp.sum((ds_Xst + grad_Xst_b) ** 2)
    weight_st = X.apply(params, s, t, method="calc_weight")

    return jnp.exp(-weight_st) * csd_loss + weight_st


def self_distill(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    u: float,
    h: float,
    key: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    calc_both_diagonals: bool,
    stopgrad_type: str,
    loss_type: str,
    rescale_lsd: bool,
    pfmm_type: str,
) -> float:
    """Self student-teacher approach."""
    rng = {"dropout": key}

    vel_loss = diagonal_term(
        params,
        x0,
        x1,
        label,
        t,
        rng,
        interp=interp,
        X=X,
    )

    if calc_both_diagonals:
        # compute the diagonal term for the second time
        vel_loss += diagonal_term(
            params,
            x0,
            x1,
            label,
            s,
            rng,
            interp=interp,
            X=X,
        )
        vel_loss /= 2.0

    if loss_type == "pfmm":
        distill_loss = pfmm_term(
            params,
            teacher_params,
            x0,
            x1,
            label,
            s,
            t,
            u,
            h,
            rng,
            interp=interp,
            X=X,
            pfmm_type=pfmm_type,
            stopgrad_type=stopgrad_type,
        )
    elif loss_type == "lsd":
        distill_loss = lsd_term(
            params,
            teacher_params,
            x0,
            x1,
            label,
            s,
            t,
            rng,
            interp=interp,
            X=X,
            stopgrad_type=stopgrad_type,
            rescale_lsd=rescale_lsd,
        )
    elif loss_type == "csd":
        distill_loss = csd_term(
            params,
            teacher_params,
            x0,
            x1,
            label,
            s,
            t,
            rng,
            interp=interp,
            X=X,
            stopgrad_type=stopgrad_type,
        )

    return vel_loss + distill_loss


def setup_loss(
    cfg: config_dict.ConfigDict, net: flow_map.FlowMap, interp: interpolant.Interpolant
) -> Tuple[Callable, Callable]:
    """Setup the loss functions."""

    print(f"Setting up loss: {cfg.training.loss_type}")
    print(f"Stopgrad type: {cfg.training.stopgrad_type}")

    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))
    def loss(params, teacher_params, x0, x1, label, s, t, u, h, dropout_keys):
        return self_distill(
            params,
            teacher_params,
            x0,
            x1,
            label,
            s,
            t,
            u,
            h,
            dropout_keys,
            interp=interp,
            X=net,
            calc_both_diagonals=cfg.training.calc_both_diagonals,
            stopgrad_type=cfg.training.stopgrad_type,
            loss_type=cfg.training.loss_type,
            rescale_lsd=cfg.training.rescale_lsd,
            pfmm_type=cfg.training.pfmm_type,
        )

    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def interp_loss(params, x0, x1, label, t, rng):
        return diagonal_term(
            params,
            x0,
            x1,
            label,
            t,
            rng,
            interp=interp,
            X=net,
        )

    return loss, interp_loss
