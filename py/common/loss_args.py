"""
Nicholas M. Boffi
3/20/25

Code for setting up arguments for loss functions.
"""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import state_utils
from . import dist_utils


def safe_resize(curr_bs: int, bs: int, x: jnp.ndarray) -> jnp.ndarray:
    """Resize the input array to the current batch size."""
    if curr_bs < bs:
        x = x[:curr_bs]
    return x


def sample_uniform_line(
    cfg: config_dict.ConfigDict,
    tkey: jnp.ndarray,
    skey: jnp.ndarray,
    delta: float,
) -> jnp.ndarray:
    """Sample uniformly from the line (s = t)."""
    del skey
    del delta

    tbatch = jax.random.uniform(
        tkey,
        shape=(cfg.optimization.bs,),
        minval=cfg.training.tmin,
        maxval=cfg.training.tmax,
    )

    return tbatch, tbatch


def sample_uniform_triangle(
    cfg: config_dict.ConfigDict,
    tkey: jnp.ndarray,
    skey: jnp.ndarray,
    delta: float,
) -> Tuple:
    """Sample uniformly from the triangle (s <= t).
    Also allow clamping for annealing stage.
    """
    temp_batch_1 = jax.random.uniform(
        tkey,
        shape=(cfg.optimization.bs,),
        minval=cfg.training.tmin,
        maxval=cfg.training.tmax,
    )
    temp_batch_2 = jax.random.uniform(
        skey,
        shape=(cfg.optimization.bs,),
        minval=cfg.training.tmin,
        maxval=cfg.training.tmax,
    )
    sbatch = jnp.minimum(temp_batch_1, temp_batch_2)
    tbatch = jnp.maximum(temp_batch_1, temp_batch_2)
    sbatch = jnp.maximum(sbatch, tbatch - delta)

    return sbatch, tbatch


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def get_loss_fn_args_randomness(
    prng_key: jnp.ndarray,
    curr_iter: int,
    cfg: config_dict.ConfigDict,
    sample_rho0: Callable,
    anneal_schedule: Callable,
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    (
        tkey,
        skey,
        ukey,
        x0key,
    ) = jax.random.split(prng_key, num=4)
    x0batch = sample_rho0(cfg.optimization.bs, x0key)
    delta = anneal_schedule(curr_iter)
    operands = (cfg, tkey, skey, delta)
    sbatch, tbatch = jax.lax.cond(
        delta == 0,
        lambda: sample_uniform_line(*operands),
        lambda: sample_uniform_triangle(*operands),
    )

    if cfg.training.pfmm_type == "shortcut":
        ubatch = 0.5 * (sbatch + tbatch)
        hbatch = (tbatch - sbatch) / 2.0
    elif cfg.training.pfmm_type == "uniform":
        minval = 0.0
        maxval = 1.0

        hbatch = jax.random.uniform(
            ukey, shape=(cfg.optimization.bs,), minval=minval, maxval=maxval
        )

        ubatch = hbatch * sbatch + (1 - hbatch) * tbatch

    elif cfg.training.pfmm_type == "lmd":
        minval = cfg.training.hmin
        maxval = cfg.training.hmax

        hbatch = jax.random.uniform(
            ukey, shape=(cfg.optimization.bs,), minval=minval, maxval=maxval
        )

        ubatch = tbatch - hbatch
    elif cfg.training.pfmm_type == None:
        ubatch = None
        hbatch = None
    else:
        raise ValueError(f"Unknown pfmm_type: {cfg.training.pfmm_type}")

    dropout_keys = jax.random.split(tkey, num=cfg.optimization.bs).reshape(
        (cfg.optimization.bs, -1)
    )
    prng_key = jax.random.split(dropout_keys[0])[0]
    return (
        tbatch,
        sbatch,
        ubatch,
        hbatch,
        x0batch,
        dropout_keys,
        prng_key,
    )


def get_batch(
    cfg: config_dict.ConfigDict, statics: state_utils.StaticArgs, prng_key: jnp.ndarray
) -> int:
    """Extract a batch based on the structure expected for image
    or non-image datasets."""
    is_image_dataset = ("imagenet" in cfg.problem.target) or (
        cfg.problem.target in ["mnist", "cifar10"]
    )

    batch = next(statics.ds)
    if is_image_dataset:
        x1batch = batch["image"]
        label_batch = batch["label"]
    else:
        x1batch = batch
        label_batch = None

    # add droput to randomly replace fraction cfg.class_dropout of labels by num_classes
    # if not conditional, we don't need the labels
    if not cfg.training.conditional:
        label_batch = None

    elif cfg.training.class_dropout > 0:
        assert cfg.network.use_cfg  # class dropout doesn't make sense without cfg
        mask = jax.random.bernoulli(
            prng_key, cfg.training.class_dropout, shape=(cfg.optimization.bs,)
        )
        mask = mask > 0
        label_batch = label_batch.at[mask].set(cfg.problem.num_classes)
        prng_key = jax.random.split(prng_key)[0]

    return x1batch, label_batch, prng_key


def get_loss_fn_args(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    force_return_diagonal_args: bool = False,
    force_return_full_args: bool = False,
) -> Tuple:

    # drew randomness needed for the objective
    (
        tbatch,
        sbatch,
        ubatch,
        hbatch,
        x0batch,
        dropout_keys,
        prng_key,
    ) = get_loss_fn_args_randomness(
        prng_key,
        dist_utils.safe_index(cfg, train_state.step),
        cfg,
        statics.sample_rho0,
        statics.anneal_schedule,
    )

    # if rescaling, make sure we dont fall below the numerical threshold
    if cfg.training.rescale_lsd:
        mask = (tbatch - sbatch) < cfg.training.min_step
        tbatch = tbatch.at[mask].set(sbatch[mask] + cfg.training.min_step)

    # grab next batch of samples and labels
    x1batch, label_batch, prng_key = get_batch(cfg, statics, prng_key)

    # set up the teacher
    if cfg.training.use_ema_teacher:
        teacher_params = train_state.ema_params[0.9999]
    else:
        teacher_params = train_state.params

    # for training interpolant alone
    diagonal_args = dist_utils.replicate_loss_fn_args(
        cfg, (x0batch, x1batch, label_batch, tbatch, dropout_keys)
    )

    # for training flow map
    full_loss_args = (
        x0batch,
        x1batch,
        label_batch,
        sbatch,
        tbatch,
        ubatch,
        hbatch,
        dropout_keys,
    )
    full_loss_args = dist_utils.replicate_loss_fn_args(cfg, full_loss_args)
    full_loss_args = (teacher_params, *full_loss_args)

    # switches to enable compilation
    if force_return_diagonal_args:
        loss_fn_args = diagonal_args
    elif force_return_full_args:
        loss_fn_args = full_loss_args
    else:
        if state_utils.use_velocity_loss(cfg, train_state):
            loss_fn_args = diagonal_args
        else:
            loss_fn_args = full_loss_args

    return loss_fn_args, prng_key
