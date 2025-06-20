"""
Nicholas M. Boffi
6/19/25

Code for setting up arguments for loss functions.
"""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import state_utils
from . import dist_utils


@functools.partial(jax.jit, static_argnums=(1, 2))
def get_loss_fn_args_randomness(
    prng_key: jnp.ndarray,
    cfg: config_dict.ConfigDict,
    sample_rho0: Callable,
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    # get needed random keys
    (
        tkey,
        x0key,
    ) = jax.random.split(prng_key, num=2)

    # sample points from base distribution
    x0batch = sample_rho0(cfg.optimization.bs, x0key)

    # sample time points
    tbatch = jax.random.uniform(
        tkey,
        shape=(cfg.optimization.bs,),
        minval=cfg.training.tmin,
        maxval=cfg.training.tmax,
    )

    # draw keys for model dropout
    dropout_keys = jax.random.split(tkey, num=cfg.optimization.bs).reshape(
        (cfg.optimization.bs, -1)
    )

    # refresh the random key
    prng_key = jax.random.split(dropout_keys[0])[0]
    return (
        tbatch,
        x0batch,
        dropout_keys,
        prng_key,
    )


def get_batch(
    cfg: config_dict.ConfigDict, statics: state_utils.StaticArgs, prng_key: jnp.ndarray
) -> int:
    """Extract a batch based on the structure expected for image or non-image datasets."""
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

    # if not conditional, we don't need the labels
    if not cfg.training.conditional:
        label_batch = None
    # add droput to randomly replace fraction cfg.class_dropout of labels by num_classes
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
    prng_key: jnp.ndarray,
) -> Tuple:

    # draw randomness needed for the objective
    (
        tbatch,
        x0batch,
        dropout_keys,
        prng_key,
    ) = get_loss_fn_args_randomness(prng_key, cfg, statics.sample_rho0)

    # grab next batch of samples and labels
    x1batch, label_batch, prng_key = get_batch(cfg, statics, prng_key)

    # for training flow map
    loss_fn_args = (
        x0batch,
        x1batch,
        label_batch,
        tbatch,
        dropout_keys,
    )
    loss_fn_args = dist_utils.replicate_loss_fn_args(cfg, loss_fn_args)

    return loss_fn_args, prng_key
