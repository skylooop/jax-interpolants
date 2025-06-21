"""
Nicholas M. Boffi
6/19/25

Code for initializing common datasets.
"""

import functools
from collections.abc import Callable

import tensorflow as tf

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from ml_collections import config_dict

from . import gmm as gmm


def unnormalize_image(image: jnp.ndarray):
    """Unnormalize an image from [-1, 1] to [0, 1] by scaling and clipping."""
    image = (image + 1) / 2
    image = jnp.clip(image, 0.0, 1.0)
    return image


def normalize_image_tf(image: tf.Tensor):
    """Normalize an image to have pixel values in the range [-1, 1]."""
    return (2 * (image / 255)) - 1


def preprocess_image(cfg, x: dict) -> dict:
    """Preprocess the image for TensorFlow datasets."""
    image = x["image"]
    label = x["label"]
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    if cfg.problem.target == "mnist" or cfg.problem.target == "cifar10":
        image = normalize_image_tf(image)
    else:
        raise ValueError("Unknown dataset type.")

    # ensure (N, C, H, W)
    image = tf.transpose(image, [2, 0, 1])

    return {"image": image, "label": label}


def get_image_dataset(cfg: config_dict.ConfigDict):
    """Assemble a TensorFlow dataset for the specified problem target."""

    if cfg.problem.target == "mnist":
        ds = tfds.load(
            "mnist",
            split="train",
            shuffle_files=True,
            data_dir=cfg.problem.dataset_location,
        )
    elif cfg.problem.target == "cifar10":
        ds = tfds.load(
            "cifar10",
            split="train",
            shuffle_files=True,
            data_dir=cfg.problem.dataset_location,
        )
    else:
        raise ValueError("Specified target dataset is not implemented.")

    ds = (
        ds.shard(num_shards=cfg.training.n_nodes, index=jax.process_index())
        .shuffle(10_000, reshuffle_each_iteration=True)
        .map(
            lambda x: preprocess_image(cfg, x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    return ds


def sample_checkerboard(
    n_samples: int, key: jnp.ndarray, *, n_squares: int
) -> np.ndarray:
    """
    Samples the checkerboard dataset on [-1,1] x [-1,1]
    with alternating squares removed.
    """
    del key
    total_samples = 0
    samples = np.array([]).reshape((0, 2))

    while total_samples < n_samples:
        # Generate uniform samples on unit square
        curr_samples = np.random.rand(
            n_samples * 2, 2
        )  # Generate extra to account for filtering

        # Determine which square each point falls into
        x_idx = (curr_samples[:, 0] * n_squares).astype(int)
        y_idx = (curr_samples[:, 1] * n_squares).astype(int)

        # Keep points that fall in "white squares" of checkerboard
        mask = (x_idx + y_idx) % 2 == 0
        curr_samples = curr_samples[mask]

        # Take only what we need
        samples = np.concatenate((samples, curr_samples))
        total_samples = samples.shape[0]

    return 2 * samples[:n_samples] - 1


def setup_base(cfg: config_dict.ConfigDict, ex_input: jnp.ndarray) -> Callable:
    """Set up the base density for the system."""
    if cfg.problem.base == "gaussian":

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: jnp.ndarray):
            return cfg.network.rescale * jax.random.normal(
                key, shape=(bs, *ex_input.shape)
            )

    else:
        raise ValueError("Specified base density is not implemented.")

    return sample_rho0


def np_to_tfds(cfg: config_dict.ConfigDict, x1s: np.ndarray) -> tf.data.Dataset:
    """Given a NumPy array, convert to a TensorFlow dataset with batching and shuffling."""
    return (
        tf.data.Dataset.from_tensor_slices(x1s)
        .shuffle(50_000, reshuffle_each_iteration=True)
        .repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )


def setup_target(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray):
    """Set up the target density for the system."""
    if "gmm" in cfg.problem.target:
        weights, means, covs = gmm.setup_gmm(cfg.problem.target, cfg.problem.d)
        sample_rho1 = functools.partial(
            gmm.sample_gmm, weights=weights, means=means, covariances=covs
        )
        n_samples = cfg.problem.n
        keys = jax.random.split(prng_key, num=(cfg.problem.n + 1))
        x1s = sample_rho1(n_samples, keys)
        rescale_value = float(np.std(x1s))
        ds = np_to_tfds(cfg, x1s)
        prng_key = jax.random.split(keys[-1])[0]

    elif cfg.problem.target == "checker":
        assert cfg.problem.d == 2, "Checkerboard only implemented for d=2."

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho1(num_samples: int, key: jnp.ndarray) -> jnp.ndarray:
            return sample_checkerboard(num_samples, key, n_squares=4)

        n_samples = cfg.problem.n
        key, prng_key = jax.random.split(prng_key)
        x1s = sample_rho1(n_samples, key)
        rescale_value = float(np.std(x1s))
        ds = np_to_tfds(cfg, x1s)

    elif (
        cfg.problem.target == "mnist"
        or cfg.problem.target == "cifar10"
        or "imagenet" in cfg.problem.target
    ):
        ds = get_image_dataset(cfg)
        print("Loaded image dataset.")

    else:
        raise ValueError("Specified target density is not implemented.")

    # compute standard deviation of the dataset
    if cfg.problem.gaussian_scale == "adaptive":
        # std on MNIST and CIFAR10 \approx 0.5 -- just hard-code
        if cfg.problem.target == "mnist" or cfg.problem.target == "cifar10":
            rescale_value = 0.5

        # imagenet already rescaled to have mean zero and channelwise std 1
        elif "imagenet" in cfg.problem.target:
            rescale_value = 1.0

        # for generated datasets, it's computed above
        cfg.network.rescale = rescale_value
    else:
        cfg.network.rescale = 1.0

    return cfg, ds, prng_key
