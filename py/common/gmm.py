"""
Nicholas M. Boffi
6/19/25

Simple class for a Gaussian mixture model.
Useful for running synthetic generative modeling experiments.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnums=0)
def sample_gmm(
    num_samples: int,
    keys: jnp.ndarray,
    *,
    weights: jnp.ndarray,  # [num_components]
    means: jnp.ndarray,  # [num_components, d]
    covariances: jnp.ndarray,  # [num_components, d, d]
) -> jnp.ndarray:
    """Sample from a Gaussian mixture model."""

    num_components = weights.size
    key1, rest = keys[0], keys[1:]

    component_indices = jax.random.choice(
        key1, a=num_components, p=weights, shape=(num_samples,)
    )

    samples = jax.vmap(jax.random.multivariate_normal, in_axes=(0, 0, 0))(
        rest, means[component_indices], covariances[component_indices]
    )

    return samples


def setup_gmm(gmm_type: str, d: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Set up some basic mixture models."""

    if gmm_type == "std_normal_gmm":
        weights = jnp.array([1.0])
        means = jnp.zeros((1, d))
        covariances = jnp.zeros((1, d, d))
        covariances[0] = jnp.eye(d)
        covariances = jnp.array(covariances)

    elif gmm_type == "basic_gmm":
        weights = jnp.ones(1)
        means = jnp.ones((4, d))
        covariances = np.zeros((1, d, d))
        covariances[0] = np.eye(d)
        covariances = jnp.array(covariances)

    elif gmm_type == "flower_gmm":
        assert d == 2, "Flower GMM only works in 2D."
        num_components = 8
        weights = jnp.ones(num_components) / num_components
        means = np.zeros((num_components, d))
        covariances = np.zeros((num_components, d, d))
        for kk in range(num_components):
            means[kk] = jnp.array(
                [
                    5 * jnp.cos(2 * jnp.pi * kk / num_components),
                    5 * jnp.sin(2 * jnp.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * np.eye(d)

        means = jnp.array(means)
        covariances = jnp.array(covariances)

    elif gmm_type == "square_gmm":
        assert d == 2, "Square GMM only works in 2D."
        num_components = 4
        weights = jnp.ones(num_components) / num_components
        means = np.zeros((num_components, d))
        covariances = np.zeros((num_components, d, d))
        for kk in range(num_components):
            means[kk] = jnp.array(
                [
                    5 * jnp.cos(2 * jnp.pi * kk / num_components),
                    5 * jnp.sin(2 * jnp.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * np.eye(d)

        means = jnp.array(means)
        covariances = jnp.array(covariances)

    elif gmm_type == "line_gmm":
        assert d == 2, "Line GMM only works in 2D."
        num_components = 2
        weights = jnp.ones(num_components) / num_components
        means = np.zeros((num_components, d))
        covariances = np.zeros((num_components, d, d))
        for kk in range(num_components):
            means[kk] = jnp.array(
                [
                    5 * jnp.cos(2 * jnp.pi * kk / num_components),
                    5 * jnp.sin(2 * jnp.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * np.eye(d)

        means = jnp.array(means)
        covariances = jnp.array(covariances)

    else:
        raise ValueError(f"Invalid GMM type: {gmm_type}")

    return weights, means, covariances
