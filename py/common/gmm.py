"""
Nicholas M. Boffi
4/12/24

Simple class for a Gaussian mixture model.
"""

import jax
import jax.numpy as np
import numpy as onp
from typing import Tuple
import functools


@functools.partial(jax.jit, static_argnums=0)
def sample_gmm(
    num_samples: int,
    keys: np.ndarray,
    *,
    weights: np.ndarray,  # [num_components]
    means: np.ndarray,  # [num_components, d]
    covariances: np.ndarray,  # [num_components, d, d]
) -> np.ndarray:
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


def setup_gmm(gmm_type: str, d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set up some basic mixture models."""

    if gmm_type == "std_normal_gmm":
        weights = np.array([1.0])
        means = np.zeros((1, d))
        covariances = np.zeros((1, d, d))
        covariances[0] = np.eye(d)
        covariances = np.array(covariances)

    elif gmm_type == "basic_gmm":
        weights = np.ones(1)
        means = np.ones((4, d))
        covariances = onp.zeros((1, d, d))
        covariances[0] = onp.eye(d)
        covariances = np.array(covariances)

    elif gmm_type == "flower_gmm":
        assert d == 2, "Flower GMM only works in 2D."
        num_components = 8
        weights = np.ones(num_components) / num_components
        means = onp.zeros((num_components, d))
        covariances = onp.zeros((num_components, d, d))
        for kk in range(num_components):
            means[kk] = np.array(
                [
                    5 * np.cos(2 * np.pi * kk / num_components),
                    5 * np.sin(2 * np.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * onp.eye(d)

        means = np.array(means)
        covariances = np.array(covariances)

    elif gmm_type == "square_gmm":
        assert d == 2, "Square GMM only works in 2D."
        num_components = 4
        weights = np.ones(num_components) / num_components
        means = onp.zeros((num_components, d))
        covariances = onp.zeros((num_components, d, d))
        for kk in range(num_components):
            means[kk] = np.array(
                [
                    5 * np.cos(2 * np.pi * kk / num_components),
                    5 * np.sin(2 * np.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * onp.eye(d)

        means = np.array(means)
        covariances = np.array(covariances)

    elif gmm_type == "line_gmm":
        assert d == 2, "Line GMM only works in 2D."
        num_components = 2
        weights = np.ones(num_components) / num_components
        means = onp.zeros((num_components, d))
        covariances = onp.zeros((num_components, d, d))
        for kk in range(num_components):
            means[kk] = np.array(
                [
                    5 * np.cos(2 * np.pi * kk / num_components),
                    5 * np.sin(2 * np.pi * kk / num_components),
                ]
            )
            covariances[kk] = 0.5 * onp.eye(d)

        means = np.array(means)
        covariances = np.array(covariances)

    else:
        raise ValueError(f"Invalid GMM type: {gmm_type}")

    return weights, means, covariances
