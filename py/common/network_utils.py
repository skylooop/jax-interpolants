"""
Nicholas M. Boffi
6/19/25

Helper routines for neural network definitions.
"""

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from ml_collections import config_dict

from . import edm2_net as edm2_net

Parameters = dict[str, dict]


class MLP(nn.Module):
    """Simple MLP network with square weight pattern."""

    n_hidden: int
    n_neurons: int
    output_dim: int
    act: Callable
    use_residual: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.n_neurons)(x)
        x = self.act(x)

        for _ in range(self.n_hidden):
            if self.use_residual:
                x = x + nn.Dense(self.n_neurons)(x)
            else:
                x = nn.Dense(self.n_neurons)(x)
            x = self.act(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class MLPVelocity(nn.Module):
    """Simple MLP network with square weight pattern, for flow map representation."""

    config: config_dict.ConfigDict

    def setup(self):
        self.mlp = MLP(
            self.config.n_hidden,
            self.config.n_neurons,
            self.config.output_dim,
            get_act(self.config),
            self.config.use_residual,
        )

    def calc_weight(self, t: float) -> float:
        del t
        return 1.0

    def __call__(
        self,
        t: float,
        x: jnp.ndarray,
        label: float | None = None,
        train: bool = True,
        calc_weight=False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, float]:
        del label
        del train

        inp = jnp.concatenate((jnp.array([t]), x / self.config.rescale))
        vel = self.mlp(inp)

        if calc_weight:
            weight = self.calc_weight(t)
            return vel, weight
        else:
            return vel


class EDM2Velocity(nn.Module):
    """Thin wrapper class for the UNet architecture based on EDM2.
    Note: assumes that there is no batch dimension, to interface with the rest of the code.
    Adds a padded batch dimension to handle this.
    """

    config: config_dict.ConfigDict

    def setup(self):
        self.one_hot_dim = (
            self.config.label_dim + 1 if self.config.use_cfg else self.config.label_dim
        )

        self.net = edm2_net.PrecondUNet(
            img_resolution=self.config.img_resolution,
            img_channels=self.config.img_channels,
            label_dim=self.one_hot_dim,
            sigma_data=self.config.rescale,
            logvar_channels=self.config.logvar_channels,
            use_bfloat16=self.config.use_bfloat16,
            unet_kwargs=self.config.unet_kwargs,
        )

    def process_inputs(
        self,
        t: jnp.ndarray | float,
        x: jnp.ndarray,
        label: jnp.ndarray | float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        # add batch dimensions
        t = jnp.asarray(t, dtype=jnp.float32)
        x = x.reshape((1, *x.shape))

        # one-hot encode
        if label is not None:
            label = jax.nn.one_hot(label, num_classes=self.one_hot_dim).reshape((1, -1))

        return t, x, label

    def calc_weight(self, t: jnp.ndarray | float) -> jnp.ndarray:
        # add batch dimension
        t = jnp.asarray(t, dtype=jnp.float32)
        return self.net.calc_weight(t)

    def __call__(
        self,
        t: jnp.ndarray | float,
        x: jnp.ndarray,
        label: jnp.ndarray | float | None = None,
        train: bool = True,
        calc_weight: bool = False,
    ):
        t, x, label = self.process_inputs(t, x, label)
        rslt = self.net(t, x, label, train, calc_weight)

        if calc_weight:
            bt, logvar = rslt
            return bt[0], logvar[0]
        else:
            bt = rslt
            return bt[0]


def get_act(
    config: config_dict.ConfigDict,
) -> Callable:
    """Get the activation function for the network.

    Args:
        config: Configuration dictionary.
    """
    if config.act == "gelu":
        return jax.nn.gelu
    elif config.act == "swish" or config.act == "silu":
        return jax.nn.silu
    else:
        raise ValueError(f"Activation function {config.activation} not recognized.")


def setup_network(
    network_config: config_dict.ConfigDict,
) -> nn.Module:
    """Setup the neural network for the system.

    Args:
        config: Configuration dictionary.
    """
    if "mlp" in network_config.network_type:
        return MLPVelocity(config=network_config)
    elif network_config.network_type == "edm2":
        return EDM2Velocity(config=network_config)
    else:
        raise ValueError(f"Network type {network_config.network_type} not recognized.")


def initialize_velocity(
    network_config: config_dict.ConfigDict, ex_input: jnp.ndarray, prng_key: jnp.ndarray
) -> tuple[nn.Module, Parameters, jnp.ndarray]:
    # define the network
    net = setup_network(network_config)

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

    print(f"Number of parameters: {ravel_pytree(params)[0].size}")

    if network_config.network_type == "edm2":
        params = edm2_net.project_to_sphere(params)

    return net, params, prng_key
