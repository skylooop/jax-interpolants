"""
Nicholas M. Boffi
3/20/25

Utilities for storing training state.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, NamedTuple, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax import struct
from flax.core import freeze, unfreeze
from flax.serialization import from_bytes, msgpack_restore
from flax.training import train_state
from ml_collections import config_dict

from . import flow_map, interpolant, velocity
from . import dist_utils


class EMATrainState(train_state.TrainState):
    """Train state including EMA parameters."""

    ema_params: Dict[float, Any] = struct.field(default_factory=dict)


class StaticArgs(NamedTuple):
    net: nn.Module
    schedule: optax.Schedule
    anneal_schedule: optax.Schedule
    loss: Callable
    interp_loss: Callable
    get_loss_fn_args: Callable
    train_step: Callable
    update_ema_params: Callable
    ds: tf.data.Dataset
    interp: interpolant.Interpolant
    sample_rho0: Callable


def load_checkpoint(
    cfg: config_dict.ConfigDict,
    train_state: EMATrainState,
) -> EMATrainState:
    """Load a training checkpoint."""
    with open(cfg.network.load_path, "rb") as f:
        raw_bytes = f.read()
        train_state = from_bytes(train_state, raw_bytes)

    return train_state


def convert_velocity_to_flow(tree):
    t = unfreeze(freeze(tree))

    v = t["params"].pop("velocity")  # {'unet': {...}, 'logvar_linear': ...}

    # Build new flow_map subtree
    flow_map_dict = {}
    for k, val in v.items():
        if k == "unet":  # outer unet goes under 'net'
            flow_map_dict["net"] = val
        else:  # keep everything else (logvar_linear, …)
            flow_map_dict[k] = val

    t["params"]["flow_map"] = flow_map_dict

    return t


def convert_flow_to_velocity(tree):
    t = unfreeze(freeze(tree))

    # pop out the flow_map subtree
    flow_map = t["params"].pop("flow_map")

    # Build new velocity subtree
    velocity_dict = {}
    for k, val in flow_map.items():
        velocity_dict[k] = val

    t["params"]["velocity"] = velocity_dict

    return t


def load_velocity_checkpoint(
    cfg: config_dict.ConfigDict,
    train_state: EMATrainState,
    tx: optax.GradientTransformation,
) -> EMATrainState:
    with open(cfg.network.load_path, "rb") as f:
        raw_bytes = f.read()

        # raw restore to avoid structure check
        raw_state = msgpack_restore(raw_bytes)

        # pull out the params / EMA dicts
        if cfg.network.load_ema_fac == 0:
            velocity_params = raw_state["params"]
        else:
            velocity_params = raw_state["ema_params"][str(cfg.network.load_ema_fac)]

        ema_buffers = raw_state["ema_params"]
        new_params = convert_velocity_to_flow(velocity_params)
        new_ema = {
            float(fac): convert_velocity_to_flow(buf)
            for fac, buf in ema_buffers.items()
        }

        # build the training state
        train_state = train_state.replace(
            params=new_params,
            ema_params=new_ema,
            opt_state=tx.init(new_params),
        )

    return train_state


def load_nvidia_checkpoint(
    cfg: config_dict.ConfigDict,
    train_state: EMATrainState,
    tx: optax.GradientTransformation,
) -> EMATrainState:
    """Load a pre-trained nvidia checkpoint."""
    with open(cfg.network.load_path, "rb") as f:
        raw_bytes = f.read()

        if cfg.network.is_velocity:
            # nvidia saved in flow map form, we aer in velocity form
            new_params = from_bytes(
                convert_velocity_to_flow(train_state.params), raw_bytes
            )
        else:
            new_params = from_bytes(train_state.params, raw_bytes)

        # pad if we allow for null token; nvidia doesn't
        if cfg.network.use_cfg:
            print("Padding emb_label.weight to add null token.")
            class_label_weight = new_params["params"]["flow_map"]["net"]["unet"][
                "emb_label"
            ]["mpconv_weight"]
            padded = jnp.pad(class_label_weight, ((0, 0), (0, 1)))  # add zero column
            new_params = unfreeze(new_params)
            new_params["params"]["flow_map"]["net"]["unet"]["emb_label"][
                "mpconv_weight"
            ] = padded
            print("Padded emb_label.weight to", padded.shape)

    # convert back to velocity form
    if cfg.network.is_velocity:
        new_params = convert_flow_to_velocity(new_params)

    new_ema = {float(fac): deepcopy(new_params) for fac in cfg.training.ema_facs}
    train_state = train_state.replace(
        params=new_params, ema_params=new_ema, opt_state=tx.init(new_params)
    )

    return train_state


def setup_schedule(
    cfg: config_dict.ConfigDict,
) -> optax.Schedule:
    """Set up the learning rate schedule."""
    if cfg.optimization.schedule_type == "cosine":
        return optax.cosine_decay_schedule(
            init_value=cfg.optimization.learning_rate,
            decay_steps=cfg.optimization.decay_steps,
            alpha=0.0,
        )
    elif cfg.optimization.schedule_type == "sqrt":
        return lambda step: cfg.optimization.learning_rate / jnp.sqrt(
            jnp.maximum(step / cfg.optimization.decay_steps, 1.0)
        )
    else:
        raise ValueError(f"Unknown schedule type: {cfg.schedule_type}")


def setup_optimizer(cfg: config_dict.ConfigDict):
    """Set up the optimizer."""
    schedule = setup_schedule(cfg)

    # optimizer mask to avoid updating fourier embeddings
    # add safety check for positional embeddings, which do not have a constants key
    def mask_fn(variables):
        masks = {
            "params": jax.tree_util.tree_map(lambda _: True, variables["params"]),
        }
        if "constants" in variables:  # network has Fourier tables
            masks["constants"] = jax.tree_util.tree_map(
                lambda _: False, variables["constants"]
            )
        return masks

    # define optimizer
    tx = optax.masked(
        optax.chain(
            optax.clip_by_global_norm(cfg.optimization.clip),
            optax.radam(learning_rate=schedule),
        ),
        mask_fn,
    )

    return tx, schedule


def setup_training_state(
    cfg: config_dict.ConfigDict,
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> Tuple[EMATrainState, flow_map.FlowMap, optax.Schedule, jnp.ndarray]:
    """Load flax training state."""

    # define and initialize the network
    init_network = (
        velocity.initialize_velocity
        if cfg.training.train_velocity
        else flow_map.initialize_flow_map
    )
    net, params, prng_key = init_network(cfg.network, ex_input, prng_key)
    ema_params = {ema_fac: deepcopy(params) for ema_fac in cfg.training.ema_facs}

    # define training state
    tx, schedule = setup_optimizer(cfg)
    train_state = EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params=ema_params,
        tx=tx,
    )

    # load training state from checkpoint, if desired
    if cfg.network.load_path != "":
        # load just the parameters from an nvidia pre-trained velocity
        if cfg.network.load_from_nvidia:
            print("Loading nvidia checkpoint.")
            train_state = load_nvidia_checkpoint(cfg, train_state, tx)
            print("Loaded nvidia checkpoint.")

        # load parameter from pre-trained drift
        elif cfg.network.load_from_velocity:
            print("Loading velocity field checkpoint.")
            train_state = load_velocity_checkpoint(cfg, train_state, tx)
            print("Loaded velocity field checkpoint.")

        # load training checkpoint to restart
        else:
            print("Loading full training state checkpoint.")
            train_state = load_checkpoint(cfg, train_state)
            print("Loaded training state checkpoint.")

        if cfg.network.reset_optimizer:
            print("Resetting optimizer state.")
            train_state = train_state.replace(
                opt_state=tx.init(train_state.params),
                step=0,
            )

    return train_state, net, schedule, prng_key


def use_velocity_loss(cfg: config_dict.ConfigDict, train_state: EMATrainState) -> bool:
    return cfg.training.train_velocity or (
        cfg.training.interp_anneal
        and dist_utils.safe_index(cfg, train_state.step) <= cfg.training.anneal_steps
    )
