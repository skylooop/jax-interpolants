"""
Nicholas M. Boffi
6/29/25

Main code for stochastic interpolant training.
"""

import os
import pathlib
import sys

import jax
import wandb

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import argparse
import importlib
import time

import common.datasets as datasets
import common.dist_utils as dist_utils
import common.interpolant as interpolant
import common.logging as logging
import common.loss_args as loss_args
import common.losses as losses
import common.state_utils as state_utils
import common.updates as updates
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import optax
from ml_collections import config_dict  # type: ignore
from tqdm.auto import tqdm as tqdm

Parameters = dict[str, dict]
mpl.rc_file(f"{pathlib.Path(__file__).resolve().parent}/matplotlibrc")


def train_loop(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: np.ndarray,
) -> None:
    """Carry out the training loop."""

    pbar = tqdm(range(cfg.optimization.total_steps))
    for _ in pbar:
        # log gradient step time
        start_time = time.time()

        # construct loss function arguments
        loss_fn_args, prng_key = statics.get_loss_fn_args(cfg, statics, prng_key)

        # take a step on the loss
        train_state, loss_value, grads = statics.train_step(
            train_state, statics.loss, loss_fn_args
        )
        end_time = time.time()

        # compute update to EMA params
        train_state = statics.update_ema_params(train_state)

        # log to wandb
        prng_key = logging.log_metrics(
            cfg,
            statics,
            train_state,
            grads,
            loss_value,
            loss_fn_args,
            prng_key,
            end_time - start_time,
        )

        pbar.set_postfix(loss=loss_value)

    # dump one final time
    logging.save_state(train_state, cfg)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Flow learning with stochastic interpolants."
    )
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--slurm_id", type=int)
    parser.add_argument("--dataset_location", type=str)
    parser.add_argument("--output_folder", type=str)
    return parser.parse_args()


def setup_config_dict():
    args = parse_command_line_arguments()
    cfg_module = importlib.import_module(args.cfg_path)
    return cfg_module.get_config(
        args.slurm_id, args.dataset_location, args.output_folder
    )


def setup_state(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray) -> tuple[
    config_dict.ConfigDict,
    state_utils.StaticArgs,
    state_utils.EMATrainState,
    jnp.ndarray,
]:
    """Construct static arguments and training state objects."""
    # define dataset
    cfg, ds, prng_key = datasets.setup_target(cfg, prng_key)
    ex_input = next(ds)
    if isinstance(ex_input, dict):  # handle image datasets
        ex_input = ex_input["image"][0]
    else:
        ex_input = ex_input[0]
    interp = interpolant.setup_interpolant(cfg)
    cfg = config_dict.FrozenConfigDict(cfg)

    # define training state
    train_state, net, schedule, prng_key = state_utils.setup_training_state(
        cfg,
        ex_input,
        prng_key,
    )

    # define the losses
    loss = losses.setup_loss(cfg, net, interp)

    # define static object
    statics = state_utils.StaticArgs(
        net=net,
        schedule=schedule,
        loss=loss,
        get_loss_fn_args=loss_args.get_loss_fn_args,
        train_step=updates.setup_train_step(cfg),
        update_ema_params=updates.setup_ema_update(cfg),
        ds=ds,
        interp=interp,
        sample_rho0=datasets.setup_base(cfg, ex_input),
    )

    train_state = dist_utils.safe_replicate(cfg, train_state)

    return cfg, statics, train_state, prng_key


if __name__ == "__main__":
    print("Setting up distributed training.")
    dist_utils.initialize_slurm()

    print("Entering main. Setting up config dict and PRNG key.")
    cfg = setup_config_dict()
    prng_key = jax.random.PRNGKey(cfg.training.seed)

    ## set up weights and biases tracking
    print("Setting up wandb.")
    if jax.process_index() == 0:
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.logging.wandb_name,
            config=cfg.to_dict(),
        )
    else:
        wandb.init(mode="disabled")

    print("Config dict set up. Setting up static arguments and training state.")
    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)

    train_loop(cfg, statics, train_state, prng_key)
