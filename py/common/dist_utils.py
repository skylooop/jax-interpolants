"""
Nicholas M. Boffi
3/20/25

Simple utilities for data parallelism.
"""

import os
import socket
import subprocess
from typing import Any, Tuple

import jax
import jax.distributed as jd
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from ml_collections import config_dict


def safe_index(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    if cfg.training.ndevices > 1:
        return x[0]
    else:
        return x


def safe_replicate(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    if cfg.training.ndevices > 1:
        return replicate(x)
    else:
        return x


def safe_unreplicate(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    if cfg.training.ndevices > 1:
        return unreplicate(x)
    else:
        return x


def replicate_batch(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Replicate the batch for data parallelism."""
    if cfg.training.ndevices > 1 and x is not None:
        x = x.reshape((cfg.training.local_ndevices, -1, *x.shape[1:]))
    return x


def unreplicate_batch(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """Unreplicate the batch for data parallelism."""
    if cfg.training.ndevices > 1 and x is not None:
        x = x.reshape((-1, *x.shape[2:]))
    return x


def replicate_loss_fn_args(cfg: config_dict.ConfigDict, loss_fn_args: Tuple) -> Tuple:
    """Replicate the loss function arguments for data parallelism."""
    return tuple(replicate_batch(cfg, arg) for arg in loss_fn_args)


def unreplicate_loss_fn_args(cfg: config_dict.ConfigDict, loss_fn_args: Tuple) -> Tuple:
    """Unreplicate the loss function arguments for data parallelism."""
    return tuple(unreplicate_batch(cfg, arg) for arg in loss_fn_args)


def _rank0_host():
    """Return the first concrete hostname from $SLURM_NODELIST."""
    hostlist = os.environ["SLURM_NODELIST"]
    # scontrol prints one host per line: pick the first
    host0 = (
        subprocess.check_output(["scontrol", "show", "hostnames", hostlist])
        .decode()
        .splitlines()[0]
    )
    return host0


def initialize_slurm():
    if jd.is_initialized():
        return

    port = int(os.getenv("JAX_COORD_PORT", "12355"))
    world = int(os.getenv("SLURM_NTASKS", "1"))
    rank = int(os.getenv("SLURM_PROCID", "0"))
    if world == 1:
        return  # single-process job, nothing to do

    print(
        f"[init] host={socket.gethostname()} "
        f"rank={rank}/{world} coord={_rank0_host()}:{port}",
        flush=True,
    )

    coord = f"{socket.gethostbyname(_rank0_host())}:{port}"
    jd.initialize(
        coordinator_address=coord,
        num_processes=world,
        process_id=rank,
    )
