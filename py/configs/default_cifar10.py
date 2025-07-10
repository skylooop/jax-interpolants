"""
Nicholas M. Boffi
6/19/25

Basic configuration for CIFAR-10.
"""

import ml_collections


def get_config(
    slurm_id: int, dataset_location: str, output_folder: str, wandb_entity: str
) -> ml_collections.ConfigDict:
    # ensure jax.device_count works (weird issue with importlib)
    import jax

    del slurm_id

    # setup overall config
    config = ml_collections.ConfigDict()

    # training config
    config.training = ml_collections.ConfigDict()
    config.training.shuffle = True
    config.training.conditional = False
    config.training.loss_type = "velocity"
    config.training.class_dropout = 0.0
    config.training.tmin = 0.0
    config.training.tmax = 1.0
    config.training.seed = 42
    config.training.ema_facs = [0.999, 0.9999]
    config.training.ndevices = jax.device_count()
    config.training.n_nodes = jax.process_count()

    # problem config
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 60000
    config.problem.d = 3072
    config.problem.target = "cifar10"
    config.problem.image_dims = (3, 32, 32)
    config.problem.num_classes = 10
    config.problem.dataset_location = dataset_location
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"
    config.problem.gaussian_scale = "adaptive"

    # optimization config
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 128
    config.optimization.learning_rate = 1e-3
    config.optimization.clip = 10.0
    config.optimization.total_samples = 200_000_000
    config.optimization.total_steps = int(
        config.optimization.total_samples // config.optimization.global_bs
    )
    config.optimization.decay_steps = 35000
    config.optimization.schedule_type = "sqrt"

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 5
    config.logging.visual_freq = 250
    config.logging.save_freq = config.optimization.total_steps // 50
    config.logging.wandb_project = "jax-interpolants-debug"
    config.logging.wandb_name = f"cifar10-debug"
    config.logging.wandb_entity = wandb_entity
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "edm2"
    config.network.load_path = ""
    config.network.reset_optimizer = False
    config.network.img_resolution = config.problem.image_dims[1]
    config.network.img_channels = config.problem.image_dims[0]
    config.network.input_dims = config.problem.image_dims
    config.network.label_dim = (
        config.problem.num_classes if config.training.conditional else 0
    )
    config.network.use_cfg = False
    config.network.logvar_channels = 128
    config.network.use_bfloat16 = False
    config.network.unet_kwargs = {
        "model_channels": 128,
        "channel_mult": [2, 2, 2],
        "num_blocks": 3,
        "attn_resolutions": [16, 8],
        "use_fourier": False,
        "block_kwargs": {"dropout": 0.13, "use_song_normalization": False},
    }

    return config
