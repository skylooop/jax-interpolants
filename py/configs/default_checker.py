"""
Nicholas M. Boffi
6/19/25

Basic configuration for the two-dimensional checker example.
"""

import ml_collections


def get_config(
    slurm_id: int, dataset_location: str, output_folder: str
) -> ml_collections.ConfigDict:
    # ensure jax.device_count works (weird issue with importlib)
    import jax

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
    config.problem.n = int(1e7)
    config.problem.d = 2
    config.problem.image_dims = None
    config.problem.num_classes = None
    config.problem.target = "checker"
    config.problem.dataset_location = None
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"
    config.problem.gaussian_scale = "adaptive"

    # optimization config
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 64000
    config.optimization.learning_rate = 1e-3
    config.optimization.clip = 5.0
    config.optimization.total_steps = 500_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 500_000
    config.optimization.schedule_type = "cosine"

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 25000
    config.logging.visual_freq = 5000
    config.logging.save_freq = config.optimization.total_steps // 50
    config.logging.wandb_project = "jax-interpolants-debug"
    config.logging.wandb_name = "checker-debug"
    config.logging.wandb_entity = "boffi"
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "mlp"
    config.network.n_hidden = 4
    config.network.n_neurons = 256
    config.network.output_dim = 2
    config.network.act = "gelu"
    config.network.use_residual = False

    ## image configs
    config.network.load_path = ""
    config.network.input_dims = (2,)
    config.network.load_from_velocity = False
    config.network.load_from_nvidia = False
    config.network.img_resolution = None
    config.network.img_channels = None
    config.network.label_dim = None
    config.network.logvar_channels = None
    config.network.is_velocity = False
    config.network.unet_kwargs = None

    return config
