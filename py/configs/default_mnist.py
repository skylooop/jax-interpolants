"""
Nicholas M. Boffi
6/19/25

Basic configuration for MNIST.
"""

import ml_collections


def get_config(
    slurm_id: int, dataset_location: str, output_folder: str
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
    config.problem.d = 784
    config.problem.target = "mnist"
    config.problem.image_dims = (1, 28, 28)
    config.problem.num_classes = 10
    config.problem.dataset_location = dataset_location
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"
    config.problem.gaussian_scale = "adaptive"

    # optimization config
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 128
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
    config.logging.visual_freq = 1000
    config.logging.save_freq = config.optimization.total_steps // 50
    config.logging.wandb_project = "jax-interpolants-debug"
    config.logging.wandb_name = f"mnist-debug"
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
    config.network.logvar_channels = 64
    config.network.use_bfloat16 = False
    config.network.unet_kwargs = {
        "model_channels": 64,
        "channel_mult": [1, 2, 2],
        "num_blocks": 2,
        "attn_resolutions": [7],
        "use_fourier": False,
    }

    return config
