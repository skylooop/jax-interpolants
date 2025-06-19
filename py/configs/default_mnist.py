"""
Nicholas M. Boffi
3/13/25
"""

import ml_collections

stopgrad_types = ["convex", "lmd", "none"]
interp_anneals = [True, False]
anneal_steps_arr = [0, 7500]


def get_config(slurm_id: int):
    # ensure jax.device_count works (weird issue with importlib)
    import jax

    # convert slurm id into indices
    ii = slurm_id % len(stopgrad_types)
    jj = (slurm_id // len(stopgrad_types)) % len(interp_anneals)
    kk = (slurm_id // (len(stopgrad_types) * len(interp_anneals))) % len(
        anneal_steps_arr
    )

    # setup overall config
    config = ml_collections.ConfigDict()

    # training config
    config.training = ml_collections.ConfigDict()
    config.training.shuffle = True
    config.training.conditional = True
    config.training.class_dropout = 0.0
    config.training.stopgrad_type = stopgrad_types[ii]
    config.training.interp_anneal = interp_anneals[jj]
    config.training.anneal_steps = anneal_steps_arr[kk]
    config.training.tmin = 0.0
    config.training.tmax = 1.0
    config.training.seed = 42
    config.training.ema_facs = [0.999, 0.9999]
    config.training.ndevices = jax.device_count()

    # problem config
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 60000
    config.problem.d = 784
    config.problem.target = "mnist"
    config.problem.dataset_location = "/n/home04/nboffi/datasets"
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"
    config.problem.gaussian_scale = "adaptive"

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 5
    config.logging.visual_freq = 2500
    config.logging.save_freq = 10000
    config.logging.wandb_project = "fmm_rebuttal"
    config.logging.wandb_name = f"3_23_25_mnist_sweep_{slurm_id}"
    config.logging.wandb_entity = "boffi"
    config.logging.output_folder = (
        "/n/home04/nboffi/results/lmm/mnist/3_23_25_mnist_sweep/"
    )
    config.logging.output_name = f"3_23_25_mnist_sweep_{slurm_id}"

    # optimization config
    config.optimization = ml_collections.ConfigDict()
    config.optimization.n_epochs = 10000
    config.optimization.bs = 128
    config.optimization.learning_rate = 1e-3
    config.optimization.clip = 5.0
    config.optimization.decay_steps = 500000

    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "edm2"
    config.network.load_path = ""
    config.network.img_resolution = 28
    config.network.img_channels = 1
    config.network.label_dim = 10
    config.network.logvar_channels = 128
    config.network.unet_kwargs = {
        "model_channels": 128,
        "channel_mult": [1, 2, 2],
        "num_blocks": 2,
        "attn_resolutions": [7],
    }

    return config
