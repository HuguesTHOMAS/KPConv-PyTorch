import os

from kpconv_torch.datasets.ModelNet40 import (
    ModelNet40Dataset,
)
from kpconv_torch.datasets.NPM3D import (
    NPM3DDataset,
)
from kpconv_torch.datasets.S3DIS import (
    S3DISDataset,
)
from kpconv_torch.datasets.SemanticKitti import (
    SemanticKittiDataset,
)
from kpconv_torch.datasets.Toronto3D import (
    Toronto3DDataset,
)
from kpconv_torch.utils.config import Config


def main(args):

    ############################
    # Initialize the environment
    ############################
    # Set which gpu is going to be used
    GPU_ID = "0"

    # Set GPU visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    # Initialize configuration class
    config = Config()
    config.load(args.chosen_log)
    if config.dataset != args.dataset:
        raise ValueError(
            f"Config dataset ({config.dataset}) "
            f"does not match provided dataset ({args.dataset})."
        )

    ##################################
    # Change model parameters for test
    ##################################
    # Change parameters for the test here. For example, you can stop augmenting the input data.

    # config.augment_noise = 0.0001
    # config.augment_symmetries = False
    # config.batch_num = 3
    # config.in_radius = 4
    config.validation_size = 200
    config.input_threads = 10

    ##############
    # Prepare Data
    ##############
    print()
    print("Data Preparation")
    print("****************")

    # Initialize datasets and samplers
    if config.dataset == "ModelNet40":
        _ = ModelNet40Dataset(args.datapath, config, train=True)
        _ = ModelNet40Dataset(args.datapath, config, train=False)
    elif config.dataset == "NPM3D":
        _ = NPM3DDataset(args.datapath, config, set="training", use_potentials=True)
        _ = NPM3DDataset(args.datapath, config, set="validation", use_potentials=True)
    elif config.dataset == "S3DIS":
        _ = S3DISDataset(args.datapath, config, set="training", use_potentials=True)
        _ = S3DISDataset(args.datapath, config, set="validation", use_potentials=True)
    elif config.dataset == "SemanticKitti":
        _ = SemanticKittiDataset(
            args.datapath, config, set="training", balance_classes=True
        )
        _ = SemanticKittiDataset(
            args.datapath, config, set="validation", balance_classes=False
        )
    elif config.dataset == "Toronto3D":
        _ = Toronto3DDataset(args.datapath, config, set="training", use_potentials=True)
        _ = Toronto3DDataset(
            args.datapath, config, set="validation", use_potentials=True
        )
    else:
        raise ValueError("Unsupported dataset : " + config.dataset)
