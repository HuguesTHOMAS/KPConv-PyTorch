from pathlib import Path

from kpconv_torch.datasets.ModelNet40 import (
    ModelNet40Config,
    ModelNet40Dataset,
)
from kpconv_torch.datasets.NPM3D import (
    NPM3DConfig,
    NPM3DDataset,
)
from kpconv_torch.datasets.S3DIS import (
    S3DISConfig,
    S3DISDataset,
)
from kpconv_torch.datasets.SemanticKitti import (
    SemanticKittiConfig,
    SemanticKittiDataset,
)
from kpconv_torch.datasets.Toronto3D import (
    Toronto3DConfig,
    Toronto3DDataset,
)


def main(args):
    preprocess(args.datapath, args.dataset)


def preprocess(datapath: Path, dataset: str) -> None:

    # ############################
    # # Initialize the environment
    # ############################
    # Option: set which gpu is going to be used and set the GPU visible device
    # By modifying the CUDA_VISIBLE_DEVICES environment variable

    # Initialize configuration class
    if dataset == "ModelNet40":
        config = ModelNet40Config()
    if dataset == "NPM3D":
        config = NPM3DConfig()
    if dataset == "S3DIS":
        config = S3DISConfig()
    if dataset == "SemanticKitti":
        config = SemanticKittiConfig()
    elif dataset == "Toronto3D":
        config = Toronto3DConfig()

    ##################################
    # Change model parameters for test
    ##################################
    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config.validation_size = 200
    config.input_threads = 10

    ##############
    # Prepare Data
    ##############
    print()
    print("Data Preparation")
    print("****************")

    # Initialize datasets
    if config.dataset == "ModelNet40":
        _ = ModelNet40Dataset(config=config, datapath=datapath, train=True)
        _ = ModelNet40Dataset(config=config, datapath=datapath, train=False)
    elif config.dataset == "NPM3D":
        _ = NPM3DDataset(
            config=config,
            datapath=datapath,
            split="training",
            use_potentials=True,
        )
        _ = NPM3DDataset(
            config=config,
            datapath=datapath,
            split="validation",
            use_potentials=True,
        )
    elif config.dataset == "S3DIS":
        _ = S3DISDataset(
            config=config,
            datapath=datapath,
            split="training",
            use_potentials=True,
        )
        _ = S3DISDataset(
            config=config,
            datapath=datapath,
            split="validation",
            use_potentials=True,
        )
    elif config.dataset == "SemanticKitti":
        _ = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            split="training",
            balance_classes=True,
        )
        _ = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            split="validation",
            balance_classes=False,
        )
    elif config.dataset == "Toronto3D":
        _ = Toronto3DDataset(
            config=config,
            datapath=datapath,
            split="training",
            use_potentials=True,
        )
        _ = Toronto3DDataset(
            config=config,
            datapath=datapath,
            split="validation",
            use_potentials=True,
        )
    else:
        raise ValueError("Unsupported dataset : " + config.dataset)
