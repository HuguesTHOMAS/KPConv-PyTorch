from pathlib import Path

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

from kpconv_torch.utils.config import load_config


def main(args):
    preprocess(args.configfile, args.datapath)


def preprocess(configfile_path: Path, datapath: Path) -> None:
    # Option: set which gpu is going to be used and set the GPU visible device
    # By modifying the CUDA_VISIBLE_DEVICES environment variable

    ##############
    # Prepare Data
    ##############
    print()
    print("Data Preparation")
    print("****************")

    # Test if the provided dataset (passed to the -d option)
    # corresponds to the one of the config file to use
    config = load_config(configfile_path)

    # Initialize datasets
    if config["dataset"] == "ModelNet40":
        # Training
        _ = ModelNet40Dataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = ModelNet40Dataset(config=config, datapath=datapath, task="validate")
    elif config["dataset"] == "NPM3D":
        # Training
        _ = NPM3DDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = NPM3DDataset(config=config, datapath=datapath, task="validate")
    elif config["dataset"] == "S3DIS":
        # Training
        _ = S3DISDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = S3DISDataset(config=config, datapath=datapath, task="validate")
    elif config["dataset"] == "SemanticKitti":
        # Training
        _ = SemanticKittiDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            task="validate",
            balance_classes=False,
        )
    elif config["dataset"] == "Toronto3D":
        # Training
        _ = Toronto3DDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = Toronto3DDataset(config=config, datapath=datapath, task="validate")
