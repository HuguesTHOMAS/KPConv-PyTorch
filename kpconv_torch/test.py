from pathlib import Path
import os
import time

from kpconv_torch.utils.config import load_config
import numpy as np
from torch.utils.data import DataLoader

from kpconv_torch.datasets.ModelNet40 import (
    ModelNet40Collate,
    ModelNet40Dataset,
    ModelNet40Sampler,
)
from kpconv_torch.datasets.S3DIS import (
    S3DISCollate,
    S3DISDataset,
    S3DISSampler,
)
from kpconv_torch.datasets.SemanticKitti import (
    SemanticKittiCollate,
    SemanticKittiDataset,
    SemanticKittiSampler,
)
from kpconv_torch.datasets.Toronto3D import (
    Toronto3DCollate,
    Toronto3DDataset,
    Toronto3DSampler,
)
from kpconv_torch.models.architectures import KPCNN, KPFCNN
from kpconv_torch.utils.tester import ModelTester, get_test_save_path


def main(args):
    test(
        args.datapath,
        args.configfile,
        args.filename,
        args.chosen_log,
    )


def test(
    datapath: Path,
    configfile: Path,
    filename: str,
    chosen_log: Path,
) -> None:

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = -1

    # Choose to test on validation or test task
    on_val = True

    # Deal with 'last_XXXXXX' choices
    output_path = get_test_save_path(filename, chosen_log)
    if configfile is not None:
        config_file_path = configfile
    else:
        config_file_path = Path(chosen_log / "config.yml")
    config = load_config(config_file_path)

    ############################
    # Initialize the environment
    ############################
    # Set which gpu is going to be used
    GPU_ID = "0"

    # Set GPU visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    ###############
    # Previous chkp
    ###############
    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, "checkpoints")
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == "chkp"]

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = "current_chkp.tar"
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, "checkpoints", chosen_chkp)

    ##############
    # Prepare Data
    ##############
    print()
    print("Data Preparation")
    print("****************")

    task = "validate" if on_val else "test"

    # Initiate dataset
    if config["dataset"] == "ModelNet40":
        test_dataset = ModelNet40Dataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            train=False,
        )
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate

    elif config["dataset"] == "S3DIS":

        test_dataset = S3DISDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            task="validate" if filename is None else "test",
        )
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate

    elif config["dataset"] == "Toronto3D":

        test_dataset = Toronto3DDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            task="test",
        )
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = Toronto3DCollate

    elif config["dataset"] == "SemanticKitti":

        test_dataset = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            task=task,
            balance_classes=False,
        )
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    else:
        raise ValueError("Unsupported dataset : " + config["dataset"])

    # Data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config["input"]["threads"],
        pin_memory=True,
    )

    # Calibrate samplers, one for each dataset
    test_sampler.calibration(test_loader, verbose=True)

    print("\nModel Preparation")
    print("*****************")

    # Define network model
    t1 = time.time()
    if config["input"]["task"] == "classification":
        net = KPCNN(config)
    elif config["input"]["task"] in ["cloud_segmentation", "slam_segmentation"]:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError("Unsupported task for testing: " + config["input"]["task"])

    # Define a visualizer class
    tester = ModelTester(net, config, chkp_path=chosen_chkp, test_path=output_path)
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("\nStart test")
    print("**********\n")

    # Testing
    if config["input"]["task"] == "classification":
        tester.classification_test(net, test_loader)
    elif config["input"]["task"] == "cloud_segmentation":
        tester.cloud_segmentation_test(net, test_loader)
    elif config["input"]["task"] == "slam_segmentation":
        tester.slam_segmentation_test(net, test_loader)
    else:
        raise ValueError("Unsupported task for testing: " + config["input"]["task"])
