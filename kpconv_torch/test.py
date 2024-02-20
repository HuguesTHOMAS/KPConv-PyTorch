from pathlib import Path
import os
import time

from utils.config import load_config
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


def model_choice(chosen_log):
    ###########################
    # Call the test initializer
    ###########################
    # Automatically retrieve the last trained model
    if chosen_log in ["last_ModelNet40", "last_ShapeNetPart", "last_S3DIS"]:
        # Dataset name
        test_dataset = "_".join(chosen_log.split("_")[1:])

        # List all training logs
        logs = np.sort(
            [os.path.join("results", f) for f in os.listdir("results") if f.startswith("Log")]
        )

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = load_config(log)
            if log_config["dataset"] == test_dataset:
                chosen_log = log
                break

        if chosen_log in ["last_ModelNet40", "last_ShapeNetPart", "last_S3DIS"]:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError("The given log does not exists: " + chosen_log)

    return chosen_log


def main(args):
    test(
        args.datapath,
        args.filename,
        args.chosen_log,
        args.n_votes,
        args.potential_increment,
    )


def test(
    config,
    datapath: Path,
    filename: str,
    chosen_log: Path,
    dataset: str,
    n_votes: int = 100,
    potential_increment: int = None,
) -> None:
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = -1

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = str(model_choice(chosen_log))

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

    split = "validation" if on_val else "test"

    # Initiate dataset
    if dataset == "ModelNet40":
        test_dataset = ModelNet40Dataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            train=False,
        )
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif dataset == "S3DIS":
        test_dataset = S3DISDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            split="validation" if filename is None else "test",
        )
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif dataset == "Toronto3D":
        test_dataset = Toronto3DDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            split="test",
        )
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = Toronto3DCollate
    elif dataset == "SemanticKitti":
        test_dataset = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            split=split,
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
        num_workers=config["model"]["input_threads"],
        pin_memory=True,
    )

    # Calibrate samplers, one for each dataset
    test_sampler.calibration(test_loader, verbose=True)

    print("\nModel Preparation")
    print("*****************")

    # Define network model
    t1 = time.time()
    if config["input"]["dataset_task"] == "classification":
        net = KPCNN(config)
    elif config["input"]["dataset_task"] in ["cloud_segmentation", "slam_segmentation"]:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError("Unsupported dataset_task for testing: " + config["input"]["dataset_task"])

    # Define a visualizer class
    output_path = get_test_save_path(filename, chosen_log)
    tester = ModelTester(net, chkp_path=chosen_chkp, test_path=output_path)
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("\nStart test")
    print("**********\n")

    # Testing
    if config["input"]["dataset_task"] == "classification":
        tester.classification_test(net, test_loader, config, num_votes=n_votes)
    elif config["input"]["dataset_task"] == "cloud_segmentation":
        tester.cloud_segmentation_test(net, test_loader, config, num_votes=n_votes)
    elif config["input"]["dataset_task"] == "slam_segmentation":
        tester.slam_segmentation_test(net, test_loader, config, num_votes=n_votes)
    else:
        raise ValueError("Unsupported dataset_task for testing: " + config["input"]["dataset_task"])
