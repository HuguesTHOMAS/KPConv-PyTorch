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
from kpconv_torch.models.architectures import KPCNN, KPFCNN
from kpconv_torch.utils.visualizer import ModelVisualizer


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
    visualize(args.dataset, args.datapath, args.chosen_log)


def visualize(dataset: str, datapath: Path, chosen_log: Path) -> None:

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Eventually you can choose which feature is visualized (index of the deform convolution in the
    # network)
    deform_idx = 0

    # Deal with 'last_XXX' choices
    chosen_log = model_choice(chosen_log)

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

    # Initialize configuration class
    config = load_config(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config["train"]["augment_noise"] = 0.0001
    config["train"]["batch_num"] = 1
    config["train"]["in_radius"] = 2.0
    config["input"]["input_threads"] = 0

    ##############
    # Prepare Data
    ##############

    print()
    print("Data Preparation")
    print("****************")

    # Initiate dataset
    if dataset == "ModelNet40":
        test_dataset = ModelNet40Dataset(config=config, datapath=datapath, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif dataset == "S3DIS":
        test_dataset = S3DISDataset(config=config, datapath=datapath, split="validation")
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    else:
        raise ValueError("Unsupported dataset : " + config["model"]["dataset"])

    # Data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config.input_threads,
        pin_memory=True,
    )

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print("\nModel Preparation")
    print("*******************")

    # Define network model
    t1 = time.time()
    if config["model"]["dataset_task"] == "classification":
        net = KPCNN(config)
    elif config["model"]["dataset_task"] in ["cloud_segmentation", "slam_segmentation"]:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError(
            "Unsupported dataset_task for deformation visu: " + config["model"]["dataset_task"]
        )

    # Define a visualizer class
    visualizer = ModelVisualizer(net, config, chkp_path=chosen_chkp, on_gpu=False)
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("\nStart visualization")
    print("*********************")

    # Training
    visualizer.show_deformable_kernels(net, test_loader, config, deform_idx)
