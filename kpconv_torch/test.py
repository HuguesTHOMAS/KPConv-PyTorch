import os
import time

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
from kpconv_torch.utils.config import Config
from kpconv_torch.utils.tester import ModelTester


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
            [
                os.path.join("results", f)
                for f in os.listdir("results")
                if f.startswith("Log")
            ]
        )

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ["last_ModelNet40", "last_ShapeNetPart", "last_S3DIS"]:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError("The given log does not exists: " + chosen_log)

    return chosen_log


def main(args):
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = -1

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = str(model_choice(args.chosen_log))

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
    config = Config()
    config.load(chosen_log)

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

    split = "validation" if on_val else "test"

    # Initiate dataset
    if config.dataset == "ModelNet40":
        test_dataset = ModelNet40Dataset(args.datapath, config, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif config.dataset == "S3DIS":
        test_dataset = S3DISDataset(
            args.datapath,
            config,
            split="validation" if args.filename is None else "test",
            use_potentials=True,
            infered_file=args.filename,
        )
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif config.dataset == "Toronto3D":
        test_dataset = Toronto3DDataset(
            args.datapath, config, split="test", use_potentials=True
        )
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = Toronto3DCollate
    elif config.dataset == "SemanticKitti":
        test_dataset = SemanticKittiDataset(
            args.datapath, config, split=split, balance_classes=False
        )
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    else:
        raise ValueError("Unsupported dataset : " + config.dataset)

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
    print("*****************")

    # Define network model
    t1 = time.time()
    if config.dataset_task == "classification":
        net = KPCNN(config)
    elif config.dataset_task in ["cloud_segmentation", "slam_segmentation"]:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError("Unsupported dataset_task for testing: " + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("\nStart test")
    print("**********\n")

    # Training
    if config.dataset_task == "classification":
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == "cloud_segmentation":
        tester.cloud_segmentation_test(net, test_loader, config)
    elif config.dataset_task == "slam_segmentation":
        tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError("Unsupported dataset_task for testing: " + config.dataset_task)
