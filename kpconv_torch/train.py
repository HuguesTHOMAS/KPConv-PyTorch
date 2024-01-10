import os
import signal
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

from kpconv_torch.datasets.ModelNet40 import (
    ModelNet40Collate,
    ModelNet40Config,
    ModelNet40Dataset,
    ModelNet40Sampler,
)
from kpconv_torch.datasets.NPM3D import (
    NPM3DCollate,
    NPM3DConfig,
    NPM3DDataset,
    NPM3DSampler,
)
from kpconv_torch.datasets.S3DIS import (
    S3DISCollate,
    S3DISConfig,
    S3DISDataset,
    S3DISSampler,
)
from kpconv_torch.datasets.SemanticKitti import (
    SemanticKittiCollate,
    SemanticKittiConfig,
    SemanticKittiDataset,
    SemanticKittiSampler,
)
from kpconv_torch.datasets.Toronto3D import (
    Toronto3DCollate,
    Toronto3DConfig,
    Toronto3DDataset,
    Toronto3DSampler,
)
from kpconv_torch.models.architectures import KPCNN, KPFCNN
from kpconv_torch.utils.trainer import ModelTrainer


def main(args):
    ############################
    # Initialize the environment
    ############################
    start = time.time()
    # Set which gpu is going to be used
    GPU_ID = "0"

    # Set GPU visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if args.chosen_log:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join(args.chosen_log, "checkpoints")
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == "chkp"]

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = "current_chkp.tar"
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(args.chosen_log, "checkpoints", chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############
    print()
    print("Data Preparation")
    print("****************")

    # Initialize configuration class
    if args.dataset == "ModelNet40":
        config = ModelNet40Config()
    if args.dataset == "NPM3D":
        config = NPM3DConfig()
    if args.dataset == "S3DIS":
        config = S3DISConfig()
    if args.dataset == "SemanticKitti":
        config = SemanticKittiConfig()
    elif args.dataset == "Toronto3D":
        config = Toronto3DConfig()

    if args.chosen_log:
        config.load(args.chosen_log)
        if config.dataset != args.dataset:
            raise ValueError(
                f"Config dataset ({config.dataset}) "
                f"does not match provided dataset ({args.dataset})."
            )
        config.saving_path = None

    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    # Initialize datasets and samplers
    if config.dataset == "ModelNet40":
        training_dataset = ModelNet40Dataset(args.datapath, config, train=True)
        test_dataset = ModelNet40Dataset(args.datapath, config, train=False)
        training_sampler = ModelNet40Sampler(training_dataset, balance_labels=True)
        test_sampler = ModelNet40Sampler(test_dataset, balance_labels=True)
        collate_fn = ModelNet40Collate
    elif config.dataset == "NPM3D":
        training_dataset = NPM3DDataset(
            args.datapath, config, split="training", use_potentials=True
        )
        test_dataset = NPM3DDataset(
            args.datapath, config, split="validation", use_potentials=True
        )
        training_sampler = NPM3DSampler(training_dataset)
        test_sampler = NPM3DSampler(test_dataset)
        collate_fn = NPM3DCollate
    elif config.dataset == "S3DIS":
        training_dataset = S3DISDataset(
            args.datapath, config, split="training", use_potentials=True
        )
        test_dataset = S3DISDataset(
            args.datapath, config, split="validation", use_potentials=True
        )
        training_sampler = S3DISSampler(training_dataset)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif config.dataset == "SemanticKitti":
        training_dataset = SemanticKittiDataset(
            args.datapath, config, split="training", balance_classes=True
        )
        test_dataset = SemanticKittiDataset(
            args.datapath, config, split="validation", balance_classes=False
        )
        training_sampler = SemanticKittiSampler(training_dataset)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    elif config.dataset == "Toronto3D":
        training_dataset = Toronto3DDataset(
            args.datapath, config, split="training", use_potentials=True
        )
        test_dataset = Toronto3DDataset(
            args.datapath, config, split="validation", use_potentials=True
        )
        training_sampler = Toronto3DSampler(training_dataset)
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = Toronto3DCollate
    else:
        raise ValueError("Unsupported dataset : " + config.dataset)

    # Initialize the dataloader
    training_loader = DataLoader(
        training_dataset,
        batch_size=1,
        sampler=training_sampler,
        collate_fn=collate_fn,
        num_workers=config.input_threads,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config.input_threads,
        pin_memory=True,
    )

    if config.dataset == "SemanticKitti":
        # Calibrate max_in_point value
        training_sampler.calib_max_in(config, training_loader, verbose=True)
        test_sampler.calib_max_in(config, test_loader, verbose=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    print("\nModel Preparation")
    print("*****************")

    # Define network model
    t1 = time.time()
    if config.dataset == "ModelNet40":
        net = KPCNN(config)
    else:
        net = KPFCNN(
            config, training_dataset.label_values, training_dataset.ignored_labels
        )

    debug = False
    if debug:
        print("\n*************************************\n")
        print(net)
        print("\n*************************************\n")
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print("\n*************************************\n")
        print(
            "Model size %i"
            % sum(param.numel() for param in net.parameters() if param.requires_grad)
        )
        print("\n*************************************\n")

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("\nStart training")
    print("**************")

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print("Forcing exit now")
    os.kill(os.getpid(), signal.SIGINT)

    end = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(end - start)))
