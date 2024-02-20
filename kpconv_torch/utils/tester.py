import os
from pathlib import Path
import time

import numpy as np
import torch

from kpconv_torch.utils.metrics import fast_confusion, IoU_from_confusions
from kpconv_torch.io.ply import write_ply


def get_test_save_path(infered_file: Path, chosen_log: Path) -> Path:
    if chosen_log is None:
        test_path = None
    elif infered_file is not None:
        test_path = Path(infered_file).parent / "test" / Path(chosen_log).name
    else:
        test_path = Path(chosen_log) / "test"
    if test_path is not None and not os.path.exists(test_path):
        os.makedirs(test_path)
    return test_path


class ModelTester:
    def __init__(self, net, chkp_path=None, test_path=None, on_gpu=True):

        ############
        # Parameters
        ############
        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################
        checkpoint = torch.load(chkp_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        self.epoch = checkpoint["epoch"]
        net.eval()
        print("Model and training state restored.")
        self.test_path = test_path

        return

    def classification_test(self, net, test_loader, config, num_votes=100, debug=False):

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config["input"]["num_classes"]

        # Initiate global prediction over test clouds
        self.test_probs = np.zeros((test_loader.dataset.num_models, nc_model))
        self.test_counts = np.zeros((test_loader.dataset.num_models, nc_model))

        t = [time.time()]
        mean_dt = np.zeros(1)
        last_display = time.time()
        while np.min(self.test_counts) < num_votes:

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            probs = []
            targets = []
            obj_inds = []

            # Start validation loop
            for batch in test_loader:

                # New time
                t = t[-1:]
                t += [time.time()]

                if "cuda" in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                # Get probs and labels
                probs += [softmax(outputs).cpu().detach().numpy()]
                targets += [batch.labels.cpu().numpy()]
                obj_inds += [batch.model_inds.cpu().numpy()]

                if "cuda" in self.device.type:
                    torch.cuda.synchronize(self.device)

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = "Test vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})"
                    print(
                        message.format(
                            np.min(self.test_counts),
                            100 * len(obj_inds) / config["train"]["validation_size"],
                            1000 * (mean_dt[0]),
                            1000 * (mean_dt[1]),
                        )
                    )
            # Stack all validation predictions
            probs = np.vstack(probs)
            targets = np.hstack(targets)
            obj_inds = np.hstack(obj_inds)

            if np.any(test_loader.dataset.input_labels[obj_inds] != targets):
                raise ValueError("wrong object indices")

            # Compute incremental average (predictions are always ordered)
            self.test_counts[obj_inds] += 1
            self.test_probs[obj_inds] += (probs - self.test_probs[obj_inds]) / (
                self.test_counts[obj_inds]
            )

            # Save/Display temporary results
            # ******************************

            test_labels = np.array(test_loader.dataset.label_values)

            # Compute classification results
            C1 = fast_confusion(
                test_loader.dataset.input_labels,
                np.argmax(self.test_probs, axis=1),
                test_labels,
            )

            ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
            print(f"Test Accuracy = {ACC:.1f}%")

        return

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config["input"]["num_classes"]

        # Initiate global prediction over test clouds
        self.test_probs = [
            np.zeros((input_label.shape[0], nc_model))
            for input_label in test_loader.dataset.input_labels
        ]

        # Test saving path
        if config["train"]["saving"]:
            if not os.path.exists(self.test_path):
                os.makedirs(self.test_path)
            if not os.path.exists(os.path.join(self.test_path, "predictions")):
                os.makedirs(os.path.join(self.test_path, "predictions"))
            if not os.path.exists(os.path.join(self.test_path, "probs")):
                os.makedirs(os.path.join(self.test_path, "probs"))
            if not os.path.exists(os.path.join(self.test_path, "potentials")):
                os.makedirs(os.path.join(self.test_path, "potentials"))

        # If on validation directly compute score
        if test_loader.dataset.set == "validation":
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum(
                        [
                            np.sum(labels == label_value)
                            for labels in test_loader.dataset.validation_labels
                        ]
                    )
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_saved_min = last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print("Initialize workers")
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print(f"Done in {t[1] - t[0]:.1f}s")

                if "cuda" in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                if "cuda" in self.device.type:
                    torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0 : i0 + length]
                    probs = stacked_probs[i0 : i0 + length]
                    inds = in_inds[i0 : i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = (
                            np.sum(points**2, axis=1)
                            < (test_radius_ratio * config["input"]["in_radius"]) ** 2
                        )
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = (
                        test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    )
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = "e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})"
                    print(
                        message.format(
                            test_epoch,
                            i,
                            100 * i / config["train"]["validation_size"],
                            1000 * (mean_dt[0]),
                            1000 * (mean_dt[1]),
                            1000 * (mean_dt[2]),
                        )
                    )

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print(
                "Test epoch {:d}, end. Min potential = {:.2f} (last: {:.1f})".format(
                    test_epoch, new_min, last_min
                )
            )

            # Save predicted cloud
            if last_min + 1 < new_min:
                print("Save predicted cloud...")

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == "validation":
                    print("\nConfusion on sub clouds")
                    Confs = []
                    for file_idx, _ in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[file_idx], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(
                            np.int32
                        )

                        # Targets
                        targets = test_loader.dataset.input_labels[file_idx]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(
                        list(enumerate(test_loader.dataset.label_values))
                    ):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = f"{100 * mIoU:5.2f} | "
                    for IoU in IoUs:
                        s += f"{100 * IoU:5.2f} "
                    print(s + "\n")

                # Save real IoU once in a while
                if last_saved_min + config["train"]["potential_increment"] < new_min:
                    last_saved_min = new_min

                    # Project predictions
                    print(f"\nReproject Vote #{int(np.floor(new_min)):d}")
                    t1 = time.time()
                    proj_probs = []
                    for file_idx, _ in enumerate(test_loader.dataset.files):

                        # Reproject probs on the evaluations points
                        probs = self.test_probs[file_idx][
                            test_loader.dataset.test_proj[file_idx], :
                        ]
                        proj_probs += [probs]

                        # Insert false columns for ignored labels
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                proj_probs[file_idx] = np.insert(
                                    proj_probs[file_idx], l_ind, 0, axis=1
                                )

                    t2 = time.time()
                    print(f"Done in {t2 - t1:.1f} s\n")

                    # Show vote results
                    if test_loader.dataset.set == "validation":
                        print("Confusion on full clouds")
                        t1 = time.time()
                        Confs = []
                        for file_idx, _ in enumerate(test_loader.dataset.files):

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[
                                np.argmax(proj_probs[file_idx], axis=1)
                            ].astype(np.int32)

                            # Confusion
                            targets = test_loader.dataset.validation_labels[file_idx]
                            Confs += [
                                fast_confusion(targets, preds, test_loader.dataset.label_values)
                            ]

                        t2 = time.time()
                        print(f"Done in {t2 - t1:.1f} s\n")

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(
                            list(enumerate(test_loader.dataset.label_values))
                        ):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = f"{100 * mIoU:5.2f} | "
                        for IoU in IoUs:
                            s += f"{100 * IoU:5.2f} "
                        print("-" * len(s))
                        print(s)
                        print("-" * len(s) + "\n")

                    # Save predictions
                    print("Saving clouds")
                    t1 = time.time()
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Get file
                        points = test_loader.dataset.load_evaluation_points(file_path)

                        # Get the predicted labels
                        preds = test_loader.dataset.label_values[
                            np.argmax(proj_probs[i], axis=1)
                        ].astype(np.int32)

                        # Save plys
                        cloud_name = file_path.name
                        test_name = os.path.join(self.test_path, "predictions", cloud_name)
                        write_ply(test_name, [points, preds], ["x", "y", "z", "preds"])
                        test_name2 = os.path.join(self.test_path, "probs", cloud_name)
                        prob_names = [
                            "_".join(test_loader.dataset.label_to_names[label].split())
                            for label in test_loader.dataset.label_values
                        ]
                        write_ply(
                            test_name2,
                            [points, proj_probs[i]],
                            ["x", "y", "z"] + prob_names,
                        )

                        # Save potentials
                        pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                        pot_name = os.path.join(self.test_path, "potentials", cloud_name)
                        pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                        write_ply(
                            pot_name,
                            [pot_points.astype(np.float32), pots],
                            ["x", "y", "z", "pots"],
                        )

                        # Save ascii preds
                        if test_loader.dataset.set == "test":
                            if test_loader.dataset.name.startswith("Semantic3D"):
                                ascii_name = os.path.join(
                                    self.test_path,
                                    "predictions",
                                    test_loader.dataset.ascii_files[cloud_name],
                                )
                            else:
                                ascii_name = os.path.join(
                                    self.test_path, "predictions", cloud_name[:-4] + ".txt"
                                )
                            np.savetxt(ascii_name, preds, fmt="%d")

                    t2 = time.time()
                    print(f"Done in {t2 - t1:.1f} s\n")

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break
            print("---")

        return

    def slam_segmentation_test(self, net, test_loader, config, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        # Test saving path
        report_path = None
        if config["train"]["saving"]:
            if not os.path.exists(self.test_path):
                os.makedirs(self.test_path)
            report_path = os.path.join(self.test_path, "reports")
            if not os.path.exists(report_path):
                os.makedirs(report_path)

        if test_loader.dataset.set == "validation":
            for folder in ["val_predictions", "val_probs"]:
                if not os.path.exists(os.path.join(self.test_path, folder)):
                    os.makedirs(os.path.join(self.test_path, folder))
        else:
            for folder in ["predictions", "probs"]:
                if not os.path.exists(os.path.join(self.test_path, folder)):
                    os.makedirs(os.path.join(self.test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == "validation":
            for seq_frames in test_loader.dataset.frames:
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print("Initialize workers")
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print(f"Done in {t[1] - t[0]:.1f}s")

                if "cuda" in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                if "cuda" in self.device.type:
                    torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0 : i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == "validation":
                        folder = "val_probs"
                        pred_folder = "val_predictions"
                    else:
                        folder = "probs"
                        pred_folder = "predictions"
                    filename = f"{seq_name}_{f_ind:7d}.npy"
                    filepath = os.path.join(self.test_path, folder, filename)
                    if os.path.exists(filepath):
                        frame_probs_uint8 = np.load(filepath)
                    else:
                        frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    np.save(filepath, frame_probs_uint8)

                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == "validation":

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(
                                    frame_probs_uint8_bis, l_ind, 0, axis=1
                                )

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[
                            np.argmax(frame_probs_uint8_bis, axis=1)
                        ].astype(np.int32)

                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = os.path.join(
                                test_loader.dataset.path,
                                "sequences",
                                test_loader.dataset.sequences[s_ind],
                            )
                            velo_file = os.path.join(
                                seq_path,
                                "velodyne",
                                test_loader.dataset.frames[s_ind][f_ind] + ".bin",
                            )
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = os.path.join(
                                self.test_path, pred_folder, filename[:-4] + ".ply"
                            )
                            pots = np.zeros((0,))  # test_loader.dataset.f_potentials[s_ind][f_ind]
                            if pots.shape[0] > 0:
                                write_ply(
                                    predpath,
                                    [
                                        frame_points[:, :3],
                                        frame_labels,
                                        frame_preds,
                                        pots,
                                    ],
                                    ["x", "y", "z", "gt", "pre", "pots"],
                                )
                            else:
                                write_ply(
                                    predpath,
                                    [frame_points[:, :3], frame_labels, frame_preds],
                                    ["x", "y", "z", "gt", "pre"],
                                )

                            # Also Save lbl probabilities
                            probpath = os.path.join(
                                self.test_path, folder, filename[:-4] + "_probs.ply"
                            )
                            lbl_names = [
                                test_loader.dataset.label_to_names[label_value]
                                for label_value in test_loader.dataset.label_values
                                if label_value not in test_loader.dataset.ignored_labels
                            ]
                            write_ply(
                                probpath,
                                [frame_points[:, :3], frame_probs_uint8],
                                ["x", "y", "z"] + lbl_names,
                            )

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Save some of the frame preds
                        if f_inds[b_i, 1] % 100 == 0:

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8 = np.insert(
                                        frame_probs_uint8, l_ind, 0, axis=1
                                    )

                            # Predicted labels
                            frame_preds = test_loader.dataset.label_values[
                                np.argmax(frame_probs_uint8, axis=1)
                            ].astype(np.int32)

                            # Load points
                            seq_path = os.path.join(
                                test_loader.dataset.path,
                                "sequences",
                                test_loader.dataset.sequences[s_ind],
                            )
                            velo_file = os.path.join(
                                seq_path,
                                "velodyne",
                                test_loader.dataset.frames[s_ind][f_ind] + ".bin",
                            )
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = os.path.join(
                                self.test_path, pred_folder, filename[:-4] + ".ply"
                            )
                            pots = np.zeros((0,))  # test_loader.dataset.f_potentials[s_ind][f_ind]
                            if pots.shape[0] > 0:
                                write_ply(
                                    predpath,
                                    [frame_points[:, :3], frame_preds, pots],
                                    ["x", "y", "z", "pre", "pots"],
                                )
                            else:
                                write_ply(
                                    predpath,
                                    [frame_points[:, :3], frame_preds],
                                    ["x", "y", "z", "pre"],
                                )

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = (
                        "e{:03d}-i{:04d} => {:.1f}% "
                        "(timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%"
                    )
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = (
                        torch.sum(test_loader.dataset.potentials > min_pot + 0.5)
                        .type(torch.int32)
                        .item()
                    )
                    current_num = (
                        pot_num
                        + (i + 1 - config["train"]["validation_size"])
                        * config["train"]["val_batch_num"]
                    )
                    print(
                        message.format(
                            test_epoch,
                            i,
                            100 * i / config["train"]["validation_size"],
                            1000 * (mean_dt[0]),
                            1000 * (mean_dt[1]),
                            1000 * (mean_dt[2]),
                            min_pot,
                            100.0 * current_num / len(test_loader.dataset.potentials),
                        )
                    )

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print(f"Test epoch {test_epoch:d}, end. Min potential = {new_min:.1f}")

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.set == "validation" and last_min % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(
                            truth, preds, test_loader.dataset.label_values
                        ).astype(np.int32)

                    # Show vote results
                    print("\nCompute confusion")

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for frame_idx, _ in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[frame_idx])]
                        val_labels += [np.hstack(all_f_labels[frame_idx])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(f" Stacking time : {t2 - t1:.1f}s")
                    print(f"Confusion time : {t3 - t2:.1f}s")

                    s1 = "\n"
                    for cc in C_tot:
                        for c in cc:
                            s1 += f"{c:7.0f} "
                        s1 += "\n"
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(
                        list(enumerate(test_loader.dataset.label_values))
                    ):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = f"{100 * mIoU:5.2f} | "
                    for IoU in val_IoUs:
                        s2 += f"{100 * IoU:5.2f} "
                    print(s2 + "\n")

                    # Save a report
                    report_file = os.path.join(
                        report_path, f"report_{int(np.floor(last_min)):04d}.txt"
                    )
                    str_report = "Report of the confusion and metrics\n"
                    str_report += "***********************************\n\n\n"
                    str_report += "Confusion matrix:\n\n"
                    str_report += s1
                    str_report += "\nIoU values:\n\n"
                    str_report += s2
                    str_report += "\n\n"
                    with open(report_file, "w") as f:
                        f.write(str_report)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return
