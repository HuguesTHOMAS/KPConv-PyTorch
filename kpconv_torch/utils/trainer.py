import os
import time
from pathlib import Path

import numpy as np
import torch

from kpconv_torch.utils.config import save_config
from kpconv_torch.utils.metrics import fast_confusion, IoU_from_confusions
from kpconv_torch.io.ply import write_ply


def get_train_save_path(output_dir: Path, chosen_log: Path) -> Path:
    if chosen_log is None and output_dir is None:
        train_path = None
    elif chosen_log is not None:
        train_path = chosen_log
    elif output_dir is not None:
        train_path = output_dir / time.strftime("Log_%Y-%m-%d_%H-%M-%S", time.gmtime())
    if train_path is not None and not os.path.exists(train_path):
        os.makedirs(train_path)
    return train_path


class ModelTrainer:
    def __init__(
        self,
        net,
        config,
        chkp_path=None,
        train_save_path=None,
        finetune=False,
        on_gpu=True,
    ):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param chosen_log: path to the folder containing the trained model
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        self.config = config
        self.num_classes = len(self.config["model"]["label_to_names"])

        # Learning rate decays, dictionary of all decay values with their epoch {epoch: decay}
        self.lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, config["train"]["max_epoch"])}

        ############
        # Parameters
        ############
        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if "offset" in k]
        other_params = [v for k, v in net.named_parameters() if "offset" not in k]
        deform_lr = config["train"]["learning_rate"] * config["train"]["deform_lr_factor"]
        self.optimizer = torch.optim.SGD(
            [{"params": other_params}, {"params": deform_params, "lr": deform_lr}],
            lr=config["train"]["learning_rate"],
            momentum=config["train"]["momentum"],
            weight_decay=config["train"]["weight_decay"],
        )

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if chkp_path is not None:
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint["model_state_dict"])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epoch = checkpoint["epoch"]
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        self.train_save_path = train_save_path
        if config["model"]["saving"]:
            save_config(self.train_save_path, config)

        return

    def train(self, net, training_loader, val_loader, chosen_log):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if self.config["model"]["saving"]:
            # Training log file
            if not (self.train_save_path / "training.txt").exists():
                with open(self.train_save_path / "training.txt", "w") as fobj:
                    fobj.write("epochs steps out_loss offset_loss train_accuracy time\n")

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = self.train_save_path / "running_PID.txt"
            if not os.path.exists(PID_file):
                with open(PID_file, "w") as fobj:
                    fobj.write(
                        "Remove this file when you want the training to stop. "
                        "The *kpconv* program uses it as a witness, and cancels the remaining "
                        "training epochs if it is removed, when config.saving is True."
                    )

            # Checkpoints directory
            checkpoint_directory = self.train_save_path / "checkpoints"
            if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        for epoch in range(self.config["train"]["max_epoch"]):

            self.step = 0
            for batch in training_loader:

                # Check kill signal (running_PID.txt deleted)
                if self.config["model"]["saving"] and not os.path.exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if "cuda" in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = net(batch, self.config)
                loss = net.loss(outputs, batch.labels)
                acc = net.accuracy(outputs, batch.labels)

                t += [time.time()]

                # Backward + optimize
                loss.backward()

                if self.config["train"]["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_value_(
                        net.parameters(), self.config["train"]["grad_clip_norm"]
                    )
                self.optimizer.step()

                torch.cuda.empty_cache()
                if "cuda" in self.device.type:
                    torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = (
                        "e{:03d}-i{:04d} => L={:.3f} acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})"
                    )
                    print(
                        message.format(
                            self.epoch,
                            self.step,
                            loss.item(),
                            100 * acc,
                            1000 * mean_dt[0],
                            1000 * mean_dt[1],
                            1000 * mean_dt[2],
                        )
                    )

                # Log file
                if self.config["model"]["saving"]:
                    with open(self.train_save_path / "training.txt", "a") as file:
                        message = "{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n"
                        file.write(
                            message.format(
                                self.epoch,
                                self.step,
                                net.output_loss,
                                net.reg_loss,
                                acc,
                                t[-1] - t0,
                            )
                        )

                self.step += 1

            ##############
            # End of epoch
            ##############

            # Remove File for kill signal if last epoch before the end
            if epoch == self.config["train"]["max_epoch"] - 1 and os.path.exists(PID_file):
                os.remove(PID_file)

            # Update learning rate
            if self.epoch in self.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= self.lr_decays[self.epoch]

            # Update epoch
            self.epoch += 1

            # Saving
            if self.config["model"]["saving"]:
                # Get current state dict
                save_dict = {
                    "epoch": self.epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "chosen_log": chosen_log,
                }

                # Save current state of the network (for restoring purposes)
                checkpoint_path = os.path.join(checkpoint_directory, "current_chkp.tar")
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if self.epoch % self.config["train"]["checkpoint_gap"] == 0:
                    checkpoint_path = os.path.join(
                        checkpoint_directory, f"chkp_{self.epoch:04d}.tar"
                    )
                    torch.save(save_dict, checkpoint_path)

            # Validation
            net.eval()
            self.validation(net, val_loader)
            net.train()

            # Check kill signal (running_PID.txt deleted)
            if self.config["model"]["saving"] and not os.path.exists(PID_file):
                break

        print("Finished Training")
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader):
        if self.config["input"]["task"] == "classification":
            self.object_classification_validation(net, val_loader)
        elif self.config["input"]["task"] == "segmentation":
            self.object_segmentation_validation(net, val_loader)
        elif self.config["input"]["task"] == "cloud_segmentation":
            self.cloud_segmentation_validation(net, val_loader)
        elif self.config["input"]["task"] == "slam_segmentation":
            self.slam_segmentation_validation(net, val_loader)
        else:
            raise ValueError("No validation method implemented for this network type")

    def object_classification_validation(self, net, val_loader):
        """
        Perform a round of validation and show/save results
        :param net: network object
        :param val_loader: data loader for validation set
        :param config: configuration object
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Number of classes predicted by the model
        nc_model = val_loader.dataset.num_classes

        softmax = torch.nn.Softmax(1)

        # Initialize global prediction over all models
        if not hasattr(self, "val_probs"):
            self.val_probs = np.zeros((val_loader.dataset.num_models, nc_model))

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in val_loader:

            # New time
            t = t[-1:]
            t += [time.time()]

            if "cuda" in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, self.config)

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
                message = "Validation : {:.1f}% (timings : {:4.2f} {:4.2f})"
                print(
                    message.format(
                        100 * len(obj_inds) / self.config["train"]["validation_size"],
                        1000 * (mean_dt[0]),
                        1000 * (mean_dt[1]),
                    )
                )

        # Stack all validation predictions
        probs = np.vstack(probs)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)

        ###################
        # Voting validation
        ###################

        self.val_probs[obj_inds] = val_smooth * self.val_probs[obj_inds] + (1 - val_smooth) * probs

        ############
        # Confusions
        ############

        validation_labels = np.array(val_loader.dataset.label_values)

        # Compute classification results
        C1 = fast_confusion(targets, np.argmax(probs, axis=1), validation_labels)

        # Compute votes confusion
        C2 = fast_confusion(
            val_loader.dataset.input_labels,
            np.argmax(self.val_probs, axis=1),
            validation_labels,
        )

        # Saving
        if self.config["model"]["saving"]:
            print("Save confusions")
            conf_list = [C1, C2]
            file_list = ["val_confs.txt", "vote_confs.txt"]
            for conf, conf_file in zip(conf_list, file_list):
                test_file = self.train_save_path / conf_file
                if os.path.exists(test_file):
                    with open(test_file, "a") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write("%d " % value)
                        text_file.write("\n")
                else:
                    with open(test_file, "w") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write("%d " % value)
                        text_file.write("\n")

        val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        vote_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
        print(f"Accuracies : val = {val_ACC:.1f}% / vote = {vote_ACC:.1f}%")

        return C1

    def cloud_segmentation_validation(self, net, val_loader, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Do not validate if dataset has no validation cloud
        # if val_loader.dataset.validation_task not in val_loader.dataset.all_tasks:
        if len(val_loader.dataset.config["train"]["validation_cloud_names"]) == 0:
            return

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = self.num_classes

        # Initiate global prediction over validation clouds
        if not hasattr(self, "validation_probs"):
            self.validation_probs = [
                np.zeros((input_label.shape[0], nc_model))
                for input_label in val_loader.dataset.input_labels
            ]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum(
                        [
                            np.sum(labels == label_value)
                            for labels in val_loader.dataset.validation_labels
                        ]
                    )
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if "cuda" in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, self.config)

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
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
                target = labels[i0 : i0 + length]
                probs = stacked_probs[i0 : i0 + length]
                inds = in_inds[i0 : i0 + length]
                c_i = cloud_inds[b_i]

                # Update current probs in whole cloud
                self.validation_probs[c_i][inds] = (
                    val_smooth * self.validation_probs[c_i][inds] + (1 - val_smooth) * probs
                )

                # Stack all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = "Validation : {:.1f}% (timings : {:4.2f} {:4.2f})"
                print(
                    message.format(
                        100 * i / self.config["train"]["validation_size"],
                        1000 * (mean_dt[0]),
                        1000 * (mean_dt[1]),
                    )
                )

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(
                np.int32
            )

        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        t5 = time.time()

        # Saving
        if self.config["model"]["saving"]:
            # Name of saving file
            test_file = self.train_save_path / "val_IoUs.txt"

            # Line to write:
            line = ""
            for IoU in IoUs:
                line += f"{IoU:.3f} "
            line = line + "\n"

            # Write in file
            if os.path.exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

            # Save potentials
            if self.config["input"]["use_potentials"]:
                pot_path = self.train_save_path / "potentials"
                if not os.path.exists(pot_path):
                    os.makedirs(pot_path)
                files = val_loader.dataset.files
                for i, file_path in enumerate(files):
                    pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
                    cloud_name = file_path.name
                    pot_name = os.path.join(pot_path, cloud_name)
                    pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
                    write_ply(
                        pot_name,
                        [pot_points.astype(np.float32), pots],
                        ["x", "y", "z", "pots"],
                    )

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        t = self.config["dataset"]
        print(f"{t} mean IoU = {mIoU:.1f}%")

        # Save predicted cloud occasionally
        if (
            self.config["model"]["saving"]
            and self.epoch % self.config["train"]["checkpoint_gap"] == 0
        ):
            val_path = self.train_save_path / f"val_preds_{self.epoch:d}"
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):

                # Get points
                points = val_loader.dataset.load_evaluation_points(file_path)

                # Get probs on our own ply points
                sub_probs = self.validation_probs[i]

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                # Get the predicted labels
                sub_preds = val_loader.dataset.label_values[
                    np.argmax(sub_probs, axis=1).astype(np.int32)
                ]

                # Reproject preds on the evaluations points
                preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)

                # Path of saved validation file
                cloud_name = file_path.name
                val_name = os.path.join(val_path, cloud_name)

                # Save file
                labels = val_loader.dataset.validation_labels[i].astype(np.int32)
                write_ply(
                    val_name, [points, preds, labels], ["x", "y", "z", "preds", "classification"]
                )

        # Display timings
        t7 = time.time()
        if debug:
            print("\n************************\n")
            print("Validation timings:")
            print(f"Init ...... {t1 - t0:.1f}s")
            print(f"Loop ...... {t2 - t1:.1f}s")
            print(f"Confs ..... {t3 - t2:.1f}s")
            print(f"Confs bis . {t4 - t3:.1f}s")
            print(f"IoU ....... {t5 - t4:.1f}s")
            print(f"Save1 ..... {t6 - t5:.1f}s")
            print(f"Save2 ..... {t7 - t6:.1f}s")
            print("\n************************\n")

        return

    def slam_segmentation_validation(self, net, val_loader, debug=True):
        """
        Validation method for slam segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Do not validate if dataset has no validation cloud
        if val_loader is None:
            return

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        softmax = torch.nn.Softmax(1)

        # Create folder for validation predictions
        if not os.path.exists(self.train_save_path / "val_preds"):
            os.makedirs(self.train_save_path / "val_preds")

        # initiate the dataset validation containers
        val_loader.dataset.val_points = []
        val_loader.dataset.val_labels = []

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        inds = []
        val_i = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if "cuda" in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, self.config)

            # Get probs and labels
            stk_probs = softmax(outputs).cpu().detach().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            f_inds = batch.frame_inds.cpu().numpy()
            r_inds_list = batch.reproj_inds
            r_mask_list = batch.reproj_masks
            labels_list = batch.val_labels
            if "cuda" in self.device.type:
                torch.cuda.synchronize(self.device)

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

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

                # Predicted labels
                preds = val_loader.dataset.label_values[np.argmax(proj_probs, axis=1)]

                # Save predictions in a binary file
                filename = f"{val_loader.dataset.sequences[s_ind]}_{f_ind:7d}.npy"
                filepath = self.train_save_path / "val_preds" / filename
                if os.path.exists(filepath):
                    frame_preds = np.load(filepath)
                else:
                    frame_preds = np.zeros(frame_labels.shape, dtype=np.uint8)
                frame_preds[proj_mask] = preds.astype(np.uint8)
                np.save(filepath, frame_preds)

                # Save some of the frame pots
                if f_ind % 20 == 0:
                    seq_path = os.path.join(
                        val_loader.dataset.path,
                        "sequences",
                        val_loader.dataset.sequences[s_ind],
                    )
                    velo_file = os.path.join(
                        seq_path,
                        "velodyne",
                        val_loader.dataset.frames[s_ind][f_ind] + ".bin",
                    )
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    frame_points = frame_points.reshape((-1, 4))
                    write_ply(
                        filepath[:-4] + "_pots.ply",
                        [frame_points[:, :3], frame_labels, frame_preds],
                        ["x", "y", "z", "gt", "pre"],
                    )

                # Update validation confusions
                frame_C = fast_confusion(
                    frame_labels,
                    frame_preds.astype(np.int32),
                    val_loader.dataset.label_values,
                )
                val_loader.dataset.val_confs[s_ind][f_ind, :, :] = frame_C

                # Stack all prediction for this epoch
                predictions += [preds]
                targets += [frame_labels[proj_mask]]
                inds += [f_inds[b_i, :]]
                val_i += 1
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = "Validation : {:.1f}% (timings : {:4.2f} {:4.2f})"
                print(
                    message.format(
                        100 * i / self.config["train"]["validation_size"],
                        1000 * (mean_dt[0]),
                        1000 * (mean_dt[1]),
                    )
                )

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(
                np.int32
            )

        t3 = time.time()

        #######################################
        # Results on this subpart of validation
        #######################################

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        C *= np.expand_dims(val_loader.dataset.class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        #####################################
        # Results on the whole validation set
        #####################################

        t4 = time.time()

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in val_loader.dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)

        if debug:
            s = "\n"
            for cc in C_tot:
                for c in cc:
                    s += f"{c:8.1f} "
                s += "\n"
            print(s)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        t5 = time.time()

        # Saving
        if self.config["model"]["saving"]:

            IoU_list = [IoUs, val_IoUs]
            file_list = ["subpart_IoUs.txt", "val_IoUs.txt"]
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = self.train_save_path / IoU_file

                # Line to write:
                line = ""
                for IoU in IoUs_to_save:
                    line += f"{IoU:.3f} "
                line = line + "\n"

                # Write in file
                if os.path.exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        t = self.config["dataset"]
        print(f"{t} : subpart mIoU = {mIoU:1f} %")
        mIoU = 100 * np.mean(val_IoUs)
        print(f"{t} :     val mIoU = {mIoU:1f} %")

        t6 = time.time()

        # Display timings
        if debug:
            print("\n************************\n")
            print("Validation timings:")
            print(f"Init ...... {t1 - t0:.1f}s")
            print(f"Loop ...... {t2 - t1:.1f}s")
            print(f"Confs ..... {t3 - t2:.1f}s")
            print(f"IoU1 ...... {t4 - t3:.1f}s")
            print(f"IoU2 ...... {t5 - t4:.1f}s")
            print(f"Save ...... {t6 - t5:.1f}s")
            print("\n************************\n")

        return
