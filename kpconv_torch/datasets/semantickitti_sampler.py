"""
SemanticKittiSampler class

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0902, R0913, R0912, R0914, R0915, R1702, C0209

import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import Sampler

from kpconv_torch.datasets.semantickitti_dataset import SemanticKittiDataset
from kpconv_torch.utils.config import BColors


class SemanticKittiSampler(Sampler):
    """
    Sampler for SemanticKitti
    """

    def __init__(self, dataset: SemanticKittiDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset
        self.calibration_path = os.path.join(self.dataset.datapath, "calibration")
        os.makedirs(self.calibration_path, exist_ok=True)

        # Number of step per epoch
        if dataset.task == "train":
            self.n_ = dataset.config.epoch_steps
        else:
            self.n_ = dataset.config.validation_size

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the
        index of batch element (input sphere) in epoch instead of the list of point indices.
        """

        if self.dataset.balance_classes:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0
            self.dataset.epoch_labels *= 0

            # Number of sphere centers taken per class in each cloud
            num_centers = self.dataset.epoch_inds.shape[0]

            # Generate a list of indices balancing classes and respecting potentials
            gen_indices = []
            gen_classes = []
            for i, c in enumerate(self.dataset.label_values):
                if c not in self.dataset.ignored_labels:

                    # Get the potentials of the frames containing this class
                    class_potentials = self.dataset.potentials[self.dataset.class_frames[i]]

                    if class_potentials.shape[0] > 0:

                        # Get the indices to generate thanks to potentials
                        used_classes = self.dataset.num_classes - len(self.dataset.ignored_labels)
                        class_n = num_centers // used_classes + 1
                        if class_n < class_potentials.shape[0]:
                            _, class_indices = torch.topk(class_potentials, class_n, largest=False)
                        else:
                            class_indices = torch.zeros((0,), dtype=torch.int64)
                            while class_indices.shape[0] < class_n:
                                new_class_inds = torch.randperm(class_potentials.shape[0]).type(
                                    torch.int64
                                )
                                class_indices = torch.cat((class_indices, new_class_inds), dim=0)
                            class_indices = class_indices[:class_n]
                        class_indices = self.dataset.class_frames[i][class_indices]

                        # Add the indices to the generated ones
                        gen_indices.append(class_indices)
                        gen_classes.append(class_indices * 0 + c)

                        # Update potentials
                        update_inds = torch.unique(class_indices)
                        self.dataset.potentials[update_inds] = torch.ceil(
                            self.dataset.potentials[update_inds]
                        )
                        self.dataset.potentials[update_inds] += torch.from_numpy(
                            np.random.rand(update_inds.shape[0]) * 0.1 + 0.1
                        )

                    else:
                        error_message = "\nIt seems there is a problem with the class statistics "
                        error_message += "of your dataset, saved in the variable "
                        error_message += "dataset.class_frames.\nHere are the current statistics:\n"
                        error_message += "{:>15s} {:>15s}\n".format("Class", "# of frames")
                        for iii, _ in enumerate(self.dataset.label_values):
                            error_message += f"{self.dataset.label_names[iii]:>15s} "
                            error_message += "{len(self.dataset.class_frames[iii]):>15d}\n"
                        error_message += "\nThis error is raised if one of the classes "
                        error_message += "is not ignored and does not appear "
                        error_message += "in any of the frames of the dataset.\n"
                        raise ValueError(error_message)

            # Stack the chosen indices of all classes
            gen_indices = torch.cat(gen_indices, dim=0)
            gen_classes = torch.cat(gen_classes, dim=0)

            # Shuffle generated indices
            rand_order = torch.randperm(gen_indices.shape[0])[:num_centers]
            gen_indices = gen_indices[rand_order]
            gen_classes = gen_classes[rand_order]

            # Update epoch inds
            self.dataset.epoch_inds += gen_indices
            self.dataset.epoch_labels += gen_classes.type(torch.int32)

        else:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0
            self.dataset.epoch_labels *= 0

            # Number of sphere centers taken per class in each cloud
            num_centers = self.dataset.epoch_inds.shape[0]

            # Get the list of indices to generate thanks to potentials
            if num_centers < self.dataset.potentials.shape[0]:
                _, gen_indices = torch.topk(
                    self.dataset.potentials, num_centers, largest=False, sorted=True
                )
            else:
                gen_indices = torch.randperm(self.dataset.potentials.shape[0])
                while gen_indices.shape[0] < num_centers:
                    new_gen_indices = torch.randperm(self.dataset.potentials.shape[0]).type(
                        torch.int32
                    )
                    gen_indices = torch.cat((gen_indices, new_gen_indices), dim=0)
                gen_indices = gen_indices[:num_centers]

            # Update potentials (Change the order for the next epoch)
            self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
            self.dataset.potentials[gen_indices] += torch.from_numpy(
                np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1
            )

            # Update epoch inds
            self.dataset.epoch_inds += gen_indices

        # Generator loop
        yield from range(self.n_)

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.n_

    def calib_max_in(self, config, dataloader, untouched_ratio=0.8, verbose=True, force_redo=False):
        """
        Method performing batch and neighbors calibration.

        Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch)
        so that the average batch size (number of stacked pointclouds) is the one asked.

        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors
        allowed in convolutions) so that 90% of the neighborhoods remain untouched. There is a
        limit for each layer.

        """

        # Previously saved calibration
        print("\nStarting Calibration of max_in_points value (use verbose=True for more details)")
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # Load max_in_limit dictionary
        max_in_lim_file = os.path.join(self.calibration_path, "max_in_limits.pkl")
        if os.path.exists(max_in_lim_file):
            with open(max_in_lim_file, "rb") as file:
                max_in_lim_dict = pickle.load(file)
        else:
            max_in_lim_dict = {}

        # Check if the max_in limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = "balanced"
        else:
            sampler_method = "random"

        t = self.dataset.config["kpconv"]["first_subsampling_dl"]
        key = f"{sampler_method}_{self.dataset.in_r:3f}_{t:3f}"
        if not redo and key in max_in_lim_dict:
            self.dataset.max_in_p = max_in_lim_dict[key]
        else:
            redo = True

        if verbose:
            print("\nPrevious calibration found:")
            print("Check max_in limit dictionary")
            if key in max_in_lim_dict:
                color = BColors.OKGREEN
                v = str(int(max_in_lim_dict[key]))
            else:
                color = BColors.FAIL
                v = "?"
            print(f'{color}"{key}": {v}{BColors.ENDC}')

        if redo:
            # Batch calib parameters
            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            all_lengths = []
            n_ = 1000

            # Perform calibration
            for _ in range(10):
                for batch in dataloader:

                    # Control max_in_points value
                    all_lengths += batch.lengths[0].tolist()

                    # Convergence
                    if len(all_lengths) > n_:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if t - last_display > 1.0:
                        last_display = t
                        message = "Collecting {:d} in_points: {:5.1f}%"
                        print(message.format(n_, 100 * len(all_lengths) / n_))

                if breaking:
                    break

            self.dataset.max_in_p = int(np.percentile(all_lengths, 100 * untouched_ratio))

            if verbose:

                # Create histogram
                pass

            # Save max_in_limit dictionary
            print("New max_in_p = ", self.dataset.max_in_p)
            max_in_lim_dict[key] = self.dataset.max_in_p
            with open(max_in_lim_file, "wb") as file:
                pickle.dump(max_in_lim_dict, file)

        # Update value in config
        if self.dataset.task == "train":
            config.max_in_points = self.dataset.max_in_p
        else:
            config.max_val_points = self.dataset.max_in_p

        print(f"Calibration done in {time.time() - t0:.1f}s\n")

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.

        Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch)
        so that the average batch size (number of stacked pointclouds) is the one asked.

        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors
        allowed in convolutions) so that 90% of the neighborhoods remain untouched. There is a
        limit for each layer.

        """
        # Previously saved calibration
        print("\nStarting Calibration (use verbose=True for more details)")
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # Load batch_limit dictionary
        batch_lim_file = os.path.join(self.calibration_path, "batch_limits.pkl")
        if os.path.exists(batch_lim_file):
            with open(batch_lim_file, "rb") as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = "balanced"
        else:
            sampler_method = "random"

        t = self.dataset.config["kpconv"]["first_subsampling_dl"]

        key = (
            f"{sampler_method}_{self.dataset.in_r:3f}_"
            f"{t:f}_"
            f"{self.dataset.batch_num:d}_{self.dataset.max_in_p:d}"
        )
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print("\nPrevious calibration found:")
            print("Check batch limit dictionary")
            if key in batch_lim_dict:
                color = BColors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = BColors.FAIL
                v = "?"
            print(f'{color}"{key}": {v}{BColors.ENDC}')

        # Neighbors limit
        # Load neighb_limits dictionary
        neighb_lim_file = os.path.join(self.calibration_path, "neighbors_limits.pkl")
        if os.path.exists(neighb_lim_file):
            with open(neighb_lim_file, "rb") as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.num_layers):

            dl = self.dataset.config["kpconv"]["first_subsampling_dl"] * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config["train"]["batch_num"]
            else:
                r = dl * self.dataset.config["kpconv"]["conv_radius"]

            key = f"{sampler_method}_{self.dataset.max_in_p:d}_{dl:.3f}_{r:.3f}"
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print("Check neighbors limit dictionary")
            for layer_ind in range(self.dataset.num_layers):
                dl = self.dataset.config["kpconv"]["first_subsampling_dl"] * (2**layer_ind)
                if self.dataset.deform_layers[layer_ind]:
                    r = dl * self.dataset.config["train"]["batch_num"]
                else:
                    r = dl * self.dataset.config["kpconv"]["conv_radius"]
                key = f"{sampler_method}_{self.dataset.max_in_p:d}_{dl:3f}_{r:3f}"

                if key in neighb_lim_dict:
                    color = BColors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = BColors.FAIL
                    v = "?"
                print(f'{color}"{key}": {v}{BColors.ENDC}')

        if redo:
            # Neighbors calib parameters
            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(
                np.ceil(4 / 3 * np.pi * (self.dataset.config["train"]["batch_num"] + 1) ** 3)
            )

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.num_layers, hist_n), dtype=np.int32)

            # Batch calib parameters
            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.batch_num

            # Calibration parameters
            low_pass_t = 10
            kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Save input pointcloud sizes to control max_in_points
            cropped_n = 0
            all_n = 0

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            # Perform calibration
            for _ in range(10):
                for batch in dataloader:

                    # Control max_in_points value
                    are_cropped = batch.lengths[0] > self.dataset.max_in_p - 1
                    cropped_n += torch.sum(are_cropped.type(torch.int32)).item()
                    all_n += int(batch.lengths[0].shape[0])

                    # Update neighborhood histogram
                    counts = [
                        np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1)
                        for neighb_mat in batch.neighbors
                    ]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.frame_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_t

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit[0] += kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_t = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = "Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}"
                        print(message.format(i, estim_b, int(self.dataset.batch_limit[0])))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print("\n**************************************************\n")
                line0 = "neighbors_num "
                for layer in range(neighb_hists.shape[0]):
                    line0 = line0.join(f"|  layer {layer:2d}  ")
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = f"     {neighb_size:4d}     "
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = BColors.FAIL
                        else:
                            color = BColors.OKGREEN
                        line0 += "|{:}{:10d}{:}  ".format(
                            color, neighb_hists[layer, neighb_size], BColors.ENDC
                        )

                    print(line0)

                print("\n**************************************************\n")
                print("\nchosen neighbors limits: ", percentiles)
                print()

            # Control max_in_points value
            print("\n**************************************************\n")
            if cropped_n > 0.3 * all_n:
                color = BColors.FAIL
            else:
                color = BColors.OKGREEN
            print(f"Current value of max_in_points {self.dataset.max_in_p:d}")
            t = BColors.ENDC
            print(f"  > {color}{100 * cropped_n / all_n:.1f}% inputs are cropped{t}")
            if cropped_n > 0.3 * all_n:
                print("\nTry a higher max_in_points value\n")
            print("\n**************************************************\n")

            # Save batch_limit dictionary
            t1 = self.dataset.config["kpconv"]["first_subsampling_dl"]
            t2 = self.dataset.config["train"]["batch_num"]
            key = (
                f"{sampler_method}_{self.dataset.in_r:3f}_"
                f"{t1:3f}_"
                f"{t2:d}_{self.dataset.max_in_p:d}"
            )
            batch_lim_dict[key] = float(self.dataset.batch_limit[0])
            with open(batch_lim_file, "wb") as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.num_layers):
                dl = self.dataset.config["kpconv"]["first_subsampling_dl"] * (2**layer_ind)
                if self.dataset.deform_layers[layer_ind]:
                    r = dl * self.dataset.config["train"]["batch_num"]
                else:
                    r = dl * self.dataset.config["kpconv"]["conv_radius"]
                key = f"{sampler_method}_{self.dataset.max_in_p:d}_{dl:.3f}_{r:.3f}"
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, "wb") as file:
                pickle.dump(neighb_lim_dict, file)

        print(f"Calibration done in {time.time() - t0:.1f}s\n")
