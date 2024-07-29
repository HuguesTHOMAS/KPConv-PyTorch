"""
NPM3DSampler class

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915

import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Sampler

from kpconv_torch.datasets.npm3d_dataset import NPM3DDataset
from kpconv_torch.utils.config import BColors


class NPM3DSampler(Sampler):
    """
    Sampler for NPM3D
    """

    def __init__(self, dataset: NPM3DDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset
        self.calibration_path = os.path.join(self.dataset.datapath, "calibration")
        os.makedirs(self.calibration_path, exist_ok=True)

        # Number of step per epoch
        if dataset.task == "train":
            self.n = dataset.config.epoch_steps
        else:
            self.n = dataset.config.validation_size

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield
        the index of batch element (input sphere) in epoch instead of the list of point indices.
        """

        if not self.dataset.use_potentials:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int64)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.n * self.dataset.config["train"]["batch_num"]
            random_pick_n = int(np.ceil(num_centers / self.dataset.num_classes))

            # Choose random points of each class for each cloud
            for label_ind, label in enumerate(self.dataset.label_values):
                if label not in self.dataset.ignored_labels:

                    # Gather indices of the points with this label in all the input clouds
                    all_label_indices = []
                    for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        all_label_indices.append(
                            np.vstack(
                                (
                                    np.full(label_indices.shape, cloud_ind, dtype=np.int64),
                                    label_indices,
                                )
                            )
                        )

                    # Stack them: [2, N1+N2+...]
                    all_label_indices = np.hstack(all_label_indices)

                    # Select a a random number amongst them
                    n_inds = all_label_indices.shape[1]
                    if n_inds < random_pick_n:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            chosen_label_inds = np.hstack(
                                (
                                    chosen_label_inds,
                                    all_label_indices[:, np.random.permutation(n_inds)],
                                )
                            )
                        warnings.warn(
                            f"When choosing random epoch indices "
                            f'(config["input"]["use_potentials"]=False), '
                            f"class {label:d}: {self.dataset.label_names[label_ind]} only had "
                            f"{n_inds:d} available points, while we needed {random_pick_n:d}. "
                            "Repeating indices in the same epoch",
                            stacklevel=2,
                        )

                    elif n_inds < 50 * random_pick_n:
                        rand_inds = np.random.choice(n_inds, size=random_pick_n, replace=False)
                        chosen_label_inds = all_label_indices[:, rand_inds]

                    else:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            rand_inds = np.unique(
                                np.random.choice(n_inds, size=2 * random_pick_n, replace=True)
                            )
                            chosen_label_inds = np.hstack(
                                (chosen_label_inds, all_label_indices[:, rand_inds])
                            )
                        chosen_label_inds = chosen_label_inds[:, :random_pick_n]

                    # Stack for each label
                    all_epoch_inds = np.hstack((all_epoch_inds, chosen_label_inds))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])[:num_centers]
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds)

        # Generator loop
        yield from range(self.n)

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.n

    def fast_calib(self):
        """
        This method calibrates the batch sizes while ensuring the potentials are well
        initialized. Indeed on a dataset like Semantic3D, before potential have been updated over
        the dataset, there are chances that all the dense area are picked in the begining and in
        the end, we will have very large batch of small point clouds.
        """

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config["train"]["batch_num"]

        # Calibration parameters
        low_pass_t = 10
        kp = 100.0
        finer = False
        breaking = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(2)

        for _ in range(10):
            for i, test in enumerate(self):

                # New time
                t = t[-1:]
                t += [time.time()]

                # batch length
                b = len(test)

                # Update estim_b (low pass filter)
                estim_b += (b - estim_b) / low_pass_t

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_t = 100
                    finer = True

                # Convergence
                if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                    breaking = True
                    break

                # Average timing
                t += [time.time()]
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = (
                        "Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d},  //  {:.1f}ms {:.1f}ms"
                    )
                    print(
                        message.format(
                            i,
                            estim_b,
                            int(self.dataset.batch_limit),
                            1000 * mean_dt[0],
                            1000 * mean_dt[1],
                        )
                    )

            if breaking:
                break

    def calibration(self, config, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.

        Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch)
        so that the average batch size (number of stacked pointclouds) is the one asked.

        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors
        allowed in convolutions) so that 90% of the neighborhoods remain untouched. There is a
        limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print("\nStarting Calibration (use verbose=True for more details)")
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = os.path.join(self.calibration_path, "batch_limits.pkl")
        if os.path.exists(batch_lim_file):
            with open(batch_lim_file, "rb") as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if config["input"]["use_potentials"]:
            sampler_method = "potentials"
        else:
            sampler_method = "random"
        t1 = self.dataset.config["input"]["sphere_radius"]
        t2 = self.dataset.config["kpconv"]["first_subsampling_dl"]
        t3 = self.dataset.config["train"]["batch_num"]
        key = f"{sampler_method}_{t1:3f}_" f"{t2:3f}_{t3:d}"
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
            if self.dataset.deform_layers[layer_ind]:
                r = dl * self.dataset.config["kpconv"]["deform_radius"]
            else:
                r = dl * self.dataset.config["kpconv"]["conv_radius"]

            key = f"{dl:.3f}_{r:.3f}"
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
                    r = dl * self.dataset.config["kpconv"]["deform_radius"]
                else:
                    r = dl * self.dataset.config["kpconv"]["conv_radius"]
                key = f"{dl:.3f}_{r:.3f}"

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
                np.ceil(4 / 3 * np.pi * (self.dataset.config["kpconv"]["deform_radius"] + 1) ** 3)
            )

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.num_layers, hist_n), dtype=np.int32)

            # Batch calib parameters
            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config["train"]["batch_num"]

            # Expected batch size order of magnitude
            expected_n = 100000

            # Calibration parameters. Higher means faster but can also become unstable.

            # Reduce kp/kd if small GPU: the total number of points per batch will be smaller
            low_pass_t = 100
            kp = expected_n / 200
            ki = 0.001 * kp
            kd = 5 * kp
            finer = False
            stabilized = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False
            error_i = 0
            error_d = 0
            last_error = 0

            debug_in = []
            debug_out = []
            debug_b = []
            debug_estim_b = []

            # Perform calibration
            # number of batch per epoch
            sample_batches = 999
            for _ in range((sample_batches // self.n) + 1):
                for batch in dataloader:

                    # Update neighborhood histogram
                    counts = [
                        np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1)
                        for neighb_mat in batch.neighbors
                    ]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.cloud_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_t

                    # Estimate error (noisy)
                    error = target_b - b
                    error_i += error
                    error_d = error - last_error
                    last_error = error

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 30:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += kp * error + ki * error_i + kd * error_d

                    # Unstability detection
                    if not stabilized and self.dataset.batch_limit < 0:
                        kp *= 0.1
                        ki *= 0.1
                        kd *= 0.1
                        stabilized = True

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
                        print(message.format(i, estim_b, int(self.dataset.batch_limit)))

                    # Debug plots
                    debug_in.append(int(batch.points[0].shape[0]))
                    debug_out.append(int(self.dataset.batch_limit))
                    debug_b.append(b)
                    debug_estim_b.append(estim_b)

                if breaking:
                    break

            # Plot in case we did not reach convergence
            if not breaking:
                print(
                    "ERROR: It seems that the calibration has not reached convergence. "
                    "Here are some plot to understand why:"
                )
                print("If you notice unstability, reduce the expected_n value")
                print("If convergece is too slow, increase the expected_n value")

                plt.figure()
                plt.plot(debug_in)
                plt.plot(debug_out)

                plt.figure()
                plt.plot(debug_b)
                plt.plot(debug_estim_b)

                plt.show()

                raise ValueError("Convergence was not reached")

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
                        line0 = line0.join(
                            f"|{color}{neighb_hists[layer, neighb_size]:10d}{BColors.ENDC}  "
                        )
                    print(line0)

                print("\n**************************************************\n")
                print("\nChosen neighbors limits: ", percentiles)
                print()

            # Save batch_limit dictionary
            if config["input"]["use_potentials"]:
                sampler_method = "potentials"
            else:
                sampler_method = "random"

            t1 = config["input"]["sphere_radius"]
            t2 = config["kpconv"]["first_subsampling_dl"]
            t3 = config["train"]["batch_num"]

            key = f"{sampler_method}_{t1:3f}_" f"{t2:3f}_" f"{t3:d}"
            batch_lim_dict[key] = float(self.dataset.batch_limit)

            with open(batch_lim_file, "wb") as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.num_layers):
                dl = self.dataset.config["kpconv"]["first_subsampling_dl"] * (2**layer_ind)
                if self.dataset.deform_layers[layer_ind]:
                    r = dl * self.dataset.config["train"]["batch_num"]
                else:
                    r = dl * self.dataset.config["kpconv"]["conv_radius"]
                key = f"{dl:.3f}_{r:.3f}"
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, "wb") as file:
                pickle.dump(neighb_lim_dict, file)

        print(f"Calibration done in {time.time() - t0:.1f}s\n")
