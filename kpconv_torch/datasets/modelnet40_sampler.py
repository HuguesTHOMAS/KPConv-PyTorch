"""
ModelNet40Sampler class

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915

import os
import pickle
import time

import numpy as np
from torch.utils.data import Sampler

from kpconv_torch.datasets.modelnet40_dataset import ModelNet40Dataset
from kpconv_torch.utils.config import BColors


class ModelNet40Sampler(Sampler):
    """
    Sampler for ModelNet40
    """

    def __init__(
        self,
        dataset: ModelNet40Dataset,
        use_potential=True,
        balance_labels=False,
    ):
        Sampler.__init__(self, dataset)

        # Does the sampler use potential for regular sampling
        self.use_potential = use_potential

        # Should be balance the classes when sampling
        self.balance_labels = balance_labels

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset
        self.calibration_path = os.path.join(self.dataset.datapath, "calibration")
        os.makedirs(self.calibration_path, exist_ok=True)

        # Create potentials
        if self.use_potential:
            self.potentials = np.random.rand(len(dataset.input_labels)) * 0.1 + 0.1
        else:
            self.potentials = None

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = 10000

    def __iter__(self):
        """
        Yield next batch indices here
        """

        # Initialize the list of generated indices
        if self.use_potential:
            if self.balance_labels:

                gen_indices = []
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                for label_value in self.dataset.label_values:

                    # Get the potentials of the objects of this class
                    label_inds = np.where(np.equal(self.dataset.input_labels, label_value))[0]
                    class_potentials = self.potentials[label_inds]

                    # Get the indices to generate thanks to potentials
                    if pick_n < class_potentials.shape[0]:
                        pick_indices = np.argpartition(class_potentials, pick_n)[:pick_n]
                    else:
                        pick_indices = np.random.permutation(class_potentials.shape[0])
                    class_indices = label_inds[pick_indices]
                    gen_indices.append(class_indices)

                # Stack the chosen indices of all classes
                gen_indices = np.random.permutation(np.hstack(gen_indices))

            else:

                # Get indices with the minimum potential
                if self.dataset.epoch_n < self.potentials.shape[0]:
                    gen_indices = np.argpartition(self.potentials, self.dataset.epoch_n)[
                        : self.dataset.epoch_n
                    ]
                else:
                    gen_indices = np.random.permutation(self.potentials.shape[0])
                gen_indices = np.random.permutation(gen_indices)

            # Update potentials (Change the order for the next epoch)
            self.potentials[gen_indices] = np.ceil(self.potentials[gen_indices])
            self.potentials[gen_indices] += np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1

        else:
            if self.balance_labels:
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                gen_indices = []
                for label_value in self.dataset.label_values:
                    label_inds = np.where(np.equal(self.dataset.input_labels, label_value))[0]
                    rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                    gen_indices += [rand_inds]
                gen_indices = np.random.permutation(np.hstack(gen_indices))
            else:
                gen_indices = np.random.permutation(self.dataset.num_models)[: self.dataset.epoch_n]

        # Generator loop
        # Initialize concatenation lists
        ti_list = []
        batch_n = 0

        # Generator loop
        for p_i in gen_indices:

            # Size of picked cloud
            n = self.dataset.input_points[p_i].shape[0]

            # In case batch is full, yield it and reset it
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            # Add data to current batch
            ti_list += [p_i]

            # Update batch size
            batch_n += n

        yield np.array(ti_list, dtype=np.int32)

        return 0

    def calibration(self, config, dataloader, untouched_ratio=0.9, verbose=False):
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

        redo = False

        # Batch limit
        # Load batch_limit dictionary
        batch_lim_file = os.path.join(self.calibration_path, "batch_limits.pkl")
        if os.path.exists(batch_lim_file):
            with open(batch_lim_file, "rb") as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        t = config["kpconv"]["first_subsampling_dl"]
        s = config["train"]["batch_num"]
        key = f"{t:.3f}_{s:d}"
        if key in batch_lim_dict:
            self.batch_limit = batch_lim_dict[key]
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

            dl = self.dataset["kpconv"]["first_subsampling_dl"] * (2**layer_ind)
            if self.dataset.deform_layers[layer_ind]:
                r = dl * self.dataset["kpconv"]["deform_radius"]
            else:
                r = dl * self.dataset["kpconv"]["conv_radius"]

            key = f"{dl:.3f}_{r:.3f}"
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == config["model"]["num_layers"]:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print("Check neighbors limit dictionary")
            for layer_ind in range(self.dataset.num_layers):
                dl = self.dataset.config["kpconv"]["first_subsampling_dl"] * (2**layer_ind)
                if self.dataset.deform_layers[layer_ind]:
                    r = dl * self.dataset["kpconv"]["deform_radius"]
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
                np.ceil(4 / 3 * np.pi * (self.dataset.config["kpconv"]["conv_radius"] + 1) ** 3)
            )

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.num_layers, hist_n), dtype=np.int32)

            # Batch calib parameters
            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config["train"]["batch_num"]

            # Calibration parameters
            low_pass_t = 10
            kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            # Perform calibration
            for _ in range(10):
                for batch in dataloader:

                    # Update neighborhood histogram
                    counts = [
                        np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1)
                        for neighb_mat in batch.neighbors
                    ]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.labels)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_t

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.batch_limit += kp * error

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
                        print(message.format(i, estim_b, int(self.batch_limit)))

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
                        line0 = line0.join(
                            f"|{color}{neighb_hists[layer, neighb_size]:10d}{BColors.ENDC}  "
                        )
                    print(line0)

                print("\n**************************************************\n")
                print("\nchosen neighbors limits: ", percentiles)
                print()

            # Save batch_limit dictionary
            t = self.dataset.config["kpconv"]["first_subsampling_dl"]
            s = self.dataset.config["train"]["batch_num"]
            key = "{t:.3f}_{s:d}"
            batch_lim_dict[key] = self.batch_limit
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
