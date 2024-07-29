"""
Toronto3D Dataset Class, used to manage data that can be downloaded here :
https://github.com/WeikaiTan/Toronto-3D

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0902, R0913, R0912, R0914, R0915, R1702, C0209

from multiprocessing import Lock
import os
import pickle
import time

import numpy as np
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import get_worker_info

from kpconv_torch.datasets.common import grid_subsampling, PointCloudDataset
from kpconv_torch.utils.config import BColors
from kpconv_torch.utils.mayavi_visu import show_input_batch
from kpconv_torch.io.ply import read_ply, write_ply


class Toronto3DDataset(PointCloudDataset):
    """
    Class to handle Toronto3D dataset.
    """

    def __init__(
        self,
        config,
        datapath,
        chosen_log=None,
        infered_file=None,
        load_data=True,
        task="train",
    ):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        super().__init__(
            config=config,
            datapath=datapath,
            ignored_labels=np.array([0]),
            chosen_log=chosen_log,
            infered_file=infered_file,
            task=task,
        )

        # Update number of class and data task in configuration
        self.num_classes = len(config["model"]["label_to_names"]) - len(self.ignored_labels)

        # Path of the training files
        self.train_files_path = "train"

        # List of files to process
        ply_path = os.path.join(self.datapath, self.train_files_path)

        # Proportion of validation scenes
        self.cloud_names = ["L001", "L002", "L003", "L004"]
        self.all_tasks = list(range(len(self.cloud_names)))
        self.validation_task = 1
        self.test_tasks = 1
        self.train_tasks = [0, 2, 3]

        # Define offset
        self.utm_offset = [627285, 4841948, 0]

        self.test_cloud_names = ["L002"]

        # Number of models used per epoch
        if self.task == "train":
            self.epoch_n = self.config["train"]["epoch_steps"] * self.config["train"]["batch_num"]
        elif self.task in ["validate", "test", "ERF"]:
            self.epoch_n = (
                self.config["train"]["validation_size"] * self.config["train"]["batch_num"]
            )
        else:
            raise ValueError("Unknown task for Toronto3D (with features) data: ", self.task)

        # Stop data is not needed
        if not load_data:
            return

        # Prepare ply files
        if infered_file is None:
            self.prepare_toronto3d_ply()

        # Load ply files
        # List of training files
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.task == "train":
                if self.all_tasks[i] != self.validation_task:
                    self.files += [os.path.join(ply_path, f + ".ply")]
            elif self.task in ["validate", "test", "ERF"]:
                if self.all_tasks[i] == self.validation_task:
                    self.files += [os.path.join(ply_path, f + ".ply")]
            else:
                raise ValueError("Unknown task for Toronto3D (with features) data: ", self.task)

        if self.task == "train":
            self.cloud_names = [
                f
                for i, f in enumerate(self.cloud_names)
                if self.all_tasks[i] != self.validation_task
            ]
        elif self.task in ["validate", "test", "ERF"]:
            self.cloud_names = [
                f
                for i, f in enumerate(self.cloud_names)
                if self.all_tasks[i] == self.validation_task
            ]

        if 0 < self.config["kpconv"]["first_subsampling_dl"] <= 0.01:
            raise ValueError("subsampling_parameter too low (should be over 1 cm")

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        # Batch selection parameters
        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if self.config["input"]["use_potentials"] is True:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for tree in self.pot_trees:
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 0.001)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(
                np.array(self.argmin_potentials, dtype=np.int64)
            )
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor(
                [0 for _ in range(self.config["input"]["threads"])], dtype=torch.int32
            )
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            self.epoch_inds = torch.from_numpy(np.zeros((2, self.epoch_n), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()

        # For ERF visualization, we want only one cloud per batch and no randomness
        if self.task == "ERF":
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in
        parallel to load a different list of indices.
        """

        if self.config["input"]["use_potentials"]:
            return self.potential_item()
        return self.random_item()

    def potential_item(self, debug_workers=False):
        """
        :param debug_workers
        """

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        r_list = []
        batch_n = 0
        failed_attempts = 0

        info = get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None

        while True:

            t += [time.time()]

            if debug_workers:
                message = ""
                for wi in range(info.num_workers):
                    if wi == wid:
                        t1 = BColors.FAIL
                        t2 = BColors.ENDC
                        message += f" {t1}X{t2} "
                    elif self.worker_waiting[wi] == 0:
                        message += "   "
                    elif self.worker_waiting[wi] == 1:
                        message += " | "
                    elif self.worker_waiting[wi] == 2:
                        message += " o "
                print(message)
                self.worker_waiting[wid] = 0

            with self.worker_lock:

                if debug_workers:
                    message = ""
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += f" {BColors.OKGREEN}v{BColors.ENDC} "
                        elif self.worker_waiting[wi] == 0:
                            message += "   "
                        elif self.worker_waiting[wi] == 1:
                            message += " | "
                        elif self.worker_waiting[wi] == 2:
                            message += " o "
                    print(message)
                    self.worker_waiting[wid] = 1

                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Add a small noise to center point
                if self.task != "ERF":
                    center_point += np.random.normal(
                        scale=self.config["input"]["sphere_radius"] / 10, size=center_point.shape
                    )

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(
                    center_point, r=self.config["input"]["sphere_radius"], return_distance=True
                )

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if self.task != "ERF":
                    tukeys = np.square(1 - d2s / np.square(self.config["input"]["sphere_radius"]))
                    tukeys[d2s > np.square(self.config["input"]["sphere_radius"])] = 0
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(
                center_point, r=self.config["input"]["sphere_radius"]
            )[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Safe check for empty spheres
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config["train"]["batch_num"]:
                    raise ValueError("It seems this dataset only contains empty input spheres")
                t += [time.time()]
                t += [time.time()]
                continue

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.task in ["test", "ERF"]:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[label] for label in input_labels])

            t += [time.time()]

            # Data augmentation
            input_points, _, scale, r_ = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config["train"]["augment_color"]:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack(
                (input_colors, input_points[:, 2:] + center_point[:, 2:])
            ).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            r_list += [r_]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        # Concatenate batch
        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(r_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config["input"]["features_dim"] == 1:
            pass
        elif self.config["input"]["features_dim"] == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config["input"]["features_dim"] == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError("Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)")

        # Create network inputs
        #   Points, neighbors, pooling indices for each layers
        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(
            stacked_points, stacked_features, labels, stack_lengths
        )

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        if debug_workers:
            message = ""
            for wi in range(info.num_workers):
                if wi == wid:
                    message += f" {BColors.OKBLUE}0{BColors.ENDC} "
                elif self.worker_waiting[wi] == 0:
                    message += "   "
                elif self.worker_waiting[wi] == 1:
                    message += " | "
                elif self.worker_waiting[wi] == 2:
                    message += " o "
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]

        # Display timings
        debug_t = False
        if debug_t:
            print("\n************************\n")
            print("Timings:")
            ti = 0
            n_ = 5
            mess = "Init ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Pots ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Sphere .... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Collect ... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Augment ... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += n_ * (len(stack_lengths) - 1) + 1
            print(f"concat .... {1000 * (t[ti+1] - t[ti]):5.1f}ms")
            ti += 1
            print(f"input ..... {1000 * (t[ti+1] - t[ti]):5.1f}ms")
            ti += 1
            print(f"stack ..... {1000 * (t[ti+1] - t[ti]):5.1f}ms")
            ti += 1
            print("\n************************\n")
        return input_list

    def random_item(self):
        """
        docstring to do
        """
        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        r_list = []
        batch_n = 0
        failed_attempts = 0

        while True:

            with self.worker_lock:

                # Get potential minimum
                cloud_ind = int(self.epoch_inds[0, self.epoch_i])
                point_ind = int(self.epoch_inds[1, self.epoch_i])

                # Update epoch indice
                self.epoch_i += 1
                if self.epoch_i >= int(self.epoch_inds.shape[1]):
                    self.epoch_i -= int(self.epoch_inds.shape[1])

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add a small noise to center point
            if self.task != "ERF":
                center_point += np.random.normal(
                    scale=self.config["input"]["sphere_radius"] / 10, size=center_point.shape
                )

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(
                center_point, r=self.config["input"]["sphere_radius"]
            )[0]

            # Number collected
            n = input_inds.shape[0]

            # Safe check for empty spheres
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config["train"]["batch_num"]:
                    raise ValueError("It seems this dataset only contains empty input spheres")
                continue

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.task in ["test", "ERF"]:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[label] for label in input_labels])

            # Data augmentation
            input_points, _, scale, r_ = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config["train"]["augment_color"]:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack(
                (input_colors, input_points[:, 2:] + center_point[:, 2:])
            ).astype(np.float32)

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            r_list += [r_]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        # Concatenate batch
        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(r_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config["input"]["features_dim"] == 1:
            pass
        elif self.config["input"]["features_dim"] == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config["input"]["features_dim"] == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError("Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)")

        # Create network inputs
        #   Points, neighbors, pooling indices for each layers
        # Get the whole input list
        input_list = self.segmentation_inputs(
            stacked_points, stacked_features, labels, stack_lengths
        )

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        return input_list

    def prepare_toronto3d_ply(self):
        """
        Prepare a PLY file after inference
        """

        print("\nPreparing ply files")
        t0 = time.time()

        # Folder for the ply files
        ply_path = os.path.join(self.datapath, self.train_files_path)
        if not os.path.exists(ply_path):
            os.mkdir(ply_path)

        for cloud_name in self.cloud_names:

            # Check if ply already exists
            if os.path.exists(os.path.join(ply_path, cloud_name + ".ply")):
                continue

            print(f"\nPreparing ply for cloud {cloud_name}\n")

            points, colors, labels = read_ply(
                os.path.join(self.datapath, "original_ply/" + cloud_name + ".ply")
            )
            xyz = np.vstack(
                (
                    points["x"] - self.utm_offset[0],
                    points["y"] - self.utm_offset[1],
                    points["z"] - self.utm_offset[2],
                )
            ).T.astype(np.float32)
            color = np.vstack((colors["red"], colors["green"], colors["blue"])).T.astype(np.uint8)
            intensity = labels["scalar_Intensity"].astype(np.uint8)
            rgbi = np.hstack((color, intensity.reshape(-1, 1)))
            labels = labels["scalar_Label"].astype(np.uint8)

            # Save as ply
            write_ply(
                os.path.join(ply_path, cloud_name + ".ply"),
                [xyz, rgbi, labels],
                [
                    "x",
                    "y",
                    "z",
                    "red",
                    "green",
                    "blue",
                    "scalar_Intensity",
                    "scalar_Label",
                ],
            )
        print(f"Done in {time.time() - t0:.1f}s")

    def load_subsampled_clouds(self):
        """
        docstring to do
        """
        # Parameter
        dl = self.config["kpconv"]["first_subsampling_dl"]

        # Create path for files
        tree_path = os.path.join(self.datapath, f"input_{dl:.3f}")
        if not os.path.exists(tree_path):
            os.mkdir(tree_path)

        # Load kdtrees
        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]

            # Name of the input files
            kdtree_file = os.path.join(tree_path, f"{cloud_name}_kdtree.pkl")
            sub_ply_file = os.path.join(tree_path, f"{cloud_name}.ply")

            # Check if inputs have already been computed
            if os.path.exists(kdtree_file):
                print(f"\nFound kdtree for cloud {cloud_name}, subsampled at {dl:.3f}")

                # read ply with data
                points, colors, labels = read_ply(sub_ply_file)
                sub_colors = np.vstack(
                    (colors["red"], colors["green"], colors["blue"], labels["scalar_Intensity"])
                ).T
                sub_labels = labels["scalar_Label"]

                # Read pkl with search tree
                with open(kdtree_file, "rb") as f:
                    search_tree = pickle.load(f)

            else:
                print(f"\nPreparing kdtree for cloud {cloud_name}, subsampled at {dl:.3f}")

                # Read ply file
                points, colors, labels = read_ply(file_path)
                points_ = np.vstack((points["x"], points["y"], points["z"])).T
                points_ = np.asarray(points_, dtype=np.float32)
                colors_ = np.vstack(
                    (colors["red"], colors["green"], colors["blue"], labels["scalar_Intensity"])
                ).T
                colors_ = np.asarray(colors_, dtype=np.float32)
                labels_ = np.array(labels["scalar_Label"], dtype=np.int32)

                # Subsample cloud
                sub_points, sub_colors, sub_labels = grid_subsampling(
                    points_, features=colors_, labels=labels_, sampledl=dl
                )

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255.0
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)

                # Save kdtree
                with open(kdtree_file, "wb") as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(
                    sub_ply_file,
                    [sub_points, sub_colors, sub_labels],
                    [
                        "x",
                        "y",
                        "z",
                        "red",
                        "green",
                        "blue",
                        "scalar_Intensity",
                        "scalar_Label",
                    ],
                )

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]

            size = sub_colors.shape[0] * 4 * 7
            print(f"{size * 1e-6:.1f} MB loaded in {time.time() - t0:.1f}s")

        # Coarse potential locations
        # Only necessary for validation and test sets
        if self.config["input"]["use_potentials"]:
            print("\nPreparing potentials")

            # Restart timer
            t0 = time.time()

            pot_dl = self.config["input"]["sphere_radius"] / 10

            for file_idx, _ in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[file_idx]

                # Name of the input files
                coarse_kdtree_file = os.path.join(tree_path, f"{cloud_name}_coarse_kdtree.pkl")

                # Check if inputs have already been computed
                if os.path.exists(coarse_kdtree_file):
                    # Read pkl with search tree
                    with open(coarse_kdtree_file, "rb") as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[file_idx].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampledl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save kdtree
                    with open(coarse_kdtree_file, "wb") as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]

            print(f"Done in {time.time() - t0:.1f}s")

        # Reprojection indices
        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.task in ["validate", "test"]:

            print("\nPreparing reprojection indices for testing")

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = os.path.join(tree_path, f"{cloud_name}_proj.pkl")

                # Try to load previous indices
                if os.path.exists(proj_file):
                    with open(proj_file, "rb") as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    points, colors, labels = read_ply(file_path)
                    points_ = np.vstack((points["x"], points["y"], points["z"])).T
                    labels_ = np.array(labels["scalar_Label"], dtype=np.int32)

                    # Compute projection inds
                    idxs = self.input_trees[i].query(points_, return_distance=False)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, "wb") as f:
                        pickle.dump([proj_inds, labels_], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels_]
                print(f"{cloud_name} done in {time.time() - t0:.1f}s")
        print()

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation task) on which the metrics should be evaluated
        """

        # Get original points
        points, _, _ = read_ply(file_path)
        return np.vstack((points["x"], points["y"], points["z"])).T


def debug_upsampling(dataset, loader):
    """
    Shows which labels are sampled according to strategy chosen
    """

    for _ in range(10):

        for batch in loader:

            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, "=>", pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0**2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print("******************")
        print("*******************************************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, loader):
    """
    Timing of generator function
    """

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.self.config["train"]["batch_num"]
    estim_n = 0

    for _ in range(10):

        for batch_i, batch in enumerate(loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_n += (batch.features.shape[0] - estim_n) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = "Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}"
                print(
                    message.format(batch_i, 1000 * mean_dt[0], 1000 * mean_dt[1], estim_b, estim_n)
                )

        print("************* Epoch ended *************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, config, loader):
    """
    docstring to do
    """

    for _ in range(10):
        layers = config["model"]["num_layers"]

        for batch in loader:

            # Print characteristics of input tensors
            print("\nPoints tensors")
            for i in range(layers):
                print(batch.points[i].dtype, batch.points[i].shape)
            print("\nNeigbors tensors")
            for i in range(layers):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print("\nPools tensors")
            for i in range(layers):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print("\nStack lengths")
            for i in range(layers):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print("\nFeatures")
            print(batch.features.dtype, batch.features.shape)
            print("\nLabels")
            print(batch.labels.dtype, batch.labels.shape)
            print("\nAugment Scales")
            print(batch.scales.dtype, batch.scales.shape)
            print("\nAugment Rotations")
            print(batch.rots.dtype, batch.rots.shape)
            print("\nModel indices")
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print("\nAre input tensors pinned")
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print("*******************************************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, loader):
    """
    Timing of generator function
    """
    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for _ in range(10):

        for batch_i, _ in enumerate(loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = "Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} "
                print(message.format(batch_i, 1000 * mean_dt[0], 1000 * mean_dt[1]))

        print("************* Epoch ended *************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)
