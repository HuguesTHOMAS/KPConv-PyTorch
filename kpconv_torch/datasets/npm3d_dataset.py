"""
NPM3D Dataset Class, used to manage data that can be downloaded here :
https://npm3d.fr/

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915

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
from kpconv_torch.io.ply import read_ply, write_ply


class NPM3DDataset(PointCloudDataset):
    """
    Class to handle NPM3D dataset.
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
        self.num_classes = config["model"]["label_to_names"] - len(self.ignored_labels)

        # Path of the training files
        self.train_files_path = "train"
        self.original_ply_path = "original_ply"

        # List of files to process
        ply_path = os.path.join(self.datapath, self.train_files_path)

        # Proportion of validation scenes
        self.cloud_names = [
            "Lille1_1",
            "Lille1_2",
            "Lille2",
            "Paris",
            "ajaccio_2",
            "ajaccio_57",
            "dijon_9",
        ]
        self.all_tasks = [0, 1, 2, 3, 4, 5, 6]
        self.validation_task = 1
        self.test_tasks = [4, 5, 6]
        self.train_tasks = [0, 2, 3]

        # Number of models used per epoch
        if self.task == "train":
            self.epoch_n = config["train"]["epoch_steps"] * config["train"]["batch_num"]
        elif self.task in ["validate", "test", "ERF"]:
            self.epoch_n = config["train"]["validation_size"] * config["train"]["batch_num"]
        else:
            raise ValueError("Unknown task for NPM3D data: ", self.task)

        # Stop data is not needed
        if not load_data:
            return

        # Prepare ply files
        if infered_file is None:
            self.prepare_npm3d_ply()

        # Load ply files
        # List of training files
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.task == "train":
                if self.all_tasks[i] in self.train_tasks:
                    self.files += [os.path.join(ply_path, f + ".ply")]
            elif self.task in ["validate", "ERF"]:
                if self.all_tasks[i] == self.validation_task:
                    self.files += [os.path.join(ply_path, f + ".ply")]
            elif self.task == "test":
                if self.all_tasks[i] in self.test_tasks:
                    self.files += [os.path.join(ply_path, f + ".ply")]
            else:
                raise ValueError("Unknown task for NPM3D data: ", self.task)
        print("The set is " + str(self.task))

        if self.task == "train":
            self.cloud_names = [
                f for i, f in enumerate(self.cloud_names) if self.all_tasks[i] in self.train_tasks
            ]
        elif self.task in ["validate", "ERF"]:
            self.cloud_names = [
                f
                for i, f in enumerate(self.cloud_names)
                if self.all_tasks[i] == self.validation_task
            ]
        elif self.task == "test":
            self.cloud_names = [
                f for i, f in enumerate(self.cloud_names) if self.all_tasks[i] in self.test_tasks
            ]
        print("The files are " + str(self.cloud_names))

        if 0 < config["kpconv"]["first_subsampling_dl"] <= 0.01:
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
        if config["input"]["use_potentials"]:
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
                [0 for _ in range(config["input"]["threads"])], dtype=torch.int32
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

        return

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
        docstring to do

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
                        message += f" {BColors.FAIL}X{BColors.ENDC} "
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
            if self.task in ["test", "ERF"]:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[label] for label in input_labels])

            t += [time.time()]

            # Data augmentation
            input_points, _, scale, r = self.augmentation_transform(input_points)

            # Get original height as additional feature
            input_features = np.hstack(input_points[:, 2:] + center_point[:, 2:]).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            r_list += [r]

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
            n = 5
            mess = "Init ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n * i + 1] - t[ti + n * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Pots ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n * i + 1] - t[ti + n * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Sphere .... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n * i + 1] - t[ti + n * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Collect ... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n * i + 1] - t[ti + n * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Augment ... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n * i + 1] - t[ti + n * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += n * (len(stack_lengths) - 1) + 1
            print(f"concat .... {1000 * (t[ti + 1] - t[ti]):5.1f}ms")
            ti += 1
            print(f"input ..... {1000 * (t[ti + 1] - t[ti]):5.1f}ms")
            ti += 1
            print(f"stack ..... {1000 * (t[ti + 1] - t[ti]):5.1f}ms")
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
            if self.task in ["test", "ERF"]:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[label] for label in input_labels])

            # Data augmentation
            input_points, _, scale, r = self.augmentation_transform(input_points)

            # Get original height as additional feature
            input_features = np.hstack(input_points[:, 2:] + center_point[:, 2:]).astype(np.float32)

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            r_list += [r]

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

    def prepare_npm3d_ply(self):
        """
        Prepares PLY result files, after the inference.

        :returns {}
        """

        print("\nPreparing ply files")
        t0 = time.time()

        # Folder for the ply files
        ply_path = os.path.join(self.datapath, self.train_files_path)
        if not os.path.exists(ply_path):
            os.makedirs(ply_path)

        for cloud_name in self.cloud_names:

            # Pass if the cloud has already been computed
            cloud_file = os.path.join(ply_path, cloud_name + ".ply")
            if os.path.exists(cloud_file):
                continue

            points, _, labels = read_ply(
                os.path.join(self.datapath, self.original_ply_path, cloud_name + ".ply")
            )

            # Initiate containers
            cloud_x = points["x"]
            cloud_y = points["y"]
            cloud_z = points["z"]
            cloud_x = cloud_x - (cloud_x.min())
            cloud_y = cloud_y - (cloud_y.min())
            cloud_z = cloud_z - (cloud_z.min())

            # Reshape
            cloud_x = cloud_x.reshape(len(cloud_x), 1)
            cloud_y = cloud_y.reshape(len(cloud_y), 1)
            cloud_z = cloud_z.reshape(len(cloud_z), 1)

            # Astype
            cloud_x = cloud_x.astype(np.float32)
            cloud_y = cloud_y.astype(np.float32)
            cloud_z = cloud_z.astype(np.float32)

            # Stack
            cloud_points = np.hstack((cloud_x, cloud_y, cloud_z))

            # Labels
            if cloud_name in ["ajaccio_2", "ajaccio_57", "dijon_9"]:

                field_names = ["x", "y", "z"]
                write_ply(os.path.join(ply_path, cloud_name + ".ply"), cloud_points, field_names)

            else:
                validation_labels = labels
                validation_labels = validation_labels.astype(np.int32)
                validation_labels = validation_labels.reshape(len(labels), 1)

                # Save as ply
                field_names = ["x", "y", "z", "classification"]
                write_ply(
                    os.path.join(ply_path, cloud_name + ".ply"),
                    [cloud_points, validation_labels],
                    field_names,
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
            os.makedirs(tree_path)

        # Load KDTrees
        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]

            # Name of the input files
            kdtree_file = os.path.join(tree_path, f"{cloud_name}_KDTree.pkl")
            sub_ply_file = os.path.join(tree_path, f"{cloud_name}.ply")

            # Check if inputs have already been computed
            if os.path.exists(kdtree_file):
                print(f"\nFound KDTree for cloud {cloud_name}, subsampled at {dl:3f}")

                # read ply with data
                points, _, labels = read_ply(sub_ply_file)
                sub_labels = labels

                # Read pkl with search tree
                with open(kdtree_file, "rb") as f:
                    search_tree = pickle.load(f)

            else:
                print(f"\nPreparing KDTree for cloud {cloud_name}, subsampled at {dl:3f}")

                # Read ply file
                points, _, labels = read_ply(file_path)
                points_ = np.vstack((points["x"], points["y"], points["z"])).T

                # Fake labels for test data
                if self.task == "test":
                    labels_ = np.zeros((len(labels),), dtype=np.int32)
                else:
                    labels_ = labels

                # Subsample cloud
                sub_points, sub_labels = grid_subsampling(points_, labels=labels_, sampledl=dl)
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)

                # Save KDTree
                with open(kdtree_file, "wb") as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file, [sub_points, sub_labels], ["x", "y", "z", "classification"])

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_labels += [sub_labels]

            size = sub_labels.shape[0] * 4 * 7
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
                coarse_kdtree_file = os.path.join(tree_path, f"{cloud_name}_coarse_KDTree.pkl")

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

                    # Save KDTree
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
                    points, _, labels = read_ply(file_path)
                    points_ = np.vstack((points["x"], points["y"], points["z"])).T

                    # Fake labels
                    if self.task == "test":
                        labels_ = np.zeros((len(labels),), dtype=np.int32)
                    else:
                        labels_ = labels

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
