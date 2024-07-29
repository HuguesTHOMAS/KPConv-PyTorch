"""
S3DIS Dataset Class, used to manage data that can be downloaded here :
https://guochengqian.github.io/PointNeXt/examples/s3dis/

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0902, R0913, R0912, R0914, R0915, R1702

from multiprocessing import Lock
import os
from pathlib import Path

import pickle
import time

import numpy as np
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import get_worker_info

from kpconv_torch.datasets.common import grid_subsampling, PointCloudDataset
from kpconv_torch.utils.config import BColors
from kpconv_torch.io.las import read_las_laz
from kpconv_torch.io.ply import read_ply, write_ply
from kpconv_torch.io.xyz import read_xyz


class S3DISDataset(PointCloudDataset):
    """
    Dataset loader for S3DIS
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
        This dataset is small enough to be stored in-memory, so load all point clouds here.

        @parameters:
        - config: YAML configuration dictionnary, coming from the config.yaml file
        - datapath: -d CLI parameter, giving the path to the labeled points clouds data files
        - chosen_log: -l CLI parameter, path to an already trained KPConv model log folder
          on the file system
        - infered_file: -f CLI parameter, path to the file on which to predict semantic labels,
          using the trained model
        - load_data: boolean, saying if loading the .ply file is needed or not
        - task: operation type to realize, can be "all", "ERF", "train", "test", "validate"
        """
        super().__init__(
            config=config,
            datapath=datapath,
            ignored_labels=np.array([]),
            chosen_log=chosen_log,
            infered_file=infered_file,
            task=task,
        )

        # Path of the training files
        self.train_files_path = datapath / self.config["train"]["train_folder"]
        if not os.path.exists(self.train_files_path):
            print("The ply folder does not exist, create it.")
            os.makedirs(self.train_files_path)

        # Create path for files
        t = self.config["kpconv"]["first_subsampling_dl"]
        self.tree_path = os.path.join(self.datapath, f"input_{t:.3f}")
        if not os.path.exists(self.tree_path):
            os.makedirs(self.tree_path)

        # Data folder management
        if self.task == "test" and infered_file is not None:
            # Inference case: a S3DIS dataset is built with the infered file
            self.cloud_names = [infered_file]
        else:
            # Any other case: the S3DIS dataset is built with the S3DIS original data
            if self.task == "all":
                self.cloud_names = (
                    self.config["model"]["train_cloud_names"]
                    + self.config["train"]["validation_cloud_names"]
                )
            elif self.task == "train":
                self.cloud_names = self.config["train"]["train_cloud_names"]
            else:
                self.cloud_names = self.config["train"]["validation_cloud_names"]
            available_cloud_data = [subfolder.name for subfolder in self.datapath.iterdir()]
            self.cloud_names = [
                cloud_name for cloud_name in self.cloud_names if cloud_name in available_cloud_data
            ]
        self.files = [
            (
                cloud_name
                if self.task == "test" and infered_file is not None
                else self.train_files_path / (cloud_name + ".ply")
            )
            for i, cloud_name in enumerate(self.cloud_names)
        ]

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.pot_trees = []
        self.test_proj = []
        self.validation_labels = []

        # Prepare ply files
        if infered_file is None:
            self.prepare_s3dis_ply()

        # Load ply files
        if load_data:
            if 0 < self.config["kpconv"]["first_subsampling_dl"] <= 0.01:
                raise ValueError("subsampling_parameter too low (should be over 1 cm)")

            for cloud_name, file_path in zip(self.cloud_names, self.files):
                cur_kdtree = self.load_kdtree(cloud_name, file_path)
                if self.config["input"]["use_potentials"]:
                    self.load_coarse_potential_locations(cloud_name, cur_kdtree.data)
                # Only necessary for validation and test sets
                if self.task in ["validate", "test"]:
                    self.load_projection_indices(cloud_name, file_path, cur_kdtree)

            # Batch selection parameters
            self.set_batch_selection_parameters()

            # For ERF visualization, we want only one cloud per batch and no randomness
            if self.task == "ERF":
                self.batch_limit = torch.tensor([1], dtype=torch.float32)
                self.batch_limit.share_memory_()
                np.random.seed(42)

    def __len__(self):
        """
        Returns the length of the data
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work
        in parallel to load a different list of indices.

        :param batch_i:
        """
        if self.config["input"]["use_potentials"]:
            return self.potential_item()

        return self.random_item()

    def potential_item(self, debug_workers=False):
        """


        :param debug_workers:
        """
        t = [time.time()]

        # Initiate concatenation lists
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
            input_points, _, scale, r = self.augmentation_transform(input_points)

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
        :returns input_lists : [scales, rots, cloud_inds, point_inds, input_inds]
        """

        # Initiate concatenation lists
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
            input_points, _, scale, r = self.augmentation_transform(input_points)

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

    def prepare_s3dis_ply(self):
        """
        Prepares PLY result files, after the inference.

        :returns nothing
        """

        print("\nPreparing ply files")
        t0 = time.time()

        for cloud_name in self.cloud_names:
            # Pass if the cloud has already been computed
            cloud_file = os.path.join(self.train_files_path, cloud_name + ".ply")
            if os.path.exists(cloud_file):
                print(f"{cloud_file} does already exist.")
                continue

            # Get rooms of the current cloud
            cloud_folder = os.path.join(self.datapath, cloud_name)
            room_folders = [
                os.path.join(cloud_folder, room)
                for room in os.listdir(cloud_folder)
                if os.path.isdir(os.path.join(cloud_folder, room))
            ]

            # Initiate containers
            cloud_points = np.empty((0, 3), dtype=np.float32)
            cloud_colors = np.empty((0, 3), dtype=np.uint8)
            cloud_classes = np.empty((0, 1), dtype=np.int32)

            # Loop over rooms
            for i, room_folder in enumerate(room_folders):

                print(
                    f"Cloud {cloud_name} - Room {i + 1}/{len(room_folders)} : \
                      {room_folder.split('/')[-1]}"
                )

                for object_name in os.listdir(os.path.join(room_folder, "Annotations")):
                    if object_name[-4:] == ".txt":
                        # Text file containing point of the object
                        object_file = os.path.join(room_folder, "Annotations", object_name)

                        # Object class and ID
                        tmp = object_name[:-4].split("_")[0]
                        if tmp in self.name_to_label:
                            object_class = self.name_to_label[tmp]
                        elif tmp in ["stairs"]:
                            object_class = self.name_to_label["clutter"]
                        else:
                            raise ValueError("Unknown object name: " + str(tmp))

                        # Correct bug in S3DIS dataset
                        if object_name == "ceiling_1.txt":
                            with open(object_file, encoding="utf-8") as f:
                                lines = f.readlines()
                            for l_i, line in enumerate(lines):
                                if "103.0\x100000" in line:
                                    lines[l_i] = line.replace("103.0\x100000", "103.000000")
                            with open(object_file, "w", encoding="utf-8") as f:
                                f.writelines(lines)

                        # Read object points and colors
                        object_data = np.loadtxt(object_file, dtype=np.float32)

                        # Stack all data
                        cloud_points = np.vstack(
                            (cloud_points, object_data[:, 0:3].astype(np.float32))
                        )
                        cloud_colors = np.vstack(
                            (cloud_colors, object_data[:, 3:6].astype(np.uint8))
                        )
                        object_classes = np.full(
                            (object_data.shape[0], 1), object_class, dtype=np.int32
                        )
                        cloud_classes = np.vstack((cloud_classes, object_classes))

            # Save as ply
            write_ply(
                cloud_file,
                (cloud_points, cloud_colors, cloud_classes),
                ["x", "y", "z", "red", "green", "blue", "classification"],
            )

        print(f"Done in {time.time() - t0:.1f}s")

    def load_kdtree(self, cloud_name, filepath):
        """
        Loads a KD-Tree from a file.

        :param cloud_name:
        :param filepath: path to the KD-Tree file.
        """
        # Restart timer
        t0 = time.time()

        # Name of the input files
        kdtree_file = os.path.join(self.tree_path, f"{cloud_name}_KDTree.pkl")
        sub_ply_file = os.path.join(self.tree_path, f"{cloud_name}.ply")

        print("KDTree file:", kdtree_file)
        print("Sub PLY file:", sub_ply_file)
        print("File path:", filepath)

        # Check if inputs have already been computed
        if os.path.exists(kdtree_file):
            t = self.config["kpconv"]["first_subsampling_dl"]
            print(f"\nFound KDTree for cloud {cloud_name}, " f"subsampled at {t:3f}")

            # Read ply with data
            _, sub_colors, sub_labels = self.read_input(sub_ply_file)

            # Read pkl with search tree
            with open(kdtree_file, "rb") as f:
                search_tree = pickle.load(f)

        else:
            t = self.config["kpconv"]["first_subsampling_dl"]
            print(f"\nPreparing KDTree for cloud {cloud_name}, " f"subsampled at {t:3f}.")

            points, colors, labels = self.read_input(filepath)

            # Subsample cloud
            sub_points, sub_colors, sub_labels = grid_subsampling(
                points,
                features=colors,
                labels=labels,
                sampledl=self.config["kpconv"]["first_subsampling_dl"],
            )

            # Rescale float color and squeeze label
            sub_colors = sub_colors / 255
            sub_labels = np.squeeze(sub_labels)

            # Get chosen neighborhoods
            search_tree = KDTree(sub_points, leaf_size=10)

            # Save KDTree
            with open(kdtree_file, "wb") as f:
                pickle.dump(search_tree, f)

            # Save ply
            write_ply(
                sub_ply_file,
                [sub_points, sub_colors, sub_labels],
                ["x", "y", "z", "red", "green", "blue", "classification"],
            )

        # Fill data containers
        self.input_trees += [search_tree]
        self.input_colors += [sub_colors]
        self.input_labels += [sub_labels]

        size = sub_colors.shape[0] * 4 * 7
        print(f"{size * 1e-6:.1f} MB loaded in {time.time() - t0:1f}s")
        return search_tree

    def load_coarse_potential_locations(self, cloud_name, kdtree_data):
        """

        :param cloud_name:
        :param kdtree_data:
        """

        # Restart timer
        t0 = time.time()

        # Name of the input files
        coarse_kdtree_file = os.path.join(self.tree_path, f"{cloud_name}_coarse_KDTree.pkl")

        # Check if inputs have already been computed
        if os.path.exists(coarse_kdtree_file):
            # Read pkl with search tree
            with open(coarse_kdtree_file, "rb") as f:
                search_tree = pickle.load(f)

        else:
            # Subsample cloud
            sub_points = np.array(kdtree_data, copy=False)
            coarse_points = grid_subsampling(
                sub_points.astype(np.float32), sampledl=self.config["input"]["sphere_radius"] / 10
            )

            # Get chosen neighborhoods
            search_tree = KDTree(coarse_points, leaf_size=10)

            # Save KDTree
            with open(coarse_kdtree_file, "wb") as f:
                pickle.dump(search_tree, f)

        # Fill data containers
        self.pot_trees += [search_tree]

        print(f"Done in {time.time() - t0:.1f}s")

    def load_projection_indices(self, cloud_name, filepath, input_tree):
        """
        Prepares reprojection indices for testing

        :param cloud_name:
        :param filepath:
        :param input_tree:
        """

        print("\nPreparing reprojection indices for testing")

        # Restart timer
        t0 = time.time()

        # File name for saving
        proj_file = os.path.join(self.tree_path, f"{cloud_name}_proj.pkl")

        # Try to load previous indices
        if os.path.exists(proj_file):
            with open(proj_file, "rb") as f:
                proj_inds, labels = pickle.load(f)
        else:
            points, _, labels = self.read_input(filepath)

            # Compute projection inds
            idxs = input_tree.query(points, return_distance=False)
            proj_inds = np.squeeze(idxs).astype(np.int32)

            # Save
            with open(proj_file, "wb") as f:
                pickle.dump([proj_inds, labels], f)

        self.test_proj += [proj_inds]
        self.validation_labels += [labels]
        print(f"{cloud_name} done in {time.time() - t0:.1f}s")

    def set_batch_selection_parameters(self):
        """
        DocString to do
        """
        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Number of models used per epoch
        if self.task == "train":
            self.epoch_n = self.config["train"]["epoch_steps"] * self.config["train"]["batch_num"]
        else:
            self.epoch_n = (
                self.config["train"]["validation_size"] * self.config["train"]["batch_num"]
            )

        # Initialize potentials
        if self.config["input"]["use_potentials"]:
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

    def load_evaluation_points(self, filepath):
        """
        Load points (from a test or validation task) on which the metrics should be evaluated

        :param filepath: path to the file
        :returns a np.array of points
        """
        points, _, _ = self.read_input(filepath)
        return points

    def read_input(self, filepath):
        """
        Read an input file that belongs to the dataset.
        PLY files are read by training and testing commands.

        :param filepath: path to the file
        :returns 3 np.arrays of points, colors and labels
        """
        file_extension = Path(filepath).suffix
        if file_extension == ".ply":
            points, colors, labels = read_ply(filepath)
        elif file_extension == ".xyz":
            points, colors, labels = read_xyz(filepath)
        elif file_extension in (".las", ".laz"):
            points, colors, labels = read_las_laz(filepath)
        else:
            raise OSError(f"Unsupported input file extension ({file_extension}).")
        return points, colors, labels
