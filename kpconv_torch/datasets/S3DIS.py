from multiprocessing import Lock
import os
from pathlib import Path

import pickle
import time
import warnings

import numpy as np
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import get_worker_info, Sampler

from kpconv_torch.datasets.common import grid_subsampling, PointCloudDataset
from kpconv_torch.utils.mayavi_visu import show_input_batch
from kpconv_torch.io.las import read_las_laz
from kpconv_torch.io.ply import read_ply, write_ply
from kpconv_torch.io.xyz import read_xyz


class S3DISDataset(PointCloudDataset):
    """Class to handle S3DIS dataset."""

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

        ###################
        # Prepare ply files
        ###################
        if infered_file is None:
            self.prepare_S3DIS_ply()

        ################
        # Load ply files
        ################
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

            ############################
            # Batch selection parameters
            ############################

            self.set_batch_selection_parameters()

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
        The main thread gives a list of indices to load a batch. Each worker is going to work
        in parallel to load a different list of indices.
        """

        if self.config["input"]["use_potentials"]:
            return self.potential_item()
        else:
            return self.random_item(batch_i)

    def potential_item(self, debug_workers=False):

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
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
                        t1 = self.config["colors"]["fail"]
                        t2 = self.config["colors"]["endc"]
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
                            t1 = self.config["colors"]["okgreen"]
                            t2 = self.config["colors"]["endc"]
                            message += f" {t1}v{t2} "
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
            input_points, scale, R = self.augmentation_transform(input_points)

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
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

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

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

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
                    t1 = self.config["colors"]["okblue"]
                    t2 = self.config["colors"]["endc"]
                    message += f" {t1}0{t2} "
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
        debugT = False
        if debugT:
            print("\n************************\n")
            print("Timings:")
            ti = 0
            N = 5
            mess = "Init ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Pots ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Sphere .... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Collect ... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Augment ... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print(f"concat .... {1000 * (t[ti + 1] - t[ti]):5.1f}ms")
            ti += 1
            print(f"input ..... {1000 * (t[ti + 1] - t[ti]):5.1f}ms")
            ti += 1
            print(f"stack ..... {1000 * (t[ti + 1] - t[ti]):5.1f}ms")
            ti += 1
            print("\n************************\n")
        return input_list

    def random_item(self, batch_i):

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
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
            input_points, scale, R = self.augmentation_transform(input_points)

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
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

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

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(
            stacked_points, stacked_features, labels, stack_lengths
        )

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        return input_list

    def prepare_S3DIS_ply(self):
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
                    "Cloud %s - Room %d/%d : %s"
                    % (cloud_name, i + 1, len(room_folders), room_folder.split("/")[-1])
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
                            with open(object_file) as f:
                                lines = f.readlines()
                            for l_i, line in enumerate(lines):
                                if "103.0\x100000" in line:
                                    lines[l_i] = line.replace("103.0\x100000", "103.000000")
                            with open(object_file, "w") as f:
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
        return

    def load_kdtree(self, cloud_name, file_path):
        # Restart timer
        t0 = time.time()

        # Name of the input files
        KDTree_file = os.path.join(self.tree_path, f"{cloud_name}_KDTree.pkl")
        sub_ply_file = os.path.join(self.tree_path, f"{cloud_name}.ply")

        print("KDTree file:", KDTree_file)
        print("Sub PLY file:", sub_ply_file)
        print("File path:", file_path)
        # Check if inputs have already been computed
        if os.path.exists(KDTree_file):
            t = self.config["kpconv"]["first_subsampling_dl"]
            print(f"\nFound KDTree for cloud {cloud_name}, " f"subsampled at {t:3f}")

            # read ply with data
            _, sub_colors, sub_labels = self.read_input(sub_ply_file)

            # Read pkl with search tree
            with open(KDTree_file, "rb") as f:
                search_tree = pickle.load(f)

        else:
            t = self.config["kpconv"]["first_subsampling_dl"]
            print(f"\nPreparing KDTree for cloud {cloud_name}, " f"subsampled at {t:3f}.")

            points, colors, labels = self.read_input(file_path)

            # Subsample cloud
            sub_points, sub_colors, sub_labels = grid_subsampling(
                points,
                features=colors,
                labels=labels,
                sampleDl=self.config["kpconv"]["first_subsampling_dl"],
            )

            # Rescale float color and squeeze label
            sub_colors = sub_colors / 255
            sub_labels = np.squeeze(sub_labels)

            # Get chosen neighborhoods
            search_tree = KDTree(sub_points, leaf_size=10)

            # Save KDTree
            with open(KDTree_file, "wb") as f:
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

        # Restart timer
        t0 = time.time()

        # Name of the input files
        coarse_KDTree_file = os.path.join(self.tree_path, f"{cloud_name}_coarse_KDTree.pkl")

        # Check if inputs have already been computed
        if os.path.exists(coarse_KDTree_file):
            # Read pkl with search tree
            with open(coarse_KDTree_file, "rb") as f:
                search_tree = pickle.load(f)

        else:
            # Subsample cloud
            sub_points = np.array(kdtree_data, copy=False)
            coarse_points = grid_subsampling(
                sub_points.astype(np.float32), sampleDl=self.config["input"]["sphere_radius"] / 10
            )

            # Get chosen neighborhoods
            search_tree = KDTree(coarse_points, leaf_size=10)

            # Save KDTree
            with open(coarse_KDTree_file, "wb") as f:
                pickle.dump(search_tree, f)

        # Fill data containers
        self.pot_trees += [search_tree]

        print(f"Done in {time.time() - t0:.1f}s")

    def load_projection_indices(self, cloud_name, file_path, input_tree):

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
            points, _, labels = self.read_input(file_path)

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

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation task) on which the metrics should be evaluated
        """
        points, _, _ = self.read_input(file_path)
        return points

    def read_input(self, filepath):
        """Read all the input files that belong to the dataset

        PLY files are read by training and testing commands.
        """
        file_extension = Path(filepath).suffix
        if file_extension == ".ply":
            points, colors, labels = read_ply(filepath)
        elif file_extension == ".xyz":
            points, colors, labels = read_xyz(filepath)
        elif file_extension == ".las" or file_extension == ".laz":
            points, colors, labels = read_las_laz(filepath)
        else:
            raise OSError(f"Unsupported input file extension ({file_extension}).")
        return points, colors, labels


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class S3DISSampler(Sampler):
    """Sampler for S3DIS"""

    def __init__(self, dataset: S3DISDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset
        self.calibration_path = os.path.join(self.dataset.datapath, "calibration")
        os.makedirs(self.calibration_path, exist_ok=True)

        # Number of step per epoch
        if dataset.task == "train":
            self.N = dataset.config["train"]["epoch_steps"]
        else:
            self.N = dataset.config["train"]["validation_size"]

        return

    def __iter__(self):
        """Yield next batch indices here. In this dataset, this is a dummy sampler that yield
        the index of batch element (input sphere) in epoch instead of the list of point indices.

        """

        if not self.dataset.config["input"]["use_potentials"]:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int64)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config["train"]["batch_num"]
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
                    N_inds = all_label_indices.shape[1]
                    if N_inds < random_pick_n:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            chosen_label_inds = np.hstack(
                                (
                                    chosen_label_inds,
                                    all_label_indices[:, np.random.permutation(N_inds)],
                                )
                            )
                        warnings.warn(
                            f"When choosing random epoch indices "
                            f'(config["input"]["use_potentials"]=False), '
                            f"class {label:d}: {self.dataset.label_names[label_ind]} only had "
                            f"{N_inds:d} available points, while we needed {random_pick_n:d}. "
                            "Repeating indices in the same epoch",
                            stacklevel=2,
                        )

                    elif N_inds < 50 * random_pick_n:
                        rand_inds = np.random.choice(N_inds, size=random_pick_n, replace=False)
                        chosen_label_inds = all_label_indices[:, rand_inds]

                    else:
                        chosen_label_inds = np.zeros((2, 0), dtype=np.int64)
                        while chosen_label_inds.shape[1] < random_pick_n:
                            rand_inds = np.unique(
                                np.random.choice(N_inds, size=2 * random_pick_n, replace=True)
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
        yield from range(self.N)

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def fast_calib(self):
        """This method calibrates the batch sizes while ensuring the potentials are well
        initialized. Indeed on a dataset like Semantic3D, before potential have been updated over
        the dataset, there are cahnces that all the dense area are picked in the begining and in
        the end, we will have very large batch of small point clouds :return:

        """

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config["train"]["batch_num"]

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
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
                estim_b += (b - estim_b) / low_pass_T

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += Kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_T = 100
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

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """Method performing batch and neighbors calibration.

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
        if self.dataset.config["input"]["use_potentials"]:
            sampler_method = "potentials"
        else:
            sampler_method = "random"

        t1 = self.dataset.config["input"]["sphere_radius"]
        t2 = self.dataset.config["kpconv"]["first_subsampling_dl"]
        t3 = self.dataset.config["train"]["batch_num"]

        key = f"{sampler_method}_{t1:3f}_" f"{t2:3f}_{t3}"
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print("\nPrevious calibration found:")
            print("Check batch limit dictionary")
            if key in batch_lim_dict:
                color = self.dataset.config["colors"]["okgreen"]
                v = str(int(batch_lim_dict[key]))
            else:
                color = self.dataset.config["colors"]["fail"]
                v = "?"
            print(f'{color}"{key}": {v}{self.dataset.config["colors"]["endc"]}')

        # Neighbors limit
        # ***************

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
                r = dl * self.dataset.config["train"]["batch_num"]
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
                    r = dl * self.dataset.config["train"]["batch_num"]
                else:
                    r = dl * self.dataset.config["kpconv"]["conv_radius"]
                key = f"{dl:.3f}_{r:.3f}"

                if key in neighb_lim_dict:
                    color = self.dataset.config["colors"]["okgreen"]
                    v = str(neighb_lim_dict[key])
                else:
                    color = self.dataset.config["colors"]["fail"]
                    v = "?"
                print(f'{color}"{key}": {v}{self.dataset.config["colors"]["endc"]}')

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(
                np.ceil(4 / 3 * np.pi * (self.dataset.config["train"]["batch_num"] + 1) ** 3)
            )

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config["train"]["batch_num"]

            # Expected batch size order of magnitude
            expected_N = 100000

            # Calibration parameters. Higher means faster but can also become unstable
            # Reduce Kp/Kd if small GPU: the total number of points per batch will be smaller
            low_pass_T = 100
            Kp = expected_N / 200
            Ki = 0.001 * Kp
            Kd = 5 * Kp
            finer = False
            stabilized = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False
            error_I = 0
            error_D = 0
            last_error = 0

            debug_in = []
            debug_out = []
            debug_b = []
            debug_estim_b = []

            #####################
            # Perform calibration
            #####################

            # number of batch per epoch
            sample_batches = 999
            for _ in range((sample_batches // self.N) + 1):
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
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b
                    error_I += error
                    error_D = error - last_error
                    last_error = error

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 30:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += Kp * error + Ki * error_I + Kd * error_D

                    # Unstability detection
                    if not stabilized and self.dataset.batch_limit < 0:
                        Kp *= 0.1
                        Ki *= 0.1
                        Kd *= 0.1
                        stabilized = True

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
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
                import matplotlib.pyplot as plt

                print(
                    "ERROR: It seems that the calibration have not reached convergence. "
                    "Here are some plot to understand why:"
                )
                print("If you notice unstability, reduce the expected_N value")
                print("If convergece is too slow, increase the expected_N value")

                plt.figure()
                plt.plot(debug_in)
                plt.plot(debug_out)

                plt.figure()
                plt.plot(debug_b)
                plt.plot(debug_estim_b)

                plt.show()

                1 / 0

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
                    line0 += f"|  layer {layer:2d}  "
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = f"     {neighb_size:4d}     "
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = self.dataset.config["colors"]["fail"]
                        else:
                            color = self.dataset.config["colors"]["okgreen"]
                        line0 += "|{:}{:10d}{:}  ".format(
                            color,
                            neighb_hists[layer, neighb_size],
                            self.dataset.config["colors"]["endc"],
                        )

                    print(line0)

                print("\n**************************************************\n")
                print("\nchosen neighbors limits: ", percentiles)
                print()

            # Save batch_limit dictionary
            if self.dataset.config["input"]["use_potentials"]:
                sampler_method = "potentials"
            else:
                sampler_method = "random"

            t1 = self.dataset.config["input"]["sphere_radius"]
            t2 = self.dataset.config["kpconv"]["first_subsampling_dl"]
            t3 = self.dataset.config["train"]["batch_num"]
            key = f"{sampler_method}_{t1:3f}_" f"{t2:3f}_{t3:d}"
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
        return


class S3DISCustomBatch:
    """Custom batch definition with memory pinning for S3DIS"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 7) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements("points", layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements("neighbors", layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements("pools", layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """Return a list of the stacked elements in the batch at a certain layer.

        If no layer is given, then return all layers.

        """

        if element_name == "points":
            elements = self.points
        elif element_name == "neighbors":
            elements = self.neighbors
        elif element_name == "pools":
            elements = self.pools[:-1]
        else:
            raise ValueError(f"Unknown element name: {element_name}")

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == "pools":
                    lengths = self.lengths[layer_i + 1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0 : i0 + length]
                    if element_name == "neighbors":
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == "pools":
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def S3DISCollate(batch_data):
    return S3DISCustomBatch(batch_data)


def debug_upsampling(dataset, loader):
    """Shows which labels are sampled according to strategy chosen"""

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
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config["train"]["batch_num"]
    estim_N = 0

    for _ in range(10):
        for batch_i, batch in enumerate(loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

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
                    message.format(batch_i, 1000 * mean_dt[0], 1000 * mean_dt[1], estim_b, estim_N)
                )

        print("************* Epoch ended *************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, config, loader):

    for _ in range(10):
        L = config["model"]["num_layers"]

        for batch in loader:

            # Print characteristics of input tensors
            print("\nPoints tensors")
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print("\nNeigbors tensors")
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print("\nPools tensors")
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print("\nStack lengths")
            for i in range(L):
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
    """Timing of generator function"""

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
