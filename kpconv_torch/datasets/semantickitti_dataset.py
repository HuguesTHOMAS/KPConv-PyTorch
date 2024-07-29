"""
SemanticKitti Dataset Class, used to manage data that can be downloaded here :
https://www.semantic-kitti.org/

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0902, R0913, R0912, R0914, R0915, R1702

from multiprocessing import Lock
import os
import pickle
import time

import numpy as np
from sklearn.neighbors import KDTree
import torch

from kpconv_torch.datasets.common import grid_subsampling, PointCloudDataset


class SemanticKittiDataset(PointCloudDataset):
    """
    Class to handle SemanticKitti dataset.
    """

    def __init__(
        self,
        config,
        datapath,
        chosen_log=None,
        infered_file=None,
        balance_classes=True,
        task="train",
    ):
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

        # Get a list of sequences
        if self.task == "train":
            self.sequences = [f"{i:02d}" for i in range(11) if i != 8]
        elif self.task == "validate":
            self.sequences = [f"{i:02d}" for i in range(11) if i == 8]
        elif self.task == "test":
            self.sequences = [f"{i:02d}" for i in range(11, 22)]
        else:
            raise ValueError("Unknown task for SemanticKitti data: ", self.task)

        # List all files in each sequence
        self.frames = []
        for seq in self.sequences:
            velo_path = os.path.join(self.datapath, "sequences", seq, "velodyne")
            frames = np.sort([vf[:-4] for vf in os.listdir(velo_path) if vf.endswith(".bin")])
            self.frames.append(frames)

        # Object classes parameters
        if self.config["kpconv"]["n_frames"] < 1:
            raise ValueError("number of frames has to be >= 1")

        # Read labels
        all_labels = self.config["specific"]["labels"]
        learning_map_inv = self.config["specific"]["learning_map_inv"]
        learning_map = self.config["specific"]["learning_map"]

        self.learning_map = np.zeros((max(learning_map) + 1), dtype=np.int32)
        for k, v in learning_map.items():
            self.learning_map[k] = v

        self.learning_map_inv = np.zeros((max(learning_map_inv) + 1), dtype=np.int32)
        for k, v in learning_map_inv.items():
            self.learning_map_inv[k] = v

        # Dict from labels to names
        self.config["model"]["label_to_names"] = {
            k: all_labels[v] for k, v in learning_map_inv.items()
        }

        # Load calibration
        # Init variables
        self.calibrations = []
        self.times = []
        self.poses = []
        self.all_inds = None
        self.class_proportions = None
        self.class_frames = []
        self.val_confs = []

        # Load everything
        self.load_calib_poses()

        # Batch selection parameters
        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize frame potentials
        self.potentials = torch.from_numpy(np.random.rand(self.all_inds.shape[0]) * 0.1 + 0.1)
        self.potentials.share_memory_()

        # If true, the same amount of frames is picked per class
        self.balance_classes = balance_classes

        # Choose batch_num in_r and max_in_p depending on validation or training
        if self.task == "train":
            self.batch_num = config["train"]["batch_num"]
            self.max_in_p = config["kpconv"]["max_in_points"]
            self.in_r = config["input"]["sphere_radius"]
        else:
            self.batch_num = config["train"]["val_batch_num"]
            self.max_in_p = config["kpconv"]["max_val_points"]
            self.in_r = config["kpconv"]["val_radius"]

        # shared epoch indices and classes (in case we want class balanced sampler)
        if self.task == "train":
            n_ = int(np.ceil(config["train"]["epoch_steps"] * self.batch_num * 1.1))
        else:
            n_ = int(np.ceil(config.validation_size * self.batch_num * 1.1))
        self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        self.epoch_inds = torch.from_numpy(np.zeros((n_,), dtype=np.int64))
        self.epoch_labels = torch.from_numpy(np.zeros((n_,), dtype=np.int32))
        self.epoch_i.share_memory_()
        self.epoch_inds.share_memory_()
        self.epoch_labels.share_memory_()

        self.worker_waiting = torch.tensor(
            [0 for _ in range(config["input"]["threads"])], dtype=torch.int32
        )
        self.worker_waiting.share_memory_()
        self.worker_lock = Lock()

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.frames)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch.
        Each worker is going to work in
        parallel to load a different list of indices.

        """

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        r_list = []
        r_inds_list = []
        r_mask_list = []
        val_labels_list = []
        batch_n = 0

        while True:

            t += [time.time()]

            with self.worker_lock:

                # Get potential minimum
                ind = int(self.epoch_inds[self.epoch_i])
                wanted_label = int(self.epoch_labels[self.epoch_i])

                # Update epoch indice
                self.epoch_i += 1
                if self.epoch_i >= int(self.epoch_inds.shape[0]):
                    self.epoch_i -= int(self.epoch_inds.shape[0])

            s_ind, f_ind = self.all_inds[ind]

            t += [time.time()]

            #########################
            # Merge n_frames together
            #########################

            # Initiate merged points
            merged_points = np.zeros((0, 3), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int32)
            merged_coords = np.zeros((0, 4), dtype=np.float32)

            # Get center of the first frame in world coordinates
            p_origin = np.zeros((1, 4))
            p_origin[0, 3] = 1
            pose0 = self.poses[s_ind][f_ind]
            p0 = p_origin.dot(pose0.T)[:, :3]
            p0 = np.squeeze(p0)
            o_pts = None
            o_labels = None

            t += [time.time()]

            num_merged = 0
            f_inc = 0
            while num_merged < self.config.n_frames and f_ind - f_inc >= 0:

                # Current frame pose
                pose = self.poses[s_ind][f_ind - f_inc]

                # Select frame only if center has moved far away (more than x_ meter).
                # Negative value to ignore.
                x_ = -1.0
                if x_ > 0:
                    diff = p_origin.dot(pose.T)[:, :3] - p_origin.dot(pose0.T)[:, :3]
                    if num_merged > 0 and np.linalg.norm(diff) < num_merged * x_:
                        f_inc += 1
                        continue

                # Path of points and labels
                seq_path = os.path.join(self.datapath, "sequences", self.sequences[s_ind])
                velo_file = os.path.join(
                    seq_path, "velodyne", self.frames[s_ind][f_ind - f_inc] + ".bin"
                )
                if self.task == "test":
                    label_file = None
                else:
                    label_file = os.path.join(
                        seq_path, "labels", self.frames[s_ind][f_ind - f_inc] + ".label"
                    )

                # Read points
                frame_points = np.fromfile(velo_file, dtype=np.float32)
                points = frame_points.reshape((-1, 4))

                if self.task == "test":
                    # Fake labels
                    sem_labels = np.zeros((frame_points.shape[0],), dtype=np.int32)
                else:
                    # Read labels
                    frame_labels = np.fromfile(label_file, dtype=np.int32)
                    sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                    sem_labels = self.learning_map[sem_labels]

                # Apply pose (without np.dot to avoid multi-threading)
                hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
                new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

                # In case of validation, keep the original points in memory
                if self.task in ["validate", "test"] and f_inc == 0:
                    o_pts = new_points[:, :3].astype(np.float32)
                    o_labels = sem_labels.astype(np.int32)

                # In case radius smaller than 50m, chose new center on a point of the wanted class
                # or not
                if self.in_r < 50.0 and f_inc == 0:
                    if self.balance_classes:
                        wanted_ind = np.random.choice(np.where(sem_labels == wanted_label)[0])
                    else:
                        wanted_ind = np.random.choice(new_points.shape[0])
                    p0 = new_points[wanted_ind, :3]

                # Eliminate points further than config["input"]["sphere_radius"]
                mask = np.sum(np.square(new_points[:, :3] - p0), axis=1) < self.in_r**2
                mask_inds = np.where(mask)[0].astype(np.int32)

                # Shuffle points
                rand_order = np.random.permutation(mask_inds)
                new_points = new_points[rand_order, :3]
                sem_labels = sem_labels[rand_order]

                # Place points in original frame reference to get coordinates
                if f_inc == 0:
                    new_coords = points[rand_order, :]
                else:
                    # We have to project in the first frame coordinates
                    new_coords = new_points - pose0[:3, 3]
                    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
                    new_coords = np.hstack((new_coords, points[rand_order, 3:]))

                # Increment merge count
                merged_points = np.vstack((merged_points, new_points))
                merged_labels = np.hstack((merged_labels, sem_labels))
                merged_coords = np.vstack((merged_coords, new_coords))
                num_merged += 1
                f_inc += 1

            t += [time.time()]

            #########################
            # Merge n_frames together
            #########################

            # Subsample merged frames
            in_pts, in_fts, in_lbls = grid_subsampling(
                merged_points,
                features=merged_coords,
                labels=merged_labels,
                sampledl=self.config["kpconv"]["first_subsampling_dl"],
            )

            t += [time.time()]

            # Number collected
            n = in_pts.shape[0]

            # Safe check
            if n < 2:
                continue

            # Randomly drop some points (augmentation process and safety for GPU memory consumption)
            if n > self.max_in_p:
                input_inds = np.random.choice(n, size=self.max_in_p, replace=False)
                in_pts = in_pts[input_inds, :]
                in_fts = in_fts[input_inds, :]
                in_lbls = in_lbls[input_inds]
                n = input_inds.shape[0]

            t += [time.time()]

            # Before augmenting, compute reprojection inds (only for validation and test)
            if self.task in ["validate", "test"]:

                # get val_points that are in range
                radiuses = np.sum(np.square(o_pts - p0), axis=1)
                reproj_mask = radiuses < (0.99 * self.in_r) ** 2

                # Project predictions on the frame points
                search_tree = KDTree(in_pts, leaf_size=50)
                proj_inds = search_tree.query(o_pts[reproj_mask, :], return_distance=False)
                proj_inds = np.squeeze(proj_inds).astype(np.int32)
            else:
                proj_inds = np.zeros((0,))
                reproj_mask = np.zeros((0,))

            t += [time.time()]

            # Data augmentation
            in_pts, _, scale, r_ = self.augmentation_transform(in_pts)

            t += [time.time()]

            # Color augmentation
            if np.random.rand() > self.config["train"]["augment_color"]:
                in_fts[:, 3:] *= 0

            # Stack batch
            p_list += [in_pts]
            f_list += [in_fts]
            l_list += [np.squeeze(in_lbls)]
            fi_list += [[s_ind, f_ind]]
            p0_list += [p0]
            s_list += [scale]
            r_list += [r_]
            r_inds_list += [proj_inds]
            r_mask_list += [reproj_mask]
            val_labels_list += [o_labels]

            t += [time.time()]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        # Concatenate batch
        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        frame_inds = np.array(fi_list, dtype=np.int32)
        frame_centers = np.stack(p0_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(r_list, axis=0)

        # Input features (Use reflectance, input height or all coordinates)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config["input"]["features_dim"] == 1:
            pass
        elif self.config["input"]["features_dim"] == 2:
            # Use original height coordinate
            stacked_features = np.hstack((stacked_features, features[:, 2:3]))
        elif self.config["input"]["features_dim"] == 3:
            # Use height + reflectance
            stacked_features = np.hstack((stacked_features, features[:, 2:]))
        elif self.config["input"]["features_dim"] == 4:
            # Use all coordinates
            stacked_features = np.hstack((stacked_features, features[:3]))
        elif self.config["input"]["features_dim"] == 5:
            # Use all coordinates + reflectance
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError("Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)")

        t += [time.time()]

        # Create network inputs
        #   Points, neighbors, pooling indices for each layers
        # Get the whole input list
        input_list = self.segmentation_inputs(
            stacked_points, stacked_features, labels.astype(np.int64), stack_lengths
        )

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [
            scales,
            rots,
            frame_inds,
            frame_centers,
            r_inds_list,
            r_mask_list,
            val_labels_list,
        ]

        t += [time.time()]

        # Display timings
        debug_t = False
        if debug_t:
            print("\n************************\n")
            print("Timings:")
            ti = 0
            n_ = 9
            mess = "Init ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Lock ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Init ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Load ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Subs ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Drop ...... {:5.1f}ms /"
            loop_times = [
                1000 * (t[ti + n_ * i + 1] - t[ti + n_ * i]) for i in range(len(stack_lengths))
            ]
            for dt in loop_times:
                mess += f" {dt:5.1f}"
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = "Reproj .... {:5.1f}ms /"
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
            ti += 1
            mess = "Stack ..... {:5.1f}ms /"
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

        return [self.config["model"]["num_layers"]] + input_list

    def load_calib_poses(self):
        """
        Load calib poses and times.
        """

        # Load data
        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in self.sequences:

            seq_folder = os.path.join(self.datapath, "sequences", seq)

            # Read Calib
            self.calibrations.append(self.parse_calibration(os.path.join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(os.path.join(seq_folder, "times.txt"), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(
                os.path.join(seq_folder, "poses.txt"), self.calibrations[-1]
            )
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

        # Prepare the indices of all frames
        seq_inds = np.hstack(
            [np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)]
        )
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T

        # For each class list the frames containing them
        if self.task in ["train", "validate"]:

            class_frames_bool = np.zeros((0, self.num_classes), dtype=np.bool)
            self.class_proportions = np.zeros((self.num_classes,), dtype=np.int32)

            for seq, seq_frames in zip(self.sequences, self.frames):

                frame_mode = "single"
                if self.config.n_frames > 1:
                    frame_mode = "multi"
                seq_stat_file = os.path.join(
                    self.datapath, "sequences", seq, f"stats_{frame_mode}.pkl"
                )

                # Check if inputs have already been computed
                if os.path.exists(seq_stat_file):
                    # Read pkl
                    with open(seq_stat_file, "rb") as f:
                        seq_class_frames, seq_proportions = pickle.load(f)

                else:

                    # Initiate dict
                    print(f"Preparing seq {seq} class frames. (Long but one time only)")

                    # Class frames as a boolean mask
                    seq_class_frames = np.zeros((len(seq_frames), self.num_classes), dtype=np.bool)

                    # Proportion of each class
                    seq_proportions = np.zeros((self.num_classes,), dtype=np.int32)

                    # Sequence path
                    seq_path = os.path.join(self.datapath, "sequences", seq)

                    # Read all frames
                    for f_ind, frame_name in enumerate(seq_frames):

                        # Path of points and labels
                        label_file = os.path.join(seq_path, "labels", frame_name + ".label")

                        # Read labels
                        frame_labels = np.fromfile(label_file, dtype=np.int32)
                        sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                        sem_labels = self.learning_map[sem_labels]

                        # Get present labels and there frequency
                        unique, counts = np.unique(sem_labels, return_counts=True)

                        # Add this frame to the frame lists of all class present
                        frame_labels = np.array(
                            [self.label_to_idx[unique_label] for unique_label in unique],
                            dtype=np.int32,
                        )
                        seq_class_frames[f_ind, frame_labels] = True

                        # Add proportions
                        seq_proportions[frame_labels] += counts

                    # Save pickle
                    with open(seq_stat_file, "wb") as f:
                        pickle.dump([seq_class_frames, seq_proportions], f)

                class_frames_bool = np.vstack((class_frames_bool, seq_class_frames))
                self.class_proportions += seq_proportions

            # Transform boolean indexing to int indices.
            self.class_frames = []
            for i, c in enumerate(self.label_values):
                if c in self.ignored_labels:
                    self.class_frames.append(torch.zeros((0,), dtype=torch.int64))
                else:
                    integer_inds = np.where(class_frames_bool[:, i])[0]
                    self.class_frames.append(torch.from_numpy(integer_inds.astype(np.int64)))

        # Add variables for validation
        if self.task == "validate":
            self.val_points = []
            self.val_labels = []
            self.val_confs = []

            for seq_frames in self.frames:
                self.val_confs.append(
                    np.zeros((len(seq_frames), self.num_classes, self.num_classes))
                )

    def parse_calibration(self, filename):
        """
        Read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        with open(filename, encoding="utf-8") as calib_file:
            for line in calib_file:
                key, content = line.strip().split(":")
                values = [float(v) for v in content.strip().split()]

                pose = np.zeros((4, 4))
                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0

                calib[key] = pose

        return calib

    def parse_poses(self, filename, calibration):
        """
        Read poses file with per-scan poses from given filename

        :returns list of poses as 4x4 numpy arrays.
        """
        with open(filename, encoding="utf-8") as file:
            poses = []

            tr = calibration["Tr"]
            tr_inv = np.linalg.inv(tr)

            for line in file:
                values = [float(v) for v in line.strip().split()]

                pose = np.zeros((4, 4))
                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0

                poses.append(np.matmul(tr_inv, np.matmul(pose, tr)))

        return poses


def debug_timing(dataset, loader):
    """
    Timing of generator function
    """

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.batch_num
    estim_n = 0

    for _ in range(10):

        for batch_i, batch in enumerate(loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.frame_inds) - estim_b) / 100
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


def debug_class_w(dataset, loader):
    """
    Timing of generator function
    """

    i = 0

    counts = np.zeros((dataset.num_classes,), dtype=np.int64)

    s = "step"
    for c in dataset.label_names:
        s += f"{c[:4]:^6}"
    print(s)
    print(6 * "-" + "|" + 6 * dataset.num_classes * "-")

    for _ in range(10):
        for batch in loader:

            # count labels
            new_counts = np.bincount(batch.labels)

            counts[: new_counts.shape[0]] += new_counts.astype(np.int64)

            # Update proportions
            proportions = 1000 * counts / np.sum(counts)

            s = f"{i:^6d}|"
            for pp in proportions:
                s += f"{pp:^6.1f}"
            print(s)
            i += 1
