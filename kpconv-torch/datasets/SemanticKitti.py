#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling SemanticKitti dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
import torch
import yaml
from multiprocessing import Lock


# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import *
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *
from utils.metrics import fast_confusion

from datasets.common import grid_subsampling
from utils.config import bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class SemanticKittiDataset(PointCloudDataset):
    """Class to handle SemanticKitti dataset."""

    def __init__(self, config, set='training', balance_classes=True):
        PointCloudDataset.__init__(self, 'SemanticKitti')

        ##########################
        # Parameters for the files
        ##########################

        # Dataset folder
        self.path = '../../Data/SemanticKitti'

        # Type of task conducted on this dataset
        self.dataset_task = 'slam_segmentation'

        # Training or test set
        self.set = set

        # Get a list of sequences
        if self.set == 'training':
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i != 8]
        elif self.set == 'validation':
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 8]
        elif self.set == 'test':
            self.sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', self.set)

        # List all files in each sequence
        self.frames = []
        for seq in self.sequences:
            velo_path = join(self.path, 'sequences', seq, 'velodyne')
            frames = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.bin')])
            self.frames.append(frames)

        ###########################
        # Object classes parameters
        ###########################

        # Read labels
        if config.n_frames == 1:
            config_file = join(self.path, 'semantic-kitti.yaml')
        elif config.n_frames > 1:
            config_file = join(self.path, 'semantic-kitti-all.yaml')
        else:
            raise ValueError('number of frames has to be >= 1')

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            learning_map_inv = doc['learning_map_inv']
            learning_map = doc['learning_map']
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v

        # Dict from labels to names
        self.label_to_names = {k: all_labels[v] for k, v in learning_map_inv.items()}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])

        ##################
        # Other parameters
        ##################

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        ##################
        # Load calibration
        ##################

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

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize frame potentials
        self.potentials = torch.from_numpy(np.random.rand(self.all_inds.shape[0]) * 0.1 + 0.1)
        self.potentials.share_memory_()

        # If true, the same amount of frames is picked per class
        self.balance_classes = balance_classes

        # Choose batch_num in_R and max_in_p depending on validation or training
        if self.set == 'training':
            self.batch_num = config.batch_num
            self.max_in_p = config.max_in_points
            self.in_R = config.in_radius
        else:
            self.batch_num = config.val_batch_num
            self.max_in_p = config.max_val_points
            self.in_R = config.val_radius

        # shared epoch indices and classes (in case we want class balanced sampler)
        if set == 'training':
            N = int(np.ceil(config.epoch_steps * self.batch_num * 1.1))
        else:
            N = int(np.ceil(config.validation_size * self.batch_num * 1.1))
        self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        self.epoch_inds = torch.from_numpy(np.zeros((N,), dtype=np.int64))
        self.epoch_labels = torch.from_numpy(np.zeros((N,), dtype=np.int32))
        self.epoch_i.share_memory_()
        self.epoch_inds.share_memory_()
        self.epoch_labels.share_memory_()

        self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
        self.worker_waiting.share_memory_()
        self.worker_lock = Lock()

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.frames)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        fi_list = []
        p0_list = []
        s_list = []
        R_list = []
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

                # Select frame only if center has moved far away (more than X meter). Negative value to ignore
                X = -1.0
                if X > 0:
                    diff = p_origin.dot(pose.T)[:, :3] - p_origin.dot(pose0.T)[:, :3]
                    if num_merged > 0 and np.linalg.norm(diff) < num_merged * X:
                        f_inc += 1
                        continue

                # Path of points and labels
                seq_path = join(self.path, 'sequences', self.sequences[s_ind])
                velo_file = join(seq_path, 'velodyne', self.frames[s_ind][f_ind - f_inc] + '.bin')
                if self.set == 'test':
                    label_file = None
                else:
                    label_file = join(seq_path, 'labels', self.frames[s_ind][f_ind - f_inc] + '.label')

                # Read points
                frame_points = np.fromfile(velo_file, dtype=np.float32)
                points = frame_points.reshape((-1, 4))

                if self.set == 'test':
                    # Fake labels
                    sem_labels = np.zeros((frame_points.shape[0],), dtype=np.int32)
                else:
                    # Read labels
                    frame_labels = np.fromfile(label_file, dtype=np.int32)
                    sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                    sem_labels = self.learning_map[sem_labels]

                # Apply pose (without np.dot to avoid multi-threading)
                hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
                #new_points = hpoints.dot(pose.T)
                new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
                #new_points[:, 3:] = points[:, 3:]

                # In case of validation, keep the original points in memory
                if self.set in ['validation', 'test'] and f_inc == 0:
                    o_pts = new_points[:, :3].astype(np.float32)
                    o_labels = sem_labels.astype(np.int32)

                # In case radius smaller than 50m, chose new center on a point of the wanted class or not
                if self.in_R < 50.0 and f_inc == 0:
                    if self.balance_classes:
                        wanted_ind = np.random.choice(np.where(sem_labels == wanted_label)[0])
                    else:
                        wanted_ind = np.random.choice(new_points.shape[0])
                    p0 = new_points[wanted_ind, :3]

                # Eliminate points further than config.in_radius
                mask = np.sum(np.square(new_points[:, :3] - p0), axis=1) < self.in_R ** 2
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
                    # new_coords = new_coords.dot(pose0[:3, :3])
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
            in_pts, in_fts, in_lbls = grid_subsampling(merged_points,
                                                       features=merged_coords,
                                                       labels=merged_labels,
                                                       sampleDl=self.config.first_subsampling_dl)

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
            if self.set in ['validation', 'test']:

                # get val_points that are in range
                radiuses = np.sum(np.square(o_pts - p0), axis=1)
                reproj_mask = radiuses < (0.99 * self.in_R) ** 2

                # Project predictions on the frame points
                search_tree = KDTree(in_pts, leaf_size=50)
                proj_inds = search_tree.query(o_pts[reproj_mask, :], return_distance=False)
                proj_inds = np.squeeze(proj_inds).astype(np.int32)
            else:
                proj_inds = np.zeros((0,))
                reproj_mask = np.zeros((0,))

            t += [time.time()]

            # Data augmentation
            in_pts, scale, R = self.augmentation_transform(in_pts)

            t += [time.time()]

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                in_fts[:, 3:] *= 0

            # Stack batch
            p_list += [in_pts]
            f_list += [in_fts]
            l_list += [np.squeeze(in_lbls)]
            fi_list += [[s_ind, f_ind]]
            p0_list += [p0]
            s_list += [scale]
            R_list += [R]
            r_inds_list += [proj_inds]
            r_mask_list += [reproj_mask]
            val_labels_list += [o_labels]

            t += [time.time()]

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
        frame_inds = np.array(fi_list, dtype=np.int32)
        frame_centers = np.stack(p0_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features (Use reflectance, input height or all coordinates)
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 2:
            # Use original height coordinate
            stacked_features = np.hstack((stacked_features, features[:, 2:3]))
        elif self.config.in_features_dim == 3:
            # Use height + reflectance
            stacked_features = np.hstack((stacked_features, features[:, 2:]))
        elif self.config.in_features_dim == 4:
            # Use all coordinates
            stacked_features = np.hstack((stacked_features, features[:3]))
        elif self.config.in_features_dim == 5:
            # Use all coordinates + reflectance
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        t += [time.time()]

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels.astype(np.int64),
                                              stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, frame_inds, frame_centers, r_inds_list, r_mask_list, val_labels_list]

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print('\n************************\n')
            print('Timings:')
            ti = 0
            N = 9
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Lock ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Load ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Subs ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Drop ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Reproj .... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Augment ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Stack ..... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print('concat .... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('input ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('stack ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('\n************************\n')

        return [self.config.num_layers] + input_list

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in self.sequences:

            seq_folder = join(self.path, 'sequences', seq)

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

        ###################################
        # Prepare the indices of all frames
        ###################################

        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T

        ################################################
        # For each class list the frames containing them
        ################################################

        if self.set in ['training', 'validation']:

            class_frames_bool = np.zeros((0, self.num_classes), dtype=np.bool)
            self.class_proportions = np.zeros((self.num_classes,), dtype=np.int32)

            for s_ind, (seq, seq_frames) in enumerate(zip(self.sequences, self.frames)):

                frame_mode = 'single'
                if self.config.n_frames > 1:
                    frame_mode = 'multi'
                seq_stat_file = join(self.path, 'sequences', seq, 'stats_{:s}.pkl'.format(frame_mode))

                # Check if inputs have already been computed
                if exists(seq_stat_file):
                    # Read pkl
                    with open(seq_stat_file, 'rb') as f:
                        seq_class_frames, seq_proportions = pickle.load(f)

                else:

                    # Initiate dict
                    print('Preparing seq {:s} class frames. (Long but one time only)'.format(seq))

                    # Class frames as a boolean mask
                    seq_class_frames = np.zeros((len(seq_frames), self.num_classes), dtype=np.bool)

                    # Proportion of each class
                    seq_proportions = np.zeros((self.num_classes,), dtype=np.int32)

                    # Sequence path
                    seq_path = join(self.path, 'sequences', seq)

                    # Read all frames
                    for f_ind, frame_name in enumerate(seq_frames):

                        # Path of points and labels
                        label_file = join(seq_path, 'labels', frame_name + '.label')

                        # Read labels
                        frame_labels = np.fromfile(label_file, dtype=np.int32)
                        sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                        sem_labels = self.learning_map[sem_labels]

                        # Get present labels and there frequency
                        unique, counts = np.unique(sem_labels, return_counts=True)

                        # Add this frame to the frame lists of all class present
                        frame_labels = np.array([self.label_to_idx[l] for l in unique], dtype=np.int32)
                        seq_class_frames[f_ind, frame_labels] = True

                        # Add proportions
                        seq_proportions[frame_labels] += counts

                    # Save pickle
                    with open(seq_stat_file, 'wb') as f:
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
        if self.set == 'validation':
            self.val_points = []
            self.val_labels = []
            self.val_confs = []

            for s_ind, seq_frames in enumerate(self.frames):
                self.val_confs.append(np.zeros((len(seq_frames), self.num_classes, self.num_classes)))

        return

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class SemanticKittiSampler(Sampler):
    """Sampler for SemanticKitti"""

    def __init__(self, dataset: SemanticKittiDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
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
                                new_class_inds = torch.randperm(class_potentials.shape[0]).type(torch.int64)
                                class_indices = torch.cat((class_indices, new_class_inds), dim=0)
                            class_indices = class_indices[:class_n]
                        class_indices = self.dataset.class_frames[i][class_indices]

                        # Add the indices to the generated ones
                        gen_indices.append(class_indices)
                        gen_classes.append(class_indices * 0 + c)

                        # Update potentials
                        update_inds = torch.unique(class_indices)
                        self.dataset.potentials[update_inds] = torch.ceil(self.dataset.potentials[update_inds])
                        self.dataset.potentials[update_inds] += torch.from_numpy(np.random.rand(update_inds.shape[0]) * 0.1 + 0.1)

                    else:
                        error_message = '\nIt seems there is a problem with the class statistics of your dataset, saved in the variable dataset.class_frames.\n'
                        error_message += 'Here are the current statistics:\n'
                        error_message += '{:>15s} {:>15s}\n'.format('Class', '# of frames')
                        for iii, ccc in enumerate(self.dataset.label_values):
                            error_message += '{:>15s} {:>15d}\n'.format(self.dataset.label_names[iii], len(self.dataset.class_frames[iii]))
                        error_message += '\nThis error is raised if one of the classes is not ignored and does not appear in any of the frames of the dataset.\n'
                        raise ValueError(error_message)

            # Stack the chosen indices of all classes
            gen_indices = torch.cat(gen_indices, dim=0)
            gen_classes = torch.cat(gen_classes, dim=0)

            # Shuffle generated indices
            rand_order = torch.randperm(gen_indices.shape[0])[:num_centers]
            gen_indices = gen_indices[rand_order]
            gen_classes = gen_classes[rand_order]

            # Update potentials (Change the order for the next epoch)
            #self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
            #self.dataset.potentials[gen_indices] += torch.from_numpy(np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)

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
                _, gen_indices = torch.topk(self.dataset.potentials, num_centers, largest=False, sorted=True)
            else:
                gen_indices = torch.randperm(self.dataset.potentials.shape[0])
                while gen_indices.shape[0] < num_centers:
                    new_gen_indices = torch.randperm(self.dataset.potentials.shape[0]).type(torch.int32)
                    gen_indices = torch.cat((gen_indices, new_gen_indices), dim=0)
                gen_indices = gen_indices[:num_centers]

            # Update potentials (Change the order for the next epoch)
            self.dataset.potentials[gen_indices] = torch.ceil(self.dataset.potentials[gen_indices])
            self.dataset.potentials[gen_indices] += torch.from_numpy(np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1)

            # Update epoch inds
            self.dataset.epoch_inds += gen_indices

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def calib_max_in(self, config, dataloader, untouched_ratio=0.8, verbose=True, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration of max_in_points value (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load max_in_limit dictionary
        max_in_lim_file = join(self.dataset.path, 'max_in_limits.pkl')
        if exists(max_in_lim_file):
            with open(max_in_lim_file, 'rb') as file:
                max_in_lim_dict = pickle.load(file)
        else:
            max_in_lim_dict = {}

        # Check if the max_in limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = 'balanced'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}'.format(sampler_method,
                                          self.dataset.in_R,
                                          self.dataset.config.first_subsampling_dl)
        if not redo and key in max_in_lim_dict:
            self.dataset.max_in_p = max_in_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check max_in limit dictionary')
            if key in max_in_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(max_in_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ########################
            # Batch calib parameters
            ########################

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            all_lengths = []
            N = 1000

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Control max_in_points value
                    all_lengths += batch.lengths[0].tolist()

                    # Convergence
                    if len(all_lengths) > N:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if t - last_display > 1.0:
                        last_display = t
                        message = 'Collecting {:d} in_points: {:5.1f}%'
                        print(message.format(N,
                                             100 * len(all_lengths) / N))

                if breaking:
                    break

            self.dataset.max_in_p = int(np.percentile(all_lengths, 100*untouched_ratio))

            if verbose:

                # Create histogram
                a = 1

            # Save max_in_limit dictionary
            print('New max_in_p = ', self.dataset.max_in_p)
            max_in_lim_dict[key] = self.dataset.max_in_p
            with open(max_in_lim_file, 'wb') as file:
                pickle.dump(max_in_lim_dict, file)

        # Update value in config
        if self.dataset.set == 'training':
            config.max_in_points = self.dataset.max_in_p
        else:
            config.max_val_points = self.dataset.max_in_p

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.balance_classes:
            sampler_method = 'balanced'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}_{:d}'.format(sampler_method,
                                                    self.dataset.in_R,
                                                    self.dataset.config.first_subsampling_dl,
                                                    self.dataset.batch_num,
                                                    self.dataset.max_in_p)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:s}_{:d}_{:.3f}_{:.3f}'.format(sampler_method, self.dataset.max_in_p, dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:s}_{:d}_{:.3f}_{:.3f}'.format(sampler_method, self.dataset.max_in_p, dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
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

            #####################
            # Perform calibration
            #####################

            #self.dataset.batch_limit[0] = self.dataset.max_in_p * (self.dataset.batch_num - 1)

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Control max_in_points value
                    are_cropped = batch.lengths[0] > self.dataset.max_in_p - 1
                    cropped_n += torch.sum(are_cropped.type(torch.int32)).item()
                    all_n += int(batch.lengths[0].shape[0])

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.frame_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit[0] += Kp * error

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
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.dataset.batch_limit[0])))

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

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Control max_in_points value
            print('\n**************************************************\n')
            if cropped_n > 0.3 * all_n:
                color = bcolors.FAIL
            else:
                color = bcolors.OKGREEN
            print('Current value of max_in_points {:d}'.format(self.dataset.max_in_p))
            print('  > {:}{:.1f}% inputs are cropped{:}'.format(color, 100 * cropped_n / all_n, bcolors.ENDC))
            if cropped_n > 0.3 * all_n:
                print('\nTry a higher max_in_points value\n'.format(100 * cropped_n / all_n))
                #raise ValueError('Value of max_in_points too low')
            print('\n**************************************************\n')

            # Save batch_limit dictionary
            key = '{:s}_{:.3f}_{:.3f}_{:d}_{:d}'.format(sampler_method,
                                                        self.dataset.in_R,
                                                        self.dataset.config.first_subsampling_dl,
                                                        self.dataset.batch_num,
                                                        self.dataset.max_in_p)
            batch_lim_dict[key] = float(self.dataset.batch_limit[0])
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:s}_{:d}_{:.3f}_{:.3f}'.format(sampler_method, self.dataset.max_in_p, dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class SemanticKittiCustomBatch:
    """Custom batch definition with memory pinning for SemanticKitti"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = int(input_list[0])

        # Extract input tensors from the list of numpy array
        ind = 1
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.frame_centers = torch.from_numpy(input_list[ind])
        ind += 1
        self.reproj_inds = input_list[ind]
        ind += 1
        self.reproj_masks = input_list[ind]
        ind += 1
        self.val_labels = input_list[ind]

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
        self.frame_inds = self.frame_inds.pin_memory()
        self.frame_centers = self.frame_centers.pin_memory()

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
        self.frame_inds = self.frame_inds.to(device)
        self.frame_centers = self.frame_centers.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
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


def SemanticKittiCollate(batch_data):
    return SemanticKittiCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_timing(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.batch_num
    estim_N = 0

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.frame_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b,
                                     estim_N))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_class_w(dataset, loader):
    """Timing of generator function"""

    i = 0

    counts = np.zeros((dataset.num_classes,), dtype=np.int64)

    s = '{:^6}|'.format('step')
    for c in dataset.label_names:
        s += '{:^6}'.format(c[:4])
    print(s)
    print(6*'-' + '|' + 6*dataset.num_classes*'-')

    for epoch in range(10):
        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # count labels
            new_counts = np.bincount(batch.labels)

            counts[:new_counts.shape[0]] += new_counts.astype(np.int64)

            # Update proportions
            proportions = 1000 * counts / np.sum(counts)

            s = '{:^6d}|'.format(i)
            for pp in proportions:
                s += '{:^6.1f}'.format(pp)
            print(s)
            i += 1

