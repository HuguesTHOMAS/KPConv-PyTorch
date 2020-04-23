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
import sys
import struct
import scipy
import time
import numpy as np
import pickle
import torch
import yaml
#from mayavi import mlab
from multiprocessing import Lock

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# OS functions
from os import listdir
from os.path import exists, join, isdir, getsize

# Dataset parent class
from datasets.common import *
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *
from utils.metrics import fast_confusion

from datasets.common import grid_subsampling
from utils.config import bcolors


def ssc_to_homo(ssc, ssc_in_radians=True):

    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    if ssc.ndim == 1:
        reduce = True
        ssc = np.expand_dims(ssc, 0)
    else:
        reduce = False

    if not ssc_in_radians:
        ssc[:, 3:] = np.pi / 180.0 * ssc[:, 3:]

    sr = np.sin(ssc[:, 3])
    cr = np.cos(ssc[:, 3])

    sp = np.sin(ssc[:, 4])
    cp = np.cos(ssc[:, 4])

    sh = np.sin(ssc[:, 5])
    ch = np.cos(ssc[:, 5])

    H = np.zeros((ssc.shape[0], 4, 4))

    H[:, 0, 0] = ch*cp
    H[:, 0, 1] = -sh*cr + ch*sp*sr
    H[:, 0, 2] = sh*sr + ch*sp*cr
    H[:, 1, 0] = sh*cp
    H[:, 1, 1] = ch*cr + sh*sp*sr
    H[:, 1, 2] = -ch*sr + sh*sp*cr
    H[:, 2, 0] = -sp
    H[:, 2, 1] = cp*sr
    H[:, 2, 2] = cp*cr

    H[:, 0, 3] = ssc[:, 0]
    H[:, 1, 3] = ssc[:, 1]
    H[:, 2, 3] = ssc[:, 2]

    H[:, 3, 3] = 1

    if reduce:
        H = np.squeeze(H)

    return H


def verify_magic(s):

    magic = 44444

    m = struct.unpack('<HHHH', s)

    return len(m)>=4 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic


def test_read_hits():

    data_path = '../../Data/NCLT'
    velo_folder = 'velodyne_data'
    day = '2012-01-08'

    hits_path = join(data_path, velo_folder, day, 'velodyne_hits.bin')

    all_utimes = []
    all_hits = []
    all_ints = []

    num_bytes = getsize(hits_path)
    current_bytes = 0

    with open(hits_path, 'rb') as f_bin:

        total_hits = 0
        first_utime = -1
        last_utime = -1

        while True:

            magic = f_bin.read(8)
            if magic == b'':
                break

            if not verify_magic(magic):
                print('Could not verify magic')

            num_hits = struct.unpack('<I', f_bin.read(4))[0]
            utime = struct.unpack('<Q', f_bin.read(8))[0]

            # Do not convert padding (it is an int always equal to zero)
            padding = f_bin.read(4)

            total_hits += num_hits
            if first_utime == -1:
                first_utime = utime
            last_utime = utime

            hits = []
            ints = []

            for i in range(num_hits):

                x = struct.unpack('<H', f_bin.read(2))[0]
                y = struct.unpack('<H', f_bin.read(2))[0]
                z = struct.unpack('<H', f_bin.read(2))[0]
                i = struct.unpack('B', f_bin.read(1))[0]
                l = struct.unpack('B', f_bin.read(1))[0]

                hits += [[x, y, z]]
                ints += [i]

            utimes = np.full((num_hits,), utime - first_utime, dtype=np.int32)
            ints = np.array(ints, dtype=np.uint8)
            hits = np.array(hits, dtype=np.float32)
            hits *= 0.005
            hits += -100.0

            all_utimes.append(utimes)
            all_hits.append(hits)
            all_ints.append(ints)

            if 100 * current_bytes / num_bytes > 0.1:
                break

            current_bytes += 24 + 8 * num_hits

            print('{:d}/{:d}  =>  {:.1f}%'.format(current_bytes, num_bytes, 100 * current_bytes / num_bytes))

        all_utimes = np.hstack(all_utimes)
        all_hits = np.vstack(all_hits)
        all_ints = np.hstack(all_ints)

        write_ply('test_hits',
                  [all_hits, all_ints, all_utimes],
                  ['x', 'y', 'z', 'intensity', 'utime'])

    print("Read %d total hits from %ld to %ld" % (total_hits, first_utime, last_utime))

    return 0


def frames_to_ply(show_frames=False):

    # In files
    data_path = '../../Data/NCLT'
    velo_folder = 'velodyne_data'

    days = np.sort([d for d in listdir(join(data_path, velo_folder))])

    for day in days:

        # Out files
        ply_folder = join(data_path, 'frames_ply', day)
        if not exists(ply_folder):
            makedirs(ply_folder)

        day_path = join(data_path, velo_folder, day, 'velodyne_sync')
        f_names = np.sort([f for f in listdir(day_path) if f[-4:] == '.bin'])

        N = len(f_names)
        print('Reading', N, 'files')

        for f_i, f_name in enumerate(f_names):

            ply_name = join(ply_folder, f_name[:-4] + '.ply')
            if exists(ply_name):
                continue


            t1 = time.time()

            hits = []
            ints = []

            with open(join(day_path, f_name), 'rb') as f_bin:

                while True:
                    x_str = f_bin.read(2)

                    # End of file
                    if x_str == b'':
                        break

                    x = struct.unpack('<H', x_str)[0]
                    y = struct.unpack('<H', f_bin.read(2))[0]
                    z = struct.unpack('<H', f_bin.read(2))[0]
                    intensity = struct.unpack('B', f_bin.read(1))[0]
                    l = struct.unpack('B', f_bin.read(1))[0]

                    hits += [[x, y, z]]
                    ints += [intensity]

            ints = np.array(ints, dtype=np.uint8)
            hits = np.array(hits, dtype=np.float32)
            hits *= 0.005
            hits += -100.0

            write_ply(ply_name,
                      [hits, ints],
                      ['x', 'y', 'z', 'intensity'])

            t2 = time.time()
            print('File {:s} {:d}/{:d} Done in {:.1f}s'.format(f_name, f_i, N, t2 - t1))

            if show_frames:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(hits[:, 0], hits[:, 1], -hits[:, 2], c=-hits[:, 2], s=5, linewidths=0)
                plt.show()

    return 0


def merge_day_pointclouds(show_day_trajectory=False, only_SLAM_nodes=False):
    """
    Recreate the whole day point cloud thks to gt pose
    Generate gt_annotation of mobile objects
    """

    # In files
    data_path = '../../Data/NCLT'
    gt_folder = 'ground_truth'
    cov_folder = 'ground_truth_cov'

    # Transformation from body to velodyne frame (from NCLT paper)
    x_body_velo = np.array([0.002, -0.004, -0.957, 0.807, 0.166, -90.703])
    H_body_velo = ssc_to_homo(x_body_velo, ssc_in_radians=False)
    H_velo_body = np.linalg.inv(H_body_velo)
    x_body_lb3 = np.array([0.035, 0.002, -1.23, -179.93, -0.23, 0.50])
    H_body_lb3 = ssc_to_homo(x_body_lb3, ssc_in_radians=False)
    H_lb3_body = np.linalg.inv(H_body_lb3)

    # Get gt files and days
    gt_files = np.sort([gt_f for gt_f in listdir(join(data_path, gt_folder)) if gt_f[-4:] == '.csv'])
    cov_files = np.sort([cov_f for cov_f in listdir(join(data_path, cov_folder)) if cov_f[-4:] == '.csv'])
    days = [d[:-4].split('_')[1] for d in gt_files]

    # Load all gt poses
    print('\nLoading days groundtruth poses...')
    t0 = time.time()
    gt_H = []
    gt_t = []
    for d, gt_f in enumerate(gt_files):

        t1 = time.time()

        gt_pkl_file = join(data_path, gt_folder, gt_f[:-4] + '.pkl')
        if exists(gt_pkl_file):
            # Read pkl
            with open(gt_pkl_file, 'rb') as f:
                day_gt_t, day_gt_H = pickle.load(f)

        else:
            # File paths
            gt_csv = join(data_path, gt_folder, gt_f)

            # Load gt
            gt = np.loadtxt(gt_csv, delimiter=',')

            # Convert gt to homogenous rotation/translation matrix
            day_gt_t = gt[:, 0]
            day_gt_H = ssc_to_homo(gt[:, 1:])

            # Save pickle
            with open(gt_pkl_file, 'wb') as f:
                pickle.dump([day_gt_t, day_gt_H], f)

            t2 = time.time()
            print('{:s} {:d}/{:d} Done in {:.1f}s'.format(gt_f, d, gt_files.shape[0], t2 - t1))

        gt_t += [day_gt_t]
        gt_H += [day_gt_H]

        if show_day_trajectory:

            cov_csv = join(data_path, cov_folder, cov_files[d])
            cov = np.loadtxt(cov_csv, delimiter=',')
            t_cov = cov[:, 0]
            t_cov_bool = np.logical_and(t_cov > np.min(day_gt_t), t_cov < np.max(day_gt_t))
            t_cov = t_cov[t_cov_bool]

            # Note: Interpolation is not needed, this is done as a convinience
            interp = scipy.interpolate.interp1d(day_gt_t, day_gt_H[:, :3, 3], kind='nearest', axis=0)
            node_poses = interp(t_cov)

            plt.figure()
            plt.scatter(day_gt_H[:, 1, 3], day_gt_H[:, 0, 3], 1, c=-day_gt_H[:, 2, 3], linewidth=0)
            plt.scatter(node_poses[:, 1], node_poses[:, 0], 1, c=-node_poses[:, 2], linewidth=5)
            plt.axis('equal')
            plt.title('Ground Truth Position of Nodes in SLAM Graph')
            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            plt.colorbar()

            plt.show()

    t2 = time.time()
    print('Done in {:.1f}s\n'.format(t2 - t0))

    # Out files
    out_folder = join(data_path, 'day_ply')
    if not exists(out_folder):
        makedirs(out_folder)

    # Focus on a particular point
    p0 = np.array([-220, -527, 12])
    center_radius = 10.0
    point_radius = 50.0

    # Loop on days
    for d, day in enumerate(days):

        #if day != '2012-02-05':
        #    continue
        day_min_t = gt_t[d][0]
        day_max_t = gt_t[d][-1]

        frames_folder = join(data_path, 'frames_ply', day)
        f_times = np.sort([float(f[:-4]) for f in listdir(frames_folder) if f[-4:] == '.ply'])

        # If we want, load only SLAM nodes
        if only_SLAM_nodes:

            # Load node timestamps
            cov_csv = join(data_path, cov_folder, cov_files[d])
            cov = np.loadtxt(cov_csv, delimiter=',')
            t_cov = cov[:, 0]
            t_cov_bool = np.logical_and(t_cov > day_min_t, t_cov < day_max_t)
            t_cov = t_cov[t_cov_bool]

            # Find closest lidar frames
            t_cov = np.expand_dims(t_cov, 1)
            diffs = np.abs(t_cov - f_times)
            inds = np.argmin(diffs, axis=1)
            f_times = f_times[inds]

        # Is this frame in gt
        f_t_bool = np.logical_and(f_times > day_min_t, f_times < day_max_t)
        f_times = f_times[f_t_bool]

        # Interpolation gt poses to frame timestamps
        interp = scipy.interpolate.interp1d(gt_t[d], gt_H[d], kind='nearest', axis=0)
        frame_poses = interp(f_times)

        N = len(f_times)
        world_points = []
        world_frames = []
        world_frames_c = []
        print('Reading', day, ' => ', N, 'files')
        for f_i, f_t in enumerate(f_times):

            t1 = time.time()

            #########
            # GT pose
            #########

            H = frame_poses[f_i].astype(np.float32)
            # s = '\n'
            # for cc in H:
            #     for c in cc:
            #         s += '{:5.2f} '.format(c)
            #     s += '\n'
            # print(s)

            #############
            # Focus check
            #############

            if np.linalg.norm(H[:3, 3] - p0) > center_radius:
                continue

            ###################################
            # Local frame coordinates for debug
            ###################################

            # Create artificial frames
            x = np.linspace(0, 1, 50, dtype=np.float32)
            points = np.hstack((np.vstack((x, x*0, x*0)), np.vstack((x*0, x, x*0)), np.vstack((x*0, x*0, x)))).T
            colors = ((points > 0.1).astype(np.float32) * 255).astype(np.uint8)

            hpoints = np.hstack((points, np.ones_like(points[:, :1])))
            hpoints = np.matmul(hpoints, H.T)
            hpoints[:, 3] *= 0
            world_frames += [hpoints[:, :3]]
            world_frames_c += [colors]

            #######################
            # Load velo point cloud
            #######################

            # Load frame ply file
            f_name = '{:.0f}.ply'.format(f_t)
            data = read_ply(join(frames_folder, f_name))
            points = np.vstack((data['x'], data['y'], data['z'])).T
            #intensity = data['intensity']

            hpoints = np.hstack((points, np.ones_like(points[:, :1])))
            hpoints = np.matmul(hpoints, H.T)
            hpoints[:, 3] *= 0
            hpoints[:, 3] += np.sqrt(f_t - f_times[0])

            # focus check
            focus_bool = np.linalg.norm(hpoints[:, :3] - p0, axis=1) < point_radius
            hpoints = hpoints[focus_bool, :]

            world_points += [hpoints]

            t2 = time.time()
            print('File {:s} {:d}/{:d} Done in {:.1f}s'.format(f_name, f_i, N, t2 - t1))

        if len(world_points) < 2:
            continue

        world_points = np.vstack(world_points)


        ###### DEBUG
        world_frames = np.vstack(world_frames)
        world_frames_c = np.vstack(world_frames_c)
        write_ply('testf.ply',
                  [world_frames, world_frames_c],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        ###### DEBUG

        print(world_points.shape, world_points.dtype)

        # Subsample merged frames
        # world_points, features = grid_subsampling(world_points[:, :3],
        #                                           features=world_points[:, 3:],
        #                                           sampleDl=0.1)
        features = world_points[:, 3:]
        world_points = world_points[:, :3]

        print(world_points.shape, world_points.dtype)

        write_ply('test' + day + '.ply',
                  [world_points, features],
                  ['x', 'y', 'z', 't'])


        # Generate gt annotations

        # Subsample day ply (for visualization)

        # Save day ply

        # a = 1/0
