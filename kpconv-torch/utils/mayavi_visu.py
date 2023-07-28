#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script for various visualization with mayavi
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


# Basic libs
import torch
import numpy as np
from sklearn.neighbors import KDTree
from os import makedirs, remove, rename, listdir
from os.path import exists, join
import time

import sys

# PLY reader
from utils.ply import write_ply, read_ply

# Configuration class
from utils.config import Config


def show_ModelNet_models(all_points):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i
    file_i = 0

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = all_points[file_i]

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    points[:, 2],
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ['g', 'G']:

            file_i = (file_i - 1) % len(all_points)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:

            file_i = (file_i + 1) % len(all_points)
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_ModelNet_examples(clouds, cloud_normals=None, cloud_labels=None):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    if cloud_labels is None:
        cloud_labels = [points[:, 2] for points in clouds]

    # Indices
    global file_i, show_normals
    file_i = 0
    show_normals = True

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Plot new data feature
        points = clouds[file_i]
        labels = cloud_labels[file_i]
        if cloud_normals is not None:
            normals = cloud_normals[file_i]
        else:
            normals = None

        # Rescale points for visu
        points = (points * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        # Show point clouds colorized with activations
        activations = mlab.points3d(points[:, 0],
                                    points[:, 1],
                                    points[:, 2],
                                    labels,
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    figure=fig1)
        if normals is not None and show_normals:
            activations = mlab.quiver3d(points[:, 0],
                                        points[:, 1],
                                        points[:, 2],
                                        normals[:, 0],
                                        normals[:, 1],
                                        normals[:, 2],
                                        scale_factor=10.0,
                                        scale_mode='none',
                                        figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i, show_normals

        if vtk_obj.GetKeyCode() in ['g', 'G']:
            file_i = (file_i - 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            file_i = (file_i + 1) % len(clouds)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            show_normals = not show_normals
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_neighbors(query, supports, neighbors):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Indices
    global file_i
    file_i = 0

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Rescale points for visu
        p1 = (query * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        p2 = (supports * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0

        l1 = p1[:, 2]*0
        l1[file_i] = 1

        l2 = p2[:, 2]*0 + 2
        l2[neighbors[file_i]] = 3

        # Show point clouds colorized with activations
        activations = mlab.points3d(p1[:, 0],
                                    p1[:, 1],
                                    p1[:, 2],
                                    l1,
                                    scale_factor=2.0,
                                    scale_mode='none',
                                    vmin=0.0,
                                    vmax=3.0,
                                    figure=fig1)

        activations = mlab.points3d(p2[:, 0],
                                    p2[:, 1],
                                    p2[:, 2],
                                    l2,
                                    scale_factor=3.0,
                                    scale_mode='none',
                                    vmin=0.0,
                                    vmax=3.0,
                                    figure=fig1)

        # New title
        mlab.title(str(file_i), color=(0, 0, 0), size=0.3, height=0.01)
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global file_i

        if vtk_obj.GetKeyCode() in ['g', 'G']:

            file_i = (file_i - 1) % len(query)
            update_scene()

        elif vtk_obj.GetKeyCode() in ['h', 'H']:

            file_i = (file_i + 1) % len(query)
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()


def show_input_batch(batch):
    from mayavi import mlab

    ###########################
    # Interactive visualization
    ###########################

    # Create figure for features
    fig1 = mlab.figure('Input', bgcolor=(1, 1, 1), size=(1000, 800))
    fig1.scene.parallel_projection = False

    # Unstack batch
    all_points = batch.unstack_points()
    all_neighbors = batch.unstack_neighbors()
    all_pools = batch.unstack_pools()

    # Indices
    global b_i, l_i, neighb_i, show_pools
    b_i = 0
    l_i = 0
    neighb_i = 0
    show_pools = False

    def update_scene():

        #  clear figure
        mlab.clf(fig1)

        # Rescale points for visu
        p = (all_points[l_i][b_i] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
        labels = p[:, 2]*0

        if show_pools:
            p2 = (all_points[l_i+1][b_i][neighb_i:neighb_i+1] * 1.5 + np.array([1.0, 1.0, 1.0])) * 50.0
            p = np.vstack((p, p2))
            labels = np.hstack((labels, np.ones((1,), dtype=np.int32)*3))
            pool_inds = all_pools[l_i][b_i][neighb_i]
            pool_inds = pool_inds[pool_inds >= 0]
            labels[pool_inds] = 2
        else:
            neighb_inds = all_neighbors[l_i][b_i][neighb_i]
            neighb_inds = neighb_inds[neighb_inds >= 0]
            labels[neighb_inds] = 2
            labels[neighb_i] = 3

        # Show point clouds colorized with activations
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      labels,
                      scale_factor=2.0,
                      scale_mode='none',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)


        """
        mlab.points3d(p[-2:, 0],
                      p[-2:, 1],
                      p[-2:, 2],
                      labels[-2:]*0 + 3,
                      scale_factor=0.16 * 1.5 * 50,
                      scale_mode='none',
                      mode='cube',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)
        mlab.points3d(p[-1:, 0],
                      p[-1:, 1],
                      p[-1:, 2],
                      labels[-1:]*0 + 2,
                      scale_factor=0.16 * 2 * 2.5 * 1.5 * 50,
                      scale_mode='none',
                      mode='sphere',
                      vmin=0.0,
                      vmax=3.0,
                      figure=fig1)
                      
        """

        # New title
        title_str = '<([) b_i={:d} (])>    <(,) l_i={:d} (.)>    <(N) n_i={:d} (M)>'.format(b_i, l_i, neighb_i)
        mlab.title(title_str, color=(0, 0, 0), size=0.3, height=0.90)
        if show_pools:
            text = 'pools (switch with G)'
        else:
            text = 'neighbors (switch with G)'
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.3)
        mlab.orientation_axes()

        return

    def keyboard_callback(vtk_obj, event):
        global b_i, l_i, neighb_i, show_pools

        if vtk_obj.GetKeyCode() in ['[', '{']:
            b_i = (b_i - 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [']', '}']:
            b_i = (b_i + 1) % len(all_points[l_i])
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in [',', '<']:
            if show_pools:
                l_i = (l_i - 1) % (len(all_points) - 1)
            else:
                l_i = (l_i - 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['.', '>']:
            if show_pools:
                l_i = (l_i + 1) % (len(all_points) - 1)
            else:
                l_i = (l_i + 1) % len(all_points)
            neighb_i = 0
            update_scene()

        elif vtk_obj.GetKeyCode() in ['n', 'N']:
            neighb_i = (neighb_i - 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['m', 'M']:
            neighb_i = (neighb_i + 1) % all_points[l_i][b_i].shape[0]
            update_scene()

        elif vtk_obj.GetKeyCode() in ['g', 'G']:
            if l_i < len(all_points) - 1:
                show_pools = not show_pools
                neighb_i = 0
            update_scene()

        return

    # Draw a first plot
    update_scene()
    fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    mlab.show()
























