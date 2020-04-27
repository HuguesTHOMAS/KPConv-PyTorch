#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the visualization
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
from mayavi import mlab
import sys

from models.blocks import KPConv

# PLY reader
from utils.ply import write_ply, read_ply

# Configuration class
from utils.config import Config, bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelVisualizer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)

        new_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'blocs' in k:
                k = k.replace('blocs', 'blocks')
            new_dict[k] = v

        net.load_state_dict(new_dict)
        self.epoch = checkpoint['epoch']
        net.eval()
        print("\nModel state restored from {:s}.".format(chkp_path))

        return

    # Main visualization methods
    # ------------------------------------------------------------------------------------------------------------------

    def show_deformable_kernels(self, net, loader, config, deform_idx=0):
        """
        Show some inference with deformable kernels
        """

        ##########################################
        # First choose the visualized deformations
        ##########################################

        print('\nList of the deformable convolution available (chosen one highlighted in green)')
        fmt_str = '  {:}{:2d} > KPConv(r={:.3f}, Din={:d}, Dout={:d}){:}'
        deform_convs = []
        for m in net.modules():
            if isinstance(m, KPConv) and m.deformable:
                if len(deform_convs) == deform_idx:
                    color = bcolors.OKGREEN
                else:
                    color = bcolors.FAIL
                print(fmt_str.format(color, len(deform_convs), m.radius, m.in_channels, m.out_channels, bcolors.ENDC))
                deform_convs.append(m)

        ################
        # Initialization
        ################

        print('\n****************************************************\n')

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        count = 0

        # Start training loop
        for epoch in range(config.max_epoch):

            for batch in loader:

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)
                original_KP = deform_convs[deform_idx].kernel_points.cpu().detach().numpy()
                stacked_deformed_KP = deform_convs[deform_idx].deformed_KP.cpu().detach().numpy()
                count += batch.lengths[0].shape[0]

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                # Find layer
                l = None
                for i, p in enumerate(batch.points):
                    if p.shape[0] == stacked_deformed_KP.shape[0]:
                        l = i

                t += [time.time()]

                # Get data
                in_points = []
                in_colors = []
                deformed_KP = []
                points = []
                lookuptrees = []
                i0 = 0
                for b_i, length in enumerate(batch.lengths[0]):
                    in_points.append(batch.points[0][i0:i0 + length].cpu().detach().numpy())
                    if batch.features.shape[1] == 4:
                        in_colors.append(batch.features[i0:i0 + length, 1:].cpu().detach().numpy())
                    else:
                        in_colors.append(None)
                    i0 += length

                i0 = 0
                for b_i, length in enumerate(batch.lengths[l]):
                    points.append(batch.points[l][i0:i0 + length].cpu().detach().numpy())
                    deformed_KP.append(stacked_deformed_KP[i0:i0 + length])
                    lookuptrees.append(KDTree(points[-1]))
                    i0 += length

                ###########################
                # Interactive visualization
                ###########################

                # Create figure for features
                fig1 = mlab.figure('Deformations', bgcolor=(1.0, 1.0, 1.0), size=(1280, 920))
                fig1.scene.parallel_projection = False

                # Indices
                global obj_i, point_i, plots, offsets, p_scale, show_in_p, aim_point
                p_scale = 0.03
                obj_i = 0
                point_i = 0
                plots = {}
                offsets = False
                show_in_p = 2
                aim_point = np.zeros((1, 3))

                def picker_callback(picker):
                    """ Picker callback: this get called when on pick events.
                    """
                    global plots, aim_point

                    if 'in_points' in plots:
                        if plots['in_points'].actor.actor._vtk_obj in [o._vtk_obj for o in picker.actors]:
                            point_rez = plots['in_points'].glyph.glyph_source.glyph_source.output.points.to_array().shape[0]
                            new_point_i = int(np.floor(picker.point_id / point_rez))
                            if new_point_i < len(plots['in_points'].mlab_source.points):
                                # Get closest point in the layer we are interested in
                                aim_point = plots['in_points'].mlab_source.points[new_point_i:new_point_i + 1]
                                update_scene()

                    if 'points' in plots:
                        if plots['points'].actor.actor._vtk_obj in [o._vtk_obj for o in picker.actors]:
                            point_rez = plots['points'].glyph.glyph_source.glyph_source.output.points.to_array().shape[0]
                            new_point_i = int(np.floor(picker.point_id / point_rez))
                            if new_point_i < len(plots['points'].mlab_source.points):
                                # Get closest point in the layer we are interested in
                                aim_point = plots['points'].mlab_source.points[new_point_i:new_point_i + 1]
                                update_scene()

                def update_scene():
                    global plots, offsets, p_scale, show_in_p, aim_point, point_i

                    # Get the current view
                    v = mlab.view()
                    roll = mlab.roll()

                    #  clear figure
                    for key in plots.keys():
                        plots[key].remove()

                    plots = {}

                    # Plot new data feature
                    p = points[obj_i]

                    # Rescale points for visu
                    p = (p * 1.5 / config.in_radius)


                    # Show point cloud
                    if show_in_p <= 1:
                        plots['points'] = mlab.points3d(p[:, 0],
                                                        p[:, 1],
                                                        p[:, 2],
                                                        resolution=8,
                                                        scale_factor=p_scale,
                                                        scale_mode='none',
                                                        color=(0, 1, 1),
                                                        figure=fig1)

                    if show_in_p >= 1:

                        # Get points and colors
                        in_p = in_points[obj_i]
                        in_p = (in_p * 1.5 / config.in_radius)

                        # Color point cloud if possible
                        in_c = in_colors[obj_i]
                        if in_c is not None:

                            # Primitives
                            scalars = np.arange(len(in_p))  # Key point: set an integer for each point

                            # Define color table (including alpha), which must be uint8 and [0,255]
                            colors = np.hstack((in_c, np.ones_like(in_c[:, :1])))
                            colors = (colors * 255).astype(np.uint8)

                            plots['in_points'] = mlab.points3d(in_p[:, 0],
                                                               in_p[:, 1],
                                                               in_p[:, 2],
                                                               scalars,
                                                               resolution=8,
                                                               scale_factor=p_scale*0.8,
                                                               scale_mode='none',
                                                               figure=fig1)
                            plots['in_points'].module_manager.scalar_lut_manager.lut.table = colors

                        else:

                            plots['in_points'] = mlab.points3d(in_p[:, 0],
                                                               in_p[:, 1],
                                                               in_p[:, 2],
                                                               resolution=8,
                                                               scale_factor=p_scale*0.8,
                                                               scale_mode='none',
                                                               figure=fig1)


                    # Get KP locations
                    rescaled_aim_point = aim_point * config.in_radius / 1.5
                    point_i = lookuptrees[obj_i].query(rescaled_aim_point, return_distance=False)[0][0]
                    if offsets:
                        KP = points[obj_i][point_i] + deformed_KP[obj_i][point_i]
                        scals = np.ones_like(KP[:, 0])
                    else:
                        KP = points[obj_i][point_i] + original_KP
                        scals = np.zeros_like(KP[:, 0])

                    KP = (KP * 1.5 / config.in_radius)

                    plots['KP'] = mlab.points3d(KP[:, 0],
                                                KP[:, 1],
                                                KP[:, 2],
                                                scals,
                                                colormap='autumn',
                                                resolution=8,
                                                scale_factor=1.2*p_scale,
                                                scale_mode='none',
                                                vmin=0,
                                                vmax=1,
                                                figure=fig1)


                    if True:
                        plots['center'] = mlab.points3d(p[point_i, 0],
                                                        p[point_i, 1],
                                                        p[point_i, 2],
                                                        scale_factor=1.1*p_scale,
                                                        scale_mode='none',
                                                        color=(0, 1, 0),
                                                        figure=fig1)

                        # New title
                        plots['title'] = mlab.title(str(obj_i), color=(0, 0, 0), size=0.3, height=0.01)
                        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
                        plots['text'] = mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
                        plots['orient'] = mlab.orientation_axes()

                    # Set the saved view
                    mlab.view(*v)
                    mlab.roll(roll)

                    return

                def animate_kernel():
                    global plots, offsets, p_scale, show_in_p

                    # Get KP locations

                    KP_def = points[obj_i][point_i] + deformed_KP[obj_i][point_i]
                    KP_def = (KP_def * 1.5 / config.in_radius)
                    KP_def_color = (1, 0, 0)

                    KP_rigid = points[obj_i][point_i] + original_KP
                    KP_rigid = (KP_rigid * 1.5 / config.in_radius)
                    KP_rigid_color = (1, 0.7, 0)

                    if offsets:
                        t_list = np.linspace(0, 1, 150, dtype=np.float32)
                    else:
                        t_list = np.linspace(1, 0, 150, dtype=np.float32)

                    @mlab.animate(delay=10)
                    def anim():
                        for t in t_list:
                            plots['KP'].mlab_source.set(x=t * KP_def[:, 0] + (1 - t) * KP_rigid[:, 0],
                                                        y=t * KP_def[:, 1] + (1 - t) * KP_rigid[:, 1],
                                                        z=t * KP_def[:, 2] + (1 - t) * KP_rigid[:, 2],
                                                        scalars=t * np.ones_like(KP_def[:, 0]))

                            yield

                    anim()

                    return

                def keyboard_callback(vtk_obj, event):
                    global obj_i, point_i, offsets, p_scale, show_in_p

                    if vtk_obj.GetKeyCode() in ['b', 'B']:
                        p_scale /= 1.5
                        update_scene()

                    elif vtk_obj.GetKeyCode() in ['n', 'N']:
                        p_scale *= 1.5
                        update_scene()

                    if vtk_obj.GetKeyCode() in ['g', 'G']:
                        obj_i = (obj_i - 1) % len(deformed_KP)
                        point_i = 0
                        update_scene()

                    elif vtk_obj.GetKeyCode() in ['h', 'H']:
                        obj_i = (obj_i + 1) % len(deformed_KP)
                        point_i = 0
                        update_scene()

                    elif vtk_obj.GetKeyCode() in ['k', 'K']:
                        offsets = not offsets
                        animate_kernel()

                    elif vtk_obj.GetKeyCode() in ['z', 'Z']:
                        show_in_p = (show_in_p + 1) % 3
                        update_scene()

                    elif vtk_obj.GetKeyCode() in ['0']:

                        print('Saving')

                        # Find a new name
                        file_i = 0
                        file_name = 'KP_{:03d}.ply'.format(file_i)
                        files = [f for f in listdir('KP_clouds') if f.endswith('.ply')]
                        while file_name in files:
                            file_i += 1
                            file_name = 'KP_{:03d}.ply'.format(file_i)

                        KP_deform = points[obj_i][point_i] + deformed_KP[obj_i][point_i]
                        KP_normal = points[obj_i][point_i] + original_KP

                        # Save
                        write_ply(join('KP_clouds', file_name),
                                  [in_points[obj_i], in_colors[obj_i]],
                                  ['x', 'y', 'z', 'red', 'green', 'blue'])
                        write_ply(join('KP_clouds', 'KP_{:03d}_deform.ply'.format(file_i)),
                                  [KP_deform],
                                  ['x', 'y', 'z'])
                        write_ply(join('KP_clouds', 'KP_{:03d}_normal.ply'.format(file_i)),
                                  [KP_normal],
                                  ['x', 'y', 'z'])
                        print('OK')

                    return

                # Draw a first plot
                pick_func = fig1.on_mouse_pick(picker_callback)
                pick_func.tolerance = 0.01
                update_scene()
                fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
                mlab.show()

        return

    # Utilities
    # ------------------------------------------------------------------------------------------------------------------


def show_ModelNet_models(all_points):

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
























