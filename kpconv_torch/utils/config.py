from enum import Enum
from os import makedirs
from os.path import exists, join
from pathlib import Path

import numpy as np
import time

# Colors for printing
class BColors(Enum):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # CLI parameters
    ##################

    command = "" # type of CLI command
    dataset = "" # pointed by the -s/--dataset option
    infered_file = None # pointed by the -f/--filename option
    data_folder = None # pointed by the -d/--datapath option
    chosen_log_folder = None # pointed by the -l/--chosen-log option
    output_folder = None # pointed by the -o/--output option

    ##################
    # Input parameters
    ##################

    # Do we need to save convergence
    saving = True

    # Type of network model
    dataset_task = ""

    # Number of classes in the dataset
    num_classes = 0

    # Dimension of input points
    in_points_dim = 3

    # Dimension of input features
    in_features_dim = 1

    # Radius of the input sphere (ignored for models, only used for point clouds)
    in_radius = 1.0

    # Number of CPU threads for the input pipeline
    input_threads = 8

    ##################
    # Model parameters
    ##################

    # Architecture definition. List of blocks
    architecture = []

    # Decide the mode of equivariance and invariance
    equivar_mode = ""
    invar_mode = ""

    # Dimension of the first feature maps
    first_features_dim = 64

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.99

    # For segmentation models : ratio between the segmented area and the input area
    segmentation_ratio = 1.0

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.02

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Kernel point influence radius
    KP_extent = 1.0

    # Influence function when d < KP_extent. ('constant', 'linear', 'gaussian') When d > KP_extent, always zero
    KP_influence = "linear"

    # Aggregation function of KPConv in ('closest', 'sum')
    # Decide if you sum all kernel point influences, or if you only take the influence of the closest KP
    aggregation_mode = "sum"

    # Fixed points in the kernel : 'none', 'center' or 'verticals'
    fixed_kernel_points = "center"

    # Use modulateion in deformable convolutions
    modulated = False

    # For SLAM datasets like SemanticKitti number of frames used (minimum one)
    n_frames = 1

    # For SLAM datasets like SemanticKitti max number of point in input cloud + validation
    max_in_points = 0
    val_radius = 51.0
    max_val_points = 50000

    #####################
    # Training parameters
    #####################

    # Network optimizer parameters (learning rate and momentum)
    learning_rate = 1e-3
    momentum = 0.9

    # Learning rate decays. Dictionary of all decay values with their epoch {epoch: decay}.
    lr_decays = {200: 0.2, 300: 0.2}

    # Gradient clipping value (negative means no clipping)
    grad_clip_norm = 100.0

    # Augmentation parameters
    augment_scale_anisotropic = True
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_symmetries = [False, False, False]
    augment_rotation = "vertical"
    augment_noise = 0.005
    augment_color = 0.7

    # Augment with occlusions (not implemented yet)
    augment_occlusion = "none"
    augment_occlusion_ratio = 0.2
    augment_occlusion_num = 1

    # Regularization loss importance
    weight_decay = 1e-3

    # The way we balance segmentation loss DEPRECATED
    segloss_balance = "none"

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    class_w = []

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = "point2point"
    deform_fitting_power = 1.0  # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.0  # Distance of repulsion for deformed kernel points

    # Number of batch
    batch_num = 10
    val_batch_num = 10

    # Maximal number of epochs
    max_epoch = 1000

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 100

    # Number of epoch between each checkpoint
    checkpoint_gap = 50
    
    def __init__(self):
        """
        Class Initialyser
        """

        # Number of layers
        self.num_layers = (
            len(
                [
                    block
                    for block in self.architecture
                    if "pool" in block or "strided" in block
                ]
            )
            + 1
        )

        ###################
        # Deform layer list
        ###################
        #
        # List of boolean indicating which layer has a deformable convolution
        #

        layer_blocks = []
        self.deform_layers = []
        for block in self.architecture:

            # Get all blocks of the layer
            if not (
                "pool" in block
                or "strided" in block
                or "global" in block
                or "upsample" in block
            ):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks and np.any(["deformable" in blck for blck in layer_blocks]):
                deform_layer = True

            if ("pool" in block or "strided" in block) and "deformable" in block:
                deform_layer = True

            self.deform_layers += [deform_layer]
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if "global" in block or "upsample" in block:
                break

    def set_chosen_log_folder(self, chosen_log_folder):
        self.chosen_log_folder = chosen_log_folder
    
    def set_output_folder(self, output_folder):
        self.output_folder = output_folder

    def load(self):

        filename = join(self.get_train_save_path(), "parameters.txt")
        with open(filename) as f:
            lines = f.readlines()
        
        # Class variable dictionary
        for line in lines:
            line_info = line.split()
            if len(line_info) > 2 and line_info[0] != "#":

                if line_info[2] == "None":
                    setattr(self, line_info[0], None)

                elif line_info[0] == "lr_decay_epochs":
                    self.lr_decays = {
                        int(b.split(":")[0]): float(b.split(":")[1])
                        for b in line_info[2:]
                    }

                elif line_info[0] == "architecture":
                    self.architecture = line_info[2:]

                elif line_info[0] == "augment_symmetries":
                    self.augment_symmetries = [bool(int(b)) for b in line_info[2:]]

                elif line_info[0] == "num_classes":
                    if len(line_info) > 3:
                        self.num_classes = [int(c) for c in line_info[2:]]
                    else:
                        self.num_classes = int(line_info[2])

                elif line_info[0] == "class_w":
                    self.class_w = [float(w) for w in line_info[2:]]

                elif hasattr(self, line_info[0]):
                    attr_type = type(getattr(self, line_info[0]))
                    if attr_type == bool:
                        setattr(self, line_info[0], attr_type(int(line_info[2])))
                    else:
                        setattr(self, line_info[0], attr_type(line_info[2]))

        self.saving = True
        self.save_path = self.get_train_save_path()
        self.__init__()
    
    def save_parameters(self):

        with open(join(self.get_train_save_path(), "parameters.txt"), "w") as text_file:
            text_file.write("# -----------------------------------#\n")
            text_file.write("# Parameters of the training session #\n")
            text_file.write("# -----------------------------------#\n\n")

            # Input parameters
            text_file.write("# Input parameters\n")
            text_file.write("# ****************\n\n")
            text_file.write(f"dataset = {self.dataset:s}\n")
            text_file.write(f"dataset_task = {self.dataset_task:s}\n")
            if isinstance(self.num_classes, list):
                text_file.write("num_classes =")
                for n in self.num_classes:
                    text_file.write(f" {n:d}")
                text_file.write("\n")
            else:
                text_file.write(f"num_classes = {self.num_classes:d}\n")
            text_file.write(f"in_points_dim = {self.in_points_dim:d}\n")
            text_file.write(f"in_features_dim = {self.in_features_dim:d}\n")
            text_file.write(f"in_radius = {self.in_radius:.6f}\n")
            text_file.write(f"input_threads = {self.input_threads:d}\n\n")

            # Model parameters
            text_file.write("# Model parameters\n")
            text_file.write("# ****************\n\n")

            text_file.write("architecture =")
            for a in self.architecture:
                text_file.write(f" {a:s}")
            text_file.write("\n")
            text_file.write(f"equivar_mode = {self.equivar_mode:s}\n")
            text_file.write(f"invar_mode = {self.invar_mode:s}\n")
            text_file.write(f"num_layers = {self.num_layers:d}\n")
            text_file.write(f"first_features_dim = {self.first_features_dim:d}\n")
            text_file.write(f"use_batch_norm = {int(self.use_batch_norm):d}\n")
            text_file.write(f"batch_norm_momentum = {self.batch_norm_momentum:.6f}\n\n")
            text_file.write(f"segmentation_ratio = {self.segmentation_ratio:.6f}\n\n")

            # KPConv parameters
            text_file.write("# KPConv parameters\n")
            text_file.write("# *****************\n\n")

            text_file.write(f"first_subsampling_dl = {self.first_subsampling_dl:6f}\n")
            text_file.write(f"num_kernel_points = {self.num_kernel_points:d}\n")
            text_file.write(f"conv_radius = {self.conv_radius:6f}\n")
            text_file.write(f"deform_radius = {self.deform_radius:.6f}\n")
            text_file.write(f"fixed_kernel_points = {self.fixed_kernel_points}\n")
            text_file.write(f"KP_extent = {self.KP_extent:.6f}\n")
            text_file.write(f"KP_influence = {self.KP_influence}\n")
            text_file.write(f"aggregation_mode = {self.aggregation_mode}\n")
            text_file.write(f"modulated = {int(self.modulated):d}\n")
            text_file.write(f"n_frames = {self.n_frames:d}\n")
            text_file.write(f"max_in_points = {self.max_in_points:d}\n\n")
            text_file.write(f"max_val_points = {self.max_val_points:d}\n\n")
            text_file.write(f"val_radius = {self.val_radius:6f}\n\n")

            # Training parameters
            text_file.write("# Training parameters\n")
            text_file.write("# *******************\n\n")

            text_file.write(f"learning_rate = {self.learning_rate:f}\n")
            text_file.write(f"momentum = {self.momentum:f}\n")
            text_file.write("lr_decay_epochs =")
            for e, d in self.lr_decays.items():
                text_file.write(f" {e:d}:{d:f}")
            text_file.write("\n")
            text_file.write(f"grad_clip_norm = {self.grad_clip_norm:f}\n\n")

            text_file.write("augment_symmetries =")
            for a in self.augment_symmetries:
                text_file.write(f" {int(a):d}")
            text_file.write("\n")
            text_file.write(f"augment_rotation = {self.augment_rotation}\n")
            text_file.write(f"augment_noise = {self.augment_noise:f}\n")
            text_file.write(f"augment_occlusion = {self.augment_occlusion}\n")
            text_file.write(
                "augment_occlusion_ratio = {:.6f}\n".format(
                    self.augment_occlusion_ratio
                )
            )
            text_file.write(f"augment_occlusion_num = {self.augment_occlusion_num:d}\n")
            text_file.write(
                "augment_scale_anisotropic = {:d}\n".format(
                    int(self.augment_scale_anisotropic)
                )
            )
            text_file.write(f"augment_scale_min = {self.augment_scale_min:.6f}\n")
            text_file.write(f"augment_scale_max = {self.augment_scale_max:.6f}\n")
            text_file.write(f"augment_color = {self.augment_color:.6f}\n\n")

            text_file.write(f"weight_decay = {self.weight_decay:f}\n")
            text_file.write(f"segloss_balance = {self.segloss_balance}\n")
            text_file.write("class_w =")
            for a in self.class_w:
                text_file.write(f" {a:.6f}")
            text_file.write("\n")
            text_file.write(f"deform_fitting_mode = {self.deform_fitting_mode}\n")
            text_file.write(f"deform_fitting_power = {self.deform_fitting_power:.6f}\n")
            text_file.write(f"deform_lr_factor = {self.deform_lr_factor:.6f}\n")
            text_file.write(f"repulse_extent = {self.repulse_extent:.6f}\n")
            text_file.write(f"batch_num = {self.batch_num:d}\n")
            text_file.write(f"val_batch_num = {self.val_batch_num:d}\n")
            text_file.write(f"max_epoch = {self.max_epoch:d}\n")
            if self.epoch_steps is None:
                text_file.write("epoch_steps = None\n")
            else:
                text_file.write(f"epoch_steps = {self.epoch_steps:d}\n")
            text_file.write(f"validation_size = {self.validation_size:d}\n")
            text_file.write(f"checkpoint_gap = {self.checkpoint_gap:d}\n")

    def set_infered_file_path(self, path):
        self.infered_file = path
    
    def get_infered_file_path(self):
        return self.infered_file

    def get_test_save_path(self):
        if self.infered_file is not None:
            test_path = join(Path(self.infered_file).parent, "test", str(self.chosen_log_folder).split("/")[-1])
        else:
            test_path = join(self.chosen_log_folder, "test")
        if not exists(test_path):
            makedirs(test_path)
        return test_path
    
    def get_train_save_path(self):
        if self.chosen_log_folder is not None:
            train_path = self.chosen_log_folder
        elif self.output_folder is not None:
            train_path = join(self.output_folder, time.strftime("Log_%Y-%m-%d_%H-%M-%S", time.gmtime()))
        if not exists(train_path):
            makedirs(train_path)
        return train_path


