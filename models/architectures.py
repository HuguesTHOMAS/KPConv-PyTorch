#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


from models.blocks import *
import numpy as np

class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))


            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.offset_loss = config.offsets_loss
        self.offset_decay = config.offsets_decay
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # TODO: Ignore unclassified points in loss for segmentation architecture

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        self.reg_loss = self.offset_regularizer()

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total

    def offset_regularizer(self):

        fitting_loss = 0
        repulsive_loss = 0

        for m in self.modules():

            if isinstance(m, KPConv) and m.deformable:

                ##############################
                # divide offset gradient by 10
                ##############################

                m.unscaled_offsets.register_hook(lambda grad: grad * 0.1)
                #m.unscaled_offsets.register_hook(lambda grad: print('GRAD2', grad[10, 5, :]))

                ##############
                # Fitting loss
                ##############

                # Get the distance to closest input point
                KP_min_d2, _ = torch.min(m.deformed_d2, dim=1)

                # Normalize KP locations to be independant from layers
                KP_min_d2 = KP_min_d2 / (m.KP_extent ** 2)

                # Loss will be the square distance to closest input point. We use L1 because dist is already squared
                fitting_loss += self.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

                ################
                # Repulsive loss
                ################

                # Normalized KP locations
                KP_locs = m.deformed_KP / m.KP_extent

                # Point should not be close to each other
                for i in range(self.K):

                    other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                    distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                    rep_loss = torch.sum(torch.clamp_max(distances - 1.5, max=0.0) ** 2, dim=1)
                    repulsive_loss += self.l1(rep_loss, torch.zeros_like(rep_loss))



        return self.offset_decay * (fitting_loss + repulsive_loss)





class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocs = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocs.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocs = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocs.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, config.num_classes, False, 0)

        ################
        # Network Losses
        ################

        # Choose segmentation loss
        if config.segloss_balance == 'none':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif config.segloss_balance == 'class':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif config.segloss_balance == 'batch':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('Unknown segloss_balance:', config.segloss_balance)
        self.offset_loss = config.offsets_loss
        self.offset_decay = config.offsets_decay
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocs):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocs):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        labels = labels.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        self.reg_loss = self.offset_regularizer()

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total

    def offset_regularizer(self):

        fitting_loss = 0
        repulsive_loss = 0

        for m in self.modules():

            if isinstance(m, KPConv) and m.deformable:

                ##############################
                # divide offset gradient by 10
                ##############################

                m.unscaled_offsets.register_hook(lambda grad: grad * 0.1)
                #m.unscaled_offsets.register_hook(lambda grad: print('GRAD2', grad[10, 5, :]))

                ##############
                # Fitting loss
                ##############

                # Get the distance to closest input point
                KP_min_d2, _ = torch.min(m.deformed_d2, dim=1)

                # Normalize KP locations to be independant from layers
                KP_min_d2 = KP_min_d2 / (m.KP_extent ** 2)

                # Loss will be the square distance to closest input point. We use L1 because dist is already squared
                fitting_loss += self.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

                ################
                # Repulsive loss
                ################

                # Normalized KP locations
                KP_locs = m.deformed_KP / m.KP_extent

                # Point should not be close to each other
                for i in range(self.K):

                    other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                    distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                    rep_loss = torch.sum(torch.clamp_max(distances - 1.5, max=0.0) ** 2, dim=1)
                    repulsive_loss += self.l1(rep_loss, torch.zeros_like(rep_loss))



        return self.offset_decay * (fitting_loss + repulsive_loss)

























