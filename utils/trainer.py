#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
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
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import KPConv


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
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

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer
        self.optimizer = torch.optim.SGD(net.parameters(),
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss offset_loss train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start training loop
        for epoch in range(config.max_epoch):

            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0
            for batch in training_loader:

                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = net(batch, config)
                loss = net.loss(outputs, batch.labels)
                acc = net.accuracy(outputs, batch.labels)

                t += [time.time()]

                # Backward + optimize
                loss.backward()

                if config.grad_clip_norm > 0:
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L={:.3f} acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss.item(),
                                         100*acc,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                        file.write(message.format(self.epoch,
                                                  self.step,
                                                  net.output_loss,
                                                  net.reg_loss,
                                                  acc,
                                                  t[-1] - t0))


                self.step += 1

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            net.eval()
            self.validation(net, val_loader, config)
            net.train()

        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config):

        if config.dataset_task == 'classification':
            self.object_classification_validation(net, val_loader, config)
        elif config.dataset_task == 'segmentation':
            self.object_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'cloud_segmentation':
            self.cloud_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'slam_segmentation':
            self.slam_segmentation_validation(net, val_loader, config)
        else:
            raise ValueError('No validation method implemented for this network type')

    def object_classification_validation(self, net, val_loader, config):
        """
        Perform a round of validation and show/save results
        :param net: network object
        :param val_loader: data loader for validation set
        :param config: configuration object
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Number of classes predicted by the model
        nc_model = config.num_classes
        softmax = torch.nn.Softmax(1)

        # Initialize global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = np.zeros((val_loader.dataset.num_models, nc_model))

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in val_loader:

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            probs += [softmax(outputs).cpu().detach().numpy()]
            targets += [batch.labels.cpu().numpy()]
            obj_inds += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * len(obj_inds) / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        probs = np.vstack(probs)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)

        ###################
        # Voting validation
        ###################

        self.val_probs[obj_inds] = val_smooth * self.val_probs[obj_inds] + (1-val_smooth) * probs

        ############
        # Confusions
        ############

        validation_labels = np.array(val_loader.dataset.label_values)

        # Compute classification results
        C1 = fast_confusion(targets,
                            np.argmax(probs, axis=1),
                            validation_labels)

        # Compute votes confusion
        C2 = fast_confusion(val_loader.dataset.input_labels,
                            np.argmax(self.val_probs, axis=1),
                            validation_labels)


        # Saving (optionnal)
        if config.saving:
            print("Save confusions")
            conf_list = [C1, C2]
            file_list = ['val_confs.txt', 'vote_confs.txt']
            for conf, conf_file in zip(conf_list, file_list):
                test_file = join(config.saving_path, conf_file)
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')
                else:
                    with open(test_file, "w") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')

        val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        vote_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
        print('Accuracies : val = {:.1f}% / vote = {:.1f}%'.format(val_ACC, vote_ACC))

        return C1

    def cloud_segmentation_validation(self, net, val_loader, config, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Do not validate if dataset has no validation cloud
        if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
            return

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        #print(nc_tot)
        #print(nc_model)

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_labels]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in val_loader.dataset.validation_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            in_inds = batch.input_inds.cpu().numpy()
            cloud_inds = batch.cloud_inds.cpu().numpy()
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                target = labels[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                inds = in_inds[i0:i0 + length]
                c_i = cloud_inds[b_i]

                # Update current probs in whole cloud
                self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                   + (1 - val_smooth) * probs

                # Stack all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)


        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)


        t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            # Name of saving file
            test_file = join(config.saving_path, 'val_IoUs.txt')

            # Line to write:
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line = line + '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

            # Save potentials
            pot_path = join(config.saving_path, 'potentials')
            if not exists(pot_path):
                makedirs(pot_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):
                pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
                cloud_name = file_path.split('/')[-1]
                pot_name = join(pot_path, cloud_name)
                pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
                write_ply(pot_name,
                          [pot_points.astype(np.float32), pots],
                          ['x', 'y', 'z', 'pots'])

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))

        # Save predicted cloud occasionally
        if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
            val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch + 1))
            if not exists(val_path):
                makedirs(val_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):

                # Get points
                points = val_loader.dataset.load_evaluation_points(file_path)

                # Get probs on our own ply points
                sub_probs = self.validation_probs[i]

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                # Get the predicted labels
                sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                # Reproject preds on the evaluations points
                preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)

                # Path of saved validation file
                cloud_name = file_path.split('/')[-1]
                val_name = join(val_path, cloud_name)

                # Save file
                labels = val_loader.dataset.validation_labels[i].astype(np.int32)
                write_ply(val_name,
                          [points, preds, labels],
                          ['x', 'y', 'z', 'preds', 'class'])

        # Display timings
        t7 = time.time()
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('Confs bis . {:.1f}s'.format(t4 - t3))
            print('IoU ....... {:.1f}s'.format(t5 - t4))
            print('Save1 ..... {:.1f}s'.format(t6 - t5))
            print('Save2 ..... {:.1f}s'.format(t7 - t6))
            print('\n************************\n')

        return

    def slam_segmentation_validation(self, net, val_loader, config, debug=True):
        """
        Validation method for slam segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Do not validate if dataset has no validation cloud
        if val_loader is None:
            return

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Create folder for validation predictions
        if not exists (join(config.saving_path, 'val_preds')):
            makedirs(join(config.saving_path, 'val_preds'))

        # initiate the dataset validation containers
        val_loader.dataset.val_points = []
        val_loader.dataset.val_labels = []

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        inds = []
        val_i = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            stk_probs = softmax(outputs).cpu().detach().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            f_inds = batch.frame_inds.cpu().numpy()
            r_inds_list = batch.reproj_inds
            r_mask_list = batch.reproj_masks
            labels_list = batch.val_labels
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                probs = stk_probs[i0:i0 + length]
                proj_inds = r_inds_list[b_i]
                proj_mask = r_mask_list[b_i]
                frame_labels = labels_list[b_i]
                s_ind = f_inds[b_i, 0]
                f_ind = f_inds[b_i, 1]

                # Project predictions on the frame points
                proj_probs = probs[proj_inds]

                # Safe check if only one point:
                if proj_probs.ndim < 2:
                    proj_probs = np.expand_dims(proj_probs, 0)

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

                # Predicted labels
                preds = val_loader.dataset.label_values[np.argmax(proj_probs, axis=1)]

                # Save predictions in a binary file
                filename = '{:s}_{:07d}.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                filepath = join(config.saving_path, 'val_preds', filename)
                if exists(filepath):
                    frame_preds = np.load(filepath)
                else:
                    frame_preds = np.zeros(frame_labels.shape, dtype=np.uint8)
                frame_preds[proj_mask] = preds.astype(np.uint8)
                np.save(filepath, frame_preds)

                # Save some of the frame pots
                if f_ind % 20 == 0:
                    seq_path = join(val_loader.dataset.path, 'sequences', val_loader.dataset.sequences[s_ind])
                    velo_file = join(seq_path, 'velodyne', val_loader.dataset.frames[s_ind][f_ind] + '.bin')
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    frame_points = frame_points.reshape((-1, 4))
                    write_ply(filepath[:-4] + '_pots.ply',
                              [frame_points[:, :3], frame_labels, frame_preds],
                              ['x', 'y', 'z', 'gt', 'pre'])

                # Update validation confusions
                frame_C = fast_confusion(frame_labels,
                                         frame_preds.astype(np.int32),
                                         val_loader.dataset.label_values)
                val_loader.dataset.val_confs[s_ind][f_ind, :, :] = frame_C

                # Stack all prediction for this epoch
                predictions += [preds]
                targets += [frame_labels[proj_mask]]
                inds += [f_inds[b_i, :]]
                val_i += 1
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        t3 = time.time()

        #######################################
        # Results on this subpart of validation
        #######################################

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        C *= np.expand_dims(val_loader.dataset.class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        #####################################
        # Results on the whole validation set
        #####################################

        t4 = time.time()

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in val_loader.dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)

        if debug:
            s = '\n'
            for cc in C_tot:
                for c in cc:
                    s += '{:8.1f} '.format(c)
                s += '\n'
            print(s)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            IoU_list = [IoUs, val_IoUs]
            file_list = ['subpart_IoUs.txt', 'val_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(config.saving_path, IoU_file)

                # Line to write:
                line = ''
                for IoU in IoUs_to_save:
                    line += '{:.3f} '.format(IoU)
                line = line + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} : subpart mIoU = {:.1f} %'.format(config.dataset, mIoU))
        mIoU = 100 * np.mean(val_IoUs)
        print('{:s} :     val mIoU = {:.1f} %'.format(config.dataset, mIoU))

        t6 = time.time()

        # Display timings
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('IoU1 ...... {:.1f}s'.format(t4 - t3))
            print('IoU2 ...... {:.1f}s'.format(t5 - t4))
            print('Save ...... {:.1f}s'.format(t6 - t5))
            print('\n************************\n')

        return



    # Saving methods
    # ------------------------------------------------------------------------------------------------------------------

    def save_kernel_points(self, model, epoch):
        """
        Method saving kernel point disposition and current model weights for later visualization
        """

        if model.config.saving:

            # Create a directory to save kernels of this epoch
            kernels_dir = join(model.saving_path, 'kernel_points', 'epoch{:d}'.format(epoch))
            if not exists(kernels_dir):
                makedirs(kernels_dir)

            # Get points
            all_kernel_points_tf = [v for v in tf.global_variables() if 'kernel_points' in v.name
                                    and v.name.startswith('KernelPoint')]
            all_kernel_points = self.sess.run(all_kernel_points_tf)

            # Get Extents
            if False and 'gaussian' in model.config.convolution_mode:
                all_kernel_params_tf = [v for v in tf.global_variables() if 'kernel_extents' in v.name
                                        and v.name.startswith('KernelPoint')]
                all_kernel_params = self.sess.run(all_kernel_params_tf)
            else:
                all_kernel_params = [None for p in all_kernel_points]

            # Save in ply file
            for kernel_points, kernel_extents, v in zip(all_kernel_points, all_kernel_params, all_kernel_points_tf):

                # Name of saving file
                ply_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.ply'
                ply_file = join(kernels_dir, ply_name)

                # Data to save
                if kernel_points.ndim > 2:
                    kernel_points = kernel_points[:, 0, :]
                if False and 'gaussian' in model.config.convolution_mode:
                    data = [kernel_points, kernel_extents]
                    keys = ['x', 'y', 'z', 'sigma']
                else:
                    data = kernel_points
                    keys = ['x', 'y', 'z']

                # Save
                write_ply(ply_file, data, keys)

            # Get Weights
            all_kernel_weights_tf = [v for v in tf.global_variables() if 'weights' in v.name
                                    and v.name.startswith('KernelPointNetwork')]
            all_kernel_weights = self.sess.run(all_kernel_weights_tf)

            # Save in numpy file
            for kernel_weights, v in zip(all_kernel_weights, all_kernel_weights_tf):
                np_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.npy'
                np_file = join(kernels_dir, np_name)
                np.save(np_file, kernel_weights)

    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def show_memory_usage(self, batch_to_feed):

            for l in range(self.config.num_layers):
                neighb_size = list(batch_to_feed[self.in_neighbors_f32[l]].shape)
                dist_size = neighb_size + [self.config.num_kernel_points, 3]
                dist_memory = np.prod(dist_size) * 4 * 1e-9
                in_feature_size = neighb_size + [self.config.first_features_dim * 2**l]
                in_feature_memory = np.prod(in_feature_size) * 4 * 1e-9
                out_feature_size = [neighb_size[0], self.config.num_kernel_points, self.config.first_features_dim * 2**(l+1)]
                out_feature_memory = np.prod(out_feature_size) * 4 * 1e-9

                print('Layer {:d} => {:.1f}GB {:.1f}GB {:.1f}GB'.format(l,
                                                                   dist_memory,
                                                                   in_feature_memory,
                                                                   out_feature_memory))
            print('************************************')

    def debug_nan(self, model, inputs, logits):
        """
        NaN happened, find where
        """

        print('\n\n------------------------ NaN DEBUG ------------------------\n')

        # First save everything to reproduce error
        file1 = join(model.saving_path, 'all_debug_inputs.pkl')
        with open(file1, 'wb') as f1:
            pickle.dump(inputs, f1)

        # First save all inputs
        file1 = join(model.saving_path, 'all_debug_logits.pkl')
        with open(file1, 'wb') as f1:
            pickle.dump(logits, f1)

        # Then print a list of the trainable variables and if they have nan
        print('List of variables :')
        print('*******************\n')
        all_vars = self.sess.run(tf.global_variables())
        for v, value in zip(tf.global_variables(), all_vars):
            nan_percentage = 100 * np.sum(np.isnan(value)) / np.prod(value.shape)
            print(v.name, ' => {:.1f}% of values are NaN'.format(nan_percentage))


        print('Inputs :')
        print('********')

        #Print inputs
        nl = model.config.num_layers
        for layer in range(nl):

            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / np.prod(pools.shape)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / np.prod(upsamples.shape)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        point_labels = inputs[ind]
        print('point labels, ', point_labels.shape, ', values : ', np.unique(point_labels))
        print(np.array([int(100 * np.sum(point_labels == l) / len(point_labels)) for l in np.unique(point_labels)]))

        ind += 1
        if model.config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        #Print inputs
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            if np.prod(pools.shape) > 0:
                max_n = np.max(pools)
                nums = np.sum(pools < max_n - 0.5, axis=-1)
                print('min pools =>', np.min(nums))
            else:
                print('pools empty')


            if np.prod(upsamples.shape) > 0:
                max_n = np.max(upsamples)
                nums = np.sum(upsamples < max_n - 0.5, axis=-1)
                print('min upsamples =>', np.min(nums))
            else:
                print('upsamples empty')


        print('\nFinished\n\n')
        time.sleep(0.5)







































