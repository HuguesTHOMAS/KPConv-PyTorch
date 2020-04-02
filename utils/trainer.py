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
from utils.metrics import IoU_from_confusions
from utils.config import Config
from sklearn.metrics import confusion_matrix
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
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch))
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
        C1 = confusion_matrix(targets,
                              np.argmax(probs, axis=1),
                              validation_labels)

        # Compute votes confusion
        C2 = confusion_matrix(val_loader.dataset.input_labels,
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

    def cloud_segmentation_validation(self, net, val_loader, config):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

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
            Confs[i, :, :] = confusion_matrix(truth, preds, val_loader.dataset.label_values)

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

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
            files = val_loader.dataset.train_files
            i_val = 0
            for i, file_path in enumerate(files):
                if val_loader.dataset.all_splits[i] == val_loader.dataset.validation_split:
                    pot_points = np.array(val_loader.dataset.pot_trees[i_val].data, copy=False)
                    cloud_name = file_path.split('/')[-1]
                    pot_name = join(pot_path, cloud_name)
                    pots = val_loader.dataset.potentials[i_val].numpy().astype(np.float32)
                    write_ply(pot_name,
                              [pot_points.astype(np.float32), pots],
                              ['x', 'y', 'z', 'pots'])

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))

        # Save predicted cloud occasionally
        if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
            val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch))
            if not exists(val_path):
                makedirs(val_path)
            files = val_loader.dataset.train_files
            i_val = 0
            for i, file_path in enumerate(files):
                if val_loader.dataset.all_splits[i] == val_loader.dataset.validation_split:

                    # Get points
                    points = val_loader.dataset.load_evaluation_points(file_path)

                    # Get probs on our own ply points
                    sub_probs = self.validation_probs[i_val]

                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                        if label_value in val_loader.dataset.ignored_labels:
                            sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                    # Get the predicted labels
                    sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                    # Reproject preds on the evaluations points
                    preds = (sub_preds[val_loader.dataset.validation_proj[i_val]]).astype(np.int32)

                    # Path of saved validation file
                    cloud_name = file_path.split('/')[-1]
                    val_name = join(val_path, cloud_name)

                    # Save file
                    labels = val_loader.dataset.validation_labels[i_val].astype(np.int32)
                    write_ply(val_name,
                              [points, preds, labels],
                              ['x', 'y', 'z', 'preds', 'class'])

                    i_val += 1

        return





    def validation_error(self, model, dataset):
        """
        Validation method for classification models
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initialize global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = np.zeros((len(dataset.input_labels['validation']), nc_model))

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []

        mean_dt = np.zeros(2)
        last_display = time.time()
        while True:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits, model.labels, model.inputs['object_inds'])
                prob, labels, inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get probs and labels
                probs += [prob]
                targets += [labels]
                obj_inds += [inds]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * len(obj_inds) / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

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

        validation_labels = np.array(dataset.label_values)

        # Compute classification results
        C1 = confusion_matrix(targets,
                              np.argmax(probs, axis=1),
                              validation_labels)

        # Compute training confusion
        C2 = confusion_matrix(self.training_labels,
                              self.training_preds,
                              validation_labels)

        # Compute votes confusion
        C3 = confusion_matrix(dataset.input_labels['validation'],
                              np.argmax(self.val_probs, axis=1),
                              validation_labels)


        # Saving (optionnal)
        if model.config.saving:
            print("Save confusions")
            conf_list = [C1, C2, C3]
            file_list = ['val_confs.txt', 'training_confs.txt', 'vote_confs.txt']
            for conf, conf_file in zip(conf_list, file_list):
                test_file = join(model.saving_path, conf_file)
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

        train_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
        val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        vote_ACC = 100 * np.sum(np.diag(C3)) / (np.sum(C3) + 1e-6)
        print('Accuracies : train = {:.1f}% / val = {:.1f}% / vote = {:.1f}%'.format(train_ACC, val_ACC, vote_ACC))

        return C1

    def segment_validation_error(self, model, dataset):
        """
        Validation method for single object segmentation models
        """

        ##########
        # Initialize
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initialize global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = [np.zeros((len(p_l), nc_model)) for p_l in dataset.input_point_labels['validation']]

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits, model.labels, model.inputs['in_batches'], model.inputs['object_inds'])
                prob, labels, batches, o_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Stack all results
                    probs += [prob[b]]
                    targets += [labels[b]]
                    obj_inds += [o_inds[b_i]]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        ###################
        # Voting validation
        ###################

        for o_i, o_probs in zip(obj_inds, probs):
            self.val_probs[o_i] = val_smooth * self.val_probs[o_i] + (1 - val_smooth) * o_probs

        ############
        # Confusions
        ############

        # Confusion matrix for each instance
        n_parts = model.config.num_classes
        Confs = np.zeros((len(probs), n_parts, n_parts), dtype=np.int32)
        for i, (pred, truth) in enumerate(zip(probs, targets)):
            parts = [j for j in range(pred.shape[1])]
            Confs[i, :, :] = confusion_matrix(truth, np.argmax(pred, axis=1), parts)

        # Objects IoU
        IoUs = IoU_from_confusions(Confs)


        # Compute votes confusion
        Confs = np.zeros((len(self.val_probs), n_parts, n_parts), dtype=np.int32)
        for i, (pred, truth) in enumerate(zip(self.val_probs, dataset.input_point_labels['validation'])):
            parts = [j for j in range(pred.shape[1])]
            Confs[i, :, :] = confusion_matrix(truth, np.argmax(pred, axis=1), parts)

        # Objects IoU
        vote_IoUs = IoU_from_confusions(Confs)

        # Saving (optionnal)
        if model.config.saving:

            IoU_list = [IoUs, vote_IoUs]
            file_list = ['val_IoUs.txt', 'vote_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(model.saving_path, IoU_file)

                # Line to write:
                line = ''
                for instance_IoUs in IoUs_to_save:
                    for IoU in instance_IoUs:
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
        mIoU2 = 100 * np.mean(vote_IoUs)
        print('{:s} : mIoU = {:.1f}% / vote mIoU = {:.1f}%'.format(model.config.dataset, mIoU, mIoU2))

        return

    def cloud_validation_error(self, model, dataset):
        """
        Validation method for cloud segmentation models
        """

        ##########
        # Initialize
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Do not validate if dataset has no validation cloud
        if dataset.validation_split not in dataset.all_splits:
            return

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Number of classes including ignored labels
        nc_tot = dataset.num_classes

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initialize global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model)) for l in dataset.input_labels['validation']]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in dataset.label_values:
                if label_value not in dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in dataset.validation_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, labels, batches, point_inds, cloud_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                                + (1-val_smooth) * probs

                    # Stack all prediction for this epoch
                    predictions += [probs]
                    targets += [dataset.input_labels['validation'][c_i][inds]]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(dataset.label_values):
                if label_value in dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = confusion_matrix(truth, preds, dataset.label_values)

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
            if label_value in dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        # Saving (optionnal)
        if model.config.saving:

            # Name of saving file
            test_file = join(model.saving_path, 'val_IoUs.txt')

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

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(model.config.dataset, mIoU))

        # Save predicted cloud occasionally
        if model.config.saving and (self.training_epoch + 1) % model.config.checkpoint_gap == 0:
            val_path = join(model.saving_path, 'val_preds_{:d}'.format(self.training_epoch))
            if not exists(val_path):
                makedirs(val_path)
            files = dataset.train_files
            i_val = 0
            for i, file_path in enumerate(files):
                if dataset.all_splits[i] == dataset.validation_split:

                    # Get points
                    points = dataset.load_evaluation_points(file_path)

                    # Get probs on our own ply points
                    sub_probs = self.validation_probs[i_val]

                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(dataset.label_values):
                        if label_value in dataset.ignored_labels:
                            sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                    # Get the predicted labels
                    sub_preds = dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                    # Reproject preds on the evaluations points
                    preds = (sub_preds[dataset.validation_proj[i_val]]).astype(np.int32)

                    # Path of saved validation file
                    cloud_name = file_path.split('/')[-1]
                    val_name = join(val_path, cloud_name)

                    # Save file
                    labels = dataset.validation_labels[i_val].astype(np.int32)
                    write_ply(val_name,
                              [points, preds, labels],
                              ['x', 'y', 'z', 'preds', 'class'])

                    i_val += 1

        return

    def multi_cloud_validation_error(self, model, multi_dataset):
        """
        Validation method for cloud segmentation models
        """

        ##########
        # Initialize
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(multi_dataset.val_init_op)

        if not hasattr(self, 'validation_probs'):

            self.validation_probs = []
            self.val_proportions = []

            for d_i, dataset in enumerate(multi_dataset.datasets):

                # Do not validate if dataset has no validation cloud
                if dataset.validation_split not in dataset.all_splits:
                    continue

                # Number of classes including ignored labels
                nc_tot = dataset.num_classes

                # Number of classes predicted by the model
                nc_model = model.config.num_classes[d_i]

                # Initialize global prediction over validation clouds
                self.validation_probs.append([np.zeros((l.shape[0], nc_model)) for l in dataset.input_labels['validation']])
                self.val_proportions.append(np.zeros(nc_model, dtype=np.float32))
                i = 0
                for label_value in dataset.label_values:
                    if label_value not in dataset.ignored_labels:
                        self.val_proportions[-1][i] = np.sum([np.sum(labels == label_value)
                                                          for labels in dataset.validation_labels])
                        i += 1

        #####################
        # Network predictions
        #####################

        pred_d_inds = []
        predictions = []
        targets = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.val_logits,
                       model.labels,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'],
                       model.inputs['dataset_inds'])
                stacked_probs, labels, batches, p_inds, c_inds, d_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get prediction (only for the concerned parts)
                    d_i = d_inds[b_i]
                    probs = stacked_probs[b, :model.config.num_classes[d_i]]
                    inds = p_inds[b]
                    c_i = c_inds[b_i]

                    # Update current probs in whole cloud
                    self.validation_probs[d_i][c_i][inds] = val_smooth * self.validation_probs[d_i][c_i][inds] \
                                                                + (1-val_smooth) * probs

                    # Stack all prediction for this epoch
                    pred_d_inds += [d_i]
                    predictions += [probs]
                    targets += [multi_dataset.datasets[d_i].input_labels['validation'][c_i][inds]]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        # Convert list to np array for indexing
        predictions = np.array(predictions)
        targets = np.array(targets)
        pred_d_inds = np.array(pred_d_inds, np.int32)

        IoUs = []
        for d_i, dataset in enumerate(multi_dataset.datasets):

            # Do not validate if dataset has no validation cloud
            if dataset.validation_split not in dataset.all_splits:
                continue

            # Number of classes including ignored labels
            nc_tot = dataset.num_classes

            # Number of classes predicted by the model
            nc_model = model.config.num_classes[d_i]

            # Extract the spheres from this dataset
            tmp_inds = np.where(pred_d_inds == d_i)[0]

            # Confusions for our subparts of validation set
            Confs = np.zeros((len(tmp_inds), nc_tot, nc_tot), dtype=np.int32)
            for i, (probs, truth) in enumerate(zip(predictions[tmp_inds], targets[tmp_inds])):

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(dataset.label_values):
                    if label_value in dataset.ignored_labels:
                        probs = np.insert(probs, l_ind, 0, axis=1)

                # Predicted labels
                preds = dataset.label_values[np.argmax(probs, axis=1)]

                # Confusions
                Confs[i, :, :] = confusion_matrix(truth, preds, dataset.label_values)

            # Sum all confusions
            C = np.sum(Confs, axis=0).astype(np.float32)

            # Remove ignored labels from confusions
            for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                if label_value in dataset.ignored_labels:
                    C = np.delete(C, l_ind, axis=0)
                    C = np.delete(C, l_ind, axis=1)

            # Balance with real validation proportions
            C *= np.expand_dims(self.val_proportions[d_i] / (np.sum(C, axis=1) + 1e-6), 1)

            # Objects IoU
            IoUs += [IoU_from_confusions(C)]

            # Saving (optionnal)
            if model.config.saving:

                # Name of saving file
                test_file = join(model.saving_path, 'val_IoUs_{:d}_{:s}.txt'.format(d_i, dataset.name))

                # Line to write:
                line = ''
                for IoU in IoUs[-1]:
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
            mIoU = 100 * np.mean(IoUs[-1])
            print('{:s} mean IoU = {:.1f}%'.format(dataset.name, mIoU))

        # Save predicted cloud occasionally
        if model.config.saving and (self.training_epoch + 1) % model.config.checkpoint_gap == 0:
            val_path = join(model.saving_path, 'val_preds_{:d}'.format(self.training_epoch))
            if not exists(val_path):
                makedirs(val_path)

            for d_i, dataset in enumerate(multi_dataset.datasets):

                dataset_val_path = join(val_path, '{:d}_{:s}'.format(d_i, dataset.name))
                if not exists(dataset_val_path):
                    makedirs(dataset_val_path)

                files = dataset.train_files
                i_val = 0
                for i, file_path in enumerate(files):
                    if dataset.all_splits[i] == dataset.validation_split:

                        # Get points
                        points = dataset.load_evaluation_points(file_path)

                        # Get probs on our own ply points
                        sub_probs = self.validation_probs[d_i][i_val]

                        # Insert false columns for ignored labels
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                        # Get the predicted labels
                        sub_preds = dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                        # Reproject preds on the evaluations points
                        preds = (sub_preds[dataset.validation_proj[i_val]]).astype(np.int32)

                        # Path of saved validation file
                        cloud_name = file_path.split('/')[-1]
                        val_name = join(dataset_val_path, cloud_name)

                        # Save file
                        labels = dataset.validation_labels[i_val].astype(np.int32)
                        write_ply(val_name,
                                  [points, preds, labels],
                                  ['x', 'y', 'z', 'preds', 'class'])

                        i_val += 1

        return

    def multi_validation_error(self, model, dataset):
        """
        Validation method for multi object segmentation models
        """

        ##########
        # Initialize
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Initialize global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = []
            for p_l, o_l in zip(dataset.input_point_labels['validation'], dataset.input_labels['validation']):
                self.val_probs += [np.zeros((len(p_l), dataset.num_parts[o_l]))]

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        objects = []
        obj_inds = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (model.logits,
                       model.labels,
                       model.inputs['super_labels'],
                       model.inputs['object_inds'],
                       model.inputs['in_batches'])
                prob, labels, object_labels, o_inds, batches = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get prediction (only for the concerned parts)
                    obj = object_labels[b[0]]
                    pred = prob[b][:, :model.config.num_classes[obj]]

                    # Stack all results
                    objects += [obj]
                    obj_inds += [o_inds[b_i]]
                    probs += [prob[b, :model.config.num_classes[obj]]]
                    targets += [labels[b]]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        ###################
        # Voting validation
        ###################

        for o_i, o_probs in zip(obj_inds, probs):
            self.val_probs[o_i] = val_smooth * self.val_probs[o_i] + (1 - val_smooth) * o_probs

        ############
        # Confusions
        ############

        # Confusion matrix for each object
        n_objs = [np.sum(np.array(objects) == l) for l in dataset.label_values]
        Confs = [np.zeros((n_obj, n_parts, n_parts), dtype=np.int32) for n_parts, n_obj in
                 zip(dataset.num_parts, n_objs)]
        obj_count = [0 for _ in n_objs]
        for obj, pred, truth in zip(objects, probs, targets):
            parts = [i for i in range(pred.shape[1])]
            Confs[obj][obj_count[obj], :, :] = confusion_matrix(truth, np.argmax(pred, axis=1), parts)
            obj_count[obj] += 1

        # Objects mIoU
        IoUs = [IoU_from_confusions(C) for C in Confs]


        # Compute votes confusion
        n_objs = [np.sum(np.array(dataset.input_labels['validation']) == l) for l in dataset.label_values]
        Confs = [np.zeros((n_obj, n_parts, n_parts), dtype=np.int32) for n_parts, n_obj in
                 zip(dataset.num_parts, n_objs)]
        obj_count = [0 for _ in n_objs]
        for obj, pred, truth in zip(dataset.input_labels['validation'],
                                    self.val_probs,
                                    dataset.input_point_labels['validation']):
            parts = [i for i in range(pred.shape[1])]
            Confs[obj][obj_count[obj], :, :] = confusion_matrix(truth, np.argmax(pred, axis=1), parts)
            obj_count[obj] += 1

        # Objects mIoU
        vote_IoUs = [IoU_from_confusions(C) for C in Confs]

        # Saving (optionnal)
        if model.config.saving:

            IoU_list = [IoUs, vote_IoUs]
            file_list = ['val_IoUs.txt', 'vote_IoUs.txt']

            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(model.saving_path, IoU_file)

                # Line to write:
                line = ''
                for obj_IoUs in IoUs_to_save:
                    for part_IoUs in obj_IoUs:
                        for IoU in part_IoUs:
                            line += '{:.3f} '.format(IoU)
                    line += '/ '
                line = line[:-2] + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in IoUs]))
        class_mIoUs = [np.mean(obj_IoUs) for obj_IoUs in IoUs]
        mcIoU = 100 * np.mean(class_mIoUs)
        print('Val  : mIoU = {:.1f}% / mcIoU = {:.1f}% '.format(mIoU, mcIoU))
        mIoU = 100 * np.mean(np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in vote_IoUs]))
        class_mIoUs = [np.mean(obj_IoUs) for obj_IoUs in vote_IoUs]
        mcIoU = 100 * np.mean(class_mIoUs)
        print('Vote : mIoU = {:.1f}% / mcIoU = {:.1f}% '.format(mIoU, mcIoU))

        return

    def slam_validation_error(self, model, dataset):
        """
        Validation method for slam segmentation models
        """

        ##########
        # Initialize
        ##########

        # Do not validate if dataset has no validation cloud
        if dataset.validation_split not in dataset.seq_splits:
            return

        # Create folder for validation predictions
        if not exists (join(model.saving_path, 'val_preds')):
            makedirs(join(model.saving_path, 'val_preds'))

        # Initialize the dataset validation containers
        dataset.val_points = []
        dataset.val_labels = []

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Number of classes including ignored labels
        nc_tot = dataset.num_classes

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        inds = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        val_i = 0
        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['points'][0],
                       model.inputs['in_batches'],
                       model.inputs['frame_inds'],
                       model.inputs['frame_centers'],
                       model.inputs['augment_scales'],
                       model.inputs['augment_rotations'])
                s_probs, s_labels, s_points, batches, f_inds, p0s, S, R = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get prediction (only for the concerned parts)
                    probs = s_probs[b]
                    labels = s_labels[b]
                    points = s_points[b, :]
                    S_i = S[b_i]
                    R_i = R[b_i]
                    p0 = p0s[b_i]

                    # Get input points in their original positions
                    points2 = (points * (1/S_i)).dot(R_i.T)

                    # get val_points that are in range
                    radiuses = np.sum(np.square(dataset.val_points[val_i] - p0), axis=1)
                    mask = radiuses < (0.9 * model.config.in_radius) ** 2

                    # Project predictions on the frame points
                    search_tree = KDTree(points2, leaf_size=50)
                    proj_inds = search_tree.query(dataset.val_points[val_i][mask, :], return_distance=False)
                    proj_inds = np.squeeze(proj_inds).astype(np.int32)
                    proj_probs = probs[proj_inds]
                    #proj_labels = labels[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)

                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(dataset.label_values):
                        if label_value in dataset.ignored_labels:
                            proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

                    # Predicted labels
                    preds = dataset.label_values[np.argmax(proj_probs, axis=1)]

                    # Save predictions in a binary file
                    filename ='{:02d}_{:07d}.npy'.format(f_inds[b_i, 0], f_inds[b_i, 1])
                    filepath = join(model.saving_path, 'val_preds', filename)
                    if exists(filepath):
                        frame_preds = np.load(filepath)
                    else:
                        frame_preds = np.zeros(dataset.val_labels[val_i].shape, dtype=np.uint8)
                    frame_preds[mask] = preds.astype(np.uint8)
                    np.save(filepath, frame_preds)

                    # Save some of the frame pots
                    if f_inds[b_i, 1] % 10 == 0:
                        pots = dataset.f_potentials['validation'][f_inds[b_i, 0]][f_inds[b_i, 1]]
                        write_ply(filepath[:-4]+'_pots.ply',
                                  [dataset.val_points[val_i], dataset.val_labels[val_i], frame_preds, pots],
                                  ['x', 'y', 'z', 'gt', 'pre', 'pots'])

                    # Update validation confusions
                    frame_C = confusion_matrix(dataset.val_labels[val_i], frame_preds, dataset.label_values)
                    dataset.val_confs[f_inds[b_i, 0]][f_inds[b_i, 1], :, :] = frame_C

                    # Stack all prediction for this epoch
                    predictions += [preds]
                    targets += [dataset.val_labels[val_i][mask]]
                    inds += [f_inds[b_i, :]]
                    val_i += 1

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):

            # Confusions
            Confs[i, :, :] = confusion_matrix(truth, preds, dataset.label_values)

        #######################################
        # Results on this subpart of validation
        #######################################

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        C *= np.expand_dims(dataset.class_proportions['validation'] / (np.sum(C, axis=1) + 1e-6), 1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
            if label_value in dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        #####################################
        # Results on the whole validation set
        #####################################

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)

        s = ''
        for cc in C_tot:
            for c in cc:
                s += '{:8.1f} '.format(c)
            s += '\n'
        print(s)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
            if label_value in dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        # Saving (optionnal)
        if model.config.saving:

            IoU_list = [IoUs, val_IoUs]
            file_list = ['subpart_IoUs.txt', 'val_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(model.saving_path, IoU_file)

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
        print('{:s} : subpart mIoU = {:.1f} %'.format(model.config.dataset, mIoU))
        mIoU = 100 * np.mean(val_IoUs)
        print('{:s} :     val mIoU = {:.1f} %'.format(model.config.dataset, mIoU))

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







































