#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
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
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd
from sklearn.metrics import confusion_matrix
import time

# My libs
from utils.config import Config
from utils.metrics import IoU_from_confusions, smooth_metrics, fast_confusion
from utils.ply import read_ply

# Datasets
from datasets.ModelNet40 import ModelNet40Dataset
from datasets.S3DIS import S3DISDataset
from datasets.SemanticKitti import SemanticKittiDataset

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def listdir_str(path):

    # listdir can return binary string instead od decoded string sometimes.
    # This function ensures a steady behavior

    f_list = []
    for f in listdir(path):
        try:
            f = f.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        f_list.append(f)

    return f_list



def running_mean(signal, n, axis=0, stride=1):
    signal = np.array(signal)
    torch_conv = torch.nn.Conv1d(1, 1, kernel_size=2*n+1, stride=stride, bias=False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.weight *= 0
    torch_conv.weight += 1 / (2*n+1)
    if signal.ndim == 1:
        torch_signal = torch.from_numpy(signal.reshape([1, 1, -1]).astype(np.float32))
        return torch_conv(torch_signal).squeeze().numpy()

    elif signal.ndim == 2:
        print('TODO implement with torch and stride here')
        smoothed = np.empty(signal.shape)
        if axis == 0:
            for i, sig in enumerate(signal):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[i, :] = sig_sum / sig_num
        elif axis == 1:
            for i, sig in enumerate(signal.T):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[:, i] = sig_sum / sig_num
        else:
            print('wrong axis')
        return smoothed

    else:
        print('wrong dimensions')
        return None


def IoU_class_metrics(all_IoUs, smooth_n):

    # Get mean IoU per class for consecutive epochs to directly get a mean without further smoothing
    smoothed_IoUs = []
    for epoch in range(len(all_IoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_IoUs))
        smoothed_IoUs += [np.mean(np.vstack(all_IoUs[i0:i1]), axis=0)]
    smoothed_IoUs = np.vstack(smoothed_IoUs)
    smoothed_mIoUs = np.mean(smoothed_IoUs, axis=1)

    return smoothed_IoUs, smoothed_mIoUs


def load_confusions(filename, n_class):

    with open(filename, 'r') as f:
        lines = f.readlines()

    confs = np.zeros((len(lines), n_class, n_class))
    for i, line in enumerate(lines):
        C = np.array([int(value) for value in line.split()])
        confs[i, :, :] = C.reshape((n_class, n_class))

    return confs


def load_training_results(path):

    filename = join(path, 'training.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    epochs = []
    steps = []
    L_out = []
    L_p = []
    acc = []
    t = []
    for line in lines[1:]:
        line_info = line.split()
        if (len(line) > 0):
            epochs += [int(line_info[0])]
            steps += [int(line_info[1])]
            L_out += [float(line_info[2])]
            L_p += [float(line_info[3])]
            acc += [float(line_info[4])]
            t += [float(line_info[5])]
        else:
            break

    return epochs, steps, L_out, L_p, acc, t


def load_single_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        all_IoUs += [np.reshape([float(IoU) for IoU in line.split()], [-1, n_parts])]
    return all_IoUs


def load_snap_clouds(path, dataset, only_last=False):

    cloud_folders = np.array([join(path, f) for f in listdir_str(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf.txt')
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir_str(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    data = read_ply(join(cloud_folder, f))
                    labels = data['class']
                    preds = data['preds']
                    Confs[c_i] += fast_confusion(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir_str(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Plot functions
#       \********************/
#


def compare_trainings(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    plot_lr = False
    smooth_epochs = 0.5
    stride = 2

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Training Logs
    # ******************

    all_epochs = []
    all_loss = []
    all_lr = []
    all_times = []
    all_RAMs = []

    for path in list_of_paths:

        print(path)

        if ('val_IoUs.txt' in [f for f in listdir_str(path)]) or ('val_confs.txt' in [f for f in listdir_str(path)]):
            config = Config()
            config.load(path)
        else:
            continue

        # Load results
        epochs, steps, L_out, L_p, acc, t = load_training_results(path)
        epochs = np.array(epochs, dtype=np.int32)
        epochs_d = np.array(epochs, dtype=np.float32)
        steps = np.array(steps, dtype=np.float32)

        # Compute number of steps per epoch
        max_e = np.max(epochs)
        first_e = np.min(epochs)
        epoch_n = []
        for i in range(first_e, max_e):
            bool0 = epochs == i
            e_n = np.sum(bool0)
            epoch_n.append(e_n)
            epochs_d[bool0] += steps[bool0] / e_n
        smooth_n = int(np.mean(epoch_n) * smooth_epochs)
        smooth_loss = running_mean(L_out, smooth_n, stride=stride)
        all_loss += [smooth_loss]
        all_epochs += [epochs_d[smooth_n:-smooth_n:stride]]
        all_times += [t[smooth_n:-smooth_n:stride]]

        # Learning rate
        if plot_lr:
            lr_decay_v = np.array([lr_d for ep, lr_d in config.lr_decays.items()])
            lr_decay_e = np.array([ep for ep, lr_d in config.lr_decays.items()])
            max_e = max(np.max(all_epochs[-1]) + 1, np.max(lr_decay_e) + 1)
            lr_decays = np.ones(int(np.ceil(max_e)), dtype=np.float32)
            lr_decays[0] = float(config.learning_rate)
            lr_decays[lr_decay_e] = lr_decay_v
            lr = np.cumprod(lr_decays)
            all_lr += [lr[np.floor(all_epochs[-1]).astype(np.int32)]]

    # Plots learning rate
    # *******************


    if plot_lr:
        # Figure
        fig = plt.figure('lr')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_lr[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('lr')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plots loss
    # **********

    # Figure
    fig = plt.figure('loss')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], all_loss[i], linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')

    # Display legends and title
    plt.legend(loc=1)
    plt.title('Losses compare')

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plot Times
    # **********

    # Figure
    fig = plt.figure('time')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], np.array(all_times[i]) / 3600, linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('time')
    # plt.yscale('log')

    # Display legends and title
    plt.legend(loc=0)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


def compare_convergences_segment(dataset, list_of_paths, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_mIoUs = []
    all_class_IoUs = []
    all_snap_epochs = []
    all_snap_IoUs = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    s = '{:^10}|'.format('mean')
    for c in class_list:
        s += '{:^10}'.format(c)
    print(s)
    print(10*'-' + '|' + 10*config.num_classes*'-')
    for path in list_of_paths:

        # Get validation IoUs
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, config.num_classes)

        # Get mean IoU
        class_IoUs, mIoUs = IoU_class_metrics(val_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_mIoUs += [mIoUs]
        all_class_IoUs += [class_IoUs]

        s = '{:^10.1f}|'.format(100*mIoUs[-1])
        for IoU in class_IoUs[-1]:
            s += '{:^10.1f}'.format(100*IoU)
        print(s)

        # Get optional full validation on clouds
        snap_epochs, snap_IoUs = load_snap_clouds(path, dataset)
        all_snap_epochs += [snap_epochs]
        all_snap_IoUs += [snap_IoUs]

    print(10*'-' + '|' + 10*config.num_classes*'-')
    for snap_IoUs in all_snap_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^10.1f}|'.format(100*np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^10.1f}'.format(100*IoU)
        else:
            s = '{:^10s}'.format('-')
            for _ in range(config.num_classes):
                s += '{:^10s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_snap_epochs[i], np.mean(all_snap_IoUs[i], axis=1), linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


def compare_convergences_classif(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_n = 12

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_OA = []
    all_train_OA = []
    all_vote_OA = []
    all_vote_confs = []


    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(list_of_paths[0])

        # Get the number of classes
        n_class = config.num_classes

        # Load epochs
        epochs, _, _, _, _, _ = load_training_results(path)
        first_e = np.min(epochs)

        # Get validation confusions
        file = join(path, 'val_confs.txt')
        val_C1 = load_confusions(file, n_class)
        val_PRE, val_REC, val_F1, val_IoU, val_ACC = smooth_metrics(val_C1, smooth_n=smooth_n)

        # Get vote confusions
        file = join(path, 'vote_confs.txt')
        if exists(file):
            vote_C2 = load_confusions(file, n_class)
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = smooth_metrics(vote_C2, smooth_n=2)
        else:
            vote_C2 = val_C1
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = (val_PRE, val_REC, val_F1, val_IoU, val_ACC)

        # Aggregate results
        all_pred_epochs += [np.array([i+first_e for i in range(len(val_ACC))])]
        all_val_OA += [val_ACC]
        all_vote_OA += [vote_ACC]
        all_vote_confs += [vote_C2]

    print()

    # Best scores
    # ***********

    for i, label in enumerate(list_of_labels):

        print('\n' + label + '\n' + '*' * len(label) + '\n')
        print(list_of_paths[i])

        best_epoch = np.argmax(all_vote_OA[i])
        print('Best Accuracy : {:.1f} % (epoch {:d})'.format(100 * all_vote_OA[i][best_epoch], best_epoch))

        confs = all_vote_confs[i]

        """
        s = ''
        for cc in confs[best_epoch]:
            for c in cc:
                s += '{:.0f} '.format(c)
            s += '\n'
        print(s)
        """

        TP_plus_FN = np.sum(confs, axis=-1, keepdims=True)
        class_avg_confs = confs.astype(np.float32) / TP_plus_FN.astype(np.float32)
        diags = np.diagonal(class_avg_confs, axis1=-2, axis2=-1)
        class_avg_ACC = np.sum(diags, axis=-1) / np.sum(class_avg_confs, axis=(-1, -2))

        print('Corresponding mAcc : {:.1f} %'.format(100 * class_avg_ACC[best_epoch]))

    # Plots
    # *****

    for fig_name, OA in zip(['Validation', 'Vote'], [all_val_OA, all_vote_OA]):

        # Figure
        fig = plt.figure(fig_name)
        for i, label in enumerate(list_of_labels):
            plt.plot(all_pred_epochs[i], OA[i], linewidth=1, label=label)
        plt.xlabel('epochs')
        plt.ylabel(fig_name + ' Accuracy')

        # Set limits for y axis
        #plt.ylim(0.55, 0.95)

        # Display legends and title
        plt.legend(loc=4)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    #for i, label in enumerate(list_of_labels):
    #    print(label, np.max(all_train_OA[i]), np.max(all_val_OA[i]))

    # Show all
    plt.show()


def compare_convergences_SLAM(dataset, list_of_paths, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_mIoUs = []
    all_val_class_IoUs = []
    all_subpart_mIoUs = []
    all_subpart_class_IoUs = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    s = '{:^6}|'.format('mean')
    for c in class_list:
        s += '{:^6}'.format(c[:4])
    print(s)
    print(6*'-' + '|' + 6*config.num_classes*'-')
    for path in list_of_paths:

        # Get validation IoUs
        nc_model = dataset.num_classes - len(dataset.ignored_labels)
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, nc_model)

        # Get Subpart IoUs
        file = join(path, 'subpart_IoUs.txt')
        subpart_IoUs = load_single_IoU(file, nc_model)

        # Get mean IoU
        val_class_IoUs, val_mIoUs = IoU_class_metrics(val_IoUs, smooth_n)
        subpart_class_IoUs, subpart_mIoUs = IoU_class_metrics(subpart_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_val_mIoUs += [val_mIoUs]
        all_val_class_IoUs += [val_class_IoUs]
        all_subpart_mIoUs += [subpart_mIoUs]
        all_subpart_class_IoUs += [subpart_class_IoUs]

        s = '{:^6.1f}|'.format(100*subpart_mIoUs[-1])
        for IoU in subpart_class_IoUs[-1]:
            s += '{:^6.1f}'.format(100*IoU)
        print(s)

    print(6*'-' + '|' + 6*config.num_classes*'-')
    for snap_IoUs in all_val_class_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^6.1f}|'.format(100*np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^6.1f}'.format(100*IoU)
        else:
            s = '{:^6s}'.format('-')
            for _ in range(config.num_classes):
                s += '{:^6s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_subpart_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_pred_epochs[i], all_val_mIoUs[i], linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    #displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_val_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



    # Show all
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
#
#           Experiments
#       \*****************/
#


def experiment_name_1():
    """
    In this function you choose the results you want to plot together, to compare them as an experiment.
    Just return the list of log paths (like 'results/Log_2020-04-04_10-04-42' for example), and the associated names
    of these logs.
    Below an example of how to automatically gather all logs between two dates, and name them.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-04-22_11-52-58'
    end = 'Log_2023-07-29_12-40-27'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Give names to the logs (for plot legends)
    logs_names = ['name_log_1',
                  'name_log_2',
                  'name_log_3',
                  'name_log_4']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def experiment_name_2():
    """
    In this function you choose the results you want to plot together, to compare them as an experiment.
    Just return the list of log paths (like 'results/Log_2020-04-04_10-04-42' for example), and the associated names
    of these logs.
    Below an example of how to automatically gather all logs between two dates, and name them.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-04-22_11-52-58'
    end = 'Log_2020-05-22_11-52-58'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2020-04-04_10-04-42')

    # Give names to the logs (for plot legends)
    logs_names = ['name_log_inserted',
                  'name_log_1',
                  'name_log_2',
                  'name_log_3']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # My logs: choose the logs to show
    logs, logs_names = experiment_name_1()

    ################
    # Plot functions
    ################

    # Check that all logs are of the same dataset. Different object can be compared
    plot_dataset = None
    config = None
    for log in logs:
        config = Config()
        config.load(log)
        if 'ShapeNetPart' in config.dataset:
            this_dataset = 'ShapeNetPart'
        else:
            this_dataset = config.dataset
        if plot_dataset:
            if plot_dataset == this_dataset:
                continue
            else:
                raise ValueError('All logs must share the same dataset to be compared')
        else:
            plot_dataset = this_dataset

    # Plot the training loss and accuracy
    compare_trainings(logs, logs_names)

    # Plot the validation
    if config.dataset_task == 'classification':
        compare_convergences_classif(logs, logs_names)
    elif config.dataset_task == 'cloud_segmentation':
        if config.dataset.startswith('S3DIS'):
            dataset = S3DISDataset(config, load_data=False)
            compare_convergences_segment(dataset, logs, logs_names)
    elif config.dataset_task == 'slam_segmentation':
        if config.dataset.startswith('SemanticKitti'):
            dataset = SemanticKittiDataset(config)
            compare_convergences_SLAM(dataset, logs, logs_names)
    else:
        raise ValueError('Unsupported dataset : ' + plot_dataset)




