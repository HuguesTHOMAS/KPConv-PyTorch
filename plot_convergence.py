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


def IoU_multi_metrics(all_IoUs, smooth_n):

    # Get mean IoU for consecutive epochs to directly get a mean
    all_mIoUs = [np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs]) for epoch_IoUs in all_IoUs]
    smoothed_mIoUs = []
    for epoch in range(len(all_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_mIoUs))
        smoothed_mIoUs += [np.mean(np.hstack(all_mIoUs[i0:i1]))]

    # Get mean for each class
    all_objs_mIoUs = [[np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs] for epoch_IoUs in all_IoUs]
    smoothed_obj_mIoUs = []
    for epoch in range(len(all_objs_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_objs_mIoUs))

        epoch_obj_mIoUs = []
        for obj in range(len(all_objs_mIoUs[0])):
            epoch_obj_mIoUs += [np.mean(np.hstack([objs_mIoUs[obj] for objs_mIoUs in all_objs_mIoUs[i0:i1]]))]

        smoothed_obj_mIoUs += [epoch_obj_mIoUs]

    return np.array(smoothed_mIoUs), np.array(smoothed_obj_mIoUs)


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

    cloud_folders = np.array([join(path, f) for f in listdir(path) if f.startswith('val_preds')])
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
            for f in listdir(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    data = read_ply(join(cloud_folder, f))
                    labels = data['class']
                    preds = data['preds']
                    Confs[c_i] += fast_confusion(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_snap_clouds(path, dataset, file_i, only_last=False):

    cloud_folders = np.array([join(path, f) for f in listdir(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    if len(cloud_folders) > 0:
        dataset_folders = [f for f in listdir(cloud_folders[0]) if dataset.name in f]
        cloud_folders = [join(f, dataset_folders[file_i]) for f in cloud_folders]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf_{:s}.txt'.format(dataset.name))
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    if np.any([cloud_path.endswith(f) for cloud_path in dataset.files]):
                        data = read_ply(join(cloud_folder, f))
                        labels = data['class']
                        preds = data['preds']
                        Confs[c_i] += confusion_matrix(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        obj_IoUs = [[float(IoU) for IoU in s.split()] for s in line.split('/')]
        obj_IoUs = [np.reshape(IoUs, [-1, n_parts[obj]]) for obj, IoUs in enumerate(obj_IoUs)]
        all_IoUs += [obj_IoUs]
    return all_IoUs


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

        if ('val_IoUs.txt' in [f for f in listdir(path)]) or ('val_confs.txt' in [f for f in listdir(path)]):
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


def compare_convergences_multisegment(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_n = 5

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_instances_mIoUs = []
    all_objs_mIoUs = []
    all_objs_IoUs = []
    all_parts = []

    obj_list = ['Air', 'Bag', 'Cap', 'Car', 'Cha', 'Ear', 'Gui', 'Kni',
                'Lam', 'Lap', 'Mot', 'Mug', 'Pis', 'Roc', 'Ska', 'Tab']
    print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
    print('-----|------|--------------------------------------------------------------------------------')
    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(path)

        # Get the number of classes
        n_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        part = config.dataset.split('_')[-1]

        # Get validation confusions
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_multi_IoU(file, n_parts)

        file = join(path, 'vote_IoUs.txt')
        vote_IoUs = load_multi_IoU(file, n_parts)

        #print(len(val_IoUs[0]))
        #print(val_IoUs[0][0].shape)

        # Get mean IoU
        #instances_mIoUs, objs_mIoUs = IoU_multi_metrics(val_IoUs, smooth_n)

        # Get mean IoU
        instances_mIoUs, objs_mIoUs = IoU_multi_metrics(vote_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_instances_mIoUs += [instances_mIoUs]
        all_objs_IoUs += [objs_mIoUs]
        all_objs_mIoUs += [np.mean(objs_mIoUs, axis=1)]

        if part == 'multi':
            s = '{:4.1f} | {:4.1f} | '.format(100 * np.mean(objs_mIoUs[-1]), 100 * instances_mIoUs[-1])
            for obj_mIoU in objs_mIoUs[-1]:
                s += '{:4.1f} '.format(100 * obj_mIoU)
            print(s)
        else:
            s = ' --  |  --  | '
            for obj_name in obj_list:
                if part.startswith(obj_name):
                    s += '{:4.1f} '.format(100 * instances_mIoUs[-1])
                else:
                    s += ' --  '.format(100 * instances_mIoUs[-1])
            print(s)
        all_parts += [part]

    # Plots
    # *****

    if 'multi' in all_parts:

        # Figure
        fig = plt.figure('Instances mIoU')
        for i, label in enumerate(list_of_labels):
            if all_parts[i] == 'multi':
                plt.plot(all_pred_epochs[i], all_instances_mIoUs[i], linewidth=1, label=label)
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

        # Figure
        fig = plt.figure('mean of categories mIoU')
        for i, label in enumerate(list_of_labels):
            if all_parts[i] == 'multi':
                plt.plot(all_pred_epochs[i], all_objs_mIoUs[i], linewidth=1, label=label)
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

    for obj_i, obj_name in enumerate(obj_list):
        if np.any([part.startswith(obj_name) for part in all_parts]):
            # Figure
            fig = plt.figure(obj_name + ' mIoU')
            for i, label in enumerate(list_of_labels):
                if all_parts[i] == 'multi':
                    plt.plot(all_pred_epochs[i], all_objs_IoUs[i][:, obj_i], linewidth=1, label=label)
                elif all_parts[i].startswith(obj_name):
                    plt.plot(all_pred_epochs[i], all_objs_mIoUs[i], linewidth=1, label=label)
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


def compare_convergences_multicloud(list_of_paths, multi, multi_datasets, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]


    # Loop on all datasets:
    for plot_dataset in multi_datasets:
        print('\n')
        print(plot_dataset)
        print('*'*len(plot_dataset))
        print()

        # Load dataset parameters
        if plot_dataset.startswith('S3DIS'):
            dataset = S3DISDataset()
        elif plot_dataset.startswith('Scann'):
            dataset = ScannetDataset()
        elif plot_dataset.startswith('Seman'):
            dataset = Semantic3DDataset()
        elif plot_dataset.startswith('NPM3D'):
            dataset = NPM3DDataset()
        else:
            raise ValueError('Unsupported dataset : ' + plot_dataset)

        # Read Logs
        # *********

        all_pred_epochs = []
        all_mIoUs = []
        all_class_IoUs = []
        all_snap_epochs = []
        all_snap_IoUs = []
        all_names = []

        class_list = [dataset.label_to_names[label] for label in dataset.label_values
                      if label not in dataset.ignored_labels]

        s = '{:^10}|'.format('mean')
        for c in class_list:
            s += '{:^10}'.format(c)
        print(s)
        print(10*'-' + '|' + 10*dataset.num_classes*'-')
        for log_i, (path, is_multi) in enumerate(zip(list_of_paths, multi)):

            n_c = None
            if is_multi:
                config = MultiConfig()
                config.load(path)
                if plot_dataset in config.datasets:
                    val_IoU_files = []
                    for d_i in np.where(np.array(config.datasets) == plot_dataset)[0]:
                        n_c = config.num_classes[d_i]
                        val_IoU_files.append(join(path, 'val_IoUs_{:d}_{:s}.txt'.format(d_i, plot_dataset)))
                else:
                    continue
            else:
                config = Config()
                config.load(path)
                if plot_dataset == config.dataset:
                    n_c = config.num_classes
                    val_IoU_files = [join(path, 'val_IoUs.txt')]
                else:
                    continue

            for file_i, file in enumerate(val_IoU_files):

                # Load validation IoUs
                val_IoUs = load_single_IoU(file, n_c)

                # Get mean IoU
                class_IoUs, mIoUs = IoU_class_metrics(val_IoUs, smooth_n)

                # Aggregate results
                all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
                all_mIoUs += [mIoUs]
                all_class_IoUs += [class_IoUs]
                all_names += [list_of_names[log_i]+'_{:d}'.format(file_i+1)]

                s = '{:^10.1f}|'.format(100*mIoUs[-1])
                for IoU in class_IoUs[-1]:
                    s += '{:^10.1f}'.format(100*IoU)
                print(s)

                # Get optional full validation on clouds
                if is_multi:
                    snap_epochs, snap_IoUs = load_multi_snap_clouds(path, dataset, file_i)
                else:
                    snap_epochs, snap_IoUs = load_snap_clouds(path, dataset)
                all_snap_epochs += [snap_epochs]
                all_snap_IoUs += [snap_IoUs]

        print(10*'-' + '|' + 10*dataset.num_classes*'-')
        for snap_IoUs in all_snap_IoUs:
            if len(snap_IoUs) > 0:
                s = '{:^10.1f}|'.format(100*np.mean(snap_IoUs[-1]))
                for IoU in snap_IoUs[-1]:
                    s += '{:^10.1f}'.format(100*IoU)
            else:
                s = '{:^10s}'.format('-')
                for _ in range(dataset.num_classes):
                    s += '{:^10s}'.format('-')
            print(s)

        # Plots
        # *****

        # Figure
        fig = plt.figure('mIoUs')
        for i, name in enumerate(all_names):
            p = plt.plot(all_pred_epochs[i], all_mIoUs[i], '--', linewidth=1, label=name)
            plt.plot(all_snap_epochs[i], np.mean(all_snap_IoUs[i], axis=1), linewidth=1, color=p[-1].get_color())

        plt.title(plot_dataset)
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
#           Main Call
#       \***************/
#


def ModelNet40_first_test():
    """
    First tries with ModelNet40
    First we compare convergence of a very very deep network on ModelNet40, with our without bn
    Then, We try the resuming of previous trainings. Which works quite well.
    However in the mean time, we change how validation worked by calling net.eval()/net.train() before/after
    validation. It seems that the network perform strange when calling net.eval()/net.train() although it should be the
    right way to do it.
    Then we try to change BatchNorm1D with InstanceNorm1D and compare with and without calling eval/train at validation.
    (Also with a faster lr decay).
    --- MISTAKE FOUND --- the batch norm momentum was inverted 0.98 instead of 0.02.
    See next experiment for correct convergences. Instance norm seems not as good
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-18_16-04-20'
    end = 'Log_2020-03-20_16-59-40'

    if end < 'Log_2020-03-22_19-30-19':
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['with_bn',
                  'without_bn',
                  'with_bn2',
                  'without_bn2',
                  'lrd_80_Inorm_eval_train',
                  'lrd_80_Inorm_always_train',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_batch_norm():
    """
    Compare different type of batch norm now that it has been fixed. Batch norm seems the best easily. Instance norm
    crewated a NAN loss so avoid this one.
    Now try fast experiments. First reduce network size. Reducing the number of convolution per layer does not affect
    results (maybe because dataset is too simple???). 5 small layers is way better that 4 big layers.
    Now reduce number of step per epoch and maybe try balanced sampler. Balanced sampler with fewer steps per epoch is
    way faster for convergence and gets nearly the same scores. so good for experimenting. However we cant really
    conclude between parameters which will get the same score (like the more layers) because the dataset my be
    limitating. We can only conclude if something is not good and reduce score.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-20_16-59-41'
    end = 'Log_2020-04-13_18-14-44'

    if end < 'Log_2020-03-22_19-30-19':
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['no_norm',
                  'IN',
                  'BN',
                  '5_small_layer-d0=0.02',
                  '3_big_layer-d0=0.02',
                  '3_big_layer-d0=0.04',
                  'small-e_n=300',
                  'small-e_n=300-balanced_train',
                  'small-e_n=300-balanced_traintest',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_fast_vs_results():
    """
    Try lr decay with fast convergence (epoch_n=300 and balanced traintest). 80 is a good value.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-21_16-09-17'
    end = 'Log_2020-03-21_16-09-36'

    if end < 'Log_2020-03-22_19-30-19':
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = np.insert(logs, 1, join(res_path, 'Log_2020-03-21_11-57-45'))

    # Give names to the logs (for legends)
    logs_names = ['lrd=120',
                  'lrd=80',
                  'lrd=40',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_grad_clipping():
    """
    Test different grad clipping. No difference so we can move on
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-21_18-21-37'
    end = 'Log_2020-03-21_18-30-01'

    if end < 'Log_2020-03-22_19-30-19':
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = np.insert(logs, 0, join(res_path, 'Log_2020-03-21_11-57-45'))

    # Give names to the logs (for legends)
    logs_names = ['value_clip_100',
                  'norm_clip_100',
                  'value_clip_10',
                  'norm_clip_10',
                  'no_clip',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_KP_extent():
    """
    Test differents mode et kp extent. sum et extent=2.0 definitivement moins bon (trop de recouvrement des kp
    influences, noyau moins versatile). les closest semble plutot bon et le sum extent=1.5 pas mal du tout () peut
    etre le meilleur. A confirmer sur gros dataset
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-21_18-30-02'
    end = 'Log_2020-03-21_23-36-18'

    if end < 'Log_2020-03-22_19-30-19':
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = np.insert(logs, 0, join(res_path, 'Log_2020-03-21_11-57-45'))

    # Give names to the logs (for legends)
    logs_names = ['KPe=1.0_sum_linear',
                  'KPe=1.5_sum_linear',
                  'KPe=2.0_sum_linear',
                  'KPe=1.5_closest_linear',
                  'KPe=2.0_closest_linear',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_gaussian():
    """
    Test different extent in gaussian mode. extent=1.5 seems the best. 2.0 is not bad. But in any case, it does not
    perform better than 1.5-linear-sum at least on this dataset.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-21_23-36-19'
    end = 'Log_2020-04-13_18-14-44'

    if end < 'Log_2020-03-22_19-30-19':
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = np.insert(logs, 4, join(res_path, 'Log_2020-03-21_19-35-11'))

    # Give names to the logs (for legends)
    logs_names = ['KPe=1.0_sum_gaussian',
                  'KPe=1.5_sum_gaussian',
                  'KPe=2.0_sum_gaussian',
                  'KPe=2.5_sum_gaussian',
                  'KPe=1.5_sum_linear',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_normals():
    """
    Test different way to add normals. Seems pretty much the same and we dont care about normals.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-22_10-18-56'
    end = 'Log_2020-03-22_13-32-51'

    if end < 'Log_2020-03-22_19-30-19':
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = np.insert(logs, 0, join(res_path, 'Log_2020-03-21_19-35-11'))

    # Give names to the logs (for legends)
    logs_names = ['no_normals',
                  'anisotropic_scale_normals',
                  'wrong_scale_normals',
                  'only_symmetries_normals(cheating)',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_radius():
    """
    Test different convolution radius. It was expected that larger radius would means slower networks but better
    performances. In fact we do not see much difference (again because of the dataset maybe?)
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-22_13-32-52'
    end = 'Log_2020-03-22_19-30-17'

    if end < 'Log_2020-03-22_19-30-19':
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = np.insert(logs, 2, join(res_path, 'Log_2020-03-21_19-35-11'))

    # Give names to the logs (for legends)
    logs_names = ['KPe=0.9_r=1.5',
                  'KPe=1.2_r=2.0',
                  'KPe=1.5_r=2.5',
                  'KPe=1.8_r=3.0',
                  'KPe=2.1_r=3.5',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_deform(old_result_limit):
    """
    Test deformable convolution with different offset decay. Without modulations 0.01 seems the best. With
    modulations 0.1 seems the best. In all cases 1.0 is to much. We need to show deformations for verification.

    It seems that deformations are not really fittig the point cloud. They just reach further away. W need to try on
    other datasets and with deformations earlier to see if fitting loss works
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-22_19-30-21'
    end = 'Log_2020-03-25_19-30-17'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, join('old_results', 'Log_2020-03-21_19-35-11'))

    # Give names to the logs (for legends)
    logs_names = ['normal',
                  'offset_d=0.01',
                  'offset_d=0.1',
                  'offset_d=1.0',
                  'offset_d=0.001',
                  'offset_d=0.001_modu',
                  'offset_d=0.01_modu',
                  'offset_d=0.1_modu',
                  'offset_d=1.0_modu',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def S3DIS_first(old_result_limit):
    """
    Test first S3DIS. First two test have all symetries (even vertical), which is not good). We corecct for
    the following.
    Then we try some experiments with different input scalea and the results are not as high as expected.
     WHY?
     FOUND IT! Problem resnet bottleneck should divide out-dim by 4 and not by 2
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-03-25_19-30-17'
    end = 'Log_2020-04-03_11-12-05'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['Fin=1_R=1.2_r=0.02 (error all symetries)',
                  'Fin=1_R=2.5_r=0.04 (error all symetries)',
                  'Fin=5_R=1.2_r=0.02',
                  'Fin=5_R=1.8_r=0.03',
                  'Fin=5_R=2.5_r=0.04',
                  'original_normal',
                  'original_deform',
                  'original_random_sampler',
                  'original_potentials_batch16',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def S3DIS_go(old_result_limit):
    """
    Test S3DIS.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-04-03_11-12-07'
    end = 'Log_2020-04-07_15-30-17'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['R=2.0_r=0.04_Din=128_potential',
                  'R=2.0_r=0.04_Din=64_potential',
                  'R=1.8_r=0.03',
                  'R=1.8_r=0.03_deeper',
                  'R=1.8_r=0.03_deform',
                  'R=2.0_r=0.03_megadeep',
                  'R=2.5_r=0.03_megadeep',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def SemanticKittiFirst(old_result_limit):
    """
    Test SematicKitti. First exps.
    Try some class weight strategies. It seems that the final score is not impacted so much. With weights, some classes
    are better while other are worse, for a final score that remains the same.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-04-07_15-30-17'
    end = 'Log_2020-04-11_21-34-16'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['R=5.0_dl=0.04',
                  'R=5.0_dl=0.08',
                  'R=10.0_dl=0.08',
                  'R=10.0_dl=0.08_20*weigths',
                  'R=10.0_dl=0.08_20*sqrt_weigths',
                  'R=10.0_dl=0.08_100*sqrt_w',
                  'R=10.0_dl=0.08_100*sqrt_w_capped',
                  'R=10.0_dl=0.08_no_w']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def SemanticKitti_scale(old_result_limit):
    """
    Test SematicKitti. Try different scales of input raduis / subsampling.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-04-11_21-34-15'
    end = 'Log_2020-04-20_11-52-58'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = logs.astype('<U50')

    # Give names to the logs (for legends)
    logs_names = ['R=10.0_dl=0.08',
                  'R=4.0_dl=0.04',
                  'R=6.0_dl=0.06',
                  'R=6.0_dl=0.06_inF=2',
                  'test',
                  'test',
                  'test',
                  'test',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def S3DIS_deform(old_result_limit):
    """
    Debug S3DIS deformable.
    At checkpoint 50, the points seem to start fitting the shape, but then, they just get further away from each other
    and do not care about input points. The fitting loss seems broken?
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-04-22_11-52-58'
    end = 'Log_2020-05-22_11-52-58'

    if end < old_result_limit:
        res_path = 'old_results'
    else:
        res_path = 'results'

    logs = np.sort([join(res_path, l) for l in listdir(res_path) if start <= l <= end])
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2020-04-04_10-04-42')

    # Give names to the logs (for legends)
    logs_names = ['off_d=0.01_baseline',
                  'off_d=0.01',
                  'off_d=0.05',
                  'off_d=0.05_corrected',
                  'off_d=0.05_norepulsive',
                  'off_d=0.05_repulsive0.5',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names






if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # TODO: test deformable on S3DIS to see of fitting loss works
    # TODO: try class weights on S3DIS (very low weight for beam)

    # Old result limit
    old_res_lim = 'Log_2020-03-25_19-30-17'

    # My logs: choose the logs to show
    logs, logs_names = S3DIS_deform(old_res_lim)
    #os.environ['QT_DEBUG_PLUGINS'] = '1'

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

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




