#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Metric utility functions
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
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

def fast_confusion(true, pred, label_values=None):
    """
    Fast confusion matrix (100x faster than Scikit learn). But only works if labels are la
    :param true:
    :param false:
    :param num_classes:
    :return:
    """

    # Ensure data is in the right format
    true = np.squeeze(true)
    pred = np.squeeze(pred)
    if len(true.shape) != 1:
        raise ValueError('Truth values are stored in a {:d}D array instead of 1D array'. format(len(true.shape)))
    if len(pred.shape) != 1:
        raise ValueError('Prediction values are stored in a {:d}D array instead of 1D array'. format(len(pred.shape)))
    if true.dtype not in [np.int32, np.int64]:
        raise ValueError('Truth values are {:s} instead of int32 or int64'.format(true.dtype))
    if pred.dtype not in [np.int32, np.int64]:
        raise ValueError('Prediction values are {:s} instead of int32 or int64'.format(pred.dtype))
    true = true.astype(np.int32)
    pred = pred.astype(np.int32)

    # Get the label values
    if label_values is None:
        # From data if they are not given
        label_values = np.unique(np.hstack((true, pred)))
    else:
        # Ensure they are good if given
        if label_values.dtype not in [np.int32, np.int64]:
            raise ValueError('label values are {:s} instead of int32 or int64'.format(label_values.dtype))
        if len(np.unique(label_values)) < len(label_values):
            raise ValueError('Given labels are not unique')

    # Sort labels
    label_values = np.sort(label_values)

    # Get the number of classes
    num_classes = len(label_values)

    #print(num_classes)
    #print(label_values)
    #print(np.max(true))
    #print(np.max(pred))
    #print(np.max(true * num_classes + pred))

    # Start confusion computations
    if label_values[0] == 0 and label_values[-1] == num_classes - 1:

        # Vectorized confusion
        vec_conf = np.bincount(true * num_classes + pred)

        # Add possible missing values due to classes not being in pred or true
        #print(vec_conf.shape)
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')
        #print(vec_conf.shape)

        # Reshape confusion in a matrix
        return vec_conf.reshape((num_classes, num_classes))


    else:

        # Ensure no negative classes
        if label_values[0] < 0:
            raise ValueError('Unsupported negative classes')

        # Get the data in [0,num_classes[
        label_map = np.zeros((label_values[-1] + 1,), dtype=np.int32)
        for k, v in enumerate(label_values):
            label_map[v] = k

        pred = label_map[pred]
        true = label_map[true]

        # Vectorized confusion
        vec_conf = np.bincount(true * num_classes + pred)

        # Add possible missing values due to classes not being in pred or true
        if vec_conf.shape[0] < num_classes ** 2:
            vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')

        # Reshape confusion in a matrix
        return vec_conf.reshape((num_classes, num_classes))

def metrics(confusions, ignore_unclassified=False):
    """
    Computes different metrics from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) precision, recall, F1 score, IoU score
    """

    # If the first class (often "unclassified") should be ignored, erase it from the confusion.
    if (ignore_unclassified):
        confusions[..., 0, :] = 0
        confusions[..., :, 0] = 0

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FP = np.sum(confusions, axis=-1)
    TP_plus_FN = np.sum(confusions, axis=-2)

    # Compute precision and recall. This assume that the second to last axis counts the truths (like the first axis of
    # a confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    PRE = TP / (TP_plus_FN + 1e-6)
    REC = TP / (TP_plus_FP + 1e-6)

    # Compute Accuracy
    ACC = np.sum(TP, axis=-1) / (np.sum(confusions, axis=(-2, -1)) + 1e-6)

    # Compute F1 score
    F1 = 2 * TP / (TP_plus_FP + TP_plus_FN + 1e-6)

    # Compute IoU
    IoU = F1 / (2 - F1)

    return PRE, REC, F1, IoU, ACC


def smooth_metrics(confusions, smooth_n=0, ignore_unclassified=False):
    """
    Computes different metrics from confusion matrices. Smoothed over a number of epochs.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param smooth_n: (int). smooth extent
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) precision, recall, F1 score, IoU score
    """

    # If the first class (often "unclassified") should be ignored, erase it from the confusion.
    if ignore_unclassified:
        confusions[..., 0, :] = 0
        confusions[..., :, 0] = 0

    # Sum successive confusions for smoothing
    smoothed_confusions = confusions.copy()
    if confusions.ndim > 2 and smooth_n > 0:
        for epoch in range(confusions.shape[-3]):
            i0 = max(epoch - smooth_n, 0)
            i1 = min(epoch + smooth_n + 1, confusions.shape[-3])
            smoothed_confusions[..., epoch, :, :] = np.sum(confusions[..., i0:i1, :, :], axis=-3)

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(smoothed_confusions, axis1=-2, axis2=-1)
    TP_plus_FP = np.sum(smoothed_confusions, axis=-2)
    TP_plus_FN = np.sum(smoothed_confusions, axis=-1)

    # Compute precision and recall. This assume that the second to last axis counts the truths (like the first axis of
    # a confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    PRE = TP / (TP_plus_FN + 1e-6)
    REC = TP / (TP_plus_FP + 1e-6)

    # Compute Accuracy
    ACC = np.sum(TP, axis=-1) / (np.sum(smoothed_confusions, axis=(-2, -1)) + 1e-6)

    # Compute F1 score
    F1 = 2 * TP / (TP_plus_FP + TP_plus_FN + 1e-6)

    # Compute IoU
    IoU = F1 / (2 - F1)

    return PRE, REC, F1, IoU, ACC


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU
