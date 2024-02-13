import numpy as np


def read_xyz(filepath):
    """Takes a file path pointing on a 3D point .xyz (text) file and returns the points,
    the associated colors and the associated classes.

    Parameters
    ----------
    filepath: path to a 3D points file with .xyz

    Returns
    -------
    points: 2D np.array with type float32
    colors: 2D np.array with type uint8
    labels: 1D np.array with type int32
    """
    data = np.loadtxt(filepath, delimiter=" ")
    points = data[:, :3].astype(np.float32)
    if data.shape[1] >= 6:
        colors = data[:, 3:6].astype(np.uint8)
    else:
        colors = colors.shape[0] > 0
    if data.shape[1] == 4:
        labels = np.squeeze(data[:, 3]).astype(np.int32)
    if data.shape[1] == 7:
        labels = np.squeeze(data[:, 6]).astype(np.int32)

    return points, colors, labels


def write_xyz(filepath, points, colors=None, labels=None):
    """Creates a .xyz file from a 3D point cloud.
    Parameters
    ----------
    filepath : pathlib.Path
        Path where to save the data
    points: 2D np.array with type float32 to save
    colors: 2D np.array with type uint8 to save
    labels: 1D np.array with type int32 to save
    """
    if colors is None:
        colors = np.zeros(points.shape[0], 3).astype(np.uint8)
    if labels is None:
        labels = np.zeros(points.shape[0]).astype(np.int32)
    data = np.column_stack((np.column_stack((points, colors)), labels))
    np.savetxt(filepath, data, delimiter=" ")

    return True
