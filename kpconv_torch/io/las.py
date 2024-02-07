import laspy
import numpy as np


def read_las_laz(filepath):
    """Takes a file path pointing on a 3D point .las or .laz file and returns the points,
    the associated colors and the associated classes.

    Parameters
    ----------
    filepath: path to a 3D points file with .laz or .las format

    Colors are original encoded on 2 bytes
    (cf. https://www.asprs.org/a/society/committees/standards/asprs_las_spec_v13.pdf), that is on a uint16.
    They are converted (divided by 256) when the las or the laz file is read by this function.

    Returns
    -------
    points: 2D np.array with type float32
    colors: 2D np.array with type uint8
    labels: 1D np.array with type int32
    """
    data = laspy.read(filepath)
    points = np.vstack([data.x, data.y, data.z]).transpose().astype(np.float32)
    dims = list(data.point_format.dimension_names)
    if "red" in dims and "blue" in dims and "green" in dims:
        colors = (
            np.vstack([data.red / 256, data.green / 256, data.blue / 256])
            .transpose()
            .astype(np.uint8)
        )
    else:
        colors = np.zeros((points.shape[0], 3))
    if "classification" in dims:
        labels = np.array(data.classification).astype(np.int32)
    else:
        labels = np.zeros(points.shape[0])

    return points, colors, labels


def write_las(filepath, points, colors=None, labels=None):
    """Creates a .las file from a 3D point cloud.
    Parameters
    ----------
    filepath: path to the .las file
    points: 2D np.array with type float32
    colors: 2D np.array with type uint8 or uint16
    labels: 1D np.array with type int32

    Uses a point_format = 7 (cf. https://pythonhosted.org/laspy/tut_background.html)
    Warning: laspy does not have support for writing files in the .laz format.
    """
    if colors is None:
        colors = np.zeros(points.shape[0], 3).astype(np.uint8)
    if labels is None:
        labels = np.zeros(points.shape[0]).astype(np.int32)
    las = laspy.create(point_format=7)
    if points.dtype == np.dtype("float32"):
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
    else:
        raise TypeError(
            f"Error while writing file {filepath}: points should be a float32 array, not {colors.dtype}"
        )
    if colors.dtype == np.dtype(np.uint8):
        las.red = colors[:, 0] * 256
        las.green = colors[:, 1] * 256
        las.blue = colors[:, 2] * 256
    elif colors.dtype == np.dtype(np.uint16):
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]
    else:
        raise TypeError(
            f"Error while writing file {filepath}: colors should be a (u)int8 or (u)int16 array, \
                not {colors.dtype}"
        )
    if labels.dtype == np.dtype("int32"):
        las.classification = labels[:]
    else:
        raise TypeError(
            f"Error while writing file {filepath}: classification should be an int32 array, \
                not {labels.dtype}"
        )
    las.write(filepath)
    return True
