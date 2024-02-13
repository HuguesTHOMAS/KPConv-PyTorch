import numpy as np

from kpconv_torch.io import las, ply, xyz


def test_write_ply_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.ply"
    res = ply.write_ply(
        str(example_filepath),
        [points_array, colors_array, classification_array],
        ["x", "y", "z", "red", "green", "blue", "classification"],
    )
    assert res and example_filepath.exists()

    read_data = ply.read_ply(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    read_classification = read_data[2]
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)
    np.testing.assert_array_equal(classification_array, read_classification)


def test_write_ply_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.ply"
    res = ply.write_ply(
        str(example_filepath),
        [points_array, colors_array],
        ["x", "y", "z", "red", "green", "blue"],
    )
    assert res and example_filepath.exists()

    read_data = ply.read_ply(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)


def test_write_las_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.las"
    res = las.write_las(str(example_filepath), points_array, colors_array, classification_array)
    assert res and example_filepath.exists()

    read_data = las.read_las_laz(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    read_classification = read_data[2]
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)
    np.testing.assert_array_equal(classification_array, read_classification)


def test_write_las_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.las"
    res = las.write_las(str(example_filepath), points_array, colors_array)
    assert res and example_filepath.exists()

    read_data = las.read_las_laz(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)


def test_write_xyz_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.xyz"
    res = xyz.write_xyz(str(example_filepath), points_array, colors_array)
    assert res and example_filepath.exists()

    read_data = xyz.read_xyz(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)


def test_write_xyz_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.xyz"
    res = xyz.write_xyz(str(example_filepath), points_array, colors_array, classification_array)
    assert res and example_filepath.exists()

    read_data = xyz.read_xyz(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    read_classification = (read_data[2]).T
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)
    np.testing.assert_array_equal(classification_array, read_classification)


def test_read_xyz_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.xyz"
    points, colors, _ = xyz.read_xyz(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)


def test_read_xyz_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.xyz"
    points, colors, classification = xyz.read_xyz(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    np.testing.assert_array_equal(classification_array, classification)


def test_read_ply_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.ply"
    points, colors, _ = ply.read_ply(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)


def test_read_ply_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.ply"
    points, colors, classification = ply.read_ply(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    np.testing.assert_array_equal(classification_array, classification)


def test_read_las_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.las"
    points, colors, _ = las.read_las_laz(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)


def test_read_las_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.las"
    points, colors, classification = las.read_las_laz(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    np.testing.assert_array_equal(classification_array, classification)
