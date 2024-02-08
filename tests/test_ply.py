import numpy as np

from kpconv_torch.utils import ply


def test_read_ply(fixture_path, ply_array):

    example_filepath = fixture_path / "example.ply"
    read_data = ply.read_ply(example_filepath)
    read_points = np.vstack((read_data["x"], read_data["y"], read_data["z"])).T
    np.testing.assert_array_equal(ply_array, read_points)


def test_write_ply(fixture_path, ply_array):

    values = np.random.randint(2, size=4, dtype=np.int8)
    example_filepath = fixture_path / "example.ply"
    res = ply.write_ply(str(example_filepath), [ply_array, values], ["x", "y", "z", "values"])
    assert res and example_filepath.exists()

    read_data = ply.read_ply(str(example_filepath))
    read_points = np.vstack((read_data["x"], read_data["y"], read_data["z"])).T
    np.testing.assert_array_equal(ply_array, read_points)
