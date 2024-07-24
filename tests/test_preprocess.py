import numpy as np

from kpconv_torch import preprocess
from kpconv_torch.io import ply


def test_preprocess(dataset_path):
    """The preprocessing produces .PLY files usable for further training process in 'original_ply'
    subfolder as well as subsampling material in 'input_{subsampling_coef}' subfolder (KDTree,
    coarse KDTree and subsampled .PLY file).

    Expected output tree:

    _ tests/fixtures/S3DIS/
        |_ input_0.030/
            |_ Area_3_coarse_KDTree.pkl
            |_ Area_3_KDTree.pkl
            |_ Area_3.ply
            |_ Area_5_coarse_KDTree.pkl
            |_ Area_5_KDTree.pkl
            |_ Area_5.ply
        |_ original_ply/
            |_ Area_3.ply
            |_ Area_5.ply
    """
    preprocess.preprocess("config_S3DIS.yml", dataset_path)
    subsampling_coef = 0.03
    assert (dataset_path / f"input_{subsampling_coef:.3f}").exists()
    assert (dataset_path / "original_ply").exists()
    for room_name, cloud_name in zip(("hallway_3", "storage_3"), ("Area_3", "Area_5")):
        # Check the output tree
        assert (
            dataset_path / f"input_{subsampling_coef:.3f}" / f"{cloud_name}_coarse_KDTree.pkl"
        ).exists()
        assert (
            dataset_path / f"input_{subsampling_coef:.3f}" / f"{cloud_name}_KDTree.pkl"
        ).exists()
        assert (dataset_path / f"input_{subsampling_coef:.3f}" / f"{cloud_name}.ply").exists()
        assert (dataset_path / "original_ply" / f"{cloud_name}.ply").exists()

        # Check the output PLY content
        raw_data = np.loadtxt(dataset_path / cloud_name / room_name / f"{room_name}.txt")
        points, colors, labels = ply.read_ply(dataset_path / "original_ply" / f"{cloud_name}.ply")
        assert points.shape == (raw_data.shape[0], 3)
        assert colors.shape == (raw_data.shape[0], 3)
        assert labels.shape == (raw_data.shape[0],)

        expected_label_count = len(
            {
                filename.name.split("_")[0]
                for filename in (dataset_path / cloud_name / room_name / "Annotations").iterdir()
            }
        )
        assert len(np.unique(labels)) == expected_label_count
