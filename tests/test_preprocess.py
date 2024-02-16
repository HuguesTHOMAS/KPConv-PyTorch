import numpy as np

from kpconv_torch import preprocess
from kpconv_torch.io import ply


def test_preprocess(fixture_path, dataset):
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
    datapath = fixture_path / dataset
    preprocess.preprocess(datapath, dataset)
    subsampling_coef = 0.03
    assert (datapath / f"input_{subsampling_coef:.3f}").exists()
    assert (datapath / "original_ply").exists()
    for room_name, cloud_name in zip(("hallway_3", "storage_3"), ("Area_3", "Area_5")):
        # Check the output tree
        assert (
            datapath / f"input_{subsampling_coef:.3f}" / f"{cloud_name}_coarse_KDTree.pkl"
        ).exists()
        assert (datapath / f"input_{subsampling_coef:.3f}" / f"{cloud_name}_KDTree.pkl").exists()
        assert (datapath / f"input_{subsampling_coef:.3f}" / f"{cloud_name}.ply").exists()
        assert (datapath / "original_ply" / f"{cloud_name}.ply").exists()

        # Check the output PLY content
        raw_data = np.loadtxt(datapath / cloud_name / room_name / f"{room_name}.txt")
        ply_data = ply.read_ply(datapath / "original_ply" / f"{cloud_name}.ply")
        assert ply_data[0].shape == (raw_data.shape[0], 3)
        assert ply_data[1].shape == (raw_data.shape[0], 3)
        assert ply_data[2].shape == (raw_data.shape[0],)

        expected_label_count = len(
            {
                filename.name.split("_")[0]
                for filename in (datapath / cloud_name / room_name / "Annotations").iterdir()
            }
        )
        assert len(np.unique(ply_data[2])) == expected_label_count
