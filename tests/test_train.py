"""Unit and functional tests related to the training and inference processes.
"""

from pathlib import Path
from shutil import rmtree

import numpy as np
from pytest import mark

from kpconv_torch.utils import tester, trainer
from kpconv_torch import test, train


def test_get_test_save_path(fixture_path):
    """Function get_test_save_path:

    - needs a "chosen_log" folder
    - optionally uses a "infered_file" path
    """
    assert tester.get_test_save_path(infered_file=None, chosen_log=None) is None
    log_dir = Path(__file__).parent / "Log_test"
    infered_file = fixture_path / "example.ply"
    test_path = tester.get_test_save_path(infered_file=infered_file, chosen_log=log_dir)
    assert test_path == fixture_path / "test" / "Log_test"
    assert test_path.exists()
    rmtree(test_path)
    test_path = tester.get_test_save_path(infered_file=None, chosen_log=log_dir)
    assert test_path == Path(log_dir) / "test"
    assert test_path.exists()
    rmtree(test_path)


def test_get_train_save_path():
    """Function get_train_save_path"""
    assert trainer.get_train_save_path(output_dir=None, chosen_log=None) is None
    log_dir = Path(__file__).parent / "Log_test"
    train_path = trainer.get_train_save_path(output_dir=None, chosen_log=log_dir)
    assert log_dir == train_path
    assert Path(train_path).exists()
    rmtree(log_dir)
    output_dir = Path(__file__).parent / "outputdir"
    train_path = trainer.get_train_save_path(output_dir=output_dir, chosen_log=None)
    assert Path(train_path).exists()
    log_dirs = list(Path(output_dir).iterdir())
    assert len(log_dirs) == 1
    rmtree(output_dir)


@mark.dependency()
def test_train(dataset_path, trained_model_path):
    """Functional test for training process

    A first training run is launched in order to test the basic usage of the process. Then a second
    run is launched in order to test the checkpoint recovering.

    Expected resulting tree:

    _ tests/fixtures/trained_models/
        |_ checkpoints/
            |_ current_chkp.tar
            |_ chkp_0001.tar
            |_ chkp_0002.tar (after second run)
        |_ potentials/
            |_ Area_5.ply
        |_ val_preds_1/
            |_ Area_5.ply
        |_ val_preds_2/ (after second run)
            |_ Area_5.ply
        |_ parameters.txt
        |_ training.txt (must contain as many rows as the epoch number times the step number)
        |_ val_IoUs.txt

    """
    max_epoch = 1
    epoch_steps = 3
    # First run
    train.train(
        dataset_path,
        chosen_log=None,
        output_dir=trained_model_path,
        dataset=dataset_path.name,
        max_epoch=max_epoch,
        checkpoint_gap=1,
        epoch_steps=epoch_steps,
    )

    log_dirs = list(trained_model_path.iterdir())
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    assert (log_dir / "parameters.txt").exists()
    assert (log_dir / "training.txt").exists()
    assert (log_dir / "checkpoints" / "current_chkp.tar").exists()
    assert (log_dir / "checkpoints" / "chkp_0001.tar").exists()
    training_results = np.loadtxt(log_dir / "training.txt", skiprows=1)
    assert training_results.shape[0] == epoch_steps * max_epoch
    assert (log_dir / "potentials" / "Area_5.ply").exists()
    assert (log_dir / f"val_preds_{max_epoch}" / "Area_5.ply").exists()
    assert (log_dir / "val_IoUs.txt").exists()

    # A second run to check the checkpoint usage
    # No need to specify max_epoch, checkpoint_gap and epoch_steps:
    # they are stored with the checkpoint
    train.train(
        dataset_path,
        chosen_log=log_dir,
        output_dir=None,
        dataset=dataset_path.name,
    )

    log_dirs = list(trained_model_path.iterdir())
    assert len(log_dirs) == 1
    new_log_dir = log_dirs[0]
    assert new_log_dir == log_dir
    assert (log_dir / "parameters.txt").exists()
    assert (log_dir / "training.txt").exists()
    assert (log_dir / "checkpoints" / "current_chkp.tar").exists()
    assert (log_dir / "checkpoints" / "chkp_0002.tar").exists()
    training_results = np.loadtxt(log_dir / "training.txt", skiprows=1)
    assert training_results.shape[0] == epoch_steps * (max_epoch + 1)
    assert (log_dir / "potentials" / "Area_5.ply").exists()
    assert (log_dir / f"val_preds_{max_epoch}" / "Area_5.ply").exists()
    assert (log_dir / "val_IoUs.txt").exists()


@mark.dependency(depends=["test_train"])
def test_test_validation_case(dataset_path, training_log):
    """Functional test for inference process

    First case: default inference on the validation dataset (Area_5.ply).

    Should produce the following tree:

    _ tests/fixtures/trained_models/Log_<date>/test/
        |_ potentials/
            |_ Area_5.ply
        |_ probs/
            |_ Area_5.ply
        |_ predictions/
            |_ Area_5.ply

    """
    test.test(
        dataset_path,
        None,
        training_log,
        dataset_path.name,
        n_votes=1,
        validation_size=5,
        potential_increment=2,  # The min potential increases until ~2.5
    )
    assert (training_log / "test" / "potentials" / "Area_5.ply").is_file()
    assert (training_log / "test" / "probs" / "Area_5.ply").is_file()
    assert (training_log / "test" / "predictions" / "Area_5.ply").is_file()


@mark.dependency(depends=["test_train"])
def test_test_inference_case(dataset_path, inference_file, training_log):
    """Functional test for inference process

    Second case: inference on an unknown dataset.

    Should produce the following tree:

    _ tests/fixtures/inference/test/Log_<date>/
        |_ potentials/
            |_ Area_5.ply
        |_ probs/
            |_ Area_5.ply
        |_ predictions/
            |_ Area_5.ply
            |_ Area_5.txt (an extra file in this case)

    """
    test.test(
        dataset_path,
        inference_file,
        training_log,
        dataset_path.name,
        n_votes=1,
        validation_size=5,
        potential_increment=5,  # The min potential increases until ~6.0
    )
    assert (
        inference_file.parent / "test" / training_log.name / "potentials" / inference_file.name
    ).is_file()
    assert (
        inference_file.parent / "test" / training_log.name / "probs" / inference_file.name
    ).is_file()
    assert (
        inference_file.parent / "test" / training_log.name / "predictions" / inference_file.name
    ).is_file()
    assert (
        inference_file.parent
        / "test"
        / training_log.name
        / "predictions"
        / inference_file.with_suffix(".txt").name
    ).is_file()

    # Clean out the inference folder
    rmtree(inference_file.parent / "test")
