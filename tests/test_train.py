from pathlib import Path
from shutil import rmtree

import numpy as np

from kpconv_torch.utils import trainer
from kpconv_torch import train


def test_get_train_save_path():
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


def test_train(fixture_path, dataset):
    datapath = fixture_path / dataset
    max_epoch = 1
    epoch_steps = 5
    # First run
    train.train(
        datapath,
        chosen_log=None,
        output_dir=fixture_path,
        dataset=dataset,
        max_epoch=max_epoch,
        checkpoint_gap=1,
        epoch_steps=epoch_steps,
        validation_size=2,
    )

    log_dirs = [
        subfolder for subfolder in fixture_path.iterdir() if subfolder.name.startswith("Log_")
    ]
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
        datapath,
        chosen_log=log_dir,
        output_dir=None,
        dataset=dataset,
    )

    log_dirs = [
        subfolder for subfolder in fixture_path.iterdir() if subfolder.name.startswith("Log_")
    ]
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

    # Clean out the fixture directory
    rmtree(log_dir)
