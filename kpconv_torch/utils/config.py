import yaml

from pathlib import Path


def save_config(train_save_path, config):
    with open(Path(train_save_path / "config.yml"), "w") as file_object:
        yaml.dump(config, file_object)


def load_config(train_save_path):
    if train_save_path is None:
        config_file_path = "../config.yml"
    else:
        config_file_path = Path(train_save_path) / "config.yml"

    with open(config_file_path) as file_object:
        config = yaml.load(file_object, Loader=yaml.SafeLoader)

    # Learning rate decays, dictionary of all decay values with their epoch {epoch: decay}
    config["train"]["lr_decays"] = {
        i: 0.1 ** (1 / 150) for i in range(1, config["train"]["max_epoch"])
    }

    return config
