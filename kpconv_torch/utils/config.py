"""
Configuration reading / writing functions and colors definition for the terminal

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, C0103, E0401

from enum import Enum
from pathlib import Path
import yaml


class BColors(Enum):
    """
    Colors used to display the code in the terminal
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def save_config(train_save_path, config):
    """
    Saves a configuration into a YAML file

    :param train_save_path: a path to a folder where to save the file
    :param config: a configuration dictionnary
    """
    with open(Path(train_save_path / "config.yml"), "w", encoding="utf-8") as file_object:
        yaml.dump(config, file_object)


def load_config(file_path):
    """
    Loads a configuration from a YAML file

    :param file_path: a path to a config file
    """
    file_path = Path(file_path)

    with open(file_path, encoding="utf-8") as file_object:
        config = yaml.load(file_object, Loader=yaml.SafeLoader)

    # Check if dataset exists
    dataset = config["dataset"]
    if dataset not in ("ModelNet40", "NPM3D", "S3DIS", "SemanticKitti", "Toronto3D"):
        raise ValueError(
            f"Error - unsupported dataset {dataset}, must be\
            among ModelNet40, NPM3D, S3DIS, SemanticKitti, Toronto3D"
        )
    return config
