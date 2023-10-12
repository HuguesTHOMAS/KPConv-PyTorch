import argparse
from pathlib import Path

from kpconv_torch import __version__ as kpconv_version
from kpconv_torch import plot_convergence, preprocess, test, train, visualize


SUPPORTED_DATASETS = {"ModelNet40", "NPM3D", "S3DIS", "SemanticKitti", "Toronto3D"}


def valid_dataset(dataset):
    if dataset not in SUPPORTED_DATASETS:
        raise argparse.ArgumentTypeError(
            f"{dataset} dataset is unknown, please choose amongst {SUPPORTED_DATASETS}."
        )
    return dataset


def valid_dir(str_dir):
    """Build a ``pathlib.Path`` object starting from the ``str_dir`` folder."""
    path_dir = Path(str_dir)
    if not path_dir.is_dir():
        raise argparse.ArgumentTypeError(f"The {str(path_dir)} folder does not exist.")
    return path_dir


def kpconv_parser(subparser, reference_func, command, command_description):
    """CLI definition for kpconv commands

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(command, help=command_description)
    parser.add_argument(
        "-d",
        "--datapath",
        required=True,
        type=valid_dir,
        help="Path of the dataset on the file system",
    )
    if command == "test":
        parser.add_argument(
            "-f",
            "--filename",
            type=str,
            help=(
                "File on which to predict semantic labels starting from a trained model "
                "(if None, use the validation split)"
            ),
        )
    #   Here you can choose which model you want to use. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > 'results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model
    if command != "preprocess":
        parser.add_argument(
            "-l",
            "--chosen-log",
            required=True,
            type=valid_dir,
            help="Path of the KPConv log folder on the file system",
        )
    parser.add_argument(
        "-s",
        "--dataset",
        default="S3DIS",
        type=valid_dataset,
        help="Name of the dataset",
    )
    parser.set_defaults(func=reference_func)


def main():
    """Main method of the module"""
    parser = argparse.ArgumentParser(
        prog="kpconv",
        description=(
            f"kpconv_torch version {kpconv_version}. "
            "Implementation of the Kernel Point Convolution (KPConv) algorithm with PyTorch."
        ),
    )
    sub_parsers = parser.add_subparsers(dest="command")
    kpconv_parser(
        sub_parsers,
        preprocess.main,
        "preprocess",
        "Preprocess a dataset to make it compliant with the program",
    )
    kpconv_parser(sub_parsers, train.main, "train", "Train a KPConv model")
    kpconv_parser(sub_parsers, test.main, "test", "Test a KPConv trained model")
    kpconv_parser(
        sub_parsers, visualize.main, "visualize", "Visualize kernel deformations"
    )
    kpconv_parser(
        sub_parsers,
        plot_convergence.main,
        "plotconv",
        "Plot convergence for a set of models",
    )

    args = parser.parse_args()

    if "func" in vars(args):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
