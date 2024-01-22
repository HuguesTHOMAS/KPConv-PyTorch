import argparse
from pathlib import Path

from kpconv_torch import __version__ as kpconv_version
from kpconv_torch import preprocess, test, train


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


def valid_file(str_path):
    path = Path(str_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"The {str(str_path)} file does not exists")
    return path


def kpconv_parser(subparser, reference_func, command, command_description):
    """CLI definition for kpconv commands

    Parameters
    ----------
    subparser: argparser.parser.SubParsersAction
    reference_func: function
    """
    parser = subparser.add_parser(command, help=command_description)

    parser.add_argument(
        "-s",
        "--dataset",
        default="S3DIS",
        type=valid_dataset,
        help="Name of the dataset",
    )

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
            type=valid_file, 
            help=(
                "File on which to predict semantic labels starting from a trained model "
                "(if None, use the validation split)"
            ),
        )
    
    if command == "train":
        parser.add_argument(
            "-o", 
            "--output", 
            type=valid_dir, 
            help=(
                "If true, starts training from an already trained model. The -l option must then point on the folder of the already trained model. \
                 Otherwise, begins from start. A folder will be created inside the one pointed by the -l option. "
            )
        )
            
    if command != "preprocess": # for train and test commands
        parser.add_argument(
            "-l",
            "--chosen-log",
            type=valid_dir,
            help="Path of the KPConv log folder on the file system",
        )
    # '.../Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model
    # 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    
    parser.set_defaults(func=reference_func)


def main():
    """Main method of the module"""
    parser = argparse.ArgumentParser(
        prog="kpconv",
        description=(
            f"kpconv_torch version {kpconv_version}."
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
    kpconv_parser(
        sub_parsers, 
        train.main, 
        "train", 
        "Train a KPConv model"
    )
    kpconv_parser(
        sub_parsers, 
        test.main, 
        "test", 
        "Test a KPConv trained model"
    )

    args = parser.parse_args()

    if args.command == "train":
        if (args.chosen_log is None and args.output is None):
            parser.error("No model and no destination folder chosen, add -l / --chosen-log or -o / --output")
        elif (args.chosen_log is not None and args.output is not None):
            parser.error("Both a model and a destination folder are chosen, remove one of them, \
                          either the -l / --chosen-log or the -o / --output option")

    elif args.command == "test":
        if (args.chosen_log is None):
            parser.error("No model chosen, add -l / --chosen-log")

    if "func" in vars(args):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
