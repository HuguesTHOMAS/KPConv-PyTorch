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
            required=False,
            type=valid_file,
            help=(
                "File on which to predict semantic labels starting from a trained model "
                "(if None, use the validation split)"
            ),
        )

        parser.add_argument(
            "-l",
            "--chosen-log",
            required=True,
            type=valid_dir,
            help=(
                "If mentioned with the test command, "
                "the test will use this folder for the inference procedure."
            ),
        )
        # '.../Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model
        # 'last_XXX': Automatically retrieve the last trained model on dataset XXX
        parser.add_argument(
            "-n",
            "--n-votes",
            type=int,
            help="Number of positive vote during inference process (stop condition to reach)",
        )
        parser.add_argument(
            "-p",
            "--potential-increment",
            type=int,
            help="Increment of inference potential at which results are saved",
        )

    if command == "train":
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "-l",
            "--chosen-log",
            type=valid_dir,
            help=(
                "If mentioned with the train command, "
                "the training starts from an already trained model, "
                "contained in the mentioned folder."
            ),
        )

        group.add_argument(
            "-o",
            "--output-dir",
            type=valid_dir,
            help=(
                "If mentioned, starts training from the begining. "
                "Otherwise, the -l option must be mentioned."
            ),
        )
        parser.add_argument(
            "-E",
            "--max-epoch",
            type=int,
            help="Upper bound for the number of training epochs (useful for functional test)",
        )
        parser.add_argument(
            "-g",
            "--checkpoint-gap",
            type=int,
            help="Frequency at which training checkpoint are saved on disk (in terms of epochs)",
        )
        parser.add_argument(
            "-e",
            "--epoch-steps",
            type=int,
            help="Number of steps per training epoch",
        )
    parser.add_argument(
        "-v",
        "--validation-size",
        type=int,
        help="Number of steps per validation process, after each epoch",
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
    kpconv_parser(
        sub_parsers,
        train.main,
        "train",
        "Train a KPConv model",
    )
    kpconv_parser(
        sub_parsers,
        test.main,
        "test",
        "Test a KPConv trained model",
    )

    args = parser.parse_args()

    if args.dataset not in ("ModelNet40", "NPM3D", "S3DIS", "SemanticKitti", "Toronto3D"):
        raise ValueError(
            f"Error - unsupported dataset {args.dataset}: --dataset or -d parameter must\
            be among ModelNet40, NPM3D, S3DIS, SemanticKitti, Toronto3D"
        )

    if "func" in vars(args):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
