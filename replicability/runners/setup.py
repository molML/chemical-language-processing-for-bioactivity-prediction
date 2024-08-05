import argparse
from typing import List

__ARGUMENTS = {
    "--representation-name": {
        "help": "Name of the representation to use",
        "choices": ["smiles", "selfies"],
        "type": str,
    },
    "--embedding-name": {
        "help": "Name of the embedding to use",
        "choices": ["learnable", "random", "onehot"],
        "type": str,
    },
    "--model-name": {
        "help": "Name of the model to train",
        "choices": [
            "cnn",
            "transformer",
            "gru",
        ],
        "type": str,
    },
    "--dataset-name": {
        "help": "Name of the model to train",
        "choices": [
            "random/CHEMBL214_Ki",
            "random/CHEMBL234_Ki",
            "random/CHEMBL233_Ki",
            "random/CHEMBL287_Ki",
            "random/CHEMBL2147_Ki",
            "distant/DRD3",
            "distant/FEN1",
            "distant/MAP4K2",
            "distant/PIN1",
            "distant/VDR",
        ],
        "type": str,
    },
    "--balance-classes": {
        "help": "Whether to balance classes in classification",
        "choices": [0, 1],
        "type": int,
    },
    "--n-experiments": {
        "help": "How many hyper-parameter experiments to run",
        "type": int,
        "default": 3,
    },
    "--starting-experiment": {
        "help": "Which experiment to start from",
        "type": int,
        "default": 0,
    },
}


def add_run_arguments(argument_list: List[str]):
    parser = argparse.ArgumentParser()

    for arg_name in argument_list:
        if arg_name not in __ARGUMENTS:
            raise ValueError(f"Invalid argument name: {arg_name}")
        parser.add_argument(arg_name, **__ARGUMENTS[arg_name])

    args, invalids = parser.parse_known_args()
    if len(invalids) > 0:
        raise ValueError(f"Invalid terminal arguments: {invalids}")
    return args
