# %%
import time
import tensorflow as tf

from emb_library.training import tuner
from runners.setup import add_run_arguments

if __name__ == "__main__":
    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
        # Invalid device or cannot modify virtual devices onc

    args = add_run_arguments(
        [
            "--representation-name",
            "--embedding-name",
            "--model-name",
            "--dataset-name",
            "--balance-classes",
        ]
    )

    representation_name = args.representation_name
    embedding_name = args.embedding_name
    model_name = args.model_name
    dataset_name = args.dataset_name
    balance_classes = bool(int(args.balance_classes))

    is_regression = "CHEMBL" in dataset_name
    print("Started", representation_name, embedding_name, model_name, dataset_name)
    start = time.time()
    hp_tuner = tuner.HyperParameterTuner(
        representation_name=representation_name,
        embedding_name=embedding_name,
        model_name=model_name,
        dataset_name=dataset_name,
        is_regression=is_regression,
        balance_classes=balance_classes,
    )
    hp_tuner.run()
    print(
        "All program took",
        time.time() - start,
        "seconds",
        representation_name,
        embedding_name,
        model_name,
        dataset_name,
    )
