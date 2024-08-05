# %%
import time

from emb_library.training import xgb_tuner
from runners.setup import add_run_arguments

if __name__ == "__main__":
    args = add_run_arguments(
        [
            "--dataset-name",
            "--balance-classes",
        ]
    )

    dataset_name = args.dataset_name
    balance_classes = bool(int(args.balance_classes))
    is_regression = "CHEMBL" in dataset_name
    print("Started")
    start = time.time()
    hp_tuner = xgb_tuner.XGBTuner(
        dataset_name=dataset_name,
        is_regression=is_regression,
        balance_classes=balance_classes,
    )
    hp_tuner.run()
    print("All program took", time.time() - start)
