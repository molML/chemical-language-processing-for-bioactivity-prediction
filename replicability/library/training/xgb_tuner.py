# %%
import datetime
import json
import os
import time

import numpy as np
import pandas as pd

from emb_library.models.xgboost import XGBoost
from emb_library.training.hpspaces import get_sampled_hp_space
from emb_library.training.training import evaluate_predictor


def read_X_y(dataset_name: str, setup_idx: int, fold: str):
    df = pd.read_csv(f"./data/{dataset_name}/setup_{setup_idx}/{fold}.csv")
    smiles = df["cleaned_smiles"].values.tolist()
    y = df["y"].values.tolist()
    return smiles, y


class XGBTuner:
    def __init__(
        self,
        dataset_name: str,
        is_regression: bool,
        balance_classes: bool,
        n_trials: int = None,
        start_trial_idx: int = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.n_trials = n_trials
        self.start_trial_idx = start_trial_idx
        self.is_regression = is_regression
        self.balance_classes = balance_classes
        model_name = "xgboost"
        if self.balance_classes:
            model_name = "xgboost_balanced"
        self.base_saving_dir = f"./models/{dataset_name}/ecfp/{model_name}"

    def run(self):
        hp_combinations = get_sampled_hp_space("smiles", "onehot", "xgboost")
        if self.n_trials is not None and self.start_trial_idx is not None:
            hp_combinations = hp_combinations[
                self.start_trial_idx : self.start_trial_idx + self.n_trials
            ]
        results_path = f"{self.base_saving_dir}_results.json"
        if os.path.exists(results_path):
            with open(results_path, "r") as results_file:
                done_combinations = json.load(results_file)
        else:
            done_combinations = dict()
        start = time.time()
        for combination_idx, hp_combination in enumerate(hp_combinations):
            if f"combination-{combination_idx}" in done_combinations:
                print(
                    f"Combination {combination_idx} already exists, skipping",
                )
                continue

            val_scores = list()
            test_scores = list()
            for setup_idx in range(5):
                X_train, y_train = read_X_y(
                    self.dataset_name,
                    setup_idx,
                    "train",
                )
                X_val, y_val = read_X_y(
                    self.dataset_name,
                    setup_idx,
                    "valid",
                )
                X_test, y_test = read_X_y(
                    self.dataset_name,
                    setup_idx,
                    "test",
                )
                hp_combination["is_regressor"] = self.is_regression
                hp_combination["balance_classes"] = self.balance_classes
                predictor = XGBoost(**hp_combination)
                predictor.fit(X_train, y_train, X_val, y_val)
                test_results = evaluate_predictor(
                    predictor, X_test, y_test, self.is_regression
                )
                test_scores.append(test_results)
                val_loss = predictor.model.evals_result()
                if self.is_regression:
                    val_scores.append(np.min(val_loss["validation_0"]["rmse"]))
                else:
                    val_scores.append(np.min(val_loss["validation_0"]["logloss"]))
                del predictor

            if combination_idx + 1 % 5 == 0:
                elapsed_time = time.time() - start
                print(
                    "Combination idx:",
                    combination_idx,
                    " Total time elapsed:",
                    str(datetime.timedelta(seconds=(elapsed_time))),
                )

            combination_dump = dict()
            combination_dump["hps"] = hp_combination
            combination_dump["misc"] = dict()
            # metric_name = "rmse" if self.is_regression else "logloss"
            combination_dump["misc"][f"val_loss_mean"] = np.mean(val_scores)
            combination_dump["misc"][f"val_loss_std"] = np.std(val_scores)
            for metric in test_scores[0].keys():
                combination_dump["misc"][f"test_loss_mean"] = np.mean(
                    [test_score[metric] for test_score in test_scores]
                )
                combination_dump["misc"][f"test_loss_std"] = np.std(
                    [test_score[metric] for test_score in test_scores]
                )

            combination_dump["misc"]["test_scores"] = {
                "setup-{}".format(setup_idx): test_score
                for setup_idx, test_score in enumerate(test_scores)
            }

            done_combinations[f"combination-{combination_idx}"] = combination_dump

            with open(results_path, "w") as results_file:
                json.dump(done_combinations, results_file, indent=4)
