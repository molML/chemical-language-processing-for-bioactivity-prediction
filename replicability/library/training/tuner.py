# %%
import datetime
import json
import os
import time

import numpy as np
import tensorflow as tf

from emb_library.models import get_predictor
from emb_library.training import get_X_y, train_predictor
from emb_library.training.hpspaces import get_sampled_hp_space
from emb_library.training.training import evaluate_predictor


class HyperParameterTuner:
    def __init__(
        self,
        representation_name: str,
        embedding_name: str,
        model_name: str,
        dataset_name: str,
        is_regression: bool,
        balance_classes: bool,
        n_trials: int = None,
        start_trial_idx: int = None,
    ) -> None:
        self.representation_name = representation_name
        self.embedding_name = embedding_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.is_regression = is_regression
        self.balance_classes = balance_classes
        self.n_trials = n_trials
        self.start_trial_idx = start_trial_idx
        model_name = self.model_name
        if self.balance_classes:
            model_name = f"{self.model_name}_balanced"
        self.base_saving_dir = f"./models/{dataset_name}/{representation_name}/{embedding_name}/{model_name}"

    def run(self):
        hp_combinations = get_sampled_hp_space(
            self.representation_name, self.embedding_name, self.model_name
        )
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

            if self.embedding_name == "onehot":
                hp_combination["embedding_dim"] = hp_combination["vocab_size"]
            val_scores = list()
            test_scores = list()
            for setup_idx in range(5):
                print(
                    f"Running setup {setup_idx} of combination {combination_idx} / {len(hp_combinations)}"
                )
                tf.keras.backend.clear_session()
                tf.random.set_seed(42)

                X_train, y_train = get_X_y(
                    self.dataset_name,
                    self.representation_name,
                    setup_idx,
                    "train",
                    maxlen=hp_combination["maxlen"],
                )
                X_val, y_val = get_X_y(
                    self.dataset_name,
                    self.representation_name,
                    setup_idx,
                    "valid",
                    maxlen=hp_combination["maxlen"],
                )
                X_test, y_test = get_X_y(
                    self.dataset_name,
                    self.representation_name,
                    setup_idx,
                    "test",
                    maxlen=hp_combination["maxlen"],
                )
                hp_combination["embedding_type"] = self.embedding_name
                hp_combination["is_regressor"] = self.is_regression
                predictor = get_predictor(self.model_name, hp_combination)
                history = train_predictor(
                    predictor,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    hp_combination["learning_rate"],
                    hp_combination["batch_size"],
                    self.is_regression,
                    self.balance_classes,
                )
                test_scores.append(
                    evaluate_predictor(predictor, X_test, y_test, self.is_regression)
                )
                val_scores.append(np.min(history["val_loss"]))
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
            combination_dump["misc"]["val_loss_mean"] = np.mean(val_scores)
            combination_dump["misc"]["val_loss_std"] = np.std(val_scores)
            for metric in test_scores[0].keys():
                combination_dump["misc"][f"test_{metric}_mean"] = np.mean(
                    [test_score[metric] for test_score in test_scores]
                )
                combination_dump["misc"][f"test_{metric}_std"] = np.std(
                    [test_score[metric] for test_score in test_scores]
                )

            combination_dump["misc"]["test_scores"] = {
                "setup-{}".format(setup_idx): test_score
                for setup_idx, test_score in enumerate(test_scores)
            }

            done_combinations[f"combination-{combination_idx}"] = combination_dump

            with open(results_path, "w") as results_file:
                json.dump(done_combinations, results_file, indent=4)
