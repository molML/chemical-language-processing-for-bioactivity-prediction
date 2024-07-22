from typing import List, Union

import numpy as np
import xgboost
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_batch_to_fp(smiles_batch: List[str]):
    mols = [Chem.MolFromSmiles(s) for s in smiles_batch]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    return np.array([list(fp) for fp in fps])


class XGBoost:
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
        n_jobs: int,
        combination_idx: int,
        maxlen: int,
        vocab_size: int,
        is_regressor: bool,
        balance_classes: bool,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.combination_idx = combination_idx
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.is_regressor = is_regressor
        self.balance_classes = balance_classes

    def fit(
        self,
        training_smiles: List[str],
        y_train: List[Union[float, int]],
        val_smiles: List[str],
        y_val: List[Union[float, int]],
    ):
        X = smiles_batch_to_fp(training_smiles)

        scale_pos_weight = 1
        if self.is_regressor:
            XGBPredictor = xgboost.XGBRegressor
            objective = "reg:squarederror"
            eval_metric = "rmse"
        else:
            XGBPredictor = xgboost.XGBClassifier
            objective = "binary:logistic"
            eval_metric = "logloss"
            if self.balance_classes:
                scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

        self.model = XGBPredictor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=1,
            scale_pos_weight=scale_pos_weight,
            objective=objective,
            n_jobs=self.n_jobs,
            eval_metric=eval_metric,
            callbacks=[
                xgboost.callback.EarlyStopping(
                    rounds=5, min_delta=1e-5, metric_name=eval_metric
                )
            ],
        )
        self.model.fit(
            X,
            y_train,
            eval_set=[(smiles_batch_to_fp(val_smiles), y_val)],
            verbose=False,
        )

    def predict(self, test_smiles: List[str]):
        X = smiles_batch_to_fp(test_smiles)
        return self.model.predict(X)
