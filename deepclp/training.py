import json
from typing import Union

import keras
import pandas as pd

from deepclp import metrics, sequence_utils


def csv_to_matrix(
    csv_path: str,
    representation_name: str,
    maxlen: int,
) -> tuple[list[list[int]], list[Union[int, float]]]:
    """Converts a CSV file to matrices for training and prediction.

    Args:
        csv_path (str): Path to the CSV file.
        representation_name (str): Name of the molecule representation.
        maxlen (int): Maximum length of the sequences.

    Returns:
        tuple[list[list[int]], list[int | float]]: Input and output matrices, X and y.
    """
    df = pd.read_csv(csv_path)
    X = df["molecule"].values.tolist()
    if representation_name == "smiles":
        with open("data/smiles_vocab.json") as f:
            smiles_vocab = json.load(f)
        X = [sequence_utils.smiles_label_encoding(s, smiles_vocab) for s in X]
    elif representation_name == "selfies":
        with open("data/selfies_vocab.json") as f:
            selfies_vocab = json.load(f)
        X = [sequence_utils.selfies_label_encoding(s, selfies_vocab) for s in X]
    else:
        raise ValueError(
            f"Invalid representation name: {representation_name}. Choose from {'smiles', 'selfies'}."
        )

    y = df["label"].values
    X = keras.preprocessing.sequence.pad_sequences(
        X, padding="post", maxlen=maxlen, value=0
    )
    return X, y


def train_predictor(
    model: keras.Model,
    X_train: list[list[int]],
    y_train: list[Union[int, float]],
    X_val: list[list[int]],
    y_val: list[Union[int, float]],
    learning_rate: float,
    batch_size: int,
    balance_loss: bool,
) -> dict[str, list[float]]:
    """Train a predictor.

    Args:
        model (keras.Model): Model to train.
        X_train (list[list[int]]): Training input matrix.
        y_train (list[int | float]): Training output vector.
        X_val (list[list[int]]): Validation input matrix.
        y_val (list[int | float]): Validation output vector.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        balance_classes (bool): Whether to balance the classes.

    Returns:
        dict[str, list[float]]: Training history.
    """
    optimizer = keras.optimizers.get("adam")
    optimizer.learning_rate.assign(learning_rate)
    loss = "binary_crossentropy" if model.is_classification else "mean_squared_error"
    metrics = (
        ["accuracy"]
        if model.is_classification
        else [keras.metrics.RootMeanSquaredError()]
    )

    model.compile(
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
    )
    model(X_val)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-5,
            restore_best_weights=True,
        ),
    ]
    class_weights = None
    if balance_loss:
        if not model.is_classification:
            raise ValueError("Loss balancing is only supported for classification.")
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        class_weights = {0: 1, 1: pos_weight}

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=500,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    return history.history


def evaluate_predictor(
    model: keras.Model,
    X_test: list[list[int]],
    y_test: list[Union[int, float]],
) -> dict[str, float]:
    """Evaluate a predictor.

    Args:
        model (keras.Model): Model to evaluate.
        X_test (list[list[int]]): Test input matrix.
        y_test (list[int | float]): Test output vector.

    Returns:
        dict[str, float]: Evaluation metrics.
    """
    predictions = model.predict(X_test)
    if model.is_classification:
        predictions = [1 if p > 0.5 else 0 for p in predictions]
        return metrics.evaluate_predictions(
            y_test,
            predictions,
            metrics=[
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
            ],
        )

    return metrics.evaluate_predictions(
        y_test, predictions, metrics=["rmse", "r2", "ci", "mse"]
    )
