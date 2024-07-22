import pandas as pd
from tensorflow import keras

from library.training.metrics import evaluate_predictions


def get_X_y(
    dataset_name: str,
    representation_name: str,
    setup_idx: int,
    fold: str,
    maxlen: int,
):
    df = pd.read_csv(f"./data/{dataset_name}/setup_{setup_idx}/{fold}.csv")
    X = df[f"encoded_{representation_name}"].values
    X = [eval(s) for s in X]
    y = df["y"].values

    X = keras.preprocessing.sequence.pad_sequences(
        X, padding="post", maxlen=maxlen, value=0
    )
    return X, y


def train_predictor(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate,
    batch_size,
    is_regressor,
    balance_classes,
):
    optimizer = keras.optimizers.get("adam")
    optimizer.learning_rate.assign(learning_rate)
    loss = "mean_squared_error" if is_regressor else "binary_crossentropy"
    metrics = [keras.metrics.RootMeanSquaredError()] if is_regressor else ["accuracy"]
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
    if balance_classes:
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


def evaluate_predictor(model, X_test, y_test, is_regressor):
    predictions = model.predict(X_test)
    if is_regressor:
        return evaluate_predictions(
            y_test, predictions, metrics=["rmse", "r2", "ci", "mse"]
        )

    predictions = [1 if p > 0.5 else 0 for p in predictions]
    return evaluate_predictions(
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
