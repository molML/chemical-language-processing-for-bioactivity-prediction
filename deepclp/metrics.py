from itertools import combinations
from typing import Dict, List

from sklearn import metrics


def ci(gold_truths: List[float], predictions: List[float]) -> float:
    """Computes concordance index (CI) between the expected values and predictions.
    See [Gönen and Heller (2005)](https://www.jstor.org/stable/20441249) for the details of the metric.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Concordance index.
    """
    gold_combs, pred_combs = combinations(gold_truths, 2), combinations(predictions, 2)
    nominator, denominator = 0, 0
    for (g1, g2), (p1, p2) in zip(gold_combs, pred_combs):
        if g2 > g1:
            nominator = nominator + 1 * (p2 > p1) + 0.5 * (p2 == p1)
            denominator = denominator + 1

    return float(nominator / denominator)


def mse(gold_truths: List[float], predictions: List[float]) -> float:
    """Computes mean squared error between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(metrics.mean_squared_error(gold_truths, predictions))


def rmse(gold_truths: List[float], predictions: List[float]) -> float:
    """Computes root mean squared error between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(metrics.root_mean_squared_error(gold_truths, predictions))


def r2(gold_truths: List[float], predictions: List[float]) -> float:
    """Compute $R^2$ (coefficient of determinant) between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        $R^2$ (coefficient of determinant) score.
    """
    return float(metrics.r2_score(gold_truths, predictions))


def accuracy(gold_truths: List[float], predictions: List[float]) -> float:
    """Compute accuracy between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Accuracy score.
    """
    return metrics.accuracy_score(gold_truths, predictions)


def precision(gold_truths: List[float], predictions: List[float]) -> float:
    """Compute precision between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Precision score.
    """
    return metrics.precision_score(
        gold_truths,
        predictions,
    )


def recall(gold_truths: List[float], predictions: List[float]) -> float:
    """Compute recall between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Recall score.
    """
    return metrics.recall_score(
        gold_truths,
        predictions,
    )


def balanced_accuracy(gold_truths: List[float], predictions: List[float]) -> float:
    """Compute balanced accuracy between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        Balanced accuracy score.
    """
    return metrics.balanced_accuracy_score(
        gold_truths,
        predictions,
        adjusted=False,
    )


def f1_score(gold_truths: List[float], predictions: List[float]) -> float:
    """Compute F1 score between expected and predicted values.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.

    Returns
    -------
    float
        F1 score.
    """
    return metrics.f1_score(
        gold_truths,
        predictions,
    )


def evaluate_predictions(
    gold_truths: List[float], predictions: List[float], metrics: List[str] = None
) -> Dict[str, float]:
    """Computes multiple metrics with a single call for convenience.

    Parameters
    ----------
    gold_truths : List[float]
        The gold labels in the dataset.
    predictions : List[float]
        Predictions of a model.
    metrics : List[str]
        Name of the evaluation metrics to compute. Possible values are: `{"ci", "r2", "rmse", "mse"}`.
        All metrics are computed if no value is provided.

    Returns
    -------
    Dict[str,float]
        A dictionary that maps each metric name to the computed value.
    """
    if metrics is None:
        metrics = ["ci", "r2", "rmse", "mse"]

    metrics = [metric.lower() for metric in metrics]
    name_to_fn = {
        "ci": ci,
        "r2": r2,
        "rmse": rmse,
        "mse": mse,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1_score,
    }
    return {metric: name_to_fn[metric](gold_truths, predictions) for metric in metrics}
