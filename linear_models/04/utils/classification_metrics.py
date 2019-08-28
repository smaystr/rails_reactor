import numpy as np
from typing import Callable, List
from prettytable import PrettyTable

metrics_naming = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall (Sensitivity)",
    "specificity": "Specificity",
    "f1": "F1 Metric",
    "roc_auc": "Roc-Auc Metric",
    "log_loss": "Log-loss",
}


def _calculate_true_positive(y: np.ndarray, y_pred: np.ndarray) -> int:
    return np.intersect1d(np.where(y == 1), np.where(y_pred == 1)).size


def _calculate_false_positive(y: np.ndarray, y_pred: np.ndarray) -> int:
    return np.intersect1d(np.where(y == 0), np.where(y_pred == 1)).size


def _calculate_true_negative(y: np.ndarray, y_pred: np.ndarray) -> int:
    return np.intersect1d(np.where(y == 0), np.where(y_pred == 0)).size


def _calculate_false_negative(y: np.ndarray, y_pred: np.ndarray) -> int:
    return np.intersect1d(np.where(y == 1), np.where(y_pred == 0)).size


def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    num_examples = y.shape[0]
    tp = _calculate_true_positive(y, y_pred)
    tn = _calculate_true_negative(y, y_pred)
    return (tp + tn) / num_examples


def precision(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    tp = _calculate_true_positive(y, y_pred)
    fp = _calculate_false_positive(y, y_pred)
    return tp / (tp + fp)


def recall(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    tp = _calculate_true_positive(y, y_pred)
    fn = _calculate_false_negative(y, y_pred)
    return tp / (tp + fn)


def specificity(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    tn = _calculate_true_negative(y, y_pred)
    fp = _calculate_false_positive(y, y_pred)
    return tn / (tn + fp)


def roc_auc(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    return (recall(y, y_pred) + specificity(y, y_pred)) / 2


def f1(y: np.ndarray, y_pred: np.ndarray, beta: float = 1.0) -> float:
    assert len(y) == len(y_pred)
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)
    return (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)


def log_loss(y: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    assert len(y) == len(y_pred)
    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()


def confusion_matrix(
    y: np.ndarray, y_pred: np.ndarray, print_result: bool = True
) -> np.ndarray:
    assert len(y) == len(y_pred)
    tp = _calculate_true_positive(y, y_pred)
    tn = _calculate_true_negative(y, y_pred)
    fn = _calculate_false_negative(y, y_pred)
    fp = _calculate_false_positive(y, y_pred)
    if print_result:
        confusion_matrix_table = PrettyTable()
        confusion_matrix_table.field_names = [
            "",
            "Actual Positive (1)",
            "Actual Negative (2)",
        ]
        confusion_matrix_table.add_row(["Predicted Positive (1)", tp, fp])
        confusion_matrix_table.add_row(["Predicted Negative (0)", fn, tn])
        print(confusion_matrix_table)
    return np.array(((tp, fp), (fn, tn)))


def get_metric(metric_type: str) -> Callable:
    if metric_type in metrics_naming:
        return eval(metric_type)
    else:
        raise NameError("No such metric")


def all_metrics(y: np.ndarray, y_pred: np.ndarray, out_print: bool = True) -> List:
    scores = []
    for metric, name in metrics_naming.items():
        scores.append([name, eval(metric)(y, y_pred)])
    if out_print:
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for i in range(len(scores)):
            table.add_row(scores[i])
        print(table)
    return scores
