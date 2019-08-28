import torch
from typing import Callable, List
from prettytable import PrettyTable

classification_metrics = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall (Sensitivity)",
    "specificity": "Specificity",
    "f1": "F1 Metric",
    "roc_auc": "Roc-Auc Metric",
    "log_loss": "Log-loss",
}

regression_metrics = {
    "mse": "MSE (Mean Squared Error)",
    "rmse": "RMSE (Root Mean Squared Error)",
    "mae": "MAE (Mean Absolute Error)",
    "r_2": "R-Squared (Coefficient of Determination)",
    "mpe": "MPE (Mean Percentage Error)",
    "mspe": "MSPE (Mean Square Percentage Error)",
    "mape": "MAPE (Mean Absolute Percentage Error)",
}

metrics = {}
metrics.update(classification_metrics)
metrics.update(regression_metrics)


def check_shapes(func):
    def wrapper(y: torch.Tensor, y_pred: torch.Tensor, *args, **kwargs):
        assert y.shape == y_pred.shape
        return func(y, y_pred, *args, **kwargs)
    return wrapper


"""         Classification Metrics      """


def _torch_intersect_1d(t1: torch.Tensor, t2: torch.Tensor) -> int:
    count_intersect = 0
    for elem in t2:
        if torch.nonzero(t1 == elem).size()[0] != 0:
            count_intersect += 1
    return count_intersect


def _calculate_true_positive(y: torch.Tensor, y_pred: torch.Tensor) -> int:
    return _torch_intersect_1d(torch.nonzero(y == 1), torch.nonzero(y_pred == 1))


def _calculate_false_positive(y: torch.Tensor, y_pred: torch.Tensor) -> int:
    return _torch_intersect_1d(torch.nonzero(y == 0), torch.nonzero(y_pred == 1))


def _calculate_true_negative(y: torch.Tensor, y_pred: torch.Tensor) -> int:
    return _torch_intersect_1d(torch.nonzero(y == 0), torch.nonzero(y_pred == 0))


def _calculate_false_negative(y: torch.Tensor, y_pred: torch.Tensor) -> int:
    return _torch_intersect_1d(torch.nonzero(y == 1), torch.nonzero(y_pred == 0))


@check_shapes
def accuracy(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    num_examples = y.shape[0]
    tp = _calculate_true_positive(y, y_pred)
    tn = _calculate_true_negative(y, y_pred)
    return (tp + tn) / num_examples


@check_shapes
def precision(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = _calculate_true_positive(y, y_pred)
    fp = _calculate_false_positive(y, y_pred)
    return tp / (tp + fp)


@check_shapes
def recall(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = _calculate_true_positive(y, y_pred)
    fn = _calculate_false_negative(y, y_pred)
    return tp / (tp + fn)


@check_shapes
def specificity(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    tn = _calculate_true_negative(y, y_pred)
    fp = _calculate_false_positive(y, y_pred)
    return tn / (tn + fp)


@check_shapes
def roc_auc(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (recall(y, y_pred) + specificity(y, y_pred)) / 2


@check_shapes
def f1(y: torch.Tensor, y_pred: torch.Tensor, beta: float = 1.0) -> float:
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)
    return (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)


@check_shapes
def log_loss(y: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> float:
    # Clipping
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    return -(y * torch.log(y_pred) + (- y + 1) * torch.log(- y_pred + 1)).mean().item()


@check_shapes
def confusion_matrix(
    y: torch.Tensor, y_pred: torch.Tensor, print_result: bool = True
) -> tuple:
    tp = _calculate_true_positive(y, y_pred)
    tn = _calculate_true_negative(y, y_pred)
    fn = _calculate_false_negative(y, y_pred)
    fp = _calculate_false_positive(y, y_pred)
    if print_result:
        confusion_matrix_table = PrettyTable()
        confusion_matrix_table.field_names = [
            "",
            "Actual Positive (1)",
            "Actual Negative (0)",
        ]
        confusion_matrix_table.add_row(["Predicted Positive (1)", tp, fp])
        confusion_matrix_table.add_row(["Predicted Negative (0)", fn, tn])
        print(confusion_matrix_table)
    return ((tp, fp), (fn, tn))


"""         Regression Metrics      """


@check_shapes
def mse(y: torch.tensor, y_pred: torch.tensor) -> float:
    return ((y - y_pred) ** 2).mean().item()


@check_shapes
def rmse(y: torch.tensor, y_pred: torch.tensor) -> float:
    return mse(y, y_pred) ** 0.5


@check_shapes
def mae(y: torch.tensor, y_pred: torch.tensor) -> float:
    return torch.abs(y - y_pred).mean().item()


@check_shapes
def r_2(y: torch.tensor, y_pred: torch.tensor) -> float:
    squared_error_regr = mse(y, y_pred)
    squared_error_y_mean = ((y - torch.mean(y)) ** 2).mean().item()
    return 1 - (squared_error_regr / squared_error_y_mean)


@check_shapes
def mpe(y: torch.tensor, y_pred: torch.tensor) -> float:
    return 100 * ((y - y_pred) / y).mean().item()


@check_shapes
def mspe(y: torch.tensor, y_pred: torch.tensor) -> float:
    return 100 * (((y - y_pred) / y) ** 2).mean().item()


@check_shapes
def mape(y: torch.tensor, y_pred: torch.tensor) -> float:
    return 100 * torch.abs((y - y_pred) / y).mean().item()


"""         Utils      """


def get_metric(metric_type: str) -> Callable:
    if metric_type in metrics:
        return eval(metric_type)
    else:
        raise NameError("No such metric")


def all_metrics(y: torch.Tensor, y_pred: torch.Tensor, metric_type: int, out_print: bool = True) -> List:
    # 1 - classification, 2 - regression
    metrics_naming = classification_metrics if metric_type == 1 else regression_metrics
    scores = []
    for metric, name in metrics_naming.items():
        scores.append([name, eval(metric)(y, y_pred)])
    if out_print:
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for i in range(len(scores)):
            table.add_row(scores[i])
        print(table)
        if metric_type == 'classification':
            confusion_matrix(y, y_pred)
    return scores
