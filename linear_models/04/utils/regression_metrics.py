import numpy as np
from typing import Callable, List
from prettytable import PrettyTable

metrics_naming = {
    "mse": "MSE (Mean Squared Error)",
    "rmse": "RMSE (Root Mean Squared Error)",
    "mae": "MAE (Mean Absolute Error)",
    "r_2": "R-Squared (Coefficient of Determination)",
    "mpe": "MPE (Mean Percentage Error)",
    "mspe": "MSPE (Mean Square Percentage Error)",
    "mape": "MAPE (Mean Absolute Percentage Error)",
}


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    return np.square(y - y_pred).mean()


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    return np.sqrt(mse(y, y_pred))


def mae(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    return np.abs(y - y_pred).mean()


def r_2(y: np.ndarray, y_pred: np.ndarray) -> float:
    squared_error_regr = np.square(y - y_pred).mean()
    squared_error_y_mean = np.square(y - y.mean()).mean()
    return 1 - (squared_error_regr / squared_error_y_mean)


def mpe(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    return 100 * ((y - y_pred) / y).mean()


def mspe(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    return 100 * np.square((y - y_pred) / y).mean()


def mape(y: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y) == len(y_pred)
    return 100 * np.abs((y - y_pred) / y).mean()


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
