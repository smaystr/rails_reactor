import sys
import argparse
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple, Union

# hw_3
from hw_3.models.linear_regression import LinearReg
from hw_3.models.logistic_regression import LogisticReg
from hw_3.utils.dataset_processing import normalize_data
from hw_3.utils.classification_metrics import all_metrics as class_metrics
from hw_3.utils.regression_metrics import all_metrics as regr_metrics

from model_selection.train_test_split import train_test_split
from model_selection.k_fold import KFold
from model_selection.leave_one_out import LeaveOneOut
from model_selection.grid_search import GridSearch
from model_selection.random_search import RandomizedSearch


def process_regression(X: pd.DataFrame, y: pd.DataFrame, train_size: Union[int, None] = None) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable, Callable, Callable]:
    X = pd.get_dummies(X, columns=["sex", "smoker", "region"]).to_numpy()
    X = normalize_data(X)
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    scoring = min
    return X_train, X_test, y_train, y_test, LinearReg, scoring, regr_metrics


def process_classification(X: pd.DataFrame, y: pd.DataFrame, train_size: Union[int, None] = None) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable, Callable, Callable]:
    X = pd.get_dummies(X, columns=["sex", "cp", "restecg", "slope", "ca", "thal"]).to_numpy()
    X = normalize_data(X, columns=[0, 1, 2, 4, 6])
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y)
    scoring = max
    return X_train, X_test, y_train, y_test, LogisticReg, scoring, class_metrics


def main(
    train_path: str,
    test_path: str,
    target_variable: str,
    task_type: int,
    output_model_path: Path,
    split_type: int,
    split_size: int,
    time_series: str,
    fitting_type: int,
) -> None:
    # python utility.py --train_path './datasets/heart_train.csv' --test_path './datasets/heart_test.csv' --target 'target' --task 1 --output 'results' --split_type 1 --split_size 90 --fitting_type 1
    # python utility.py --train_path './datasets/insurance_train.csv' --test_path './datasets/insurance_test.csv' --target 'charges' --task 2 --output 'results' --split_type 1 --split_size 90 --fitting_type 1

    task_types = {
        1: process_classification,
        2: process_regression
    }
    cross_validation_algorithms = {
        1: KFold,
        2: LeaveOneOut
    }
    fitting_algorithms = {
        1: GridSearch,
        2: RandomizedSearch
    }

    dataset = pd.read_csv(train_path)
    if test_path:
        test_dataset = pd.read_csv(test_path)
        dataset = pd.concat((dataset, test_dataset))
    X, y = dataset.drop(columns=target_variable), dataset[target_variable]

    if task_type in task_types:
        X_train, X_test, y_train, y_test, model, scoring_func, metrics_func = task_types[task_type](X, y, split_size)
    else:
        raise Exception(f"No such task with index: {task_type}")

    if split_type in cross_validation_algorithms:
        cv = cross_validation_algorithms[split_type]
    else:
        raise Exception(f"No such cross-validation with index: {split_type}")

    if fitting_type in fitting_algorithms:
        fitting_func = fitting_algorithms[fitting_type]
    else:
        raise Exception(f"No such fitting algorithm with index: {fitting_type}")

    with open(Path('parameter_candidates.json'), 'r', encoding="utf-8") as parameter_file:
        parameter_candidates = json.load(parameter_file)

    model_selection = fitting_func(estimator=model, param_grid=parameter_candidates, cv=cv, scoring=scoring_func)
    model_selection.fit(X_train, y_train)
    print(f"Best score: {model_selection.best_score_}")
    print(f"Best params: {model_selection.best_params_}")
    print(f"Search time: {model_selection.search_time_}")
    print(f"Refit time: {model_selection.refit_time_}")

    y_pred = model_selection.predict(X_test)
    score = model_selection.score(X_test, y_test)
    print(f"Test score: {score}")
    scores = metrics_func(y_test, y_pred, out_print=True)
    scores_str = [f"{sc[0]}: {sc[1]}\n" for sc in scores]
    out_metrics = (
        f"Program runned with command: {' '.join(sys.argv)}\n\n"
        f"Search time: {model_selection.search_time_}\n"
        f"Refit time: {model_selection.refit_time_}\n\n"
        f"Scores:\n"
        f"Best fitting score: {model_selection.best_score_}\n"
        f"{''.join(scores_str)}"
    )
    task_str = "Classification" if task_type == 1 else "Regression"
    weights = model_selection.best_estimator_.theta
    out_model_info = (
        f"Program runned with command: {' '.join(sys.argv)}\n\n"
        f"Task: {task_str}\n"
        f"Best fitting params: {model_selection.best_params_}\n"
        f"Weights:\n{weights}"
    )
    output_model_path.mkdir(mode=0o777, exist_ok=True)
    with open(output_model_path / "metrics.txt", "w", encoding="utf-8") as metrics_file:
        metrics_file.write(out_metrics)
    with open(output_model_path / "model.txt", "w", encoding="utf-8") as model_info_file:
        model_info_file.write(out_model_info)
    joblib.dump(model_selection.best_estimator_, output_model_path / "model.sav")

    model_from_file = joblib.load(output_model_path / 'model.sav')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for end2end model training lifecycle"
    )

    parser.add_argument(
        "--train_path",
        help="path/url to download train dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_path",
        help="path/url to download test dataset",
        type=str,
        required=False,
        default='',
    )
    parser.add_argument(
        "--target", help="target variable name", type=str, required=True
    )
    parser.add_argument(
        "--task_type",
        help="classification - 1, regression - 2",
        type=int,
        required=True,
        choices=[1, 2],
    )
    parser.add_argument(
        "--output", help="path for output model", type=Path, required=True
    )
    parser.add_argument(
        "--split_type",
        help="cross-validation: k-fold - 1, leave one-out - 2",
        type=int,
        required=False,
        default=1,
        choices=[1, 2],
    )
    parser.add_argument(
        "--split_size",
        help="validation split size",
        type=int,
        required=False,
        default=80,
        choices=range(0, 101),
    )
    parser.add_argument(
        "--time_series",
        help="parameter for specifying time series column to perform time series validation",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "--fitting_type",
        help="parameter for hyperparameter fitting algorithm: grid search - 1, random search - 2",
        type=int,
        required=False,
        default=1,
        choices=[1, 2],
    )
    args = parser.parse_args()

    main(
        args.train_path,
        args.test_path,
        args.target,
        args.task_type,
        args.output,
        args.split_type,
        args.split_size,
        args.time_series,
        args.fitting_type,
    )
