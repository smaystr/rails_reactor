import sys
import torch
import json
import joblib
import argparse
import pandas as pd
from pathlib import Path
from torch.autograd import Variable

from scripts import script_1, script_2
from utils.dataset_processing import normalize_data, add_ones_column
from utils.metrics import all_metrics


def process_regression(train: pd.DataFrame, test: pd.DataFrame, device: torch.device):
    train = pd.get_dummies(train, columns=["sex", "smoker", "region"])
    test = pd.get_dummies(test, columns=["sex", "smoker", "region"])
    X_train, y_train = (
        torch.tensor(train.drop(columns="charges").values, device=device),
        torch.tensor(train["charges"].values, device=device).float(),
    )
    X_test, y_test = (
        torch.tensor(test.drop(columns="charges").values, device=device),
        torch.tensor(test["charges"].values, device=device).float(),
    )
    return X_train, X_test, y_train, y_test


def process_classification(train: pd.DataFrame, test: pd.DataFrame, device: torch.device):
    train = pd.get_dummies(train, columns=["sex", "cp", "restecg", "slope", "ca", "thal"])
    test = pd.get_dummies(test, columns=["sex", "cp", "restecg", "slope", "ca", "thal"])
    X_train, y_train = (
        torch.tensor(train.drop(columns=["target", "restecg_2", "ca_4"]).values, device=device),
        torch.tensor(train["target"].values, device=device).float(),
    )
    X_test, y_test = (
        torch.tensor(test.drop(columns="target").values, device=device),
        torch.tensor(test["target"].values, device=device).float(),
    )
    return X_train, X_test, y_train, y_test


def main(script: int, task: int, config_path: Path, on_gpu: bool, output_path: Path) -> None:
    device = torch.device("cuda") if on_gpu and torch.cuda.is_available() else torch.device("cpu")

    with open(config_path, 'r', encoding="utf-8") as config_file:
        configs = json.load(config_file)

    train = pd.read_csv(Path(configs['train_path']))
    test = pd.read_csv(Path(configs['test_path']))

    if task == 1:
        X_train, X_test, y_train, y_test = process_classification(train, test, device)
    elif task == 2:
        X_train, X_test, y_train, y_test = process_regression(train, test, device)
    else:
        raise Exception(f"No such type of tasks: {task}")

    X_train = add_ones_column(X_train, device)
    X_test = add_ones_column(X_test, device)
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    if script == 1:
        model, out_model_info = script_1(X_train, y_train, task, configs['model_params'], device)
        y_pred = model.predict(X_test)
    elif script == 2:
        model, out_model_info = script_2(X_train, y_train, task, configs['model_params'], device)
        y_pred = model(Variable(X_test.float())).reshape((-1,)).round()
    else:
        raise Exception(f"No such type of scripts: {script}")

    scores = all_metrics(y_test, y_pred, task, out_print=False)
    scores_str = [f"- **{sc[0]}**: *{sc[1]}*\n" for sc in scores]
    task_str = "classification" if task == 1 else "regression"
    out_metrics = (
        f"**Program started with command**:\n```\n{' '.join(sys.argv)}\n```\n"
        f"**Task**: *{task_str}*\n\n"
        f"**Original values**:\n```\n{y_test[:10]}\n```\n"
        f"**Predicted values**:\n```\n{y_pred[:10]}\n```\n"
        f"**Scores**:\n"
        f"{''.join(scores_str)}"
    )

    output_path.mkdir(mode=0o777, exist_ok=True)
    with open(output_path / f"{task_str}_metrics.md", "w", encoding="utf-8") as metrics_file:
        metrics_file.write(out_metrics)
    with open(output_path / f"{task_str}_model.md", "w", encoding="utf-8") as model_info_file:
        model_info_file.write(out_model_info)
    joblib.dump(model, output_path / f"{task_str}_model.sav")
    model_from_file = joblib.load(output_path / f"{task_str}_model.sav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Linear models")

    parser.add_argument(
        "--script",
        help="script: 1 - port of NumPy models to PyTorch, 2 - idiomatic implementation of  models in PyTorch",
        required=True,
        type=int,
        choices=[1, 2],
    )
    parser.add_argument(
        "--task",
        help="task type: 1 - classification, 2 - regression",
        required=True,
        type=int,
        choices=[1, 2],
    )
    parser.add_argument(
        "--config",
        help="path to JSON configuration file",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--on_gpu",
        help="True - train model on your GPU, False - on CPU",
        required=False,
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--output",
        help="path for output model",
        type=Path,
        required=False,
        default=Path('./results')
    )
    args = parser.parse_args()
    main(args.script, args.task, args.config, args.on_gpu, args.output)
