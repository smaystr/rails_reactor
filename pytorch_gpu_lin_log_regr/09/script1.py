import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import models
import torch
import matplotlib.pyplot as plt
import time
import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to dataset", type=Path)
    parser.add_argument("--target", help="Target column name", type=str)
    parser.add_argument(
        "--task",
        help="Type of task. Classification/regression",
        type=str,
        choices=("classification", "regression"),
    )
    parser.add_argument(
        "--cuda", help="Use gpu", type=bool, choices=(True, False), default=False
    )
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.03)
    parser.add_argument(
        "--batch_size",
        help="Batch size for SGD to use (int > 0).",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--tol",
        help="Model tolerance towards low loss changes. Stops if is lower than tol.",
        type=float,
        default=1e-3
    )
    return parser.parse_args()


def run():
    args = arg_parse()
    if args.task == "classification":
        model = models.LogisticRegression(lr=args.lr, batch_size=args.batch_size, tol=args.tol)
        metr = [metrics.accuracy_score, metrics.precision_score, metrics.recall_score]
        metric_names = ["accuracy", "precision", "recall"]
    else:
        model = models.LinearRegression(lr=args.lr, batch_size=args.batch_size, tol=args.tol)
        metr = [metrics.mean_squared_error, metrics.mean_absolute_error, metrics.explained_variance_score]
        metric_names = ["Mean squared error", "Mean absolute error", "Explained Variance Score"]

    data = pd.get_dummies(pd.read_csv(args.path))
    features, target = data.drop([args.target], axis=1).values, data[args.target].values

    scaler = StandardScaler()
    features, target = (
        torch.Tensor(scaler.fit_transform(features)),
        torch.Tensor(target),
    )
    start = time.time()

    model.fit(features, target)

    print("Elapsed time: ", time.time() - start)

    for key, metric in enumerate(metr):
        print(f"{metric_names[key]} : {metric(target,model.predict(features))}")

    plt.plot(range(0, len(model.loss_history)), np.array(model.loss_history) * 2)
    plt.title(f"lr {args.lr} batch_size{args.batch_size}")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f"hw6/images/learning_curve{args.batch_size}_{args.lr}.png")


if __name__ == "__main__":
    plt.style.use('seaborn')
    run()
