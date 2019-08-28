import numpy as np
import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from lin_regression import LinearRegression
from log_regression import LogRegression

import time
import stuff


def main(
    ds: str, target: int, task: str,
    lr: float, max_iter: int,
    batch: int, device: str, regulization: str
):
    dataset = np.genfromtxt(ds, delimiter=',', dtype=str)

    device = device.lower()
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA isn\'t available. Running on cpu...')
    else:
        print(f'Running on {device}...')

    if task == 'classification':
        model = LogRegression
        metric = stuff.classification_metrics
    else:
        model = LinearRegression
        metric = stuff.regression_metrics

    X_train, y_train, X_test, y_test = stuff.train_test_split(
        dataset[1:, :], 0.2, target, device
    )

    print(f'\nModel with parametrs:\n  lr = {lr}\n  max_iter = {max_iter}')
    mdl = model(
        lr, max_iter,
        batch, regulization, device
        )

    print("\nTraining in progress...")
    start_time = time.time()
    mdl.fit(X_train, y_train)
    total_time = time.time() - start_time
    print(f"...total training time is {str(total_time)[:9]}s\n")

    print("__Train__")
    metric(y_train.to(device), mdl.predict(X_train))

    print("\n__Test__")
    metric(y_test.to(device), mdl.predict(X_test))

    print(f"\nFinal weight:\n {mdl.get_theta()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', help='path to local .csv file or url to file', type=str,
        default=None, metavar='ds', required=True
        )
    parser.add_argument(
        '--target', help='target variable column index', type=int,
        default=None, metavar='t', required=True
        )
    parser.add_argument(
        '--task', help='classification / regression', type=str,
        default=None, required=True
        )
    parser.add_argument(
        '--lr', help='learning rate for model', type=float,
        default=0.001
        )
    parser.add_argument(
        '--num_epoch', help='number of iterations', type=int,
        default=100000
        )
    parser.add_argument(
        '--batch', help='sets the batch size for SGD', type=int,
        default=24,
        )
    parser.add_argument(
        '--device', help='GPU (if available) or CPU using', type=str,
        default='CPU',
        )
    parser.add_argument(
        '--regulization', help='l1, l2, or l1_l2', type=str,
        default='',
        )

    args = parser.parse_args()

    main(
        args.dataset, args.target, args.task,
        args.lr, args.num_epoch, args.batch,
        args.device, args.regulization,
        )
