import numpy as np
from torch.utils.data import DataLoader, random_split
import torch
import argparse
from torch.multiprocessing import cpu_count

from lin_regression import LinearRegressionTorch
from log_regression import LogRegressionTorch

import time
import stuff


def main(
    ds: str, target: int, task: str,
    lr: float, max_iter: int, optmzr: str,
    batch: int, device: str, shuffle: int
):
    device = device.lower()
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA isn\'t available. Running on cpu...')
    else:
        print(f'Running on {device}...')

    dataset = stuff.DatasetReader(ds, target)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    shuffle = True if shuffle == 1 else False

    train_iter = DataLoader(
        train, batch_size=batch, shuffle=shuffle, num_workers=cpu_count()
        )
    test_iter = DataLoader(
        test, batch_size=batch, shuffle=shuffle, num_workers=cpu_count()
        )

    if task == 'classification':
        model = LogRegressionTorch
        metric = stuff.classification_metrics
        loss_fn = torch.nn.BCELoss()
    elif task == 'regression':
        model = LinearRegressionTorch
        metric = stuff.regression_metrics
        loss_fn = torch.nn.MSELoss()

    mdl = model(dataset.size[1], device)

    if optmzr == "adam":
        optimizer = torch.optim.Adam(mdl.parameters())
    elif optmzr == "nesterov":
        optimizer = torch.optim.SGD(
            mdl.parameters(), lr=lr, momentum=0.9,
            nesterov=True
        )
    else:
        optimizer = torch.optim.SGD(mdl.parameters(), lr=lr)

    trainer = stuff.TrainModel(
        mdl, lr, max_iter, loss_fn, optimizer, device
    )

    print("\nTraining in progress...")
    start_time = time.time()
    trained_mdl = trainer.fit(train_iter, task)
    total_time = time.time() - start_time
    print(f"...total training time is {str(total_time)[:9]}s\n")

    print("__Metrics for train__")
    y_train = torch.cat([y for x, y in train_iter], dim=0)
    X_train = torch.cat([x for x, y in train_iter], dim=0)
    metric(y_train.squeeze(dim=1).to(device), trained_mdl.predict(X_train))

    print("\n__Metrics for test__")
    y_test = torch.cat([y for x, y in test_iter], dim=0)
    X_test = torch.cat([x for x, y in test_iter], dim=0)
    metric(y_test.squeeze(dim=1).to(device), trained_mdl.predict(X_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', help='path to local .csv file or url to file', type=str,
        default=None, required=True
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
        '--optimizer', help='ADAM or SGD', type=str,
        default='SGD'
        )
    parser.add_argument(
        '--batch', help='sets the batch size for SGD', type=int,
        default=24,
        )
    parser.add_argument(
        '--device', help='GPU (if available) or CPU using', type=str,
        default='cpu',
        )
    parser.add_argument(
        '--shuffle', help='1 - shuffled, 0 - non shuffled', type=int,
        default=None,
        )

    args = parser.parse_args()

    main(
        args.dataset, args.target, args.task,
        args.lr, args.num_epoch, args.optimizer,
        args.batch, args.device, args.shuffle
        )
