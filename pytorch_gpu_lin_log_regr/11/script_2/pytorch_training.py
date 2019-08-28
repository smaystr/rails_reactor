import argparse
import json
import pathlib
import torch
import torch.nn.functional as F
import time

from utils.metrics import rmse, f_score
from utils.preprocessing import load_data
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def fit(num_epochs, model, loss_fn, opt, train_dl, train_x, train_y, test_x, test_y):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()


class LogisticRegression(nn.Module):
    def __init__(self, size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class LinearRegression(nn.Module):
    def __init__(self, size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(size, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def main(task, train, test, config, device_type, output_path):
    start_time = time.time()
    if device_type == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config.update({'device': device})

    X_train, y_train, X_test, y_test = load_data(train, test, task, device)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, config['batch_size'], shuffle=False)

    if task == 'Classification':
        model = LogisticRegression(X_train.shape[1])
        opt = torch.optim.SGD(model.parameters(), lr=config['alpha'])
        loss_fn = torch.nn.BCELoss()
    else:
        model = LinearRegression(X_train.shape[1])
        opt = torch.optim.SGD(model.parameters(), lr=config['alpha'])
        loss_fn = F.mse_loss

    fit(config['iters'], model, loss_fn, opt, train_dl, X_train, y_train, X_test, y_test)

    y_pred_train = model(X_train)
    y_pred_test = model(X_test)

    if task == 'Classification':
        print(
            f"""                     
    Time : {time.time() - start_time}  sec                 
    F1-Score fot training set: {f_score(y_train, y_pred_train)} 
    F1-Score for test set: {f_score(y_test, y_pred_test)}                 
    """)

    else:
        print(f"""
    Time : {time.time() - start_time}
    RMSE fot training set: {rmse(y_train, y_pred_train)}
    RMSE for test set: {rmse(y_test, y_pred_test)}
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='type of task', choices=['Regression', 'Classification'])
    parser.add_argument('train', type=str, help='path to the train dataset')
    parser.add_argument('test', type=str, help='path to the test dataset')
    parser.add_argument('-dt', '--device_type', type=str, default='cpu', choices=['cpu', 'gpu'],
                        help='type of processing unit')
    parser.add_argument('-op', '--output_path', type=pathlib.Path, default='./output.txt',
                        help='path for saving weights')
    parser.add_argument('-cp', '--config_path', type=pathlib.Path, default='./config.json',
                        help='path to the configuration file')
    args = parser.parse_args()

    config = json.loads(args.config_path.read_text())
    print(args.task, args.train, args.test, config, args.device_type, args.output_path)
    main(args.task, args.train, args.test, config, args.device_type, args.output_path)
