import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import torch_models
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import numpy as np
from torch.autograd import Variable


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help="Path to train dataset", type=Path)
    parser.add_argument("--test_path", help="Path to test dataset", type=Path)
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
        "--epochs",
        help="Number of epochs",
        type=int,
        default=15,
    )
    return parser.parse_args()


def train_model(model, optimizer, criterion, train_loader, test_loader, epochs, stopping_rounds):
    train_loss_hist = []
    val_loss_hist = []
    min_val_loss = 1e20
    for epoch in range(epochs):
        for data in train_loader:

            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        train_loss_hist.append(loss.item())
        val_loss = 0

        for i, test in enumerate(test_loader, 0):
            inputs, labels = test
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
        val_loss /= i + 1
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        val_loss_hist.append(val_loss)
        if min(val_loss_hist[max([epoch - stopping_rounds, 0]):epoch + 1]) > min_val_loss:
            print('Early stopping at epoch ', epoch)
            break
    return (model, train_loss_hist, val_loss_hist)


def preprocess_data(path, target):
    data = pd.get_dummies(pd.read_csv(path))
    return torch.Tensor(data.drop([target], axis=1).values), torch.Tensor(data[target].values)


def load_data(features, target, scaler, batch_size, shuffle=False):
    features, target = (
        torch.Tensor(scaler.transform(features)),
        torch.Tensor(target).reshape((len(features), 1)),
    )

    dataset = TensorDataset(features, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run():
    args = arg_parse()

    train_features, train_target = preprocess_data(args.train_path, args.target)
    test_features, test_target = preprocess_data(args.test_path, args.target)

    scaler = StandardScaler()
    scaler.fit(train_features)

    train_loader = load_data(train_features, train_target, scaler, args.batch_size, shuffle=True)
    test_loader = load_data(test_features, test_target, scaler, args.batch_size, shuffle=False)

    if args.task == "classification":
        model = torch_models.LogisticRegression(train_features.shape[1], 1)
        loss = torch.nn.BCELoss()
        metr = [metrics.accuracy, metrics.precision, metrics.recall]
        metric_names = ["accuracy", "precision", "recall"]

    else:
        model = torch_models.LinearRegression(train_features.shape[1], 1)
        loss = torch.nn.MSELoss()
        metr = [metrics.mean_squared_error, metrics.mean_absolute_error, metrics.explained_variance_score]
        metric_names = ["Mean squared error", "Mean absolute error", "Explained Variance Score"]

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    start = time.time()

    model, loss_history, val_loss_history = train_model(model, optimizer, loss, train_loader, test_loader, args.epochs, 10)

    print("Elapsed time: ", time.time() - start)

    model.training = False
    for key, metric in enumerate(metr):
        print(f"{metric_names[key]} : {metric(train_target,model(train_features).detach())}")

    plt.plot(range(len(loss_history)), loss_history)
    plt.plot(range(len(loss_history)), val_loss_history)
    plt.title(f"lr {args.lr} batch_size{args.batch_size}")
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f"hw6/images/learning_curve_torch_SGD{args.batch_size}_{args.lr}.png")


if __name__ == "__main__":
    plt.style.use('seaborn')
    run()
