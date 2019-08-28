import torch
from torch.utils.data import Dataset

import numpy as np
import metrics

import tensorboardX

np.random.seed(42)


class DatasetReader(Dataset):

    def __init__(self, file_path, target):
        data = np.genfromtxt(file_path, delimiter=',', dtype=str)[1:]
        self.X = torch.from_numpy(
            preprocessing(
                np.delete(data, target, axis=1)
            )
        )
        self.y = torch.from_numpy(
            data[:, target].reshape((-1, 1)).astype(np.float32)
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    @property
    def size(self):
        return self.X.size()


class TrainModel:

    def __init__(
        self, model, lr, max_iter,
        loss_fn, optimizer, device
    ):
        self.mdl = model
        self.lr = lr
        self.epoch = max_iter
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = optimizer

    def fit(self, data, task):
        j = 1
        writer = tensorboardX.SummaryWriter()

        for i in range(1, self.epoch + 1):
            # process checking
            if i / (self.epoch * 0.1 * j) == 1.:
                print(' [', '#' * j + '.' * (10-j), ']', f'{j}0%')
                j += 1

            for features, label in data:
                X_batch, y_batch = (
                    features.to(self.device), label.to(self.device)
                )

                self.optimizer.zero_grad()

                y_hat = self.mdl.forward(X_batch)
                loss = self.loss_fn(y_hat, y_batch)

                loss.backward()
                self.optimizer.step()

                writer.add_scalar('Loss', loss.item(), i)

                if task == 'classification':
                    y_pred = (y_hat > 0.5).float()
                    writer.add_scalar("Accuracy", metrics.accuracy(y_batch, y_pred).item(), i)
                    writer.add_scalar("Recall", metrics.recall(y_batch, y_pred).item(), i)
                    writer.add_scalar("Precision", metrics.precision(y_batch, y_pred).item(), i)
                    writer.add_scalar("F1", metrics.f1(y_batch, y_pred).item(), i)
                else:
                    y_pred = y_hat
                    writer.add_scalar("RMSE", metrics.rmse(y_batch, y_pred).item(), i)
                    writer.add_scalar("MSE", metrics.mse(y_batch, y_pred).item(), i)
                    writer.add_scalar("MAE", metrics.mae(y_batch, y_pred).item(), i)

        writer.close()
        return self.mdl


def classification_metrics(y_true, y_pred):
    print("Accuracy:  ", metrics.accuracy(y_true, y_pred).item())
    print("Recall:    ", metrics.recall(y_true, y_pred).item())
    print("Precision: ", metrics.precision(y_true, y_pred).item())
    print("F1:        ", metrics.f1(y_true, y_pred).item())


def regression_metrics(y_true, y_pred):
    print("RMSE: ", metrics.rmse(y_true, y_pred).item())
    print("MSE:  ", metrics.mse(y_true, y_pred).item())
    print("MAE:  ", metrics.mae(y_true, y_pred).item())


def train_test_split(dataset, size, target, device):
    n = dataset.shape[0]
    random_col = np.random.choice(n, int(n * size), replace=False)

    dataset_test = dataset[random_col, :]
    dataset_train = np.delete(dataset.copy(), random_col, axis=0)

    X_train = torch.from_numpy(
        preprocessing(
            np.delete(dataset_train, target, axis=1)
            )
        ).to(device)

    y_train = torch.from_numpy(
        dataset_train[:, target].astype(np.float32)
        ).view(-1, 1).squeeze(dim=1).to(device)

    X_test = torch.from_numpy(
        preprocessing(
            np.delete(dataset_test, target, axis=1)
            )
        ).to(device)

    y_test = torch.from_numpy(
        dataset_test[:, target].astype(np.float32)
        ).view(-1, 1).squeeze(dim=1).to(device)

    return (X_train, y_train, X_test, y_test, )


def preprocessing(X):
    for i in range(X.shape[1]):
        try:
            X[:, i] = np.float32(X[:, i])

        except Exception:
            uniq = np.unique(X[:, i])
            classes = dict()

            cl_num = 1
            for cl in uniq:
                classes[cl] = cl_num
                cl_num += 1

            for j in range(X.shape[0]):
                X[j, i] = classes[X[j, i]]

    X = X.astype(np.float32)

    np.seterr(divide='ignore', invalid='ignore')
    min_ = X.min(axis=0)
    max_ = X.max(axis=0)

    return np.nan_to_num((X - min_) / (max_ - min_))
