import argparse
import utils
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

import time


class LinearModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = self.fc(x)
        return out


class LinearRegression(LinearModel):
    def __init__(self, n_features):
        super().__init__(n_features)


class LogisticRegression(LinearModel):
    def __init__(self, n_features):
        super().__init__(n_features)

    def forward(self, x):
        x = self.fc(x)

        out = F.softmax(x, dim=0)

        return out


def fit(model, trainloader, epochs, optimizer, criterion, device, tb):
    t = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        tb.add_scalar('Loss', running_loss, epoch)

    print('Finished Training', running_loss)
    print('time', time.time() - t)


def to_torch(arr):
    return Variable(torch.from_numpy(arr)).float()


def main():

    args = utils.parse_arguments()

    mode = args.mode
    config = args.config

    params = utils.Parameters(config)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode == 'GPU':
        if torch.cuda.is_available():
            use_cuda = True
        else:
            use_cuda = False
            print('GPU mode is not available. Using CPU...')
    else:
        use_cuda = False

    tb = SummaryWriter()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Using this device: ', device)

    if params.model == 'linear_regression':
        criterion = torch.nn.MSELoss()
        train, test = utils.download_train_test(
            'insurance_train.csv', 'insurance_test.csv', url=utils.URL_DATA)

        X_train, y_train, X_test, y_test = utils.preprocess_medicalcost(
            train, test)
        n_features = X_train.shape[1]
        model = LinearRegression(n_features).to(device)
        metric = mean_squared_error

    elif params.model == 'logistic_regression':
        criterion = torch.nn.BCELoss()
        train, test = utils.download_train_test(
            'heart_train.csv', 'heart_test.csv', url=utils.URL_DATA)

        X_train, y_train, X_test, y_test = utils.preprocess_heart(train, test)
        n_features = X_train.shape[1]

        model = LogisticRegression(n_features).to(device)
        metric = roc_auc_score

    else:
        raise Exception('Incorrect model type is provided.')

    optimizer = optim.SGD(model.parameters(), lr=params.lr)

    X_train, y_train = to_torch(X_train).float(), to_torch(y_train)
    X_test, y_test = to_torch(X_test).float(), to_torch(y_test)

    trainloader = Data.DataLoader(
        dataset=TensorDataset(X_train, y_train),
        batch_size=params.batch_size)

    fit(model, trainloader, params.epochs, optimizer, criterion, device, tb)

    res = model(X_test.to(device))
    y_test = y_test.cpu()

    t = time.time()
    res = res.detach().cpu()
    print(time.time() - t)

    print('Score', metric(y_test, res))
    tb.close()


if __name__ == '__main__':
    main()
