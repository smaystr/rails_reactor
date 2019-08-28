import argparse
import utils
import numpy as np

import torch
from torch.autograd import Variable

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

import time
import linear_models


def to_torch(arr):
    return torch.tensor(arr).double()


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

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Using this device: ', device)

    if params.model == 'linear_regression':
        train, test = utils.download_train_test(
            'insurance_train.csv', 'insurance_test.csv', url=utils.URL_DATA)

        X_train, y_train, X_test, y_test = utils.preprocess_medicalcost(
            train, test)
        n_features = X_train.shape[1]
        model = linear_models.LinearRegression(
            params.epochs, params.lr, params.batch_size, device)

    elif params.model == 'logistic_regression':
        train, test = utils.download_train_test(
            'heart_train.csv', 'heart_test.csv', url=utils.URL_DATA)

        X_train, y_train, X_test, y_test = utils.preprocess_heart(train, test)
        n_features = X_train.shape[1]

        model = linear_models.LogisticRegression(
            params.epochs, params.lr, params.batch_size, device)
    else:
        raise Exception('Incorrect model type is provided.')

    X_train, y_train = to_torch(X_train), to_torch(y_train)
    X_test, y_test = to_torch(X_test), to_torch(y_test)

    t = time.time()

    model.fit(X_train, y_train)
    print('Time for execution', time.time() - t)

    print(model.score(X_test, y_test))


if __name__ == '__main__':
    main()
