import argparse
from pathlib import Path
import numpy as np
import requests
import sys
import json

import metrics

# TODO: fill na


class StandardScaler():
    def __init__(self):
        self.var_ = 1
        self.mean_ = 0

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.var_ = X.std(axis=0)

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.var_

    def fit_transform(self, X: np.ndarray):
        self.fit(X)

        return self.transform(X)


def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    # required
    parser.add_argument('path_data', type=str,
                        help='path to dataset (local .csv file or url)')
    parser.add_argument('target', type=str, help='target variable name')
    parser.add_argument('task_type', help='task: classification/regression',
                        choices=['classification', 'regression'])
    parser.add_argument('path_output', type=Path,
                        help='path for output model: output.info output.model')
    # optional
    parser.add_argument('--split_type', help='parameter for validation split \
                        type: train-test split/k-fold/leave one-out',
                        choices=['train-test', 'k-fold', 'leave one-out'],
                        default='train-test')
    parser.add_argument('--validation_size', help='parameter for validation split size',
                        default=0.33)
    # parser.add_argument('--time_column', help='parameter for specifying time series \
    #                     column to perform timeseries validation', default='')
    parser.add_argument('--hyper_param_fit', help='parameter for hyperparameter fitting algo: \
                        grid search/random search (also add parameter for this',
                        choices=['grid_search', 'random_search', None],
                        default=None)

    args = parser.parse_args()

    return args


def preprocess_csv(path_data, target):

    if Path(path_data).is_file():
        filename = path_data
    else:
        try:
            r = requests.get(path_data)

        except requests.exceptions.RequestException as e:
            print(f'Neither local file nor url, please try again')

        filename = list(filter(None, path_data.split('/')))[-1]
        print(f'Saving {filename} to local directory...')
        with open(filename, 'w') as f:
            f.write(r.text)
    try:
        data = np.array(np.genfromtxt(filename, delimiter=',', names=True))

    except:
        print(f'Problems with reading {filename}')
        sys.exit()
    else:
        columns = np.array(data.dtype.names)

        target_ind = np.where(columns == target)

        if len(target_ind) > 0:
            target_ind = target_ind[0][0]
        else:
            raise Exception(f'No {target} column in data.')

        X = data[columns[columns != target]]
        X = X.view(np.float).reshape(X.shape + (-1,))
        y = data[target]

        return X, y, columns[columns != target]


def create_info_file(path_dir, task_type, model, X, y, columns):
    data = {}
    scores = []
    if task_type == 'regression':
        metrics_funcs = [metrics.mean_squared_error]
    else:
        metrics_funcs = [metrics.precision_score,
                         metrics.recall_score,
                         metrics.f1_score,
                         metrics.accuracy_score]

    y_true = y
    y_pred = model.predict(X)

    for metric in metrics_funcs:
        scores.append((metric.__name__, metric(y_true, y_pred)))
    # print(scores)
    data['metrics'] = scores
    data['time'] = model.time
    data['loss'] = model.cost

    data['feature_importance'] = sorted(
        zip(columns, model.coef_.flatten()), key=lambda x: x[1], reverse=True)

    # print(data)
    filename = path_dir / 'output.info'
    save_json(data, filename)


def create_model_file(path_dir, task_type, model):
    data = {}

    hyper_params = {}
    hyper_params['C'] = model.C
    hyper_params['lr'] = model.lr
    hyper_params['max_iter'] = model.max_iter
    hyper_params['penalty'] = model.penalty

    data['type'] = task_type
    data['best hyperparameters'] = tuple(hyper_params.items())
    data['weights'] = model.coef_.tolist()

    filename = path_dir / 'output.model'
    save_json(data, filename)


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
