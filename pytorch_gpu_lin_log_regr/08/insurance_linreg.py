import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from models import LinearRegression, LinRegTorch, LinearTrainer
import metrics
import preprocessing as prep

import warnings
warnings.filterwarnings("ignore")

TRAIN, TEST = 'insurance_train.csv', 'insurance_test.csv'


def prepare_data(features, train, test):
    TO_NUMERIC = [1, 4, 5]
    STANDARDIZE = [0, 2, 3]
    ONEHOT = [5]
    data, classes = prep.to_numeric_multiple(np.vstack([train, test]), TO_NUMERIC)
    d_train = data[:train.shape[0], :].astype(np.float32)
    d_test = data[train.shape[0]:, :].astype(np.float32)

    d_train, params1 = prep.normalize(d_train, [-1])
    d_test, _ = prep.normalize(d_test, [-1], params1[0], params1[1])
    d_train, params2 = prep.standardize_columns(d_train, STANDARDIZE)
    d_test, _ = prep.standardize_columns(d_test, [0, 2, 3], params=params2)
    data, features, _ = prep.ohe_columns(np.vstack([d_train, d_test]), ONEHOT, features)
    param_list = {'classes': classes, 'target': params1, 'features': params2}
    return features, data[:train.shape[0]], data[train.shape[0]:], param_list


if __name__ == "__main__":
    parser = ArgumentParser(description='Linear Regression on insurance dataset.')
    parser.add_argument('--path', type=Path, required=True, help=f'path to "{TRAIN}" and "{TEST} datasets"')
    parser.add_argument('--config', type=str, required=True, help='path to model config file')
    args = parser.parse_args()
    model_params = prep.read_model_config(args.config)

    if args.path.is_dir():
        Path('./logs').mkdir(exist_ok=True)
        features = prep.read_feature_names(args.path / TRAIN, skip_last=True)
        features, d_train, d_test, params = prepare_data(features,
                                                         prep.read_data(args.path / TRAIN, X_Y=False, dtype=str),
                                                         prep.read_data(args.path / TEST, X_Y=False, dtype=str))
        d_max = params['target'][0]
        d_min = params['target'][1]

        model = LinearRegression(
            lr=model_params['lr'],
            batch=model_params['batch'],
            num_iter=model_params['epoch'],
            penalty=model_params['penalty'],
            C=model_params['C'],
            is_cuda=model_params['cuda'])
        model.fit(d_train[:, :-1], d_train[:, -1])
        predict = model.predict(d_test[:, :-1]).float()
        print('Low level LinearRegression RMSE is',
              metrics.rmse(predict, torch.tensor(d_test[:, -1], dtype=torch.float)).item())

        model = LinRegTorch(d_train.shape[1] - 1, cuda=model_params['cuda'])
        trainer = LinearTrainer(
            model=model,
            lr=model_params['lr'],
            batch=model_params['batch'],
            epoch=model_params['epoch'])
        model = trainer.train(d_train[:, :-1], d_train[:, -1])
        predict = model.forward(torch.tensor(d_test[:, :-1], dtype=torch.float))
        print('High level LinearRegression RMSE is',
              metrics.rmse(predict, torch.tensor(d_test[:, -1], dtype=torch.float)).item())
    else:
        print('Try again with valid dataset directory path')
