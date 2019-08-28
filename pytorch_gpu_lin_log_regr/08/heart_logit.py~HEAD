import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from models import LogisticRegression, LogitTorch, LinearTrainer
import preprocessing as prep
import metrics
import warnings
warnings.filterwarnings("ignore")

TRAIN, TEST = 'heart_train.csv', 'heart_test.csv'


def prepare_data(features, train_path: Path, test_path: Path):
    STANDARDIZE = [0, 3, 4, 7, 9]
    ONEHOT = [2, 6, 10, 11, 12]
    d_train, feature_parameters = prep.standardize_columns(train_path, STANDARDIZE)
    d_test, _ = prep.standardize_columns(test_path, STANDARDIZE, params=feature_parameters)
    data, features, _ = prep.ohe_columns(np.vstack([d_train, d_test]), ONEHOT, features)
    param_list = {'feature_parameters': feature_parameters}
    return features, data[:train_path.shape[0]], data[train_path.shape[0]:], param_list


if __name__ == "__main__":
    parser = ArgumentParser(description='Logistic Regression implementation on heart diseases dataset.')
    parser.add_argument('--path', type=Path, required=True, help=f'path to "{TRAIN}" and "{TEST}" data')
    parser.add_argument('--config', type=str, required=True, help='path to model config file')
    args = parser.parse_args()
    model_params = prep.read_model_config(args.config)

    if args.path.is_dir():
        Path('./logs').mkdir(exist_ok=True)
        features = prep.read_feature_names(args.path / TRAIN, skip_last=True)
        features, d_train, d_test, _ = prepare_data(features, prep.read_data(args.path / TRAIN, X_Y=False),
                                                    prep.read_data(args.path / TEST, X_Y=False))

        model = LogisticRegression(
            lr=model_params['lr'],
            batch=model_params['batch'],
            num_iter=model_params['epoch'],
            penalty=model_params['penalty'],
            C=model_params['C'],
            is_cuda=model_params['cuda'])
        model.fit(d_train[:, :-1], d_train[:, -1])
        predict = model.predict(d_test[:, :-1]).float()
        print('Low level LogisticRegression F1 score is',
              metrics.f1(predict, torch.tensor(d_test[:, -1], dtype=torch.float)).item())

        model = LogitTorch(d_train.shape[1] - 1, cuda=model_params['cuda'])
        trainer = LinearTrainer(
            model=model,
            lr=model_params['lr'],
            batch=model_params['batch'],
            epoch=model_params['epoch'])
        model = trainer.train(d_train[:, :-1], d_train[:, -1])
        predict = (model.forward(torch.tensor(d_test[:, :-1], dtype=torch.float)) > .5).float()
        print('High level LogisticRegression F1 score is',
              metrics.f1(predict, torch.tensor(d_test[:, -1], dtype=torch.float)).item())
    else:
        print('Try again with valid dataset directory path')
