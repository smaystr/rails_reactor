from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")
from models import LinearRegression
import numpy as np
import metrics
import preprocessing as prep


TRAIN = 'insurance_train.csv'
TEST  = 'insurance_test.csv'

def prepare_data(features: list, train: Path, test: Path) -> Tuple[list, np.ndarray, np.ndarray, dict]:
    data, classes = prep.to_numeric_multiple(np.vstack([train, test]), [1, 4, 5])
    d_train = data[:train.shape[0], :].astype(np.float32)
    d_test  = data[train.shape[0]:, :].astype(np.float32)

    d_train, target_parameters = prep.normalize(d_train, [-1])
    d_test, _ = prep.normalize(d_test, [-1], target_parameters[0], target_parameters[1])
    d_train, feature_parameters = prep.standardize_columns(d_train, [0, 2, 3])
    d_test, _ = prep.standardize_columns(d_test, [0, 2, 3], params=feature_parameters)
    data, features, _ = prep.onehot_columns(np.vstack([d_train, d_test]), [5], features)
    param_list = {'classes': classes, 'target':target_parameters, 'features':feature_parameters}
    return features, data[:train.shape[0]], data[train.shape[0]:], param_list

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Linear Regression implementation for "insurance" dataset.',
        epilog='Vladyslav Rudenko')
    parser.add_argument('--path', type=str, required=True, help=f'path to datasets "{TRAIN}" and "{TEST}"')
    path = Path(parser.parse_args().path)

    features = prep.read_feature_names(path / TRAIN, skip_last=True)
    features, d_train, d_test, params = prepare_data(
        features,
        prep.read_data(path / TRAIN, X_Y=False, dtype=str),
        prep.read_data(path / TEST, X_Y=False, dtype=str)
    )
    d_max = params['target'][0]
    d_min = params['target'][1]

    model = LinearRegression(lr=0.05, epoch=1000, penalty='l2', C=0.01).fit(d_train[:, :-1], d_train[:, -1])
    print('TRANSFORMED\nMSE:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='mse')))
    print('RMSE:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='rmse')))
    print('MAE:{0:.5f}\n'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='mae')))

    print('INVERSE TRANSFORMED\nMSE:{0:.5f}'.format(metrics.mse(
            model.predict(d_test[:, :-1]) * (d_max - d_min) + d_min,
            (d_test[:, -1] * (d_max - d_min) + d_min).reshape(-1, 1)
            )))
    print('RMSE:{0:.5f}'.format(metrics.rmse(
            model.predict(d_test[:, :-1]) * (d_max - d_min) + d_min,
            (d_test[:, -1] * (d_max - d_min) + d_min).reshape(-1, 1)
            )))
    print('MAE:{0:.5f}\n'.format(metrics.mae(
            model.predict(d_test[:, :-1]) * (d_max - d_min) + d_min,
            (d_test[:, -1] * (d_max - d_min) + d_min).reshape(-1, 1)
            )))

    for k, v in prep.feature_importance(model.w):
        print(f'FEATURE {features[int(k)]} with importance {v}')
