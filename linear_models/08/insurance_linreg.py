import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from models import LinearRegression
import metrics
import preprocessing as prep
# ignore warnings to make console output cleaner
import warnings
warnings.filterwarnings("ignore")

TRAIN = 'insurance_train.csv'
TEST = 'insurance_test.csv'


def prepare_data(train, test):
    data, classes = prep.to_numeric_multiple(np.vstack([train, test]), [1, 4, 5])
    d_train = data[:train.shape[0], :].astype(np.float32)
    d_test = data[train.shape[0]:, :].astype(np.float32)

    d_train, params1 = prep.normalize(d_train, [-1])
    d_test, _ = prep.normalize(d_test, [-1], params1[0], params1[1])
    d_train, params2 = prep.standardize_columns(d_train, [0, 2, 3])
    d_test, _ = prep.standardize_columns(d_test, [0, 2, 3], params=params2)
    data, _ = prep.ohe_columns(np.vstack([d_train, d_test]), [5])
    param_list = {'classes': classes, 'target': params1, 'features': params2}
    return data[:train.shape[0]], data[train.shape[0]:], param_list


if __name__ == "__main__":
    parser = ArgumentParser(description='Linear Regression on insurance dataset.')
    parser.add_argument('--path', type=Path, required=True, help=f'path to "{TRAIN}" and "{TEST} datasets"')
    args = parser.parse_args()

    if args.path.is_dir():
        d_train, d_test, params = prepare_data(
        prep.read_data(path / TRAIN, X_Y=False, dtype=str),
        prep.read_data(path / TEST, X_Y=False, dtype=str)
        )
        d_max = params['target'][0]
        d_min = params['target'][1]

        model = LinearRegression(lr=0.01, num_iter=100000, penalty='l2', C=0.017)
        model.fit(d_train[:, :-1], d_train[:, -1])
        print('TRANSFORMED\nMSE:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='mse')))
        print('RMSE:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='rmse')))
        print('MAE:{0:.5f}\n'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='mae')))

        print('INVERSE TRANSFORMED\n MSE:{0:.5f}'.format(metrics.mse(
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

        for k, v in prep.get_important_feats(model.theta):
            print(f'FEATURE {int(k)} with importance {v}')
    else:
        print('Try again with valid dataset directory path')
