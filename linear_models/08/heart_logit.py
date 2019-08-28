import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from models import LogisticRegression
import preprocessing as prep

TRAIN, TEST = 'heart_train.csv', 'heart_test.csv'


def prepare_data(train, test):
    d_train, params1 = prep.standardize_columns(train, [0, 3, 4, 7, 9])
    d_test, _ = prep.standardize_columns(test, [0, 3, 4, 7, 9], params=params1)
    data, _ = prep.ohe_columns(np.vstack([d_train, d_test]), [2, 6, 10, 11, 12])
    param_list = {'features': params1}
    return data[:train.shape[0]], data[train.shape[0]:], param_list


if __name__ == "__main__":
    parser = ArgumentParser(description='Logit on heart disease dataset.')
    parser.add_argument('--path', type=Path, required=True, help=f'path to "{TRAIN}" and "{TEST} datasets"')
    args = parser.parse_args()

    if args.path.is_dir():
        d_train, d_test, _ = prepare_data(
            prep.read_data(path / TRAIN, X_Y=False),
            prep.read_data(path / TEST, X_Y=False)
        )

        model = LogisticRegression(lr=0.01, num_iter=100000, penalty='l2', C=0.017)
        model.fit(d_train[:, :-1], d_train[:, -1])
        print('ACCURACY:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='accuracy')))
        print('PRECISION:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='precision')))
        print('RECALL:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='recall')))
        print('F1:{0:.5f}\n'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='f1')))

        for k, v in prep.get_important_feats(model.theta):
            print(f'FEATURE {int(k)} with importance {v}')
    else:
        print('Try again with valid dataset directory path')
