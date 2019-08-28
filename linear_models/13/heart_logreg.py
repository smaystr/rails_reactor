from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
from models import LogisticRegression
import numpy as np
import metrics
import preprocessing as prep


TRAIN = 'heart_train.csv'
TEST  = 'heart_test.csv'

def prepare_data(features: list, train: Path, test: Path) -> Tuple[list, np.ndarray, np.ndarray, dict]:
    d_train, feature_parameters = prep.standardize_columns(train, [0, 3, 4, 7, 9])
    d_test, _ = prep.standardize_columns(test, [0, 3, 4, 7, 9], params=feature_parameters)
    data, features, _ = prep.onehot_columns(np.vstack([d_train, d_test]), [2, 6, 10, 11, 12], features)
    param_list = {'feature_parameters': feature_parameters}
    return features, data[:train.shape[0]], data[train.shape[0]:], param_list

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Logistic Regression implementation for "heart" dataset.',
        epilog='Vladyslav Rudenko')
    parser.add_argument('--path', type=str, required=True, help=f'path to datasets "{TRAIN}" and "{TEST}"')
    path = Path(parser.parse_args().path)

    features = prep.read_feature_names(path / TRAIN, skip_last=True)
    features, d_train, d_test, _ = prepare_data(
        features,
        prep.read_data(path / TRAIN, X_Y=False),
        prep.read_data(path / TEST, X_Y=False)
    )

    model = LogisticRegression(lr=0.05, epoch=1000, penalty='l2', C=0.01).fit(d_train[:, :-1], d_train[:, -1])
    print('ACCURACY:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='accuracy')))
    print('PRECISION:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='precision')))
    print('RECALL:{0:.5f}'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='recall')))
    print('F1-SCORE:{0:.5f}\n'.format(model.score(d_test[:, :-1], d_test[:, -1], metric='f_score')))

    for k, v in prep.feature_importance(model.w):
        print(f'FEATURE {features[int(k)]} with importance {v}')
