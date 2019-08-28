from preprocess import Dataset
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from bayes_opt import BayesianOptimization
import numpy as np
from pathlib import Path


def optimize_parameters(train, target, folds, init_points=15, n_iter=5):

    def LGB_bayesian(
            num_leaves,  # int
            min_data_in_leaf,  # int
            min_sum_hessian_in_leaf,    # int
            feature_fraction,
            lambda_l1,
            bagging_fraction,
            lambda_l2,
            min_gain_to_split,
            max_depth,
            max_cat_threshold,
            cat_l2,
            cat_smooth):

        num_leaves = int(num_leaves)
        min_data_in_leaf = int(min_data_in_leaf)
        max_depth = int(max_depth)
        max_cat_threshold = int(max_cat_threshold)

        assert type(num_leaves) == int
        assert type(min_data_in_leaf) == int
        assert type(max_depth) == int

        param = {
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
            'feature_fraction': feature_fraction,
            'lambda_l1': lambda_l1,
            'bagging_fraction': bagging_fraction,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split,
            'max_depth': max_depth,
            'max_cat_threshold': max_cat_threshold,
            'cat_l2': cat_l2,
            'cat_smooth': cat_smooth,
            'objective': 'regression',
            'seed': 30,
            'feature_fraction_seed': 30,
            'bagging_seed': 30,
            'drop_seed': 30,
            'data_random_seed': 30,
            'boosting_type': 'gbdt',
            'verbose': -1,
            'boost_from_average': True,
            'metric': 'mse',
        }

        oof = np.zeros(len(train))

        for fold, (train_idx, val_idx) in enumerate(folds.split(train)):
            lgb_train = lgb.Dataset(train[train_idx], label=target[train_idx], silent=True)
            lgb_valid = lgb.Dataset(train[val_idx], label=target[val_idx], silent=True)

            model = lgb.train(param, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_valid],
                              verbose_eval=0, early_stopping_rounds=20)

            oof[val_idx] = model.predict(train[val_idx], num_iteration=model.best_iteration)

        score = -mean_absolute_error(target, oof)
        return score

    bounds_LGB = {
        'num_leaves': (15, 500),
        'min_data_in_leaf': (20, 200),
        'min_sum_hessian_in_leaf': (0.0001, 0.01),
        'lambda_l1': (0, 5.0),
        'bagging_fraction': (0.1, 0.9),
        'feature_fraction': (0.1, 0.9),
        'lambda_l2': (0, 5.0),
        'min_gain_to_split': (0, 1.0),
        'max_depth': (3, 40),
        'max_cat_threshold': (5, 70),
        'cat_l2': (3, 25),
        'cat_smooth': (3, 25)
    }

    optimize = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=30)

    print('-' * 130)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        optimize.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

    return optimize.max['params']


def get_preprocessed_data():
    data = Dataset()
    data.get_text_features()
    data.encode_labels()
    data.encode_text()
    data.remove_outliers()
    data.encode_text()
    data.drop_columns()
    data.one_hot_encode()
    data.to_numpy()
    return (data.data, data.target)


def process_params(params):
    return {
        'num_leaves': int(params['num_leaves']),
        'min_data_in_leaf': int(params['min_data_in_leaf']),
        'min_sum_hessian_in_leaf': params['min_sum_hessian_in_leaf'],
        'feature_fraction': params['feature_fraction'],
        'lambda_l1': params['lambda_l1'],
        'bagging_fraction': params['bagging_fraction'],
        'lambda_l2': params['lambda_l2'],
        'min_gain_to_split': params['min_gain_to_split'],
        'max_depth': int(params['max_depth']),
        'max_cat_threshold': int(params['max_cat_threshold']),
        'cat_l2': params['cat_l2'],
        'cat_smooth': params['cat_smooth'],
        'objective': 'regression',
        'seed': 30,
        'feature_fraction_seed': 30,
        'bagging_seed': 30,
        'drop_seed': 30,
        'data_random_seed': 30,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'boost_from_average': True,
        'metric': 'mse',
    }


def train_model(params, train, target, folds, save=True):
    if save:
        path = Path('models/')
        path.mkdir(parents=True, exist_ok=True)

    folds = KFold(n_splits=4, shuffle=True, random_state=30)
    score = np.zeros((2))
    oof = np.zeros(len(train))
    for fold, (train_idx, val_idx) in enumerate(folds.split(train)):
        lgb_train = lgb.Dataset(train[train_idx], label=target[train_idx], silent=True)
        lgb_valid = lgb.Dataset(train[val_idx], label=target[val_idx], silent=True)

        model = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_valid],
                          verbose_eval=0, early_stopping_rounds=20)
        score[0] += mean_absolute_error(
            target[train_idx],
            model.predict(train[train_idx], num_iteration=model.best_iteration)
        )
        oof[val_idx] = model.predict(train[val_idx], num_iteration=model.best_iteration)

        if save:
                # lightgbm doesn't want to work with pathlib Path
            model.save_model(f'models/boosting_{fold}.txt', num_iteration=model.best_iteration)

    score = score / (fold + 1)

    score[1] = mean_absolute_error(target, oof)
    print(f"Train MAE: {score[0]}; Test MAE: {score[1]}")


def run():
    train, target = get_preprocessed_data()
    folds = KFold(n_splits=4, shuffle=True, random_state=30)
    params = process_params(optimize_parameters(train, target, folds))
    train_model(params, train, target, folds)


if __name__ == '__main__':
    run()
