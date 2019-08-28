import sys
import json
import time
from pathlib import Path
import argparse
import numpy as np
from hyperparam import GridSearch, RandomSearch
from validation import train_test_split
import metrics
from utils import prepare_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End to end model trainning lifecycle')
    parser.add_argument('dataset', help='Dataset path/url', type=Path)
    parser.add_argument('target', help='Target variable name')
    parser.add_argument('task', help='Type of model for the task: regression/classification')
    parser.add_argument('--out', help='Path to output files', type=Path, default=Path())
    parser.add_argument('--validation', help='Validation split type:train_test/kfold/loocv', default='kfold')
    parser.add_argument('--split_size', help='Validation split size:\
                        Use float for the test size percentage/amount of folds for kfold/None for loocv', default=5)
    parser.add_argument('--hyper_search', help='Hyperparameter fitting algorithm: grid/random', default='grid')
    parser.add_argument('--hyperparams', help='Hyperparameters to tune. use a json format string',
                        default='{"tol": [0.1, 0.001, 0.0001]}')
    args = parser.parse_args()
    columns = np.genfromtxt(args.dataset, delimiter=',', dtype=str)[0]  # get column names
    target = np.where(columns == args.target)[0][0]
    data = np.genfromtxt(args.dataset, delimiter=',')
    X = np.delete(data, target, axis=1)[1:, :]
    y = data[1:, target]
    sys.path.append(str(Path().cwd().parent.joinpath('hw3')))
    if args.task == 'regression':
        from linear_regression import LinearRegression
        model = LinearRegression
        loss = metrics.mse
    elif args.task == 'classification':
        from logistic_regression import LogisticRegression
        model = LogisticRegression
        loss = metrics.log_loss
    else:
        print('Parameter error: task must be regression or classification')
        exit(0)
    params = json.loads(args.hyperparams)
    if args.hyper_search == 'random':
        search = RandomSearch(model, params, cv=args.validation, cv_size=args.split_size)
        search.fit(X, y)
    else:
        search = GridSearch(model, params, cv=args.validation, cv_size=args.split_size)
        search.fit(X, y)
    out_params = 'Hyperparameters: {}\n'.format(search.best_params_)
    model = search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    start_time = time.time()
    model.fit(X_train, y_train)
    out_weights = 'Weights: {}\n'.format(model.coef_)
    out_time = 'Training time: {}\n'.format(time.time() - start_time)
    train_pred = model.predict(prepare_data(X_train, y_train, True)[0])
    out_loss = 'Training loss: {}\n'.format(loss(y_train, train_pred))
    baseline = model.score(X_test, y_test)
    importance = []
    columns = np.delete(columns, target)
    out_importance = 'Feature importance:\n'
    for i, feature in enumerate(X_train.T):
        X_new = np.array(X_train, copy=True)
        np.random.shuffle(X_new[i])
        importance.append(baseline - model.score(X_new, y_train))
        out_importance += '{}: {}\n'.format(columns[i], importance[i])
    test_pred = model.predict(prepare_data(X_test, y_test, True)[0])
    if args.task == 'regression':
        out_model = 'Model: Linear Regression (regressor)\n'
        out_metrics = 'MAE: {}\nMSE: {}\nRMSE: {}\nMAPE: {}\nMPE: {}\n'.format(metrics.mae(y_test, test_pred),
                                                                               metrics.mse(y_test, test_pred),
                                                                               metrics.rmse(y_test, test_pred),
                                                                               metrics.mape(y_test, test_pred),
                                                                               metrics.mpe(y_test, test_pred))
    elif args.task == 'classification':
        out_model = 'Model: Logistic Regression (classificator)\n'
        out_metrics = 'Log Loss: {}\nAccuracy: {}\nPrecision: {}' \
                      '\nRecall: {}\nF1: {}\n'.format(metrics.log_loss(y_test, test_pred),
                                                      metrics.accuracy(y_test, test_pred),
                                                      metrics.precision(y_test, test_pred),
                                                      metrics.recall(y_test, test_pred),
                                                      metrics.f1(y_test, test_pred))
    with open(str(args.out.joinpath('output_info.txt')), 'w', encoding='utf-8') as f:
        f.write(out_metrics + out_time + out_loss + out_importance)
    with open(str(args.out.joinpath('output_model.txt')), 'w', encoding='utf-8') as f:
        f.write(out_model + out_params + out_weights)
