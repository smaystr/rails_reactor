import argparse
import pathlib

from regressions import LinearRegression, LogisticRegression
import utils
import preprocessing
import metrics
import search

OUTPUT_INFO = "output.info"
OUTPUT_MODEL = "output.model"


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('train_dataset', metavar='tnds', type=str, help="Train dataset local file (*.csv) or URL")
    parser.add_argument('test_dataset', metavar='ttds', type=str, help="Test dataset local file (*.csv) or URL")
    parser.add_argument('target', metavar='tg', type=str, help="Target column in dataset")
    parser.add_argument('task', metavar='ts', type=str, choices=['classification', 'regression'],
                        help="Implementing task: classification or regression")
    parser.add_argument('output', metavar='out', type=str,
                        help="Output directory for files output.info and output.model")
    parser.add_argument('-s', '--split', type=str, default='k-fold',
                        choices=['0', '1', '2'],
                        help="Split type: train-test split (0) OR k-fold (1) OR leave one-out (2)")
    parser.add_argument('-vs', '--validation_size', type=float, default=0.1, help="Size of validation dataset")
    parser.add_argument('-ts', '--time_series', type=str, help="Time series column in dataset")
    parser.add_argument('-hf', '--hyperparameter_fit', type=str, default='grid_search',
                        choices=['grid_search', 'random_search'],
                        help="Hyperparameter fitting algorithm: grid_search OR random_search")
    args = parser.parse_args()

    directory = pathlib.Path(args.output)
    if not directory.exists():
        directory.mkdir(parents=True)

    output_info = (directory / pathlib.Path(OUTPUT_INFO))
    output_info.touch()
    output_model = (directory / pathlib.Path(OUTPUT_MODEL))
    output_model.touch()

    print(f'Task: {args.task}')
    print(f'Validation size: {args.validation_size * 100}%')
    print(f'Hyperparameter fit: {args.hyperparameter_fit} with split type {args.split}')
    if args.task == 'regression':
        data_train = utils.upload_dataset(args.train_dataset)
        data_test = utils.upload_dataset(args.test_dataset)
        data_train = preprocessing.one_hot_encoding(data_train, columns=['sex', 'smoker', 'region'])
        data_test = preprocessing.one_hot_encoding(data_test, columns=['sex', 'smoker', 'region'])
        X_train, y_train = utils.get_X_y(data_train, args.target)
        X_test, y_test = utils.get_X_y(data_test, args.target)
        X_train = preprocessing.normalize(X_train, columns=[0, 1])
        X_test = preprocessing.normalize(X_test, columns=[0, 1])
        lr = LinearRegression(max_iter=10000, alpha=1e-2, regularisation=True, lmd=1e-3)
        lr.fit(X_train, y_train)
        parameter_grid = {
                "alpha": [1e-1, 3e-1, 1e-2, 3e-2, 1e-3, 3e-3, 1e-4, 3e-4, 1e-5, 3e-5],
                "lmd": [1e-1, 3e-1, 1e-2, 3e-2, 1e-3, 3e-3, 1e-4, 3e-4, 1e-5, 3e-5]
        }
        if args.hyperparameter_fit == 'grid_search':
            sch = search.GridSearch(
                model=lr,
                parameter_grid=parameter_grid,
                n_splits=10,
                test_size=.1,
                iterations=1,
                split_type=args.split,
                metric=metrics.RMSE
            )
            lr = sch.fit(X_train, y_train)
        else:
            sch = search.RandomSearch(
                model=lr,
                parameter_grid=parameter_grid,
                n_splits=10,
                test_size=.1,
                iterations=1,
                split_type=args.split,
                metric=metrics.RMSE
            )
            lr = sch.fit(X_train, y_train)
        lr = sch.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        loss_history, prediction, working_time = lr.predict(X_test)

    else:
        data_train = utils.upload_dataset(args.train_dataset)
        data_test = utils.upload_dataset(args.test_dataset)
        X_train, y_train = utils.get_X_y(data_train, args.target)
        X_test, y_test = utils.get_X_y(data_test, args.target)
        X_train = preprocessing.standardize(X_train, columns=[0, 3, 4, 7])
        X_test = preprocessing.standardize(X_test, columns=[0, 3, 4, 7])
        lr = LogisticRegression(max_iter=10000, alpha=1e-2, regularisation=True, lmd=1e-3)
        lr.fit(X_train, y_train)
        parameter_grid = {
            "alpha": [1e3, 3e3, 1e4, 3e4, 1e2, 3e2, 1e1, 3e1, 1e0, 3e0, 1e-1, 3e-1, 1e-2, 3e-2, 1e-3, 3e-3, 1e-4, 3e-4, 1e-5, 3e-5],
            "lmd": [1e3, 3e3, 1e4, 3e4, 1e2, 3e2, 1e1, 3e1, 1e0, 3e0, 1e-1, 3e-1, 1e-2, 3e-2, 1e-3, 3e-3, 1e-4, 3e-4, 1e-5, 3e-5],
            "max_iter": [1000, 2000, 5000, 10000]
        }
        if args.hyperparameter_fit == 'grid_search':
            sch = search.GridSearch(
                model=lr,
                parameter_grid=parameter_grid,
                n_splits=10,
                test_size=args.validation_size,
                iterations=1,
                split_type=args.split,
                metric=metrics.F1_score
            )
            lr = sch.fit(X_train, y_train)
        else:
            sch = search.RandomSearch(
                model=lr,
                parameter_grid=parameter_grid,
                n_splits=10,
                test_size=args.validation_size,
                iterations=1,
                split_type=args.split,
                metric=metrics.F1_score
            )
            lr = sch.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        loss_history, prediction, working_time = lr.predict(X_test)

    with pathlib.Path.open(output_info, 'w') as info:
        info.write(f"1. All metrics:\n")
        if args.task == 'classification':
            info.write(f"Accuracy: {metrics.accuracy(y_test, prediction)}\n")
            info.write(f"Precision: {metrics.precision(y_test, prediction)}\n")
            info.write(f"Recall: {metrics.recall(y_test, prediction)}\n")
            info.write(f"F1-score: {metrics.F1_score(y_test, prediction)}\n")
        else:
            info.write(f"RMSE: {metrics.RMSE(y_test, prediction)}\n")
            info.write(f"MSE: {metrics.MSE(y_test, prediction)}\n")
            info.write(f"MAE: {metrics.MAE(y_test, prediction)}\n")
            info.write(f"MAPE: {metrics.MAPE(y_test, prediction)}\n")
            info.write(f"MAP: {metrics.MPE(y_test, prediction)}\n")
        info.write(f"2. Info about training phase\n")
        info.write(f"Working time: {working_time}\n")
        info.write(f"Loss: \n")
        info.write(f"{[loss for iteration, loss in enumerate(loss_history) if iteration % 100 == 0]}\n")
        info.write(f"3. Info about feature importance\n")
        most_important = list(filter(lambda weight: weight > lr.get_weights().mean(), lr.get_weights()))
        important_features_indices = [index for index, weight in enumerate(lr.get_weights()) if
                                      weight in most_important]
        info.write(f"{data_train[0, important_features_indices]}")

    with pathlib.Path.open(output_model, 'w') as model:
        model.write(f"1. Type: {args.task}\n")
        model.write(f"2. Best hyperparameters:\n")
        model.write(f"Hyperparameters: {sch.get_best_parameters()}\n")
        model.write(f"Best Score by metric {sch.get_metric()}: {sch.get_best_score()}\n")
        model.write(f"3. Weights:\n")
        model.write(f"{lr.get_weights()}\n")


if __name__ == '__main__':
    main()
