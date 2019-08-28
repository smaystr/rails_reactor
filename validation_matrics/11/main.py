import numpy as np
import argparse
import time
import models
import utils
import validate
import param_optimization
from pathlib import Path
import json
import metrics


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to dataset", type=Path)
    parser.add_argument("--target", help="Target column name", type=str)
    parser.add_argument(
        "--task",
        help="Type of task. Classification/regression",
        type=str,
        choices=("classification", "regression"),
    )
    parser.add_argument(
        "--validation",
        help="Type of validation.",
        choices=("train-test-split", "kfold", "leave-one-out"),
    )
    parser.add_argument(
        "--split_size", help="Percentage of data to validate on.", type=float
    )
    parser.add_argument(
        "--number_of_folds", help="Number of folds for kfold.", type=int
    )
    parser.add_argument(
        "--t_series", help="Time series column.", default=None, type=str
    )
    parser.add_argument(
        "--hyp_opt",
        help="Hyperparameter fitting algorithm",
        default="grid-search",
        type=str,
        choices=("grid-search", "random-search"),
    )
    parser.add_argument(
        "--hyp_opt_range",
        help='Parameters and values to iterate over "{"lr":[...],..}',
        default='{"lr":[0.01,0.02,0.03],"max_iter":[200,400,600]}',
        type=str,
    )
    return parser.parse_args()


def model_param_check(params):
    expected_types = {
        "fit_intercept": bool,
        "lr": float,
        "tol": float,
        "max_iter": int,
        "verbose_rounds": int,
        "verbose": bool,
        "eps": float,
        "threshold": float,
    }
    for key in params.keys():
        if isinstance(params[key], list):
            for parameter in params[key]:
                if not isinstance(parameter, expected_types[key]):
                    raise Exception(
                        f"Got {parameter}, expected of type {expected_types[key]}"
                    )
        else:
            if not isinstance(params[key], expected_types[key]):
                raise Exception(
                    f"Got {params[key]}, expected of type {expected_types[key]}"
                )


def get_models(args):
    if args.task == "classification":
        model = models.LogisticRegression()
    else:
        model = models.LinearRegression()
    if args.validation == "kfold":
        validated = validate.KFold
    elif args.validation == "leave-one-out":
        if args.t_series:
            raise Exception("Only k-fold or train-test-split for time series")
        validated = validate.LeaveOneOut
    else:
        validated = validate.train_test_split
    if args.hyp_opt == "grid-search":
        hyp_opt = param_optimization.GridSearch
    else:
        hyp_opt = param_optimization.RandomSearch
    time_series = False
    if args.t_series is not None:
        time_series = True
    return (model, validated, hyp_opt, time_series)


def read_and_preprocess_csv(PATH, time_col=None):
    try:
        strings = np.genfromtxt(PATH, delimiter=",", dtype=str)
        columns = strings[0, :]
        floats = np.genfromtxt(PATH, delimiter=",", dtype=float)[1:, :]
    except:
        raise Exception(f"{PATH} not found or cannot be read.")
    str_ind = [key for key, val in enumerate(floats.T) if np.isnan(val).all()]

    if time_col is not None:

        if time_col not in set(columns):
            raise Exception(f"Time column {time_col} not found.")
        time_column_index = np.where(columns == time_col)
        string_cols = strings[0, :].copy()
        strings = np.delete(strings, 0, axis=0)
        time = np.copy(strings[:, time_column_index])

        time = time.reshape((len(time),))
        strings, floats = (
            np.delete(strings, time_column_index, axis=1),
            np.delete(floats, time_column_index, axis=1),
        )
        try:
            time = time.astype("datetime64")
        except:
            raise Exception(f"Datetime has incorrect format like {time[0]}")

        strings, floats = (
            strings[np.argsort(time, axis=0)],
            floats[np.argsort(time, axis=0)],
        )
        strings = np.vstack((string_cols, strings))

    encoder = utils.LabelEncoder()
    floats[:, str_ind] = encoder.fit_transform(
        strings[1:, str_ind], strings[0, str_ind]
    )

    return floats, columns


def remove_nans(data):
    rows_nan = [key for key, val in enumerate(data) if np.isnan(val).any()]
    return rows_nan


def split_features_target(data, columns, target_col):
    try:
        target = np.where(columns == target_col)[0][0]
    except IndexError:
        raise Exception(f"Target column {target_col} not found.")

    xs = list(set(range(len(columns))) - set([target]))

    return data[:, xs], data[:, target].reshape(len(data), 1)


def parse_parameters(params):
    return json.loads(params)


def setup_model(model, params):
    for key in params.keys():
        setattr(model, key, params[key])
    return model


def run():
    start = time.time()
    args = parse()
    model, validation, hype_opt, time_series = get_models(args)

    data, cols = read_and_preprocess_csv(Path(args.path), args.t_series)
    if args.task == "classification":
        metr = [metrics.accuracy, metrics.precision, metrics.recall]
        names = ["accuracy", "precision", "recall"]

    else:
        metr = [metrics.MSE, metrics.MAE, metrics.RMSE]
        names = ["MSE", "MAE", "RMSE"]

    if np.isnan(data).any():
        data = np.delete(remove_nans(data), axis=0)

    features, target = split_features_target(data, cols, args.target)
    scaler = utils.StandardScaler()
    features = scaler.fit_transform(features)

    params = parse_parameters(args.hyp_opt_range)
    model_param_check(params)

    optimize = hype_opt(
        model,
        validate.KFold(shuffle=(not time_series)),
        params,
        metr[0],
        task=args.task,
        jobs=4,
    )

    res = optimize.fit(features, target)

    if args.validation == "train-test-split":
        model = setup_model(model, res["params"])
        X_train, X_test, y_train, y_test = validate.train_test_split(
            features, target, test_size=args.split_size, shuffle=(not time_series)
        )

        model.fit(X_train, y_train)
        history = np.zeros((len(metr), 2))

        for key, score in enumerate(metr):
            history[key] += (
                score(y_train, model.predict(X_train)),
                score(y_test, model.predict(X_test)),
            )

    else:
        if args.t_series:
            validation = validate.TimeKFold(folds=int(args.number_of_folds))
        elif args.validation == "kfold":
            validation = validation(
                folds=int(args.number_of_folds), shuffle=(not time_series)
            )
        else:
            validation = validation()

        history = np.zeros((len(metr), 2))
        for fold, (train_idx, val_idx) in enumerate(validation.split(features)):
            model = setup_model(model, res["params"])

            X_train, X_test = features[train_idx], features[val_idx]
            y_train, y_test = target[train_idx], target[val_idx]

            model.fit(X_train, y_train)

            for key, score in enumerate(metr):
                history[key] += (
                    score(y_train, model.predict(X_train)),
                    score(y_test, model.predict(X_test)),
                )

        history /= fold + 1

    with open("model.info", "w+") as file:
        file.write(f"Time spent: {time.time()-start} seconds\n")
        file.write("Metrics\n")
        for key, name in enumerate(names):
            file.write(f"{name} train: {history[key][0]}; test: {history[key][1]}\n")
        file.write("Feature importance:\n")
        start = 0
        if model.fit_intercept:
            file.write(f"Intercept: {model.weights[0]}\n")
        start = 1
        for key, name in enumerate(cols):
            file.write(f"{name}: {model.weights[key+start-1][0]}\n")

    with open("model.model", "w+") as file:
        file.write(f"Type: {args.task}\n")
        file.write(f'Best parameters: {res["params"]}\n')
        file.write(f"Weights:\n")
        file.write(str(model.weights))


if __name__ == "__main__":
    run()
