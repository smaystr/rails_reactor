import requests
import numpy as np
import tempfile
from pathlib import Path
import preprocessing as prep
import hp_search as hps
import cross_val as cv
import models
from argument_parser import get_arguments


def load_dataset(path):
    if path.startswith("http"):
        response = requests.get(path)
        if response.status_code != 200:
            raise RuntimeError(f"Got error: {response.text}")
        content = response.content.decode("utf-8")
        name = next(tempfile._get_candidate_names())
        file_path = Path(f"{tempfile._get_default_tempdir()}/{name}")
        file_path.write_text(content, "utf-8")
    else:
        file_path = Path(path)
    features = prep.read_feature_names(file_path, skip_last=False)
    data = prep.read_data(file_path, X_Y=False, dtype=str)
    return features, data


def split_types(dataset):
    numerical_feats = []
    categorical_feats = []
    boolean_feats = []
    for col in range(dataset.shape[0]):
        uniques = np.unique(dataset[:, col])
        if uniques.shape[0] <= 2:
            boolean_feats.append(col)
        else:
            if uniques.shape <= 6:
                categorical_feats.append(col)
            else:
                numerical_feats.append(col)

    return numerical_feats, categorical_feats, boolean_feats


def clean_data(args, features, data):
    target = np.copy(data[:, features == args.target]).astype(np.float32)
    data = np.copy(data[:, features != args.target])
    features = features[features != args.target]

    if args.time_column:
        time = np.copy(data[:, features == args.time_column])
        time = time.astype(np.datetime64)
        data = np.copy(data[:, features != args.time_column])
        features = features[features != args.time_column]
    else:
        time = None

    numerical, categorical, boolean = split_types(data)
    data, _ = prep.to_numeric_multiple(data, categorical + boolean)
    data, features, _ = prep.onehot_columns(
        data.astype(np.float32), categorical, features)
    data, _ = prep.standardize_columns(data, numerical)

    return data, target, features, time


def create_validator(args, data, dates=None):
    if args.validation == "train_test":
        if args.time_column:
            validator = cv.TimeTrainTest(dates, args.test_size)
        else:
            validator = cv.TrainTestCV(data, args.test_size)
    elif args.validation == "k_fold":
        if args.time_column:
            validator = cv.TimeKFoldCV(dates, args.n_splits)
        else:
            validator = cv.KFoldCV(data, args.n_splits)
    else:
        if args.time_column:
            validator = cv.TimeLeaveOneOutCV(dates)
        else:
            validator = cv.LeaveOneOutCV(data)
    return validator


def from_dist(params):
    result = [1]
    if params[0] == "uniform":
        result = np.random.uniform(params[1], params[2], params[3])
    elif params[0] == "normal":
        result = np.random.normal(params[1], params[2], params[3])
    return result


def create_searcher(args, model, validator):
    scoring = "f_score" if args.task == "classification" else "rmse"
    if args.param_search == "grid":
        param_grid = {"lr": args.lr, "epoch": args.epoch, "penalty": args.penalty, "C": args.C}
        searcher = hps.GridSearchCV(model, param_grid, validator, scoring)
    else:
        param_grid = {"lr": from_dist(args.lr), "epoch": from_dist(args.epoch), "penalty": args.penalty,
                      "C": from_dist(args.C)}
        searcher = hps.RandomSearchCV(model, param_grid, validator, scoring)
    return searcher


def calc_metrics(model, X, y):
    metrics = dict()
    for k in model.metrics.keys():
        metrics[k.upper()] = model.score(X, y, metric=k)
    return metrics


def write_info(path, model, features, X, y):
    info_string = ""
    for k, v in calc_metrics(model, X, y).items():
        info_string += "{}: {0:.5f}\n".format(k, v)
    info_string += f"\nTIME: {model.time_}\tLOSS: {model.loss_}\n"
    for k, v in prep.feature_importance(model.w):
        info_string += f"\nFEATURE {features[int(k)]} with importance {v}\n"

    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "output.txt").write_text(info_string, "utf-8")


if __name__ == "__main__":
    np.random.seed(42)
    args = get_arguments()
    features, data = load_dataset(args.dataset)

    data, target, features, time = clean_data(args, features, data)
    if time:
        train_test = list(cv.TimeTrainTestCV(time, 1))[0]
    else:
        train_test = list(cv.TrainTestCV(data, args.test_size))[0]
    train_ind, test_ind = train_test[0], train_test[1]

    if args.task == "classification":
        model = models.LogisticRegression()
    else:
        model = models.LinearRegression()

    validator = create_validator(args, time if time else data)
    searcher = create_searcher(args, model, validator)
    searcher.fit(data[train_ind], target[train_ind])
    model = searcher.best_estimator_
    write_info(args.output, model, features, data[test_ind], target[test_ind])
    model.serialize(args.output)
