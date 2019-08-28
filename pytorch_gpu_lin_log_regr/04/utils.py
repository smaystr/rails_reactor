import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob
import torch
import time
import json
import settings


def load_config():
    with open(settings.CONFIG_PATH) as config_file:
        data = json.load(config_file)
    return data


config = load_config()


def accuracy(Y_pred, Y_true):
    pred_classes = (Y_pred > 0.5).float()
    correct = (pred_classes == Y_true).float().sum()
    return correct / Y_true.shape[0]


def preprocess_data(full_df, categorical_columns):

    num_columns = list(set(full_df.columns) - set(categorical_columns))

    scaler = StandardScaler()

    full_df[num_columns] = scaler.fit_transform(full_df[num_columns])

    full_df_encoded = pd.get_dummies(full_df, columns=categorical_columns)

    return full_df_encoded


def prepare_data(path_name, categorical_columns, target_column="target"):

    paths_for_csv = sorted(
        glob.glob(settings.DATA_PATH + path_name + "*.csv"), reverse=True
    )

    if not len(paths_for_csv):
        raise FileNotFoundError(f"No .csv files in specified dir {settings.DATA_PATH}")

    print(f"Files will be parsed {paths_for_csv}")
    train_df, test_df = [pd.read_csv(path) for path in paths_for_csv]

    full_df = pd.concat((train_df, test_df)).reset_index().drop(["index"], axis=1)

    targets = full_df[[target_column]].values

    full_df.drop([target_column], axis=1, inplace=True)

    preprocessed_data = preprocess_data(full_df, categorical_columns)

    X_train, X_test, Y_train, Y_test = (
        preprocessed_data[: len(train_df)],
        preprocessed_data[len(train_df) :],
        targets[: len(train_df)],
        targets[len(train_df) :],
    )

    X_train, X_test = map(
        lambda x: torch.from_numpy(x.values).float(), [X_train, X_test]
    )

    Y_train, Y_test = map(lambda x: torch.from_numpy(x).float(), [Y_train, Y_test])

    return X_train, X_test, Y_train, Y_test


def load_data_for_classification():
    cat_columns_heart_df = ["sex", "exang", "slope", "fbs", "cp", "restecg", "thal"]
    return prepare_data(config.get("DATA_PATH_CLASSIFICATION"), cat_columns_heart_df)


def load_data_for_regression():
    cat_columns_insurance = ["region", "sex", "smoker"]
    return prepare_data(
        config.get("DATA_PATH_REGRESSION"),
        cat_columns_insurance,
        target_column="charges",
    )


def rmse(Y_pred, Y_true):
    return torch.mean(torch.sqrt(torch.pow(Y_pred - Y_true, 2)))


def timeit(method):

    """
    By virtue of https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print(f"{method.__name__} {(te - ts) * 1000:.4f} ms elapsed")
        return result

    return timed
