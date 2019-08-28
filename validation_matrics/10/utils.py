import requests
import logging
from pathlib import Path
import pandas as pd
from metrics import Metrics


def split_to_X_and_y(df, target_name):
    y = df[target_name]
    X = df.loc[:, df.columns != target_name]

    return X, y


data_path = Path(Path(__file__).parent) / 'data'
data_path.mkdir(exist_ok=True)


def get_dataset(path):
    if path.startswith('http'):
        try:
            text = requests.get(path).text
            with open(data_path / 'dataset.csv', 'w+', encoding='utf-8') as dataset_file:
                dataset_file.write(text)
        except Exception as e:
            logging.warning(f'Cannot download file via URL: {path}. Error: {e}')
        df = pd.read_csv(data_path / 'dataset.csv')
    else:
        df = pd.read_csv(path)

    return df


def randomize(df):
    return df.sample(frac=1, random_state=42)


def get_fold(df, k):
    N = df.shape[0]
    fold_size = N // k

    for fold_num in range(k):
        start = fold_num * fold_size
        end = min((fold_num + 1) * fold_size, N)
        train = pd.concat([df[0: start], df[end:]])
        test = df[start: end]

        yield train, test


def get_scores(task, y_true, y_pred):
    mapTaskToMetric = {
        'classification': [
            'f1_score',
            'precision_score',
            'recall_score',
        ],
        'regression': [
            'mean_squared_error',
            'mean_squared_log_error',
            'mean_absolute_error',
            'median_absolute_error',
            'r2_score',
            'explained_variance_score',
        ],
    }

    metrics_output = {}

    for metric in mapTaskToMetric[task]:
        metrics_output[metric] = getattr(Metrics(), metric)(y_true, y_pred)

    return metrics_output
