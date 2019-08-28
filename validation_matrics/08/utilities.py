import argparse
import logging
import pathlib
import traceback

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).parent

PARAMS = {
    "C": [0.01, 0.1, 0.5, 0.05],
    "num_iterations": [2000, 3000, 4000, 1000],
    "learning_rate": [0.01, 0.1, 0.5, 0.05]
}


def parse_args():
    """
    Program argument parsing
    :return: all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Fourth homework. Models with metrics, etc.')
    parser.add_argument('--dataset',
                        help='Path or url to the dataset.',
                        type=str,
                        required=True)
    parser.add_argument('--target', '--T',
                        help='Define target column in csv file, default "target"',
                        type=str,
                        required=True)
    parser.add_argument('--task',
                        help='Define model task: linear/regression',
                        type=str,
                        required=True)
    parser.add_argument('--path', '--P',
                        help='Path folder for output model',
                        type=str,
                        required=True)
    parser.add_argument('--split_type',
                        help='parameter for validation split type: k-fold/leave one-out',
                        type=str,
                        default='k-fold')
    parser.add_argument('--split_size',
                        help='parameter for validation split size',
                        type=int,
                        default=5)
    parser.add_argument('--time_series',
                        help='parameter for specifying time series column to perform timeseries validation',
                        type=str,
                        default=None)
    parser.add_argument('--algo',
                        help='parameter for hyperparameter fitting algo: grid search/random search (also add '
                             'parameter for this)',
                        type=str,
                        default='grid')
    parser.add_argument('--categorical',
                        help='model categorical values passed as one string comma-separated without spaces',
                        type=str,
                        default=None)
    parser.add_argument('--na',
                        help='model na values that will be fit with median passed as one string comma-separated without'
                             ' spaces',
                        type=str,
                        default=None)
    parser.add_argument('--verbose', '--V',
                        help='Printing whole program log to the console.',
                        action='store_true')
    parser.add_argument('--log',
                        help='Path to log file.',
                        type=str,
                        default='model.log')

    return parser.parse_args()


def set_up_logging(log_file: str, verbose: bool):
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        filename=log_file,
                        filemode='a')
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

    logging.info('ARGS PARSED, LOGGING CONFIGURED.')


def read_file(dataset_path: str, target: str, na: str, categorical: str):
    dataset_path = pathlib.Path(dataset_path)
    if not (dataset_path.exists() and dataset_path.is_file()):
        try:
            _download_file(url=dataset_path)
        except Exception as e:
            tb = traceback.TracebackException.from_exception(e)
            logging.error(''.join(tb.format()))
        dataset_path = pathlib.Path(PROJECT_ROOT / dataset_path.name)

    data = pd.read_csv(dataset_path).reset_index().drop(["index"], axis=1)

    logging.info(f'DATASET LOADED FROM {dataset_path}, target: {target}, NA: {na}, categorical: {categorical}')

    targets = data[[target]].values
    data.drop([target], axis=1, inplace=True)

    if na:
        for n in na.split(','):
            median = data[n].median()
            data[n] = data[n].fillna(median)

    standard_scaler = StandardScaler()

    if categorical:
        num_columns = list(set(data.columns) - set(categorical.split(',')))

        data[num_columns] = standard_scaler.fit_transform(data[num_columns])
        data = pd.get_dummies(data, columns=categorical.split(','))
        columns = data.columns
        data = np.asarray(data)
    else:
        data = pd.get_dummies(data)
        columns = data.columns
        data = standard_scaler.fit_transform(data)

    return data, targets, columns


def _download_file(url: pathlib.Path):
    logging.info(f'Downloading file from {url}')
    file_name = url.name
    r = requests.get(url=str(url))
    if r.status_code == 200:
        with open(PROJECT_ROOT / file_name, 'wb') as f:
            f.write(r.content)
            logging.info(f'File {file_name} successfully downloaded')
    else:
        logging.warning(f'Status code of request is not 200 OK. URL: {url}')
