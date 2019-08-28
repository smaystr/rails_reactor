import argparse
import logging
import pathlib

import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

__all__ = ('parse_args', 'Dataset', 'DatasetException')

PROJECT_ROOT = pathlib.Path(__file__).parent


class DatasetException(Exception):
    pass


class StatusCodeWarning(Warning):
    pass


class OneHotEncoder:
    def __init__(self, option_keys):
        length = len(option_keys)
        self.__dict__ = {option_keys[j]: [0 if i != j else 1 for i in range(length)] for j in range(length)}

    def __repr__(self):
        return f'OneHotEncoder({[x for x in self.__dict__.keys()]})'

    def __getitem__(self, item):
        return self.__dict__[item]


class Dataset:

    def __init__(
        self, *, force_download: bool,
        dataset_path: str,
        test_url: str,
        train_url: str,
        categorical_features: list,
        target_column='target'
    ):
        self._force_download = force_download
        self._dataset_path = pathlib.Path(dataset_path)
        self._test_url = test_url
        self._train_url = train_url
        self._categorical_features = categorical_features
        self._target_col = target_column

    def load_dataset(self):
        logging.info(
            f'Preparing dataset from {self._dataset_path} with train url '
            f'{self._train_url} and test url {self._test_url}'
        )
        try:
            _download_file(self._test_url, self._dataset_path, self._force_download)
            _download_file(self._train_url, self._dataset_path, self._force_download)
        except StatusCodeWarning as e:
            logging.error(f'ERROR DURING DOWNLOADING DATASET FILES FROM {self._train_url, self._test_url}')
            raise DatasetException(e)
        self._preprocess_data()
        logging.info(f'All data preprocessed and loaded. Dataset is ready for use.')

    def _preprocess_data(self):
        logging.info(f'Preparing model (parsing, extracting and preprocess data)')

        train_data = pd.read_csv(self._dataset_path / pathlib.Path(self._train_url).name)
        test_data = pd.read_csv(self._dataset_path / pathlib.Path(self._test_url).name)

        full_data = pd.concat((train_data, test_data)).reset_index().drop(["index"], axis=1)

        targets = full_data[[self._target_col]].values
        full_data.drop([self._target_col], axis=1, inplace=True)

        num_columns = list(set(full_data.columns) - set(self._categorical_features))

        standard_scaler = StandardScaler()

        full_data[num_columns] = standard_scaler.fit_transform(full_data[num_columns])
        full_data_encoded = pd.get_dummies(full_data, columns=self._categorical_features)

        self.X_train, self.X_test, self.Y_train, self.Y_test = (
            full_data_encoded[: len(train_data)],
            full_data_encoded[len(train_data):],
            targets[: len(train_data)],
            targets[len(train_data):],
        )


def parse_args() -> dict:
    """
    Program argument parsing
    :return: dict object with all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Third homework. Logistic and linear regression implementation.')
    parser.add_argument('--train_url',
                        help='Link to the train dataset url, default:'
                             'http://ps2.railsreactor.net/datasets/medicine/heart_train.csv',
                        type=str,
                        default='http://ps2.railsreactor.net/datasets/medicine/heart_train.csv')
    parser.add_argument('--test_url',
                        help='Link to the test dataset url, default: '
                             'http://ps2.railsreactor.net/datasets/medicine/heart_test.csv',
                        type=str,
                        default='http://ps2.railsreactor.net/datasets/medicine/heart_test.csv')
    parser.add_argument('--dataset_path',
                        help='Path to the dataset folder inside the project folder, default: dataset',
                        type=str,
                        default='dataset')
    parser.add_argument('--force_download', '--F',
                        help='Force program to re download train and test data, by default it will use files from '
                             'dataset_path folder or download them if they are not exist',
                        action='store_true')
    parser.add_argument('--verbose', '--V',
                        help='Printing whole program log to the console.',
                        action='store_true')
    parser.add_argument('--target', '--T',
                        help='Define target column in csv file, default "target"',
                        type=str,
                        default='target')
    args = vars(parser.parse_args())

    return args


def _download_file(url: str, dataset_path: pathlib.Path, force=False):
    """
    Downloading file by url and saving it in dataset_path
    :param force: force re download file
    :param url: file url
    :param dataset_path: path to where store this dataset file
    :return:
    """
    if not (PROJECT_ROOT / dataset_path).exists():  # check if dataset folder is exists
        (PROJECT_ROOT / dataset_path).mkdir(parents=True, exist_ok=True)
        logging.info('Dataset folder created')
    file_name = pathlib.Path(url).name
    if not (PROJECT_ROOT / dataset_path / file_name).exists() or force:
        logging.info(f'Downloading file from {url} to {dataset_path} with force {force}')
        try:
            r = requests.get(url=url)
            if r.status_code == 200:
                with open(PROJECT_ROOT / dataset_path / file_name, 'wb') as f:
                    f.write(r.content)
                    logging.info(f'File {file_name} successfully downloaded')
            else:
                raise StatusCodeWarning(f'Status code of request is not 200 OK. URL: {url}')
        except requests.exceptions.Timeout:
            logging.info(f'HTTP Timeout error. URL: {url}')
    else:
        logging.info(f'Used existing dataset file for {url} in {dataset_path}/{file_name}')
