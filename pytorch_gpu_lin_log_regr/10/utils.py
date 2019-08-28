import argparse
from pathlib import Path
from urllib.parse import urljoin
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


URL_DATA = 'http://ps2.railsreactor.net/datasets/medicine/'


def parse_arguments():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        'mode', type=str, help='specify "GPU" or "CPU" mode',
        choices=['CPU', 'GPU'])
    parser.add_argument('config', type=Path,
                        help='path to model configuration file')

    return parser.parse_args()


class Parameters:
    def __init__(self, config_path):
        with open(config_path) as json_file:
            data = json.load(json_file)

            if len({'model', 'epochs', 'lr', 'batch_size'} - set(data)) != 0:
                raise Exception('incorrect format of json config')

        self.model = str(data['model'])
        self.lr = float(data['lr'])
        self.epochs = int(data['epochs'])
        self.batch_size = int(data['batch_size'])


def download_train_test(train_name, test_name, url):

    train_url = urljoin(url, train_name)
    test_url = urljoin(url, test_name)

    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)

    return train, test


def preprocess_dataset(train, test, numerical, categorical, boolean):

    df = pd.concat((train, test), axis=0)

    # working with numerical data
    scaler = StandardScaler()
    scaler.fit(train[numerical])

    scaled = scaler.transform(df[numerical])
    scaled = pd.DataFrame(scaled, columns=numerical, index=df.index)

    # working with categorical data
    one_hot = pd.get_dummies(df[categorical], columns=categorical)
    res_df = pd.concat([scaled, one_hot, df[boolean]], axis=1)
    return res_df[:train.shape[0]].values, res_df[train.shape[0]:].values


def preprocess_heart(train, test):
    target = 'target'
    y_train = train[target].values.reshape(-1, 1)
    train.drop(target, axis=1, inplace=True)

    y_test = test[target].values.reshape(-1, 1)
    test.drop(target, axis=1, inplace=True)

    numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    categorical = ['cp', 'restecg', 'slope', 'thal']
    boolean = ['sex', 'fbs', 'exang']

    X_train, X_test = preprocess_dataset(
        train, test, numerical, categorical, boolean)

    return X_train, y_train, X_test, y_test


def preprocess_medicalcost(train, test):
    target = 'charges'
    y_train = train[target].values.reshape(-1, 1)
    train.drop(target, axis=1, inplace=True)

    y_test = test[target].values.reshape(-1, 1)
    test.drop(target, axis=1, inplace=True)

    train['sex'] = train['sex'].map({'female': 1, 'male': 0})
    train['smoker'] = train['smoker'].map({'yes': 1, 'no': 0})

    test['sex'] = test['sex'].map({'female': 1, 'male': 0})
    test['smoker'] = test['smoker'].map({'yes': 1, 'no': 0})

    numerical = ['age', 'bmi', 'children']
    categorical = ['region']
    boolean = ['sex', 'smoker']

    X_train, X_test = preprocess_dataset(
        train, test, numerical, categorical, boolean)

    return X_train, y_train, X_test, y_test
