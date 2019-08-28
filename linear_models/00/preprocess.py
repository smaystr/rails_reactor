import logging
import pandas as pd
import numpy as np

from linearregression import LinearRegression
from logisticregression import LogisticRegression
from pathlib import Path
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

def replace_column_values(df):
    df.replace("?", np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)


def get_xy(df, target):
    replace_column_values(df)
    return df.drop(target, axis=1), df.loc[:, [target]]


def normalize(train, test, numeric_columns):
    train_n = train[numeric_columns]
    test_n = test[numeric_columns]

    train[numeric_columns] = (train_n - np.min(train_n)) / (np.max(train_n) - np.min(train_n)).values
    test[numeric_columns] = (test_n - np.min(train_n)) / (np.max(train_n) - np.min(train_n)).values

    return train, test

def get_path(train_data, test_data):
    data_path = Path('data')
    return pd.read_csv(data_path.joinpath(train_data), low_memory=False, na_values=['nan', '?']), \
                        pd.read_csv(data_path.joinpath(test_data), low_memory=False, na_values=['nan', '?'])

def linearregression_preprocess():
    df_train, df_test = get_path('insurance_train.csv', 'insurance_test.csv')

    df_train['sex'].replace(['female', 'male'], [0, 1], inplace=True)
    df_test['sex'].replace(['female', 'male'], [0, 1], inplace=True)
    df_train['smoker'].replace(['no', 'yes'], [0, 1], inplace=True)
    df_test['smoker'].replace(['no', 'yes'], [0, 1], inplace=True)

    df_train = pd.concat([df_train, pd.get_dummies(df_train['region'])], axis=1)
    df_test = pd.concat([df_test, pd.get_dummies(df_test['region'])], axis=1)
    df_train.drop(['region'], axis=1, inplace=True)
    df_test.drop(['region'], axis=1, inplace=True)

    x_train, y_train = get_xy(df_train, 'charges')
    x_test, y_test = get_xy(df_test, 'charges')

    x_train, x_test = normalize(x_train, x_test, ['age', 'bmi', 'children'])
    y_train, y_test = normalize(y_train, y_test, ['charges'])
    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values

    model = LinearRegression(1e-3, 1000)
    model.fit(x_train, y_train)

    y_predict_train, y_predict_test = model.predict(x_train, y_train), model.predict(x_test, y_test)

    train_mse, test_mse = model.calculate_mse(y_train, y_predict_train), model.calculate_mse(y_test, y_predict_test)
    logger.info(f'MSE on train set: {train_mse :.4f} and on test set: {test_mse :.4f}')


def logisticregression_preprocess():
    df_train, df_test = get_path('heart_train.csv', 'heart_test.csv')


    x_train, y_train = get_xy(df_train, 'target')
    x_test, y_test = get_xy(df_test, 'target')

    x_train, x_test = normalize(x_train, x_test, ['ca', 'age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
    # y_train, y_test = normalize(y_train, y_test, ['target'])
    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values

    model = LogisticRegression(1, 100)
    fit = model.fit(x_train, y_train)

    logger.info(f'Accuracy on train set: {fit.accuracy(x_train, y_train) :.4f} '
                f'on test set: {model.accuracy(x_test, y_test) :.4f}')


def get_arguments():
    parser = ArgumentParser(description='Linear models')
    parser.add_argument("--num", type=int, metavar="N",
                        default=0, help="number of model: 0 - Linear, 1 - Logistic regression",
                        required=True)

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pd.set_option('display.expand_frame_repr', False)

    args = get_arguments()

    linearregression_preprocess() if args.num == 0 else logisticregression_preprocess()


if __name__ == '__main__':
    main()