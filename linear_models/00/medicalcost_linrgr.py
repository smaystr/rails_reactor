import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

logger = logging.getLogger(__name__)

plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams['font.size'] = 14

class LinearRegression:
    def __init__(self, learning_rate = 0.01, iters = 1000):
        self._learning_rate = learning_rate
        self._iters = iters
        self._theta = None
        self._bias = None

    def initialize_weights(self, data_x):
        # self._theta = np.zeros([1, data_x.shape[1]]).T
        self._theta = np.random.rand(data_x.shape[1], 1)
        self._bias = np.zeros((1,))

    def predicts(self, data_x):
        return (data_x @ self._theta) + self._bias

    def fit(self, x_train, y_train):
        self.initialize_weights(x_train)

        for i in range(self._iters):
            # make normalized predictions
            # find error
            dff = self.predicts(x_train) - y_train

            # compute d/dw and d/db of MSE
            delta_w = np.mean(dff * x_train, axis=0, keepdims=True).T
            delta_b = np.mean(dff)

            # update weights and biases
            self._theta = self._theta - self._learning_rate * delta_w
            self._bias = self._bias - self._learning_rate * delta_b
        return self

    def predict(self, data_x, data_y):
        return self.predicts(data_x) * data_y.std() + data_y.mean()


    @staticmethod
    def calculate_mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


def replace_column_values(data, value):
    # just for fun
    rg_set = set(data[value])
    rg_keys = list(rg_set)
    rg_value = [elem for elem in range(len(rg_set))]

    data[value].replace(rg_keys, rg_value, inplace=True)

def prepare_data(data, *args):
    pd.set_option('display.expand_frame_repr', False)
    data = pd.concat([data, pd.get_dummies(data['region'])], axis=1)
    data.drop(['region'], axis=1, inplace=True)
    [replace_column_values(data, arg) for arg in args]
    data = (data - data.mean()) / data.std()
    x = data.iloc[:, np.r_[:5,-4:0]]
    ones = np.ones([x.shape[0], 1])
    x = np.concatenate((ones, x), axis=1)
    y = data.iloc[:, np.r_[5]].values

    return x, y


def main():
    _format = '%(message)s'
    logging.basicConfig(level=logging.INFO, format=_format)

    data_path = Path('data')
    train_data = 'insurance_train.csv'
    test_data = 'insurance_test.csv'

    df_train, df_test = pd.read_csv(data_path.joinpath(train_data)), pd.read_csv(data_path.joinpath(test_data))

    x_train, y_train = prepare_data(df_train, 'sex', 'smoker')
    x_test, y_test = prepare_data(df_test, 'sex', 'smoker')

    model = LinearRegression(1e-2, 100)
    model.fit(x_train, y_train)

    y_predict_train, y_predict_test = model.predict(x_train, y_train), model.predict(x_test, y_test)

    train_mse, test_mse = model.calculate_mse(y_train, y_predict_train), model.calculate_mse(y_test, y_predict_test)
    logger.info(f'MSE on train: {train_mse :.2f} and on test: {test_mse :.2f}')


if __name__ == '__main__':
    main()
