import argparse
import numpy as np
from linear_regression import LinearRegression
from utils import mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression on Medical Cost Personal Dataset')
    parser.add_argument('--fit_intercept', help='Specifies if bias be added to the decision function.', default=True)
    train = np.genfromtxt('insurance_train.csv', delimiter=',')
    test = np.genfromtxt('insurance_test.csv', delimiter=',')
    X_train = train[1:, [0, 2, 3]]  # for this task i'm only using numerical features(age, bmi, children)
    y_train = train[1:, 6]
    X_test = test[1:, [0, 2, 3]]
    y_test = test[1:, 6]
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred_test = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    test_mean_squared = mse(y_test, y_pred_test)
    train_mean_squared = mse(y_train, y_pred_train)
    print(f'Train MSE for this model: {train_mean_squared}')
    print(f'Test MSE for this model: {test_mean_squared}')
