import argparse
import numpy as np
from sklearn.metrics import precision_score
from logistic_regression import LogisticRegression
from utils import prepare_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression on Heart Disease UCI Dataset')
    parser.add_argument('--tol', help='Tolerance for stopping criteria', default=0.0001)
    parser.add_argument('--fit_intercept', help='Specifies if bias be added to the decision function.', default=True)
    parser.add_argument('--max_iter', help='Maximum number of iterations for gradient descent', default=10000)
    args = parser.parse_args()
    train = np.genfromtxt('heart_train.csv', delimiter=',')
    test = np.genfromtxt('heart_test.csv', delimiter=',')
    X_train = train[1:, :-1]
    y_train = train[1:, -1]
    X_test = test[1:, :-1]
    y_test = test[1:, -1]
    clf = LogisticRegression(tol=args.tol, fit_intercept=args.fit_intercept, max_iter=args.max_iter)
    clf.fit(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    train_accuracy = clf.score(X_train, y_train)
    X_test, y_test = prepare_data(X_test, y_test, args.fit_intercept)
    X_train, y_train = prepare_data(X_train, y_train, args.fit_intercept)
    test_pred = clf.predict(X_test)
    train_pred = clf.predict(X_train)
    test_precision = precision_score(y_test, test_pred)
    train_precision = precision_score(y_train, train_pred)
    print(f'Train accuracy for this model: {train_accuracy}')
    print(f'Test accuracy for this model: {test_accuracy}')
    print(f'Train precision for this model: {train_precision}')
    print(f'Test precision for this model: {test_precision}')
