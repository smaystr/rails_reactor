import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from logistic import LogisticRegression
from linear import LinearRegression

def linear():
    train = pd.read_csv('insurance_train.csv')
    test = pd.read_csv('insurance_test.csv')

    gender = {'male': 1,'female': 0}
    train.sex = train.sex.map(gender)
    test.sex = test.sex.map(gender)

    smoker = {'yes': 1,'no': 0}
    train.smoker = train.smoker.map(smoker)
    test.smoker = test.smoker.map(smoker)

    train = pd.concat([train.drop('region', axis=1), pd.get_dummies(train['region'])], axis=1)
    test = pd.concat([test.drop('region', axis=1), pd.get_dummies(test['region'])], axis=1)
    columns = list(train.drop('charges', axis=1).columns)
    
    y_train = train['charges']
    x_train = train.drop('charges', axis=1).values
    y_test = test['charges']
    x_test = test.drop('charges', axis=1).values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    print('MSE train: {}'.format(mean_squared_error(y_train, linear_reg.predict(x_train))))
    print('MSE test: {}'.format(mean_squared_error(y_test, linear_reg.predict(x_test))))
    print('======'*10)

def logistic():
    train = pd.read_csv('heart_train.csv')
    test = pd.read_csv('heart_test.csv')

    y_train = train['target'].values
    x_train = train.drop('target', axis=1).values
    y_test = test['target'].values
    x_test = test.drop('target', axis=1).values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    log_regression = LogisticRegression(learning_rate=0.01, num_iterations=10000)
    log_regression.fit(x_train, y_train)
    print('Accuracy test: {}'.format(accuracy_score(y_test, log_regression.predict(x_test, 0.5))))
    print('Accuracy train: {}'.format(accuracy_score(y_train, log_regression.predict(x_train, 0.5))))

if __name__ == '__main__':
    print('Linear model')
    linear()

    print('Logistic model')
    logistic()