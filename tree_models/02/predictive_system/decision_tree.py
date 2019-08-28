from sklearn.tree import DecisionTreeRegressor
from preprocess import Dataset
from sklearn.metrics import mean_absolute_error
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split


def get_preprocessed_data():
    data = Dataset()
    data.get_text_features()
    data.encode_labels()
    data.encode_text()
    data.remove_outliers()
    data.encode_text()
    data.drop_columns()

    data.one_hot_encode()
    data.replace_nans()
    data.to_numpy()
    return (data.data, data.target)


def train_model(train, target, test_size=0.25, save=True):
    if save:
        path = Path('models/')
        path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, test_size=test_size, random_state=30
    )
    regressor = DecisionTreeRegressor(
        criterion='mae', random_state=30, max_depth=200, max_features=200
    )
    regressor.fit(X_train, y_train)
    score = np.zeros((2))
    score[0] = mean_absolute_error(y_train, regressor.predict(X_train))
    score[1] = mean_absolute_error(y_test, regressor.predict(X_test))
    print(f"Train MAE: {score[0]}; Test MAE: {score[1]}")


def run():
    train, target = get_preprocessed_data()
    train_model(train, target)


if __name__ == '__main__':
    run()
