from pathlib import Path
from linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as SKLR
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

data_path = Path(Path(__file__).parent) / 'data'


def get_X_y(filename):
    data = pd.read_csv(data_path / filename)
    data['sex_oh'] = LabelBinarizer().fit_transform(data['sex'])
    data['smoker_oh'] = LabelBinarizer().fit_transform(data['smoker'])
    X = data[['sex_oh', 'age', 'bmi', 'children', 'smoker_oh']]
    y = data['charges']

    return X, y

X_train, y_train = get_X_y('insurance_train.csv')
X_test, y_test = get_X_y('insurance_test.csv')
#
clf = LinearRegression().fit(X_train, y_train, epochs=1000)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)


sklearn_clf = SKLR().fit(X_train, y_train)
y_pred_sklearn = sklearn_clf.predict(X_test)
y_pred_sklearn_train = sklearn_clf.predict(X_train)

print('sklearn LR coef', sklearn_clf.coef_)

mse_own_impelementaion = mean_squared_error(y_pred, y_test)
mse_sklearn_impelementaion = mean_squared_error(y_pred_sklearn, y_test)
diff = (mse_own_impelementaion - mse_sklearn_impelementaion) / mse_sklearn_impelementaion

print('own implementation MSE', mse_own_impelementaion)
print('sklearn prediction MSE', mse_sklearn_impelementaion)
print(f'MSE diff {round(diff)} %' )

mse_own_impelementaion_train = mean_squared_error(y_pred_train, y_train)
mse_sklearn_impelementaion_train = mean_squared_error(y_pred_sklearn_train, y_train)
diff_train = (mse_own_impelementaion_train - mse_sklearn_impelementaion_train) / mse_sklearn_impelementaion_train

print('own implementation MSE train', mse_own_impelementaion_train)
print('sklearn prediction MSE train', mse_sklearn_impelementaion_train)
print(f'MSE diff train {round(diff_train)} %' )



